import json
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agentic_system.genai_layer.generator import InterventionGenerator
from agentic_system.genai_layer.critic import CriticAgent

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    _API_KEY = os.getenv("GEMINI_API_KEY", "")
    if _API_KEY and _API_KEY != "your_gemini_api_key_here":
        _thought_model = genai.GenerativeModel("gemini-1.5-flash")
        _GENAI_AVAILABLE = True
        logger.info("ReActPlanner: Gemini available for reasoning.")
    else:
        _GENAI_AVAILABLE = False
except Exception:
    _GENAI_AVAILABLE = False


@dataclass
class StudentState:
    """S_t representation of the student's current status and context."""
    drift_score: float
    drift_vector: List[float]         # [f_pace, f_lag, f_hesitation, f_volatility]
    dropout_prob: float
    time_to_dropout: int
    context: Dict
    intervention_history: List[Dict]


class ReActPlanner:
    """
    The Agentic AI Planner using the ReAct (Reasoning → Acting → Reflecting) framework.
    When Gemini is available, the Thought phase uses a real LLM. Otherwise it falls
    back to deterministic heuristics.
    """
    def __init__(self):
        self.generator = InterventionGenerator()
        self.critic    = CriticAgent()
        self.strategies = ["micro_nudge", "content_simplification", "schedule_restructure",
                           "peer_sync", "human_escalation"]
        self.strategy_mapping = {
            "micro_nudge":            "Motivation Boost",
            "content_simplification": "Conceptual Breakdown",
            "schedule_restructure":   "Micro-Plan",
            "peer_sync":              "Motivation Boost",
            "human_escalation":       "Human Evaluation"
        }

    def _thought_phase(self, state: StudentState) -> str:
        """
        THOUGHT: Diagnose the root psychological cause from the student's feature vector.
        Uses Gemini when available; falls back to deterministic heuristics.
        """
        f_pace, f_lag, f_hesitation, f_volatility = state.drift_vector

        if _GENAI_AVAILABLE:
            prompt = f"""
You are an expert educational psychologist AI performing a root-cause diagnosis for an at-risk student.

Student behavioural signals:
- Learning Pace Index:      {f_pace:.2f}   (0=stopped, 1=on-track)
- Assignment Lag (days):   {f_lag:.1f}
- Hesitation Time (sec):   {f_hesitation:.1f}
- Volatility Index:        {f_volatility:.2f}
- Drift Score (D_t):       {state.drift_score:.2f}
- Dropout Probability:     {state.dropout_prob:.2%}
- Days Until Predicted Dropout: {state.time_to_dropout}

Based on these signals, reason about the most likely primary root cause of the student's struggle.
Choose ONE of: "Burnout", "Confusion / Cognitive Overload", "Disengagement / Apathy", "General Risk"

Return JSON: {{"thought": "one-sentence reasoning", "root_cause": "<chosen category>"}}
"""
            try:
                response = _thought_model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                data = json.loads(response.text.strip())
                thought = data.get("thought", "")
                root_cause = data.get("root_cause", "General Risk")
                logger.info(f"[Thought] LLM: {thought}  →  {root_cause}")
                print(f"[Thought] {thought}")
                return root_cause
            except Exception as e:
                logger.warning(f"Gemini thought phase failed, using heuristic: {e}")

        # ── Heuristic fallback ──
        if f_volatility > 1.5 and state.drift_score > 2.5:
            return "Burnout"
        elif f_hesitation > 100 or f_lag > 5.0:
            return "Confusion / Cognitive Overload"
        elif f_pace < 0.5:
            return "Disengagement / Apathy"
        return "General Risk"

    def _compute_effectiveness_score(self, history: List[Dict], proposed_strategy: str) -> float:
        """
        Computes the historical composite effectiveness score for a specific strategy.
        Formula: (0.3 * normalized_engagement) + (0.3 * normalized_quiz) + (0.4 * normalized_risk_reduction)
        """
        strategy_logs = [log for log in history if log.get("strategy_used") == proposed_strategy]
        
        if not strategy_logs:
            return 1.0 # Default positive assumption if untried
            
        total_score = 0.0
        for log in strategy_logs:
            # If it's a legacy flat 'success_score', use it. Otherwise, compute composite.
            if "delta_risk_reduction" in log:
                # Normalize dummy values for the formula
                # Assume max risk drop is 1.0, max quiz jump is 100, max engagement jump is 60 mins
                norm_eng = min(1.0, max(0.0, log.get("delta_engagement", 0) / 60.0))
                norm_quiz = min(1.0, max(0.0, log.get("delta_quiz", 0) / 100.0))
                norm_risk = min(1.0, max(0.0, log.get("delta_risk_reduction", 0)))
                
                score = (0.3 * norm_eng) + (0.3 * norm_quiz) + (0.4 * norm_risk)
                total_score += score
            else:
                total_score += log.get("success_score", 1.0)
                
        return total_score / len(strategy_logs)

    def _reflect_phase(self, proposed_strategy: str, state: StudentState) -> str:
        """
        Reflects on the student's intervention history memory block.
        If a strategy failed recently, pivot to a different one.
        """
        history = state.intervention_history
        if not history:
            return proposed_strategy
            
        # 1. Compute aggregate historical effectiveness for this strategy
        historical_score = self._compute_effectiveness_score(history, proposed_strategy)
        
        # 2. Check frequency of attempts
        attempts = len([log for log in history if log.get("strategy_used") == proposed_strategy])
        
        # Policy Adjustment Logic
        if historical_score < 0.4 and attempts >= 1:
            print(f"  [Reflect] Strategy '{proposed_strategy}' has poor historical success ({historical_score:.2f}). Pivoting.")
            pivot_map = {
                "micro_nudge": "peer_sync",
                "peer_sync": "content_simplification",
                "content_simplification": "schedule_restructure",
                "schedule_restructure": "human_escalation"
            }
            return pivot_map.get(proposed_strategy, "human_escalation")
        
        return proposed_strategy

    def _action_phase(self, root_cause: str, state: StudentState) -> Dict:
        """
        Selects the baseline intervention strategy parameters based on root cause.
        """
        if state.time_to_dropout <= 2:  # Critical emergency constraint override
            proposed = "human_escalation"
        elif root_cause == "Burnout":
            proposed = "schedule_restructure"
        elif root_cause == "Confusion / Cognitive Overload":
            proposed = "content_simplification"
        elif root_cause == "Disengagement / Apathy":
            proposed = "micro_nudge"
        else:
            proposed = "micro_nudge"
            
        # Reflection Step: Adapt proposed strategy based on memory
        final_strategy = self._reflect_phase(proposed, state)
        
        action = {
            "strategy": final_strategy,
            "genai_strategy": self.strategy_mapping.get(final_strategy, "Motivation Boost")
        }
        return action

    def execute_react_loop(self, state: StudentState, top_features: list) -> Dict:
        """
        The main Agentic loop: Thought → Reflect & Act → Generate → Critic Validate.
        """
        print("\n--- Agentic ReAct Loop Started ---")

        # 1. THOUGHT — LLM diagnoses root cause
        root_cause = self._thought_phase(state)
        print(f"[Reason] Diagnosed Cause: {root_cause}")

        # 2. REFLECT & ACT — rule-based strategy selection with memory
        action_params = self._action_phase(root_cause, state)
        print(f"[Reflect & Act] Selected Strategy: {action_params['strategy']}")

        # 3. GENERATE — LLM drafts personalised intervention
        payload = self.generator.generate(action_params["genai_strategy"], root_cause, top_features)

        # 4. CRITIC — Multi-Agent Validation Protocol
        student_context = {
            "risk_score":       state.dropout_prob,
            "dropout_type":     root_cause,
            "demographic_group": state.context.get("demographic_group", "unspecified")
        }
        critic_result = self.critic.validate(payload, student_context)
        critic_verdict = critic_result.get("verdict", "pass")
        critic_msg     = critic_result.get("reasoning", "")

        if not critic_result.get("safe_to_deliver", True):
            print(f"[Critic] REJECTED — {critic_msg}")
            payload = {
                "alert": "Safety Override: payload rejected by Critic Agent.",
                "message": "We've noticed some challenges. Please schedule a sync with your advisor.",
                "critic_revision": critic_result.get("suggested_revision", "")
            }
        else:
            print(f"[Critic] APPROVED — {critic_msg}")

            # 5. TRIGGER: Revision notes if confused and high-risk
            if state.dropout_prob > 0.65 and root_cause == "Confusion / Cognitive Overload" and top_features:
                target_topic = top_features[0] if top_features[0] != "general_drift" else "Module 4.2 Concepts"
                payload["revision_notes"] = self.generator.generate_revision_notes(target_topic)
                print(f"[Trigger] Auto-generating revision notes for {target_topic}.")

            # 6. TRIGGER: Adaptive schedule if burnout detected
            if root_cause == "Burnout" and state.drift_vector[3] > 1.5:
                current_topic = top_features[0] if top_features else "Current Module"
                payload["adaptive_schedule"] = self.generator.generate_adaptive_schedule(current_topic)
                print(f"[Trigger] Auto-restructuring weekly schedule.")

        return {
            "root_cause": root_cause,
            "action_parameters": action_params,
            "generated_payload": payload,
            "critic_evaluation": {
                "verdict": critic_verdict,
                "message": critic_msg,
                "safe_to_deliver": critic_result.get("safe_to_deliver", True)
            },
            "metadata": {"time_to_dropout": state.time_to_dropout}
        }

if __name__ == "__main__":
    # Test the Agent with Reflection
    state = StudentState(
        drift_score=3.2,
        drift_vector=[0.2, 5.0, 50.0, 0.4], 
        dropout_prob=0.85,
        time_to_dropout=5,
        context={"student_id": "STU_1001", "current_module": 4},
        intervention_history=[
            {
                "strategy_used": "schedule_restructure", 
                "delta_engagement": 5.0,   # minimal change
                "delta_quiz": 2.0,         # minimal change
                "delta_risk_reduction": 0.05, # minimal drop
                "timestamp": "2026-02-20"
            }
        ]
    )
    
    agent = ReActPlanner()
    result = agent.execute_react_loop(state, ["low_pace"])
    print("\nResult Payload:")
    print(json.dumps(result, indent=2))

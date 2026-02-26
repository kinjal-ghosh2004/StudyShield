import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys
import os
import random

# Ensure we can import from the sibling package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from agentic_system.genai_layer.generator import InterventionGenerator

class CriticAgent:
    """
    Acts as a circuit-breaker to evaluate the Generator Agent's output against
    strict pedagogical and safety bounds.
    """
    def evaluate_safety(self, payload: Dict, strategy: str) -> tuple[bool, str]:
        # Mock constraint check
        # Hard bounds: Tone Sentiment Score > 0.3, No mention of financial constraints, Reading Level < Grade 10
        
        # Simulate a 15% chance for the generated payload to fail the strict safety bounds
        if random.random() > 0.85:
            return False, "Failed Tone Sentiment Score: Flagged for high anxiety-inducing language."
            
        return True, "Passed bounds (Tone > 0.3, Reading Level < Grade 10)"

@dataclass
class StudentState:
    """
    S_t representation of the student's current status and context.
    """
    drift_score: float                # D(t)
    drift_vector: List[float]         # X_t [f_pace, f_lag, f_hesitation, f_volatility]
    dropout_prob: float               # P(Drop)
    time_to_dropout: int              # T_d (days)
    context: Dict                     # C (Demographics, course progression)
    intervention_history: List[Dict]  # H (Past interventions and outcomes)


class ReActPlanner:
    """
    The Agentic AI Planner using ReAct (Reasoning, Acting, Reflecting) framework.
    Interprets risk signals, diagnoses root causes, reflects on past memory, 
    and generates multi-modal payloads.
    """
    def __init__(self):
        self.generator = InterventionGenerator()
        self.critic = CriticAgent()
        
        self.strategies = ["micro_nudge", "content_simplification", "schedule_restructure", "peer_sync", "human_escalation"]
        
        # Mapping generic agent conceptual strategies to GenAI specific prompt strategies
        self.strategy_mapping = {
            "micro_nudge": "Motivation Boost",
            "content_simplification": "Conceptual Breakdown",
            "schedule_restructure": "Micro-Plan",
            "peer_sync": "Motivation Boost",
            "human_escalation": "Human Evaluation"
        }

    def _thought_phase(self, state: StudentState) -> str:
        """
        Diagnoses root psychological cause based on the feature vector X_t.
        Returns the hypothesized root cause.
        """
        f_pace, f_lag, f_hesitation, f_volatility = state.drift_vector
        
        if f_volatility > 1.5 and state.drift_score > 2.5:
            return "Burnout"
        elif f_hesitation > 100 or f_lag > 5.0:
            return "Confusion / Cognitive Overload"
        elif f_pace < 0.5:
            return "Disengagement / Apathy"
        else:
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
        The main Agentic loop combining Reason (Thought), Reflect, Act, and Generation.
        """
        print("\n--- Agentic ReAct Loop Started ---")
        
        # 1. Reason (Diagnose)
        root_cause = self._thought_phase(state)
        print(f"[Reason] Diagnosed Cause: {root_cause}")
        
        # 2. Reflect & Act (Select Strategy)
        action_params = self._action_phase(root_cause, state)
        print(f"[Reflect & Act] Selected Strategy: {action_params['strategy']}")
        
        # 3. Generation (LLM drafts payload)
        payload = self.generator.generate(action_params["genai_strategy"], root_cause, top_features)
        
        # 4. Critic Agent Validation (Multi-Agent Protocol)
        is_safe, critic_msg = self.critic.evaluate_safety(payload, action_params["strategy"])
        if not is_safe:
            print(f"[Trigger] Critic rejected payload: {critic_msg}. Using safe fallback.")
            payload = {
                "alert": "Safety Override: System generated payload rejected by Critic Agent.",
                "message": "We have noticed some challenges in your recent activity. Please schedule a quick sync with your advisor."
            }
        else:
            print(f"[Critic] Payload validated: {critic_msg}")
            
            # 5. Structured Revision Notes Trigger Logic
            # Trigger when dropout probability > 0.65 and it's a conceptual issue
            if state.dropout_prob > 0.65 and root_cause == "Confusion / Cognitive Overload" and top_features:
                # For demo purposes, we treat the first 'feature' as the weak topic
                target_topic = top_features[0] if top_features[0] != "general_drift" else "Module 4.2 Concepts"
                payload["revision_notes"] = self.generator.generate_revision_notes(target_topic)
                print(f"[Trigger] High Risk & Confusion Detected. Auto-generating revision notes for {target_topic}.")
                
            # 6. Personalized Learning Pace Optimizer Trigger
            # Trigger when Burnout is detected alongside high fragmentation/volatility
            if root_cause == "Burnout" and state.drift_vector[3] > 1.5:  # volatility index
                # Extract the presumed current topic from context or features
                current_topic = top_features[0] if top_features else "Current Module"
                payload["adaptive_schedule"] = self.generator.generate_adaptive_schedule(current_topic)
                print(f"[Trigger] Burnout & Fragmentation Detected. Auto-restructuring weekly schedule.")
        
        return {
            "root_cause": root_cause,
            "action_parameters": action_params,
            "generated_payload": payload,
            "critic_evaluation": {"is_safe": is_safe, "message": critic_msg},
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

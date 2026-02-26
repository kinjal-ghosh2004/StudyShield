import numpy as np

from agentic_system.behavioral_drift.drift_detector import BehavioralDriftDetector
from agentic_system.react_planner.agent import ReActPlanner, StudentState
from agentic_system.rl_intervention.environment import ContextualBanditRLEngine

class AgenticDropoutPreventionSystem:
    """
    The orchestrator that wires together the Behavioral Engine, the Agentic Planner, and the RL Subsystem.
    """
    def __init__(self):
        # 1. Initialize Behavioral Drift Monitor
        self.behavioral_engine = BehavioralDriftDetector(alpha=0.3, baseline_window=10)
        
        # 2. Initialize ReAct Agent
        self.react_agent = ReActPlanner()
        
        # 3. Initialize RL Policy Engine
        self.rl_engine = ContextualBanditRLEngine()
        
        # Mocks
        self.mock_survival_engine = lambda xt: max(1, int(20 - np.sum(xt))) # Fake Time-to-dropout
        self.mock_risk_classifier = lambda xt: min(0.99, np.sum(xt) / 10.0) # Fake Dropout Prob %

    def process_student_event(self, student_id: str, new_activity_vector: np.ndarray):
        """
        Main pipeline execution flow triggered by a daily batch or real-time event.
        """
        print(f"\n[PIPELINE] Processing Event for {student_id}")
        
        # Step 1: DRIFT DETECTION
        # (Assuming baseline is already trained)
        drift_score = self.behavioral_engine.update_drift_score(student_id, new_activity_vector)
        zone, action_suggestion = self.behavioral_engine.evaluate_threshold(drift_score)
        print(f"  -> Drift Score: {drift_score:.2f} | {zone}")
        
        if drift_score > 2.5: # Zone 2/3: Triggers Intervention
            print("  -> High Risk Detected! Triggering Agentic Planner...")
            
            # Step 2: GATHER FULL CONTEXT (Predictive Intelligence Layer)
            t_drop = self.mock_survival_engine(new_activity_vector)
            p_drop = self.mock_risk_classifier(new_activity_vector)
            
            state = StudentState(
                drift_score=drift_score,
                drift_vector=new_activity_vector.tolist(),
                dropout_prob=p_drop,
                time_to_dropout=t_drop,
                context={"student_id": student_id, "current_module": 4},
                intervention_history=[]
            )
            
            # Step 3: RL STRATEGY SELECTION
            # RL engine picks the best strategy index for this state context
            # In a real system, the state is embedded into a fixed size vector first.
            state_emb = np.array([drift_score, p_drop, 1.0/t_drop]) 
            action_idx = self.rl_engine.select_action(state_emb, time_to_dropout=t_drop)
            
            # We map RL discrete action to the Agent's strategy name
            rl_strategy = self.react_agent.strategies[action_idx]
            
            # Step 4: REACT LOOP GENERATION
            # Diagnose root cause
            root_cause = self.react_agent._thought_phase(state)
            
            # Format Action Parameters
            action_params = self.react_agent._action_phase(root_cause, state)
            
            # Override Agent's default baseline strategy with the RL Engine's dynamically learned strategy
            # (Unless the Agent overrode it for safety e.g. T_d < 2)
            if state.time_to_dropout >= 2:
                action_params["strategy"] = rl_strategy
                
            payload = self.react_agent._generate_payload(action_params, state, root_cause)
            
            print(f"  -> [AGENT REASONING] Diagnosed Cause: {root_cause}")
            print(f"  -> [RL POLICY] Picked Strategy: {action_params['strategy']}")
            print(f"  -> [GENERATED PAYLOAD] {payload}")
            
            # The system would now send the payload and wait delta T to evaluate the reward...
            # return action taken to log for reward later
            return action_idx, state_emb, t_drop, drift_score
        else:
            print("  -> Student behavior nominal. No intervention required.")
            return None


if __name__ == "__main__":
    system = AgenticDropoutPreventionSystem()
    student = "STU_2042"
    
    # Train mock baseline
    normal_data = np.random.normal(1.0, 0.1, size=(10, 4))
    system.behavioral_engine.train_baseline(student, normal_data)
    
    # Simulate a sudden disengagement drop
    bad_behavior = np.array([0.2, 4.0, 10.0, 2.0]) # low pace, high lag, high volatility
    system.process_student_event(student, bad_behavior)

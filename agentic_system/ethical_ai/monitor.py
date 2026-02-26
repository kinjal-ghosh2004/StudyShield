import json
from datetime import datetime
from typing import Dict, List, Tuple

class EthicalMonitor:
    """
    Governance Layer to ensure fair, transparent, and non-intrusive AI interventions.
    """
    def __init__(self):
        self.transparency_log = []
        self.max_interventions_per_week = 2

    def check_fatigue(self, student_id: str, intervention_history: List[Dict], current_day: int) -> Tuple[bool, str]:
        """
        Prevents over-intervening and causing notification fatigue.
        """
        if not intervention_history:
            return False, "Clear to intervene."
            
        # Mocking time: assuming 'Day X' is mapped to integer current_day
        recent_interventions = [
            i for i in intervention_history 
            if i.get("day", 0) >= (current_day - 7)
        ]
        
        if len(recent_interventions) >= self.max_interventions_per_week:
            return True, f"Fatigue Limit Reached: {len(recent_interventions)} interventions in last 7 days."
            
        return False, "Clear to intervene."

    def log_transparency(self, student_id: str, day: int, risk_score: float, cause: str, strategy: str, features: list):
        """
        Stores an immutable record of why an AI decision was made.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "student_id": student_id,
            "simulated_day": day,
            "risk_score": risk_score,
            "top_contributing_features": features,
            "agent_diagnosed_cause": cause,
            "selected_strategy": strategy
        }
        self.transparency_log.append(log_entry)
        print(f"  [Ethical Monitor] Logged reasoning trace for auditing.")

    def generate_fairness_report(self):
        """
        Mock generation of a batch fairness report aggregating logged decisions. 
        In practice, maps logged predictions to demographic data to find disparate impact.
        """
        return {
            "total_logs": len(self.transparency_log),
            "bias_check": "Pass (Simulated)",
            "false_positive_rate": "0.15 (Simulated)",
            "fatigue_blocks_issued": 0
        }

if __name__ == "__main__":
    monitor = EthicalMonitor()
    is_fatigued, msg = monitor.check_fatigue("STU_1", [{"day": 1, "strategy_used": "micro_nudge"}], 4)
    print("Fatigue Test:", is_fatigued, msg)

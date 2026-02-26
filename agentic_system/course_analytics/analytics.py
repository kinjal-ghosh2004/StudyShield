import json

class CourseIntelligenceModule:
    """
    Analyzes aggregated cohort telemetry to identify structural syllabus failures
    and recommend course-level improvements to faculty.
    """
    def __init__(self):
        # Weights for the Difficulty Impact Formula
        self.w_error = 0.3
        self.w_hesitation = 0.2
        self.w_dropout = 0.5

    def compute_difficulty_score(self, error_rate: float, hesitation_ratio: float, dropout_corr: float) -> float:
        """
        Calculates the composite Difficulty Impact Score [0, 1]
        """
        # Normalize hesitation ratio (assuming baseline is 1.0, max reasonable spike is ~5.0)
        norm_hesitation = min(1.0, (hesitation_ratio - 1.0) / 4.0) if hesitation_ratio > 1.0 else 0.0
        
        score = (self.w_error * error_rate) + (self.w_hesitation * norm_hesitation) + (self.w_dropout * dropout_corr)
        return min(1.0, max(0.0, score))

    def generate_recommendation(self, topic_name: str, error_rate: float, hesitation_ratio: float, dropout_corr: float) -> dict:
        """
        Maps the specific telemetry distribution to pedagogical recommendations.
        """
        impact_score = self.compute_difficulty_score(error_rate, hesitation_ratio, dropout_corr)
        
        # If the impact is minimal, no action needed
        if impact_score < 0.6:
            return {"topic_name": topic_name, "status": "Nominal"}
            
        # Generative mapping logic
        if hesitation_ratio > 3.0:
            action = "Additional Prerequisite Modules"
            rationale = "High hesitation indicates missing foundational knowledge causing cognitive lock-up."
        elif error_rate > 0.7 and hesitation_ratio < 1.5:
            action = "Topic Restructuring & Extra Practice"
            rationale = "Students are moving fast but getting it wrong, implying a fundamental misconception in how the material is phrased."
        else:
            action = "Reduced Content Load"
            rationale = "Broadly high failure and correlation to dropout suggest this module is structurally overloaded."
            
        return {
            "topic_name": topic_name,
            "difficulty_score": round(impact_score, 2),
            "dropout_correlation": round(dropout_corr, 2),
            "suggested_action": action,
            "metrics_breakdown": {
                "error_density": round(error_rate, 2),
                "hesitation_spike": round(hesitation_ratio, 2)
            },
            "generated_rationale": rationale
        }

if __name__ == "__main__":
    analytics = CourseIntelligenceModule()
    
    # Mock aggregated data for a problematic topic
    topic = "Module 4.2: Dynamic Programming"
    # 75% error rate, students pausing 4.5x longer than normal, high dropout correlation (0.8)
    report = analytics.generate_recommendation(topic, error_rate=0.75, hesitation_ratio=4.5, dropout_corr=0.8)
    
    print("\n[Faculty Dashboard] AI Syllabus Architect Report:")
    print(json.dumps(report, indent=2))

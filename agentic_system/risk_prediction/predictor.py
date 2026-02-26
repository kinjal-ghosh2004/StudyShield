import numpy as np

class RiskPredictor:
    """
    Risk Prediction Layer mocking the outputs of XGBoost, LSTM, and Survival Models.
    """
    def __init__(self):
        # Feature names corresponding to the incoming drift vector (Pace, Lag, Volatility, Pace_Var)
        self.feature_names = ["pace", "lag", "volatility", "pace_variance"]
        
        # Efficacy heuristics (expected % reduction in dropout risk) mapped to strategies
        self.efficacy_map = {
            "micro_nudge": 0.15,
            "content_simplification": 0.25,
            "schedule_restructure": 0.35,
            "peer_sync": 0.20,
            "human_escalation": 0.60
        }

    def simulate_intervention_impact(self, base_risk: float, strategy: str) -> dict:
        """
        Calculates the counterfactual risk delta: What happens if we apply this intervention?
        Returns the original risk, the new theoretical risk, and the % reduction.
        """
        efficacy_factor = self.efficacy_map.get(strategy, 0.10)
        
        # Add slight stochastic variance for realism (-5% to +5% of the factor)
        variance = np.random.uniform(-0.05, 0.05)
        applied_efficacy = max(0.01, min(0.99, efficacy_factor + variance))
        
        new_risk = base_risk * (1.0 - applied_efficacy)
        reduction_percentage = ((base_risk - new_risk) / base_risk) * 100 if base_risk > 0 else 0
        
        return {
            "strategy_simulated": strategy,
            "risk_without_intervention": round(base_risk, 4),
            "risk_with_intervention": round(new_risk, 4),
            "risk_reduction_percentage": round(reduction_percentage, 2)
        }

    def classify_dropout_type(self, pace: float, lag: float, hesitation: float, volatility: float, accuracy_decay: float) -> dict:
        """
        Classifies the student into one of 5 dropout archetypes based on a heuristic weighting of features.
        """
        # Feature normalization heuristics
        n_pace = max(0, 1.0 - pace) # Low pace is bad
        n_lag = min(1.0, lag / 10.0)
        n_hes = min(1.0, hesitation / 300.0)
        n_vol = min(1.0, volatility / 3.0)
        n_acc = min(1.0, accuracy_decay)

        # Class scoring (Weights mapping to the design doc matrix)
        scores = {
            "Burnout Dropout": (n_lag * 0.3) + (n_vol * 0.5) + (n_acc * 0.2),
            "Cognitive Overload Dropout": (n_pace * 0.2) + (n_hes * 0.4) + (n_acc * 0.4),
            "Conceptual Confusion Dropout": (n_hes * 0.3) + (n_acc * 0.7),
            "Motivation Dropout": (n_pace * 0.4) + (n_lag * 0.6),
            "Time-Management Dropout": (n_lag * 0.4) + (n_vol * 0.6)
        }

        # Select top class
        primary_class = max(scores, key=scores.get)
        confidence = min(0.99, scores[primary_class] + 0.1) # Boost for demo display
        
        # Determine supporting evidence
        evidence = []
        if n_vol > 0.5: evidence.append(f"High Session Volatility ({volatility:.1f})")
        if n_lag > 0.4: evidence.append(f"Significant Assignment Lag ({lag:.1f} days)")
        if n_hes > 0.5: evidence.append(f"High Hesitation Time ({hesitation:.0f}s)")
        if n_pace > 0.4: evidence.append(f"Pace Dropped (-{n_pace*100:.0f}%)")
        if not evidence: evidence.append("General accuracy decay")

        return {
            "dropout_type": primary_class,
            "confidence_score": round(confidence, 2),
            "supporting_features": evidence
        }

    def predict(self, activity_vector: np.ndarray, drift_score: float) -> dict:
        """
        Takes the current activity vector and the calculated drift score to predict dropout risk.
        """
        # 1. XGBoost Mock: Dropout Probability (0.0 to 1.0)
        # Higher lag and volatility generally increase probability. Lower pace increases probability.
        # activity_vector is [pace, lag, volatility, pace_variance]
        pace, lag, vol, p_var = activity_vector
        
        # Heuristic for probability
        # Base risk heavily tied to drift score, scaled by lag and volatility.
        base_risk = min(0.99, (drift_score * 0.2) + (lag * 0.1) + (vol * 0.05))
        dropout_prob = max(0.01, min(0.99, base_risk))

        # 2. Survival Model Mock: Time-to-dropout (days)
        # Higher probability = fewer days. If doing fine, a large number of days.
        if dropout_prob > 0.8:
            predicted_days = int(np.random.normal(3, 1)) # Drops out in ~3 days
        elif dropout_prob > 0.5:
            predicted_days = int(np.random.normal(10, 2))
        else:
            predicted_days = int(np.random.normal(30, 5))
            
        predicted_days = max(1, predicted_days)

        # 3. LSTM Mock: Engagement Decline Forecast
        # Is the trend getting worse? 
        decline_trend = "Accelerating Decline" if vol > 2.0 else "Stable"

        # 4. Top Contributing Features
        # Determine which feature is causing the biggest issue
        contributions = {
            "lag": lag * 0.5,
            "volatility": vol * 0.3,
            "low_pace": max(0, 1.0 - pace) * 0.6
        }
        # Sort by impact
        sorted_features = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        top_features = [k for k, v in sorted_features if v > 0.5]
        
        if not top_features:
            top_features = ["general_drift"]

        # 5. Dropout Type Classification
        # For demo purposes, we infer 'hesitation' and 'accuracy_decay' from existing features or randomize slightly
        mock_hesitation = p_var * 100.0 # using pace_variance as proxy for hesitation
        mock_acc_decay = min(1.0, vol * 0.2)
        dropout_class = self.classify_dropout_type(pace, lag, mock_hesitation, vol, mock_acc_decay)

        return {
            "risk_score": dropout_prob,
            "predicted_dropout_days": predicted_days,
            "engagement_trend": decline_trend,
            "top_contributing_features": top_features,
            "classification": dropout_class
        }

if __name__ == "__main__":
    # Test the mock
    predictor = RiskPredictor()
    # Bad behavior vector
    risk_info = predictor.predict(np.array([0.2, 4.0, 3.5, 2.0]), 3.8)
    print("Risk Prediction Output:")
    print(risk_info)

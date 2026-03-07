import sys, os, numpy as np
sys.path.insert(0, os.path.abspath('.'))

from agentic_system.risk_prediction.predictor import RiskPredictor

p = RiskPredictor()
print("Models loaded:")
print(f"  XGBoost:   {'YES' if p._xgb_model else 'NO (heuristic)'}")
print(f"  Survival:  {'YES' if p._survival_model else 'NO (heuristic)'}")

print("\n--- Test 1: High-risk student (low pace, high lag, high volatility) ---")
result = p.predict(np.array([0.1, 8.0, 3.5, 2.0]), drift_score=4.2)
print(result)

print("\n--- Test 2: Low-risk student (good pace, no lag, low volatility) ---")
result = p.predict(np.array([0.9, 0.0, 0.2, 0.1]), drift_score=0.3)
print(result)

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI(title="Dropout Prediction API", description="OULAD Predictive Engine serving XGBoost, LSTM, and Survival Ensembles.")

# Define input schema for the API
class StudentState(BaseModel):
    id_student: int
    gender: str
    region: str
    highest_education: str
    imd_band: str
    age_band: str
    sum_click_history: list[int]
    volatility_history: list[float]
    hesitation_history: list[float]
    drift_history: list[float]

# Note: In production, these would be loaded from an MLflow registry or local .pt / .model files
loaded_models = {
    'xgboost': None, 
    'lstm': None,
    'survival': None
}

@app.on_event("startup")
def load_artifacts():
    print("Loading scalers and trained model weights into memory...")
    # e.g., loaded_models['xgb'] = xgb.Booster(model_file='xgb_best.model')

@app.post("/predict")
def predict_dropout_risk(student: StudentState):
    """
    Takes live state data for a student, prepares it into the different formats 
    (Tabular for XGB, 3D Tensor for LSTM) and returns an ensembled risk probability 
    and estimated hazard ratio.
    """
    # 1. Feature Engineering API Side (Process history -> lag features & tensors)
    
    # Placeholder outputs for demonstration
    xgb_risk = np.random.uniform(0, 1)
    lstm_risk = np.random.uniform(0, 1)
    hazard_ratio = np.random.uniform(0.5, 3.0)
    
    # Static Ensemble Weighting
    final_risk_score = (0.6 * xgb_risk) + (0.4 * lstm_risk)
    
    # Threshold interpretation
    is_at_risk = bool(final_risk_score > 0.65)
    
    return {
        "student_id": student.id_student,
        "final_risk_probability": round(final_risk_score, 4),
        "is_at_risk": is_at_risk,
        "hazard_ratio": round(hazard_ratio, 4),
        "model_breakdown": {
            "xgb_contribution": round(xgb_risk, 4),
            "lstm_contribution": round(lstm_risk, 4)
        }
    }

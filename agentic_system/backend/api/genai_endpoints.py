"""
API Endpoint — POST /api/v1/genai/intervene
Chains: Risk Profile → ReAct Plan → Generator → Critic → MongoDB save
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class InterveneRequest(BaseModel):
    student_id: str
    drift_score: float = 2.5
    drift_vector: List[float] = [0.4, 3.0, 80.0, 1.8]  # [pace, lag, hesitation, volatility]
    dropout_prob: float = 0.75
    time_to_dropout: int = 10
    top_features: List[str] = ["volatility", "lag"]
    context: Optional[Dict] = {}
    intervention_history: Optional[List[Dict]] = []


@router.post("/genai/intervene")
async def create_intervention(req: InterveneRequest):
    """
    Runs the full Agentic ReAct loop for a student and returns a structured intervention.
    The output is validated by the CriticAgent before being returned.
    """
    try:
        # Lazy import to avoid circular deps
        from agentic_system.react_planner.agent import ReActPlanner, StudentState

        state = StudentState(
            drift_score=req.drift_score,
            drift_vector=req.drift_vector,
            dropout_prob=req.dropout_prob,
            time_to_dropout=req.time_to_dropout,
            context={"student_id": req.student_id, **(req.context or {})},
            intervention_history=req.intervention_history or []
        )

        planner = ReActPlanner()
        result  = planner.execute_react_loop(state, req.top_features)

        # Persist to MongoDB asynchronously (best-effort)
        try:
            from agentic_system.backend.db.session import get_mongo_db
            db = get_mongo_db()
            await db["interventions"].insert_one({
                "student_id": req.student_id,
                "root_cause": result["root_cause"],
                "strategy":   result["action_parameters"]["strategy"],
                "payload":    result["generated_payload"],
                "critic":     result["critic_evaluation"],
            })
            logger.info(f"Intervention persisted to MongoDB for {req.student_id}")
        except Exception as e:
            logger.warning(f"MongoDB persist skipped: {e}")

        return {
            "student_id":    req.student_id,
            "root_cause":    result["root_cause"],
            "strategy":      result["action_parameters"]["strategy"],
            "intervention":  result["generated_payload"],
            "critic_verdict": result["critic_evaluation"],
        }

    except Exception as e:
        logger.error(f"GenAI intervention failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

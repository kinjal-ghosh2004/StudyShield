from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

# Internal Imports
from agentic_system.backend.db.session import get_db, get_mongo_db, get_influx_client
from agentic_system.backend.models.user_models import User, Course
from agentic_system.backend.schemas.schemas import UserCreate, UserResponse, DBConnectivityResponse
from agentic_system.backend.core.config import settings

router = APIRouter()

@router.get("/diagnostics/db-health", response_model=DBConnectivityResponse)
async def check_db_health(
    pg_db: AsyncSession = Depends(get_db)
):
    """
    Checks connectivity to PostgreSQL, MongoDB, and InfluxDB.
    Useful after running docker-compose up.
    """
    statuses = {
        "postgres": "Unreachable",
        "mongo": "Unreachable",
        "influx": "Unreachable"
    }

    # 1. Check Postgres
    try:
        await pg_db.execute(text("SELECT 1"))
        statuses["postgres"] = "Connected"
    except Exception as e:
        statuses["postgres"] = f"Error: {e}"

    # 2. Check Mongo
    try:
        mongo_db = get_mongo_db()
        await mongo_db.command("ping")
        statuses["mongo"] = "Connected"
    except Exception as e:
        statuses["mongo"] = f"Error: {e}"

    # 3. Check InfluxDB
    try:
        influx_client = get_influx_client()
        is_ready = await influx_client.ping()
        if is_ready:
            statuses["influx"] = "Connected"
        else:
            statuses["influx"] = "Ping Failed (Not Ready)"
    except Exception as e:
        statuses["influx"] = f"Error: {e}"

    return statuses


# --- User Endpoints (Replaces Mock Behavior) ---

@router.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Creates a new student profile in PostgreSQL"""
    
    # Check if student exists
    result = await db.execute(select(User).where(User.student_id == user.student_id))
    db_user = result.scalars().first()
    if db_user:
        raise HTTPException(status_code=400, detail="Student ID already registered")
        
    db_user = User(**user.model_dump())
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@router.get("/users/{student_id}", response_model=UserResponse)
async def get_user(student_id: str, db: AsyncSession = Depends(get_db)):
    """Fetches a student profile from PostgreSQL"""
    result = await db.execute(select(User).where(User.student_id == student_id))
    db_user = result.scalars().first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return db_user

# --- Intervention Endpoints (Mongo Hook Example) ---

@router.post("/interventions/{student_id}/log")
async def log_intervention(student_id: str, payload: dict):
    """
    Writes GenAI interaction logs directly to MongoDB.
    Expects payload: {"strategy": "simplication", "content": "..."}
    """
    mongo_db = get_mongo_db()
    collection = mongo_db["interventions"]
    
    document = {
        "student_id": student_id,
        "payload": payload,
        "timestamp": "Now (will use proper datetime later)"
    }
    
    await collection.insert_one(document)
    return {"status": "Logged successfully to Mongo"}


@router.get("/interventions/{student_id}")
async def get_interventions(student_id: str):
    """Returns the intervention history for a student from MongoDB."""
    try:
        mongo_db = get_mongo_db()
        cursor = mongo_db["interventions"].find(
            {"student_id": student_id}, {"_id": 0}
        ).sort("timestamp", -1).limit(50)
        docs = await cursor.to_list(length=50)
    except Exception:
        docs = []
    return docs


@router.post("/risk_prediction/predict")
async def predict_risk(payload: dict):
    """
    Accepts a student activity vector and drift score, returns ML risk prediction.
    Used by the dashboard to power the Risk Overview panel.
    """
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.abspath('.'))

    from agentic_system.risk_prediction.predictor import RiskPredictor

    try:
        predictor = RiskPredictor()
        activity_vector = np.array(payload.get("activity_vector", [0.5, 1.0, 30.0, 0.5]))
        drift_score = float(payload.get("drift_score", 1.0))
        result = predictor.predict(activity_vector, drift_score)
        return result
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

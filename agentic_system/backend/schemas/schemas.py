from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any

# --- User Schemas ---
class UserBase(BaseModel):
    student_id: str
    name: str
    email: Optional[str] = None
    demographic_group: Optional[str] = None

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

# --- General Responses ---
class CourseHealthResponse(BaseModel):
    course_id: str
    active_cohort_size: int
    at_risk_count: int
    dropout_type_distribution: Dict[str, int]
    intervention_effectiveness: Dict[str, Any]

class DBConnectivityResponse(BaseModel):
    postgres: str
    mongo: str
    influx: str

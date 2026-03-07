from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
from agentic_system.backend.streaming.producer import KafkaProducerManager
import time

router = APIRouter()

class TelemetryEvent(BaseModel):
    student_id: str
    event_type: str # 'video_pause', 'mouse_hesitation', 'login', 'quiz_submit'
    course_id: str
    timestamp: Optional[float] = None
    payload: Dict[str, Any] # e.g., {"duration_seconds": 12, "module": "M_2"}

@router.post("/telemetry/event")
async def ingest_event(event: TelemetryEvent, background_tasks: BackgroundTasks):
    """
    Simulates a webhook or LMS pushing raw interaction telemetry.
    The data is immediately validated and pushed to the Kafka queue for stream processing.
    """
    if event.timestamp is None:
        event.timestamp = time.time()
        
    event_dict = event.model_dump()
    
    # We await the send immediately here for demonstration, 
    # but in extreme high throughput, this could be a background_task.
    success = await KafkaProducerManager.send_event(event_dict)
    
    if not success:
        # In a real system, you might fall back to writing to a local dead-letter file
        raise HTTPException(status_code=500, detail="Failed to write telemetry to stream")
        
    return {"status": "ok", "event_type": event.event_type}

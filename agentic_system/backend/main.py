from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agentic_system.backend.api.endpoints import router as api_router
from agentic_system.backend.api.telemetry_endpoints import router as telemetry_router
from agentic_system.backend.api.genai_endpoints import router as genai_router
from agentic_system.backend.streaming.producer import KafkaProducerManager
from agentic_system.backend.streaming.consumer_worker import KafkaConsumerWorker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    await KafkaProducerManager.start()
    await KafkaConsumerWorker.start()
    yield
    # Shutdown actions
    await KafkaProducerManager.stop()
    await KafkaConsumerWorker.stop()

app = FastAPI(
    title="Dropout Prevention API",
    description="Agentic AI system for identifying and intervening with at-risk students.",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(telemetry_router, prefix="/api/v1")
app.include_router(genai_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Dropout Prevention API"}


@app.get("/dashboard", response_class=None)
async def serve_dashboard():
    """Serves the StudyShield interactive dashboard HTML."""
    import os
    from fastapi.responses import FileResponse
    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "agentic_system", "dashboard.html")
    html_path = os.path.abspath(html_path)
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Dashboard file not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agentic_system.backend.main:app", host="0.0.0.0", port=8000, reload=True)

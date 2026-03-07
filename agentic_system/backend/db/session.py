from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from motor.motor_asyncio import AsyncIOMotorClient
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from agentic_system.backend.core.config import settings
import logging

logger = logging.getLogger(__name__)

# --- PostgreSQL (SQLAlchemy) ---
engine = create_async_engine(settings.async_database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for initial setup/migrations
sync_engine = create_engine(settings.sync_database_url)

async def get_db():
    """Dependency injection to get Postgres DB session in endpoints."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# --- MongoDB (Motor) ---
class MongoDBManager:
    client: AsyncIOMotorClient = None

def get_mongo_db():
    if MongoDBManager.client is None:
        MongoDBManager.client = AsyncIOMotorClient(settings.mongo_uri)
    return MongoDBManager.client["dropout_prevention_docs"]

# --- InfluxDB ---
class InfluxDBManager:
    client: InfluxDBClientAsync = None

def get_influx_client():
    if InfluxDBManager.client is None:
        InfluxDBManager.client = InfluxDBClientAsync(
            url=settings.INFLUXDB_URL, 
            token=settings.INFLUXDB_TOKEN, 
            org=settings.INFLUXDB_ORG
        )
    return InfluxDBManager.client

# Startup/Shutdown hooks can be mapped to these
async def close_connections():
    if MongoDBManager.client:
        MongoDBManager.client.close()
    if InfluxDBManager.client:
        await InfluxDBManager.client.close()

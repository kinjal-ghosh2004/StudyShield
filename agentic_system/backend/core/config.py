import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Dropout Prevention Agentic API"

    # PostgreSQL Database Settings
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "ai_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "ai_password")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5433")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "dropout_prevention")
    
    @property
    def sync_database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # MongoDB Database Settings
    MONGO_USER: str = os.getenv("MONGO_USER", "ai_admin")
    MONGO_PASSWORD: str = os.getenv("MONGO_PASSWORD", "ai_admin_password")
    MONGO_SERVER: str = os.getenv("MONGO_SERVER", "localhost")
    MONGO_PORT: str = os.getenv("MONGO_PORT", "27017")
    
    @property
    def mongo_uri(self) -> str:
        return f"mongodb://{self.MONGO_USER}:{self.MONGO_PASSWORD}@{self.MONGO_SERVER}:{self.MONGO_PORT}/"
    
    # InfluxDB Database Settings
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "supersecret_influx_token_123")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "agentic_ai")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "student_telemetry")

settings = Settings()

from agentic_system.backend.models.user_models import Base
from agentic_system.backend.db.session import sync_engine

print("Creating database tables...")
Base.metadata.create_all(bind=sync_engine)
print("Tables created successfully.")

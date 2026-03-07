from fastapi.testclient import TestClient
from agentic_system.backend.main import app
import uuid

client = TestClient(app)

student_id = f"STU_{uuid.uuid4().hex[:6]}"
payload = {
    "student_id": student_id,
    "name": "Test User",
    "email": "test@example.com",
    "demographic_group": "A"
}

try:
    response = client.post("/api/v1/users/", json=payload)
    print("STATUS CODE:", response.status_code)
    print("RESPONSE BODY:", response.json())
except Exception as e:
    import traceback
    traceback.print_exc()

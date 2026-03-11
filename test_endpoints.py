import urllib.request
import json
import uuid

# Test Postgres User Creation
student_id = f"STU_MOCK_{uuid.uuid4().hex[:6]}"
user_data = json.dumps({
    "student_id": student_id,
    "name": "Test User",
    "email": f"test_{student_id}@example.com",
    "demographic_group": "A"
}).encode('utf-8')

req1 = urllib.request.Request("http://localhost:8000/api/v1/users/", data=user_data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req1) as response:
        print("Postgres User Created:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("Postgres HTTP Error:", e.read().decode('utf-8'))
except Exception as e:
    print("Postgres Error:", e)

# Test Mongo Intervention Logging
mongo_data = json.dumps({
    "strategy": "simplification",
    "content": "Test intervention content"
}).encode('utf-8')

req2 = urllib.request.Request(f"http://localhost:8000/api/v1/interventions/{student_id}/log", data=mongo_data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req2) as response:
        print("Mongo Log Created:", response.read().decode('utf-8'))
except Exception as e:
    print("Mongo Error:", e)

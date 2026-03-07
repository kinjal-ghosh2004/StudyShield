import urllib.request, json

payload = {
    "student_id": "STU_TEST_001",
    "drift_score": 3.8,
    "drift_vector": [0.15, 6.0, 150.0, 2.2],
    "dropout_prob": 0.82,
    "time_to_dropout": 5,
    "top_features": ["volatility", "lag"],
    "context": {"demographic_group": "group_A"},
    "intervention_history": []
}

data = json.dumps(payload).encode()
req = urllib.request.Request(
    "http://localhost:8000/api/v1/genai/intervene",
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.read().decode())
except Exception as e:
    print("Error:", e)

import urllib.request
import json
import time
import uuid

student_id = f"STU_{uuid.uuid4().hex[:6]}"
print(f"Testing telemetry pipeline for student {student_id}")

events = [
    {
        "student_id": student_id,
        "event_type": "login",
        "page_id": "home",
        "duration_sec": 0,
        "metadata": {"device": "desktop", "os": "windows"}
    },
    {
        "student_id": student_id,
        "event_type": "video_play",
        "page_id": "module_3",
        "duration_sec": 120,
        "metadata": {"video_id": "vid_01"}
    },
    {
        "student_id": student_id,
        "event_type": "mouse_hesitation",
        "page_id": "assignment_submit",
        "duration_sec": 15,
        "metadata": {"dom_element": "submit_btn"}
    }
]

for event in events:
    data = json.dumps(event).encode('utf-8')
    req = urllib.request.Request(
        "http://localhost:8000/api/v1/telemetry/event", 
        data=data, 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print(f"Sent {event['event_type']} -> Response:", response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error on {event['event_type']}:", e.read().decode('utf-8'))
    except Exception as e:
        print(f"Fatal Error on {event['event_type']}:", e)
        
    time.sleep(1)

print("Finished sending test events.")

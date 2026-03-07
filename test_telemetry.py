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
        "course_id": "CS101",
        "payload": {"device": "desktop", "os": "windows"}
    },
    {
        "student_id": student_id,
        "event_type": "video_play",
        "course_id": "CS101",
        "payload": {"video_id": "vid_01", "module": "M_1"}
    },
    {
        "student_id": student_id,
        "event_type": "mouse_hesitation",
        "course_id": "CS101",
        "payload": {"hesitation_time_seconds": 15.5, "dom_element": "assignment_submit_btn"}
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

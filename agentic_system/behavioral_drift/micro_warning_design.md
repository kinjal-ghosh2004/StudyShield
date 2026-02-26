# Early Micro-Warning Detection System Design

## 1. Overview
The Early Micro-Warning System acts as the preventative front-line of the Agentic AI platform. It intercepts subtle behavioral decays (Zone 1 Drift) *before* they mature into the Zone 2/3 high-risk classifications that require ReAct Agent interventions.

This module is directly integrated into the `BehavioralDriftDetector`. Instead of binary silence until a major risk occurs, it issues low-intensity nudges to correct minor trajectory deviations.

## 2. Detection Logic & Drift Thresholds
The system computes an an `Engagement Stability Index (ESI)` natively within the drift architecture.
- **Normal Zone ($D_t < 1.0$)**: ESI is stable. No action.
- **Micro-Warning Zone ($1.0 \le D_t \le 2.5$)**: ESI drops by $X\%$ (e.g., login irregularity spikes, or quiz consistency drops by 10%). Triggers a Micro-Warning.
- **Intervention Zone ($D_t > 2.5$)**: High-risk. Bypasses warnings and triggers the full ReAct Agentic Pipeline.

### Triggers:
1. **Consistency Drop**: Moving Average of Quiz Accuracy falls by $>15\%$ in a 3-day window.
2. **Login Irregularity**: Standard deviation of login intervals exceeds $36$ hours.
3. **Pace Decay**: Session pace drops by $20\%$ compared to the localized baseline.

## 3. Alert Hierarchy Design
Micro-warnings differ fundamentally from ReAct interventions:
- **Cost**: Almost zero friction. Delivered as soft UI toasts or brief native push notifications.
- **Tone**: Friendly, non-alarmist, and peer-benchmarked.
- **No Heavy LLM Generation**: Uses templated heuristics rather than full GenAI reasoning to save compute for actual $D_t > 2.5$ risks.

**Hierarchy of Action:**
1. **Low ($1.0 < D_t \le 1.5$)**: "Encouragement Nudge" (e.g., *“You’re on a 3-day streak! Just 15 mins today will keep it alive.”*)
2. **Medium ($1.5 < D_t \le 2.0$)**: "Reminder Message" (e.g., *“Noticed you usually log in on Tuesdays. The Chapter 4 quiz is waiting when you’re ready.”*)
3. **High ($2.0 < D_t \le 2.5$)**: "Minor Schedule Adjustment" (e.g., *“Looks like a busy week. We’ve bumped your reading deadline to Thursday to give you breathing room.”*)

## 4. API & Orchestration
If a behavior vector hits the Micro-Warning interval, `drift_detector.py` generates the alert payload. The Orchestrator (`demo_runner.py`) will log this alert and route it to the student **without** spinning up the Risk Predictor or ReAct logic, ensuring a highly scalable, low-latency safeguard.

## 5. Example Micro-Warning Output

```json
{
  "warning_type": "Minor Schedule Adjustment",
  "drift_score_trigger": 2.1,
  "detected_anomaly": "Login Irregularity (Variance > 36h)",
  "suggested_action": "extend_deadline_24h",
  "student_notification_text": "Hey there! We noticed your login times have been a bit scattered this week. To help you balance your schedule, we've automatically shifted your Assignment 3 deadline back by 24 hours. No stress, take your time!"
}
```

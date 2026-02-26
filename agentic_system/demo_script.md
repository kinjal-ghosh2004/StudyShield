# Agentic AI Dropout Prevention System: Live Demo Script

## General Setup
**Objective**: Walk stakeholders through the end-to-end capabilities of the Agentic AI pipeline.
**Duration**: ~5 Minutes
**Target Student**: "Jane_452", taking "Advanced Data Structures".

---

## Step 1: Normal Student Baseline (T = 0)
**Presenter Script**: "We start our demo watching Jane. For the first two weeks, Jane is engaging perfectly with the system. Take a look at the Student Panel."
- **On Screen**: Student Panel graph shows a steady green line. Risk Meter is flat at `5%`. Predicted dropout days is `80+ Days`.
- **Backend Trigger**: `demo_runner.py` processes nominal behavior vector `[pace=1.0, lag=0.5, hes=30, vol=0.2]`.
- **JSON Output Snippet**:
```json
{
  "drift_score": 0.15,
  "system_reaction": "Passive Monitoring",
  "anomaly_flag": false
}
```

## Step 2: Simulated Engagement Drift (T + 2 Days)
**Presenter Script**: "Now, let's fast forward. Over the last 48 hours, Jane's data has begun to drift. She's logging in, but she's not completing modules."
- **On Screen**: The Engagement Graph dips into the yellow zone. The AI Reasoning Panel's radar chart spikes exclusively in "Hesitation Time". 
- **Backend Trigger**: Inject `[pace=0.4, lag=1.5, hes=150, vol=0.5]`.
- **JSON Output Snippet**:
```json
{
  "drift_score": 0.62,
  "top_flags": ["hesitation_increase: 400%"],
  "zone": "Warning: Micro-Drift"
}
```

## Step 3: Risk Spike Detection (T + 4 Days)
**Presenter Script**: "By Day 4, the situation fragments. The Drift Score breaks the critical threshold. Notice the Risk Predictor fires, drastically cutting her predicted survival time on the platform."
- **On Screen**: Student Panel turns orange. Risk Meter spikes to `78%`. Predicted Dropout flashes `T-Minus 8 Days`.
- **Backend Trigger**: Risk Predictor Layer runs inference on high drift.
- **JSON Output Snippet**:
```json
{
  "risk_probability": 0.78,
  "predicted_dropout_days": 8,
  "decline_trend": "Accelerating"
}
```

## Step 4: Agent Reasoning Explanation (T + 4 Days)
**Presenter Script**: "Unlike black-box models that just alert an advisor, our ReAct Agent diagnoses *why* this is happening. Looking at the AI Panel, we see the agent isolating the root cause."
- **On Screen**: SHAP-style explanation bars light up: `Massive Hesitation` pulls the risk up. The ReAct Agent Diagnosis text appears: *Root Cause: Conceptual Confusion*.
- **Backend Trigger**: Agent `_thought_phase()` logic fires based on hesitation dominance.
- **JSON Output Snippet**:
```json
{
  "thought_trace": "High hesitation without high login variance indicates student is present but stuck. Diagnosing Conceptual Confusion."
}
```

## Step 5: Autonomous Intervention Generation (T + 4 Days)
**Presenter Script**: "The RL Policy Maps 'Conceptual Confusion' to a 'Content Simplification' strategy. Watch the Intervention Panel draft a personalized payload instantly."
- **On Screen**: The Generative AI component types out a message to Jane.
- **Text on UI**: *"Hey Jane, Graphs and Trees can be overwhelming. I noticed you paused on the Dijkstra algorithm module. Let's break it down using a subway map analogy..."* + Auto-assigned a 5-minute micro-quiz.
- **Backend Trigger**: GenAI layout formatting based on `strategy="content_simplification"`.

## Step 6: Simulated Improvement (T + 7 Days)
**Presenter Script**: "We simulate Jane interacting with that subway analogy. Let's step forward 3 days."
- **On Screen**: The Engagement graph curls back upward out of the yellow zone into green. Hesitation drops back to normal levels.
- **Backend Trigger**: `demo_runner.py` injects stabilized recovery vector `[pace=0.9, lag=0.6, hes=35, vol=0.3]`.

## Step 7: Risk Reduction Calculation (T + 7 Days)
**Presenter Script**: "As her behavior normalizes, the Risk Prediction Layer instantly re-evaluates her."
- **On Screen**: Risk Meter plummets from `78%` down to `22%`. Predicted Dropout pushes back up to `65 Days`.
- **JSON Output Snippet**:
```json
{
  "risk_probability": 0.22,
  "status_change": "-56% Risk Reduction",
  "predicted_dropout_days": 65
}
```

## Step 8: Policy Update via RL (T + 7 Days)
**Presenter Script**: "Finally, we loop back. The Feedback Panel calculates the success of the intervention we deployed on Day 4. The reinforcement learning policy updates, becoming more confident that Content Simplification works for this exact profile."
- **On Screen**: Feedback Panel flashes green. `Score: +85`. Policy Update: `Weight for A2 (Content Simplification) increased by 14% for State (High Drift, Stable Login, High Hesitation)`.
- **Backend Trigger**: RL Engine calculates reward based on $\Delta$ Drift and updates Q-Table/Beta parameters.
- **JSON Output Snippet**:
```json
{
  "reward_calculated": 85.0,
  "policy_action_updated": 2,
  "new_q_value": 45.5,
  "memory_logged": "Success"
}
```

**Presenter Script**: "And just like that, the system monitored, diagnosed, intervened, saved the student, and learned from the experience entirely autonomously. Thank you."

# Working Demo Interface Design

## 1. Overview
The Demo Interface is a React-based web dashboard designed to visualize the AI dropout prevention pipeline. It allows stakeholders to observe how the AI processes student signals, reasons about root causes, generates personalized interventions, and tracks effectiveness via reinforcement learning over simulated time scenarios.

## 2. Interface Layout (Four-Panel Layout)

### Panel 1: Student Panel (Top Left)
**Purpose:** Visualizes raw and modeled metrics of the user's current status.
- **Engagement Trend Graph:** A dynamic line chart plotting "Days" vs. "Engagement Level". The line color shifts from green (Nominal) to orange (Warning) to red (Critical) based on recent behavioral matrices.
- **Risk Score Meter:** A circular gauge chart indicating the $P(Dropout) \in [0, 100\%]$.
- **Predicted Dropout Days:** A bold textual read-out: "Estimated time to dropout: X Days".

### Panel 2: AI Reasoning Panel (Top Right)
**Purpose:** Breaks the "black box" of the ML predictions and Agentic reasoning.
- **Drift Score Breakdown:** A radar chart displaying the 4 core drift anchors: Login Variance, Session Drop, Quiz Decay, and Hesitation Increase.
- **Top Contributing Factors:** A sorted list of features driving the risk score (via SHAP-like approximation).
- **SHAP Explanation Visualization Logic:** Instead of raw SHAP values, this is abstracted into an impact bar chart. Red bars dragging the score right (Risk Increase), Blue bars dragging the score left (Risk Mitigation). For example, "Very Low Pace (Pushing Risk +20%)".
- **Agent Diagnosis:** "ReAct Agent Root Cause: Cognitive Overload."

### Panel 3: Intervention Panel (Bottom Left)
**Purpose:** Displays the output of the Generative AI component.
- **Trigger Strategy:** e.g., "Strategy Selected: Micro-Plan"
- **Generated Payload:** A multi-tab or card UI displaying:
  - **Simplified Explanation** (Only populated if the root cause is Conceptual Confusion).
  - **Study Plan** (If Burnout or Schedule restructure).
  - **Motivation Message** (If Disengagement).
  - **Auto-assigned Task** (e.g., "Review Module 4.1").

### Panel 4: Feedback Panel (Bottom Right)
**Purpose:** Visualizes the RL Effectiveness loop $\Delta(t \rightarrow t+1)$.
- **Pre-Intervention Risk:** Displays Risk Probability and Drift Score from Day T.
- **Post-Intervention Risk:** Displays New Risk Probability and Drift Score on Day T+3.
- **Intervention Effectiveness Score:** A badge rating the success: "Success Score: 0.8 / 1.0".
- **RL Policy Update Status:** "Policy Alpha increased. Strategy weight +12%."

## 3. Scenario Simulation Controls
To allow users to interact with the demo, a global control bar exists at the top:
- **Scenario Selector Dropdown:**
  1. *Scenario 1: Normal Engagement* (Everything stays green/stable).
  2. *Scenario 2: Gradual Decline* (Engagement graph slowly dips, risk slowly rises, triggers ReAct mid-way).
  3. *Scenario 3: Sudden Performance Drop* (Immediate spike in risk, triggers emergency escalation).
- **Time Controls:** "Next Day Step", "Auto-Play", "Reset Scenario".
- **Student Selector:** Dropdown to switch between mock student profiles.

## 4. Backend-to-Frontend Data Flow

1. **Frontend Init:** User selects a `student_id` and Scenario.
2. **Step Action:** User clicks "Next Day". Frontend dispatches `POST /api/demo/step`.
3. **Backend Processing:**
   - The backend runs `demo_runner.py` logic for that specific day, feeding the predetermined scenario vector into the system.
   - Computes Drift, Predicts Risk, Runs ReAct loop (if necessary), Generates GenAI payload (if necessary), Calculates RL Reward (if an intervention happened 3 days ago).
4. **Response Parsing:** Backend returns a monolithic JSON state object.
5. **Frontend Render:** React updates the global state store, re-rendering all four panels simultaneously.

## 5. API Endpoints

### `POST /api/demo/init`
**Payload:** `{"scenario": "sudden_drop", "student_id": "STU_Charlie"}`
**Response:** `{"status": "ready", "baseline_established": true, "day": 0}`

### `POST /api/demo/step`
**Payload:** `{"student_id": "STU_Charlie"}`
**Response Concept:**
```json
{
  "day": 1,
  "student_panel": {
    "history_points": [{"day": 1, "engagement": 0.95}],
    "risk_score": 0.05,
    "predicted_dropout_days": 45
  },
  "ai_reasoning_panel": {
    "drift_breakdown": {"v_login": 0.1, "d_session": 0.05, "r_quiz": 0, "t_hes": 0.02},
    "shap_impacts": [{"feature": "Good Pace", "impact": -15}],
    "root_cause_diagnosis": "Nominal"
  },
  "intervention_panel": {
    "intervention_triggered": false,
    "payload": null
  },
  "feedback_panel": {
    "pending_evaluation": false
  }
}
```
*(When an intervention triggers, `intervention_triggered` flips true, populating the payload and logging the state for the Feedback Panel to evaluate 3 steps later).*

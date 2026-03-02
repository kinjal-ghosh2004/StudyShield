# Agentic AI-Based Dropout Prevention System

This repository contains the architecture, design documents, and a working technical demo of an autonomous, Agentic AI-driven system designed to intervene and prevent student dropout in e-learning platforms.

Unlike traditional predictive models that simply flag a student as "At Risk" based on static global cohort averages, this system establishes a continuous, personalized behavioral baseline, diagnoses the psychological root cause of drift using an autonomous ReAct agent, generates personalized micro-interventions via LLMs, and updates its intervention strategy mapping through a reinforcement learning feedback loop.

## 📂 Project Structure

```text
agentic_system/
│
├── architecture_design.md          # High-level component diagrams, data flow, ML integration
├── demo_interface_design.md        # UI/UX specifications for the 4-panel dashboard demo
├── demo_runner.py                  # EXECUTABLE: End-to-end simulated pipeline runner
├── demo_script.md                  # Step-by-step 5-minute live presentation script
├── patent_claims.md                # Breakdown of the novel, patentable components
│
├── behavioral_drift/
│   ├── drift_detector.py           # Uses LSTM Autoencoders & EWMA for anomaly scoring
│   ├── drift_detection_design.md   # Mathematical breakdown of continuous baseline metrics
│   └── micro_warning_design.md     # Specifications for early low-intensity nudges
│
├── ethical_ai/
│   ├── monitor.py                  # Governance layer handling fatigue, bias, and logging
│   └── ethical_ai_layer_design.md  # Design document for fairness and transparency schemas
│
├── course_analytics/
│   ├── analytics.py                # Aggregates cohort telemetry for syllabus improvements
│   └── course_intelligence_design.md # Difficulty scoring & macro-recommendation specs
│
├── genai_layer/
│   ├── generator.py                # Mocks LLM text generation for personalized payloads
│   ├── revision_notes_design.md    # Specification for auto-generated structured study guides
│   └── pace_optimizer_design.md    # Specification for adaptive weekly schedule restructuring
│
├── react_planner/
│   ├── agent.py                    # Core ReAct Loop (Reason -> Act -> Reflect)
│   ├── react_planner_design.md     # State space, heuristics, and agent memory logic
│   └── memory_design.md            # Intervention memory tracker & component scoring
│
├── risk_prediction/
│   ├── predictor.py                # Mocks XGBoost, LSTM Forecasting, and Survival Models
│   ├── counterfactual_design.md    # Specification for estimating post-intervention risk reduction
│   └── dropout_classifier_design.md# Specification for the 5-class behavioral taxonomy
│
└── rl_intervention/
    ├── environment.py              # Contextual Bandit environment evaluating proxy rewards
    └── rl_simulation_design.md     # Q-update logic and exploration vs exploitation math
```

## 🚀 How to Run the Demo

To test the multi-layered logic, execute the interactive python demo runner. Ensure you have `numpy`, `torch`, and `streamlit` installed.

```bash
# From the root directory
pip install -r agentic_system/requirements.txt
python demo.py
```

The script will launch a **Streamlit Dashboard** simulating 3 distinct student scenarios:
1. **Normal Engagement**: A stable baseline with no alerts.
2. **Gradual Decline**: A slow dip triggering RL evaluation ($R_{intervene}$), ReAct memory reflection, and a pivot in strategy.
3. **Sudden Performance Drop**: A critical rupture triggering an immediate emergency Human Escalation.

---

## 🏆 Novel Components & Potential Patent Claims

This system introduces specific, non-obvious mathematical methodologies and architectural constraints that differ significantly from current black-box educational data mining practices. The following four technical mechanisms represent the core patent claims of the system:

### 1. Cognitive Load vs. Behavioral Engagement Indexing Engine
**Novelty**: Moving beyond binary predictive models to a continuous, mathematically generated dimension of cognitive burden.

*   **Methodology**: Calculating a real-time **Cognitive Struggle Index ($CSI_t$)** by normalizing asynchronous video playback micro-events (rewinds) against real-time quiz hesitation decay.
*   **Formula**: $CSI_t = \gamma * ( (R_k * W_{diff}) / H_k ) + (1-\gamma) * CSI_{t-1}$

### 2. Self-Correcting Intervention Router (Multi-Agent RL Framework)
**Novelty**: An Actor-Critic multi-agent routing architecture governed by a proprietary proximal reward scalar that mathematically prevents generative "over-intervention" fatigue.

*   **Methodology**: When an anomaly is detected, an Actor Agent proposes an intervention. A Critic Agent dynamically maps this intervention against the student's *Hazard Shift* ($\Delta h(t)$) over the successive 48-hour epoch ($T_{eval}$).
*   **Reward Function**: The Actor's policy updates based on $R_{intervene} = \lambda_1 (\frac{S - \hat{S}}{\hat{S}}) - \lambda_2 \cdot P_{fatigue}$, balancing strict retention gains against the annoyance factor of repeated nudges.

### 3. Sandboxed Counterfactual Risk Analysis Engine
**Novelty**: The autonomous execution of parallel simulation projections across the feature space prior to deploying any generative action.

*   **Methodology**: Upon triggering an intervention threshold, the system executes three parallel sandboxes generating counterfactual vectors ($X'_{strategy}$) (e.g., Do Nothing vs. Simplification vs. Syllabus Downgrade).
*   **Execution**: By running Temporal Convolution Networks on each projection, the system autonomously routes the actual API response to the strategy simulating the minimal cumulative future hazard.

### 4. Autonomous Pacing Governor
**Novelty**: A closed-loop gating mechanism that structurally intercepts standard LMS syllabus API calls to forcefully regulate cognitive volume.

*   **Methodology**: The governor intercepts `API GET` requests for subsequent syllabus modules. If a calculated exponential error rate exceeds an overload threshold ($\tau_{load}$), the Governor overrides the API response.
*   **Execution**: It autonomously replaces the requested module with an autogenerated, logically synthesized sub-module containing specific variations of failed concepts from the preceding learning graph.

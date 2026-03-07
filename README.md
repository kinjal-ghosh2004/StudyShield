# Agentic AI-Based Dropout Prevention System

This repository contains the architecture, design documents, and a working technical demo of an autonomous, Agentic AI-driven system designed to intervene and prevent student dropout in e-learning platforms.

Unlike traditional predictive models that simply flag a student as "At Risk" based on static global cohort averages, this system establishes a continuous, personalized behavioral baseline, diagnoses the psychological root cause of drift using an autonomous ReAct agent, generates personalized micro-interventions via LLMs, and updates its intervention strategy mapping through a reinforcement learning feedback loop.

## 📂 Project Structure

```text
StudyShield/
│
├── agentic_system/                 # Core AI intervention and ReAct agent system
│   ├── backend/                    # FastAPI backend, Kafka Streaming, Postgres/Mongo/Influx integrations
│   ├── behavioral_drift/           # Anomaly scoring and continuous baseline metrics
│   ├── course_analytics/           # Telemetry aggregation for syllabus improvements
│   ├── ethical_ai/                 # Governance layer handling fatigue and bias
│   ├── genai_layer/                # Auto-generation of structured study guides (Gemini API)
│   ├── react_planner/              # Core ReAct Loop (Reason -> Act -> Reflect)
│   ├── risk_prediction/            # XGBoost, LSTM Forecasting, and Survival Models
│   ├── rl_intervention/            # Contextual Bandit environment evaluating proxy rewards
│   └── dashboard.html              # Premium Interactive Web Dashboard
│
├── ml_pipeline/                    # Prepared datasets and model training scripts (OULAD)
├── oulad_augmentation/             # Data augmentation and synthesizing routines
├── Plans/                          # Project planning and architecture documents
│
├── demo.py                         # EXECUTABLE: Streamlit interactive simulation dashboard
├── train_models.py                 # EXECUTABLE: Trains the XGBoost/LSTM/Survival models
├── init_db.py                      # EXECUTABLE: Initializes PostgreSQL tables
├── test_*.py                       # Unit and integration test scripts
├── docker-compose.yml              # Docker configuration for Postgres, Mongo, InfluxDB, Kafka, Zookeeper
├── requirements.txt                # Root project dependencies
├── .env                            # API Keys and database credentials
└── README.md                       # This document
```

## 🚀 How to Run the Project

This project contains two distinct operational modes: the **Full Agentic Backend Stack** (connecting the real ML models, Kafka streaming, and Gemini AI) and the **Technical Simulation Demo** (a standalone Streamlit app).

### Option 1: Run the Full Agentic AI Backend & Dashboard (Actual Architecture)

This mode runs the true production-like infrastructure:
1. **Infrastructure**: Start the databases and event streaming cluster via Docker.
   ```bash
   docker compose up -d
   ```
2. **Environment**: Ensure your `.env` file is configured with the necessary database credentials and a `GEMINI_API_KEY`.
3. **Database Setup**: Initialize the PostgreSQL tables:
   ```bash
   python init_db.py
   ```
4. **Model Training**: Train the XGBoost, LSTM, and Survival Analysis models on the synthesized OULAD dataset:
   ```bash
   python train_models.py
   ```
5. **Start API Server**: Launch the Uvicorn/FastAPI backend:
   ```bash
   python -m uvicorn agentic_system.backend.main:app --host 0.0.0.0 --port 8000
   ```
6. **Access Dashboard**: Open your browser and navigate to **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)**. 
   - From the dashboard, you can view the live ML telemetry feed, simulate Kafka streams, and trigger the GenAI Multi-Agent Reason-Act-Critic Loop.

### Option 2: Run the Technical Simulation Demo (Streamlit)

For a quick, standalone visual demonstration of the proprietary logic without needing Docker or actual databases:
```bash
pip install -r requirements.txt
python demo.py
```
This launches a Streamlit Dashboard simulating 3 scenarios:
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

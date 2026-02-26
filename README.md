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

To test the multi-layered logic, execute the interactive python demo runner. Ensure you have `numpy` and `torch` installed.

```bash
cd P7
python demo.py
```

The script simulates 3 distinct student scenarios:
1. **Normal Engagement**: A stable baseline with no alerts.
2. **Gradual Decline**: A slow dip triggering RL evaluation, ReAct memory reflection, and a pivot in strategy.
3. **Sudden Performance Drop**: A critical rupture triggering an immediate emergency Human Escalation.

---

## 🏆 Novel Components & Potential Patent Claims

This system introduces several novel methodologies differing significantly from current black-box educational data mining practices. The following components represent strong candidates for patent claims:

### 1. Dynamic Behavioral Drift Modeling via Dual-Window Continuous Baseline
**Novelty**: Isolating the $n$-th derivative (rate of change) in behavior using dual sliding windows against a user's *own* historical norm, rather than global cohort means.

*   **Methodology for Personalized Continuous Telemetry Assessment**: A computerized method for detecting disengagement by establishing a static personalized baseline window (e.g., $t-17$ to $t-3$) and continuously comparing it against an active behavioral window (e.g., $t-2$ to $t$).
*   **Derivative-Based Anomaly Score Generation**: Creating a unified `Normalized Drift Score` formulated specifically by calculating the proportional variance and targeted decline rates (e.g., login variance, session contraction rate) against localized historical behavior.
*   **Cognitive Burden Isolation (Hesitation Metric Algorithm)**: A specific method for mathematically isolating "Active Hesitation Time" from general "Idle Time" using the formula: $H_t = (\text{Total Session Time} - \text{Active Scrolling Time}) \times (1 + \text{DOM Erratic Click Variance})$. This constructs a proprietary telemetry proxy vector for conceptual overload.

### 2. Agentic ReAct (Reason-Act-Reflect) Intervention Loop
**Novelty**: The employment of an autonomous Reasoning-Acting-Reflecting agent layer strictly decoupled from the underlying risk prediction ML models, moving away from simple threshold-based trigger rules.

*   **Decoupled Causal Diagnosis Layer**: A system that receives a unified `Normalized Drift Score` and applies heuristic mapping rules to classify behavioral decay into specific psychological categories (e.g., Burnout, Cognitive Overload, Conceptual Confusion, Disengagement).
*   **Reflective Memory Constraint for Intervention Scaling**: A mechanism where the autonomous agent queries a localized memory structure of previous interventions applied to the precise user, mandating a pivot to an escalated intervention channel if the immediate prior action logged a `Success Score` below a defined threshold.
*   **Critic vs Generator Multi-Agent Validation Protocol**: An architectural circuit-breaker where an independent 'Critic Agent' mathematically evaluates the LLM 'Generator Agent' output against strict pedagogical bounds (e.g., maximum reading grade level, negative sentiment thresholds), falling back to a deterministic safe escalation track if the generative output fails safety parameters.

### 3. Closed-Loop Contextual Bandit Reinforcement Optimization
**Novelty**: The translation of delayed educational outcomes (retention vs. dropout weeks later) into immediate, proximal reinforcement proxy rewards using real-time behavioral vectors.

*   **Proximal Reward Formulation for Educational Interventions**: A system architecture calculating feedback loops by comparing pre-intervention predictions against post-intervention metrics ($t+3$). It creates an immediate quantitative reward ($R_t$) fed directly to the Bandit using the multivariate formula: $R_t = \alpha(\Delta T_d) + \beta(\Delta E) - \gamma(\text{decay\_penalty})$, where $\Delta T_d$ represents the survival engine delta and $\Delta E$ represents immediate engagement shifts.
*   **Contextual Policy Update Mechanism**: Utilizing the proxy reward ($R_t$) to update the continuous probability distributions of a Contextual Bandit, linking the identified psychological root cause to the optimal action category globally across cohorts.
*   **Safety-Bound Exploration Override**: A method to govern exploratory RL intervention behaviors by forcing deterministic action exploitation (e.g., immediate Human Escalation) anytime the calculated survival function (Predicted Dropout Days) falls beneath a critical safety threshold.

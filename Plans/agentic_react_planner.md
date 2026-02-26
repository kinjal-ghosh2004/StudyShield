# Agentic AI Planner for Dropout Prevention (ReAct Architecture)

This document formalizes the design of the autonomous **Agentic AI Planner** using the ReAct (Reasoning and Acting) framework. This core intelligence layer interprets risk signals, determines root causes, and generates hyper-personalized interventions while continuously learning through reinforcement.

---

## 1. State Representation ($S_t$)

The state $S_t$ encapsulates everything the system knows about the student at the moment an intervention is triggered. It is passed into the Agent's prompt context.

$$S_t = \{ D(t), X_t, P(Drop), T_d, C, H \}$$

*   $D(t)$: Current Behavioral Drift Score (e.g., 2.8 - Structural Drift).
*   $X_t$: High-resolution behavioral vector $[f_{pace}, f_{lag}, f_{hesitation}, f_{volatility}]$.
*   $P(Drop)$: Overall dropout probability from the ML Classifier (e.g., 78%).
*   $T_d$: Estimated Time-to-Dropout from the Survival Engine (e.g., 4 days).
*   $C$: Student Context (Demographics, current course progress, module difficulty).
*   $H$: History of previous interventions (What was sent, when, and did it work?).

---

## 2. The ReAct Loop: Interpreting Risk & Identifying Root Cause

When $D(t)$ crosses a critical threshold, the LLM-based planner is invoked using the ReAct paradigm (**Thought $\rightarrow$ Action $\rightarrow$ Observation**).

### A. The "Thought" Phase (Root Cause Diagnosis)
The Agent analyzes the components of $X_t$ to categorize the psychological root cause:

1.  **Burnout:**
    *   *Signal*: High $f_{volatility}$ (erratic login times) + High overall activity + Dropping quiz scores.
    *   *Diagnosis*: Student is spending a lot of time but at irregular hours and getting poor results.
2.  **Confusion / Cognitive Overload:**
    *   *Signal*: High $f_{hesitation}$ (pausing videos frequently, re-watching) + High time on specific concepts.
    *   *Diagnosis*: Student is stuck on a specific complex module.
3.  **Disengagement / Apathy:**
    *   *Signal*: High $f_{lag}$ (waiting days to start assignments) + Low $f_{pace}$.
    *   *Diagnosis*: Student is losing interest or prioritizing other life events.

### B. The "Action Space" ($A_t$)
The action space is a complex tuple selected by the agent based on the root cause.
$A_t = (\text{Strategy}, \text{Tone}, \text{Channel}, \text{Payload})$

*   **Strategy $\in$**
    *   `micro_nudge`: Brief motivational push.
    *   `content_simplification`: Generate simpler analogies or practice questions.
    *   `schedule_restructure`: Generate a revised, extended deadline plan.
    *   `human_escalation`: Alert a human tutor.
*   **Tone $\in$** $\{$Empathetic, Urgent, Analytical, Encouraging$\}$
*   **Channel $\in$** $\{$Email, SMS, In-App Modal$\}$
*   **Payload:** The actual LLM-generated text (e.g., a personalized email draft).

---

## 3. Generating Personalized Study Plan Adjustments

Once a strategy is selected, the Generative component of the agent creates the payload.

*   *Example (Confusion):* "I noticed you spent extra time on the 'Derivatives' module. That's a notoriously tough concept! I've automatically generated a 3-question micro-quiz using sports analogies to help clarify it, and pushed your main assignment deadline back by 24 hours. Keep at it!"
*   *Example (Disengagement):* "We are 4 days away from the midterm and you haven't started module 4. As an adjustment, I've broken module 4 into three 15-minute daily chunks for you. Click here to add them to your calendar."

---

## 4. Evaluating Effectiveness over Time

To close the loop, the system evaluates the intervention's success after a temporal window $\Delta t$ (e.g., 3 days). 

### Reward Function ($R_t$)
The reward quantifies the success of action $A_t$ taken in state $S_t$.

$$R_t = w_1 \Delta(\text{Engagement}) + w_2 \Delta(\text{Performance}) + w_3 (T_{d, t+1} - T_{d, t})$$

*   $\Delta(\text{Engagement})$: Did $D(t)$ normalize? Did $f_{lag}$ decrease? (Positive if drift reduces).
*   $\Delta(\text{Performance})$: Did the subsequent quiz score improve?
*   **Survival Time Extension** $(T_{d, t+1} - T_{d, t})$: Did the Survival Engine's new estimation push the expected dropout date further into the future? (Crucial metric).
*   *Penalty:* If the student formally drops out within $\Delta t$, $R_t = -100$.

---

## 5. Policy Improvement Approach (Reinforcement Learning)

The goal is to map $S_t \rightarrow A_t$ to maximize the cumulative expected reward.

1.  **Meta-Controller (Contextual Bandit Architecture):**
    Instead of running RL directly on the LLM weights (which is computationally expensive), we use a smaller, highly efficient **Contextual Bandit model** (e.g., using LinUCB or a lightweight Neural Bandit layer).
2.  **The Feedback Loop:**
    *   The Bandit model takes $S_t$ and outputs the optimal *Prompt Parameters* (`Strategy`, `Tone`, `Channel`).
    *   These parameters are fed into the LLM as instructions.
    *   The LLM generates the text payload.
    *   The reward $R_t$ is observed 3 days later.
    *   The Bandit model updates its weights to favor that (`Strategy`, `Tone`, `Channel`) combination for similar $S_t$ profiles in the future.
3.  **LLM Fine-Tuning (DPO):**
    Periodically (e.g., monthly), the highest-reward payloads generated by the LLM are curated into a dataset. The base LLM can be fine-tuned via Direct Preference Optimization (DPO) so its baseline generation style naturally aligns with historically successful interventions.

# Reinforcement Learning Framework for Intervention Strategy Selection

This document details the Reinforcement Learning (RL) framework used to optimize the selection of interventions by the Agentic AI Planner in the Dropout Prevention System. By framing intervention selection as a sequential decision-making process, the system learns long-term policies that maximize student retention.

---

## 1. State Definition ($\mathcal{S}$)

The state space $\mathcal{S}$ represents the complete, actionable context of a student at time $t$ when an intervention decision is required. To make reinforcement learning tractable, we synthesize the high-dimensional logs into a condensed state vector.

**State Vector $s_t \in \mathcal{S}$ includes:**
1.  **Behavioral Drift Features ($X_t$):** $[f_{pace}, f_{lag}, f_{hesitation}, f_{volatility}]$
2.  **Risk Metrics:**
    *   $P_{drop}(t)$: Probability of dropout (from XGBoost/LSTM).
    *   $T_d(t)$: Predicted time-to-dropout (from Survival Engine).
3.  **Student Profile Embeddings ($C$):** Demographics, past performance, and course difficulty index (encoded into a low-dimensional vector).
4.  **Intervention History ($H_t$):** A condensed representation of the last $k$ interventions (e.g., [Type of last intervention, time since last intervention, previous reward]).

---

## 2. Action Space ($\mathcal{A}$)

The action space $\mathcal{A}$ defines the "Prompt Parameters" that dictate how the LLM will generate the intervention. We use a **discrete, factored action space** to constrain the RL problem.

An action $a_t = (\text{Strategy}, \text{Tone}, \text{Channel, Timing_Delay})$ where:
*   **Strategy** $\in \{ \text{Micro-nudge, Content Simplification, Schedule Restructure, Peer Sync, Human Escalation} \}$
*   **Tone** $\in \{ \text{Empathetic, Urgent, Analytical, Encouraging, Neutral} \}$
*   **Channel** $\in \{ \text{Email, SMS, In-App Modal, Dashboard Alert} \}$
*   **Timing\_Delay** $\in \{ \text{Immediate, Next Morning, Next Weekend} \}$

The meta-controller selects $a_t$, which is then translated by the LLM into a specific text payload sent to the student.

---

## 3. Reward Structure ($\mathcal{R}$)

The reward function $\mathcal{R}(s_t, a_t, s_{t+1})$ is the critical component that aligns the RL agent's goals with the educational objective: long-term retention. 

Because educational outcomes are delayed, we use a **composite reward function** calculated over an observation window $\Delta t$ (e.g., 3 to 7 days).

$$R_t = \lambda_{proximal} R_{proximal} + \lambda_{survival} R_{survival} + \lambda_{terminal} R_{terminal} - Penalty$$

1.  **Proximal Reward ($R_{proximal}$): Immediate Behavioral Shift**
    *   $R_{proximal} = \max(0, D(t) - D(t+\Delta t))$ 
    *   *Reward for reducing the Behavioral Drift Score.*
2.  **Survival Reward ($R_{survival}$): Risk Reduction**
    *   $R_{survival} = (T_d(t+\Delta t) - T_d(t)) \times w_{time}$
    *   *Reward for extending the predicted time-to-dropout from the Survival Engine.*
    *   Alternatively, $R_{survival} = P_{drop}(t) - P_{drop}(t+\Delta t)$.
3.  **Terminal Reward ($R_{terminal}$): Milestone Achievement**
    *   $+50$ if the student passes a course milestone (completes a module, passes a midterm).
4.  **Penalty:**
    *   $-100$ if the student officially drops out.
    *   Small negative penalty ($-1$) for high-friction actions (like Human Escalation) to encourage the system to solve problems autonomously unless necessary.

---

## 4. Exploration vs. Exploitation Balance

To discover new, effective intervention strategies for unique student cohorts without severely harming current at-risk students, we must carefully balance exploration and exploitation.

### Contextual Epsilon-Greedy with Thompson Sampling
Standard RL (like PPO or DQN) can be sample-inefficient for education. We model this as a **Contextual Multi-Armed Bandit** or use **Soft Actor-Critic (SAC)** to handle the stochastic nature of human behavior.

1.  **Thompson Sampling (Preferred for Bandits):**
    *   Instead of point estimates for the value of an action, the agent maintains a probability distribution (e.g., Beta or Gaussian) for the expected reward of each action given the state $s_t$.
    *   It samples from these distributions and picks the action with the highest sampled value. This naturally explores actions with high uncertainty while exploiting known good actions.
2.  **Upper Confidence Bound (UCB):**
    *   $a_t = \arg\max_{a} \left( Q(s_t, a) + c \sqrt{\frac{\ln N(s_t)}{N(s_t, a)}} \right)$
    *   Explores strategies rarely tried for a specific student profile.
3.  **Safety Bounds (Constrained Exploration):**
    *   If $T_d(t) < 48 \text{ hours}$ (Critical Risk), $\epsilon \rightarrow 0$. The system *must* exploit the known best action (e.g., Human Escalation). Exploration is only permitted when the student is in lower risk zones.

---

## 5. Long-Term Policy Learning

The learning objective is to find a policy $\pi_\theta(a|s)$ that maximizes the expected cumulative discounted reward: $J(\theta) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right]$.

### Offline-to-Online Learning Approach

1.  **Phase 1: Behavioral Cloning (Offline Pre-training)**
    *   Before deploying the RL agent, we train it on historical intervention logs.
    *   We use Supervised Learning to mimic the historical decisions made by successful human counselors/teachers: $\mathcal{L}(\theta) = -\log \pi_\theta(a_{expert} | s_t)$.
2.  **Phase 2: Offline RL (Conservative Q-Learning)**
    *   We use Conservative Q-Learning (CQL) on historical data to learn a policy that improves upon the human baseline by heavily penalizing actions that lead to dropouts in the historical dataset.
3.  **Phase 3: Online Fine-tuning (PPO / SAC)**
    *   Once deployed, the agent interacts with live students.
    *   We use **Proximal Policy Optimization (PPO)** to update the policy weights. PPO is chosen for its stability, ensuring the agent doesn't take catastrophic update steps that could harm student retention.
    *   Experience Replay Buffer stores $(s_t, a_t, r_t, s_{t+1})$ tuples. Periodically, the meta-controller samples mini-batches to update the Actor and Critic networks.

This offline-to-online pipeline ensures the system is safe on Day 1 while continuously adapting to new cohort behaviors over time.

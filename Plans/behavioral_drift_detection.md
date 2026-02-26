# Behavioral Drift Detection Model Design

This document outlines the design of the **Behavioral Drift Detection Model** for the Agentic AI-Based Dropout Prevention System. This model continuously monitors a student's engagement pattern and compares it to their historical baseline to detect early signs of disengagement (drift) before traditional dropout classifiers would flag them.

---

## 1. Feature Extraction Methods

Instead of relying solely on static or aggregate metrics (e.g., total logins, average quiz score), this model extracts high-resolution, continuous-time behavioral vectors.

Let the student's activity stream be a sequence of events $E = \{(e_1, t_1), (e_2, t_2), ..., (e_n, t_n)\}$.

**Temporal Features (Extracted dynamically per time window $\Delta t$, e.g., daily):**
*   **$f_{pace}(t)$:** Velocity of content consumption (Modules completed / Expected modules).
*   **$f_{lag}(t)$:** Time elapsed between assignment release and first interaction.
*   **$f_{hesitation}(t)$:** Dwell time on complex concepts (e.g., pausing lecture videos at specific timestamps without progressing).
*   **$f_{volatility}(t)$:** Variance in login times (shifting from consistent morning logins to erratic, late-night, short-duration logins).

**Feature Vector:** For a given day $t$, the student's state is represented as a multidimensional vector $X_t = [f_{pace}(t), f_{lag}(t), f_{hesitation}(t), f_{volatility}(t), ...]^T$.

---

## 2. Mathematical Formulation & Time-Series Modeling

We frame drift detection as a continuous anomaly detection problem over sequential data.

### Step 2a: Establishing the Individual Baseline
Instead of comparing exclusively to the cohort, we build a personalized **Hidden Markov Model (HMM)** or use an **LSTM-based Autoencoder** to learn the student's *normal* behavioral distribution during their peak engagement periods (usually the first 2-3 weeks).

Let $f_\theta$ be an LSTM Autoencoder parameterized by $\theta$.
During the baseline period ($t = 1 \dots T_{base}$), the model reconstructs the sequence:
$$\hat{X}_t = f_\theta(X_t)$$

The acceptable individual reconstruction error (baseline variance) is:
$$\mu_{error} = \frac{1}{T_{base}} \sum_{t=1}^{T_{base}} || X_t - \hat{X}_t ||^2$$

### Step 2b: Formulating the Drift
As time progresses ($t > T_{base}$), we inject the new daily vector $X_t$ into the trained model.
The instantaneous deviation is the current reconstruction error:
$$d_t = || X_t - f_\theta(X_t) ||^2$$

To prevent noisy spikes (e.g., the student took one day off) from triggering false alarms, we apply an **Exponentially Weighted Moving Average (EWMA)** to smooth the instantaneous deviations into a continuous drift score.

---

## 3. Drift Scoring Function

The **Behavioral Drift Score $D(t)$** at time $t$ is defined as:

$$D(t) = \alpha \left( \frac{d_t - \mu_{error}}{\sigma_{error}} \right) + (1 - \alpha) D(t-1)$$

Where:
*   $d_t$ is the instantaneous behavioral deviation.
*   $\mu_{error}$ and $\sigma_{error}$ are the mean and standard deviation of errors from the student's baseline period.
*   $\alpha \in [0, 1]$ is the decay factor controlling sensitivity (higher $\alpha$ reacts faster to recent changes).
*   $D(t)$ is bounded via a sigmoid function if normalized outputs $[0,1]$ are required for downstream agents: $D_{norm}(t) = \sigma(W \cdot D(t) + b)$.

---

## 4. Early Warning Thresholds

Instead of a single binary threshold, we employ a **Multi-Tiered Z-Score Thresholding Mechanism** to trigger different types of interventions from the ReAct agent based on the Drift Score $D(t)$.

Assuming $D(t)$ is standardized (Z-score representing standard deviations from the student's normal behavior):

*   **Zone 0: Nominal ($D(t) \le 1.5$)**
    *   *System Action:* Passive monitoring. Log telemetry.
*   **Zone 1: Micro-Drift ($1.5 < D(t) \le 2.5$)**
    *   *System Action:* Trigger **Subtle Generative Nudge**. The RL Engine suggests adding a motivational overlay or gamified notification upon next login.
*   **Zone 2: Structural Drift ($2.5 < D(t) \le 3.5$)**
    *   *System Action:* Trigger **ReAct Reasoning Loop**. The LLM analyzes the specific feature causing the drift (e.g., high $f_{lag}$) and generates a personalized email proposing a dynamically generated catch-up schedule.
*   **Zone 3: Critical Rupture ($D(t) > 3.5$ for $> 48$ hours)**
    *   *System Action:* Trigger **Emergency Escalation**. System alerts human counselors/tutors directly and executes an aggressive retention protocol.

---

## 5. Why this Approach is Novel (Patent Perspective)

Standard dropout classifiers (e.g., running Logistic Regression or XGBoost on tabular mid-term data) suffer from several fatal flaws that this architecture solves, constituting its novelty:

1.  **"Self-Referential Relativity" vs. "Absolute Thresholds":**
    *   *Standard:* Flags a student if they log in $< 3$ times a week.
    *   *Novelty:* A student who historically logs in once a week but studies for 8 hours is not flagged, while a student who drops from 7 logins to 4 logins is flagged. *The model detects the shift in the derivative of behavior, not just the absolute behavior.*
2.  **Continuous Time-to-Event Focus over Binary Classification:**
    *   *Standard:* Predicts "Will Drop" vs "Won't Drop" at the end of the term.
    *   *Novelty:* The EWMA-smoothed Drift Score $D(t)$ acts as a continuous hazard rate. It is mathematically designed to be an input to the downstream Survival Engine (Cox/DeepSurv), predicting precisely *when* the rupture will occur.
3.  **Explainable Vector Attribution for Agentic Action:**
    *   *Standard:* Black-box output ("80% risk").
    *   *Novelty:* Because $d_t$ is broken down by feature reconstruction error, the system knows *exactly* which behavioral vector broke the baseline. The system passes this specific vector formulation to the LLM ReAct agent (e.g., "Drift caused by $f_{hesitation}$ spike"), allowing the LLM to ground its generative intervention in exact mathematical reality rather than generic encouragement.

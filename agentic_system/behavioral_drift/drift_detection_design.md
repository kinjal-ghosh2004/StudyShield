# Behavioral Drift Detection System Design

## 1. Overview
The Behavioral Drift Detection System serves as the first-response sensory layer in the e-learning dropout prevention engine. Instead of predicting absolute dropout probability using global cohort data, it continuously monitors a student against their **own historical baseline** to detect anomalous deceleration in engagement. 

It outputs a **Normalized Drift Score (0–1)** that acts as an early warning trigger for downstream Agentic Interventions.

## 2. Mathematical Formulation

Let $B$ represent the **Baseline Time Window** (e.g., Days $t-17$ to $t-3$).
Let $C$ represent the **Current Time Window** (e.g., Days $t-2$ to $t$).

For each time window, we extract four primary behavioral signals:
1. $L$: Daily Login Frequency
2. $S$: Average Session Duration
3. $Q$: Average Quiz Accuracy
4. $H$: Average Hesitation Time (seconds paused on video/task without interaction)

### Deviation Metrics Computation:
**1. Login Frequency Variance ($v_{login}$):**
How much more unpredictable has their attendance become?
$$ v_{login} = \frac{\sigma_{C}(L)^2 + \epsilon}{\sigma_{B}(L)^2 + \epsilon} - 1 $$
*(Where $\sigma^2$ is variance and $\epsilon$ prevents division by zero. Constrained to $\ge 0$)*

**2. Session Duration Drop % ($d_{session}$):**
Are they spending less time per session compared to their norm?
$$ d_{session} = \max\left(0, \frac{\mu_{B}(S) - \mu_{C}(S)}{\mu_{B}(S)}\right) $$

**3. Quiz Accuracy Decay Rate ($r_{quiz}$):**
Is their performance trending downward?
$$ r_{quiz} = \max\left(0, \frac{\mu_{B}(Q) - \mu_{C}(Q)}{\mu_{B}(Q)}\right) $$

**4. Hesitation Time Increase ($t_{hesitation}$):**
Are they staring at the screen without progressing (Cognitive Overload)?
$$ t_{hesitation} = \max\left(0, \frac{\mu_{C}(H) - \mu_{B}(H)}{\mu_{B}(H)}\right) $$

### Normalized Drift Score ($D_{score}$ ∈ [0, 1]):
We calculate a weighted sum of the deviations:
$$ Z_t = w_1 v_{login} + w_2 d_{session} + w_3 r_{quiz} + w_4 t_{hesitation} $$
*(Example Weights: $w_1=0.15, w_2=0.35, w_3=0.30, w_4=0.20$)*

To normalize this into a strictly bounded (0 to 1) score, we pass $Z_t$ through a scaled **Sigmoid Function**:
$$ D_{score} = \frac{1}{1 + e^{-k(Z_t - \beta)}} $$
*(Where $k$ determines the steepness/sensitivity of the alert, and $\beta$ is the center threshold where drift is considered 0.5)*

## 3. Time Window Comparison Logic
The system relies on a **Sliding Dual-Window** approach:
- **Baseline Window (14 days)**: Lags behind the current date by 3 days. This ensures the baseline represents "established normal behavior" and is not immediately contaminated if the student started drifting two days ago.
- **Current Window (3 days)**: Captures immediate short-term trends to enable rapid intervention before a full drop occurs.
- **Update Frequency**: Computed nightly via a batch job or triggered in real-time upon session end.

## 4. Early Warning Threshold Logic
The $D_{score}$ triggers tiered agentic responses based on the following bands:

| Score Range | Zone | Meaning | System Action / Intervention |
|-------------|------|---------|------------------------------|
| `0.00 - 0.35` | 🟢 Nominal | Stable engagement. | No action. Append data to baseline. |
| `0.36 - 0.65` | 🟡 Warning | Micro-drift detected. Engagement softening. | **Soft Nudge:** Gentle GenAI motivational message. |
| `0.66 - 0.85` | 🟠 High Risk | Structural drift. Noticeable decline in habits. | **ReAct Loop:** Agent analyzes root cause, suggests Micro-Plan or Content Simplification. |
| `0.86 - 1.00` | 🔴 Critical | Immediate rupture trajectory. Student paralyzed. | **Human Escalation:** Alert advisor, pause deadlines. |

## 5. Implementation Pseudocode

```python
def compute_drift_score(baseline_data, current_data):
    # baseline_data and current_data are DataFrames/Dicts of the 4 features
    
    # 1. Login Frequency Variance
    var_b = variance(baseline_data['login_freq']) + 1e-5
    var_c = variance(current_data['login_freq']) + 1e-5
    v_login = max(0, (var_c / var_b) - 1)
    
    # 2. Session Duration Drop
    mu_b_session = mean(baseline_data['session_duration'])
    mu_c_session = mean(current_data['session_duration'])
    d_session = max(0, (mu_b_session - mu_c_session) / mu_b_session)
    
    # 3. Quiz Accuracy Decay
    mu_b_quiz = mean(baseline_data['quiz_accuracy'])
    mu_c_quiz = mean(current_data['quiz_accuracy'])
    r_quiz = max(0, (mu_b_quiz - mu_c_quiz) / mu_b_quiz)
    
    # 4. Hesitation Time Increase
    mu_b_hes = mean(baseline_data['hesitation_time'])
    mu_c_hes = mean(current_data['hesitation_time'])
    t_hes = max(0, (mu_c_hes - mu_b_hes) / mu_b_hes)  # Note: increasing is bad
    
    # Weighted Sum Configuration
    weights = {'v': 0.15, 'd': 0.35, 'r': 0.30, 't': 0.20}
    
    Z_t = (weights['v'] * v_login + 
           weights['d'] * d_session + 
           weights['r'] * r_quiz + 
           weights['t'] * t_hes)
           
    # Sigmoid Normalization
    k = 5.0      # Sensitivity 
    beta = 0.5   # Midpoint threshold
    drift_score = 1 / (1 + math.exp(-k * (Z_t - beta)))
    
    return min(1.0, max(0.0, drift_score))

def evaluate_threshold(drift_score):
    if drift_score < 0.35: return "Nominal"
    if drift_score < 0.65: return "Warning"
    if drift_score < 0.85: return "High Risk"
    return "Critical"
```

## 6. Why This Differs from Traditional Dropout Classifiers
Traditional machine learning classifiers (like static XGBoost or Random Forests trained on demographics and mid-term grades) fail to act dynamically. 

**Key Differences:**
1. **Personalized Context vs. Global Average**: A traditional model might flag a student logging in 2 times a week as "At Risk" because the cohort average is 5. The Drift System understands that if this specific student *always* logged in 2 times a week and maintains good quiz scores, there is no drift. It only flags when *they* deviate from *their own* norm.
2. **Gradient vs. Absolute Status**: Traditional models predict an absolute probability $P(Y=1 | X)$ (Will they drop out?). The Drift System measures the *rate of change* $\Delta X$. A student with an A grade can experience severe behavioral drift weeks before their grade actually drops.
3. **Actionability**: A traditional model outputs a black-box probability. The Drift System calculates exactly *how* the behavior is decaying (e.g., high hesitation time increase), which allows the next layer (ReAct Agent) to immediately hypothesize "Cognitive Overload" rather than just guessing.

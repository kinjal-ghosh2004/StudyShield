# Reinforcement Learning Simulation Module Design

## 1. Overview
The RL Simulation Module serves as the critical feedback loop in the Agentic Dropout Prevention System. Instead of hardcoding which intervention strategy works best for every student configuration, the system learns mapping probabilities dynamically. This design uses a simplified Q-Learning / Contextual Bandit approach that is robust, easily visualizable, and computationally feasible for a working demo.

## 2. Environment Definitions

### State Space ($S$)
To prevent state-space explosion, continuous metrics are bucketed into discrete bins.
- **Drift Level**: `Low`, `Medium`, `High`
- **Performance Trend (Quiz Accuracy)**: `Improving`, `Stable`, `Declining`
- **Hesitation / Lag Time**: `Nominal`, `High`

*Example State*: `(High Drift, Declining Trend, High Lag)`

### Action Space ($A$)
The intervention strategies the agent can deploy:
1. $a_1$: `Micro-Nudge` (Motivation)
2. $a_2$: `Content Simplification` (Academic)
3. $a_3$: `Schedule Restructure` (Time Management)
4. $a_4$: `Peer Sync` (Social)
5. $a_5$: `Human Escalation` (Emergency)

### Reward Function ($R_{t+1}$)
The reward is evaluated $t=3$ days after the action is taken ($a_t$) at state ($s_t$), observing the transition to state ($s_{t+1}$).

- **+50**: Significant engagement improvement (Drift drops $\ge 1$ level).
- **+30**: Quiz score increase or stabilization.
- **-20**: Continued decline (Drift level increases or stays maxed without improvement).
- **-100**: Student fully drops out/goes inactive for 7 days.

$$ R(s, a, s') = w_{eng}\Delta \text{Engagement} + w_{perf}\Delta \text{Performance} + \text{Penalty} $$

## 3. Q-Value Update Logic
Since the state transitions in e-learning aren't strictly deterministic Markov processes (an intervention today affects behavior 2 weeks from now), we use continuous updating based on immediate proximal proxy metrics ($R$).

The system maintains a Q-Table $Q(s, a)$. The core Q-Learning formula:
$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$

Where:
- $\alpha$: Learning rate (e.g., 0.1 for the demo).
- $\gamma$: Discount factor (e.g., 0.8, valuing long-term retention).
- $R_{t+1}$: The reward generated from observing $s \rightarrow s'$.

For a simplified Contextual Bandit approach (no sequential $Q(s')$ consideration, just state-to-reward mapping), the update simplifies to:
$$ Q(s, a) \leftarrow (1-\alpha)Q(s, a) + \alpha R $$

## 4. Policy Update Simulation Logic ($\epsilon$-Greedy Policy)
How does the system choose an action given $s$?

With probability **$1 - \epsilon$ (Exploitation)**:
- Select $\arg\max_a Q(s, a)$
- "Do what we know works best for this student profile."

With probability **$\epsilon$ (Exploration)**:
- Select a random action from $a_1 \dots a_5$.
- "Try a new strategy to see if it yields a higher reward."

For the demo, we set $\epsilon = 0.2$. In a production environment, $\epsilon$ degrades over time as the system becomes confident.

## 5. Example Policy Improvement Simulation
**Setup**: A student is in state $S_1=$ `(High Drift, Declining Trend, High Lag)`.
Initial $Q(S_1, a) = 0$ for all actions. Learning rate $\alpha = 0.5$.

### Iteration 1 (Exploration)
- System explores and selects $a_1$ (`Micro-Nudge`).
- **Result**: Student ignores the nudge. Engagement continues to decline.
- **Reward**: -20.
- **Update**: $Q(S_1, a_1) = (0.5)(0) + (0.5)(-20) = \mathbf{-10}$. 
- **Policy Preference**: Anything but $a_1$.

### Iteration 2 (Exploration)
- System explores a different student in same state $S_1$ and selects $a_4$ (`Peer Sync`).
- **Result**: Student logs in but doesn't do quizzes.
- **Reward**: -5 (Slight stabilization, but still poor).
- **Update**: $Q(S_1, a_4) = (0.5)(0) + (0.5)(-5) = \mathbf{-2.5}$.

### Iteration 3 (Exploration)
- System tries $a_2$ (`Content Simplification`) on $S_1$.
- **Result**: Student watches the simplified video, attempts the quiz, and scores well.
- **Reward**: +40.
- **Update**: $Q(S_1, a_2) = (0.5)(0) + (0.5)(40) = \mathbf{+20}$.
- **Policy Preference shifted**: $\arg\max_a Q(s, a)$ is now firmly $a_2$.

### Iteration 4 (Exploitation)
- A new student hits $S_1$. System exploits and selects $a_2$.
- **Result**: It works moderately well.
- **Reward**: +20.
- **Update**: $Q(S_1, a_2) = (0.5)(20) + (0.5)(20) = \mathbf{+20}$ (Maintains confidence).

### Iteration 5 (Exploitation)
- System encounters $S_1$ again. Selects $a_2$.
- **Result**: Massive success.
- **Reward**: +60.
- **Update**: $Q(S_1, a_2) = (0.5)(20) + (0.5)(60) = \mathbf{+40}$.

### Simulation Result summary:
Over 5 iterations, the system learned that for a student suffering from high lag/hesitation and dropping scores, a purely motivational nudge (`a1`: -10) is useless. Delivering academic `Content Simplification` (`a2`: +40) is the optimal strategy. 

The demo UI will actively visualize these Q-values morphing as the mock timeline progresses, proving the system is dynamically learning rather than utilizing static rule engines.

# Intervention Memory System Design

## 1. Overview
The Intervention Memory System upgrades the ReAct Planner's "Reflect" phase from a simple 1-step lookback into a comprehensive historical effectiveness tracker. It continuously calculates how specific interventions impact a particular student over time, allowing the agent to permanently pivot away from intervention types that mathematically do not work for that specific individual.

## 2. Data Schema Addition
The student's `intervention_history` array is expanded to store comprehensive deltas rather than a single boolean success score.

```sql
CREATE TABLE intervention_memory (
    log_id SERIAL PRIMARY KEY,
    student_id VARCHAR,
    timestamp TIMESTAMP,
    intervention_type VARCHAR,
    delta_engagement FLOAT,    -- e.g., +20 mins session duration
    delta_quiz FLOAT,          -- e.g., +15% accuracy
    delta_risk_reduction FLOAT -- e.g., -0.22 absolute risk drop
);
```

**Memory State Representation (JSON Payload):**
```json
{
  "student_id": "STU_1001",
  "historical_effectiveness": {
    "micro_nudge": {
      "attempts": 3,
      "avg_engagement_change": 5.0,
      "avg_quiz_improvement": 2.0,
      "avg_risk_reduction": 0.05,
      "composite_score": 0.22 
    },
    "schedule_restructure": {
      "attempts": 1,
      "avg_engagement_change": 30.0,
      "avg_quiz_improvement": 15.0,
      "avg_risk_reduction": 0.40,
      "composite_score": 0.85
    }
  }
}
```

## 3. Scoring Formula
When evaluating an intervention $k$, the system retrieves all past instances $H_k$ and calculates the mean deltas. The **Composite Effectiveness Score ($E_k$)** is computed using weighted normalization:

$$E_k = (W_{eng} \cdot \Delta_{engagement}) + (W_{quiz} \cdot \Delta_{quiz}) + (W_{risk} \cdot \Delta_{risk})$$

*(Example Weights: $W_{eng} = 0.3$, $W_{quiz} = 0.3$, $W_{risk} = 0.4$)*

Scores $< 0.5$ indicate a historically failing strategy for this specific user.

## 4. Policy Adjustment Logic (The Reflect Phase)
The ReAct Agent's `_reflect_phase` is augmented to execute the following logic:

1. **Reason -> Act**: The agent proposes `strategy = X`.
2. **Reflect**:
   - The agent queries the Intervention Memory for $E_X$.
   - **Rule 1 (Historical Failure)**: If $E_X < 0.4$ and `attempts >= 2`, the strategy is completely blacklisted for this student. The agent escalates to the next tier immediately.
   - **Rule 2 (Diminishing Returns)**: If $E_X$ is dropping over the last 3 attempts, the agent adds a "Fatigue Warning" to the generative context.
   - **Rule 3 (Historical Success)**: If $E_X > 0.8$, the agent proceeds with maximal confidence.

If escalated due to blacklisting, the pivot path is:
`micro_nudge` $\rightarrow$ `peer_sync` $\rightarrow$ `content_simplification` $\rightarrow$ `schedule_restructure` $\rightarrow$ `human_escalation`.

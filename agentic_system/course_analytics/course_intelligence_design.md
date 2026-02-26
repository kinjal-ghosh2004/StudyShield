# Course-Level Intelligence Module Design

## 1. Overview
While the core Agentic System focuses on micro-interventions for individual students, the **Course-Level Intelligence Module** operates at the macro scale. It aggregates anonymous telemetry data (hesitation time, repeated quiz failures, dropout correlations) across the entire cohort to identify structural flaws in the course syllabus itself.

Instead of fighting symptoms (saving 100 struggling students), this module aims to cure the disease (fixing the module that caused 100 students to struggle).

## 2. Difficulty Scoring Algorithm
To rank topics by priority, we compute a composite `Difficulty Impact Score` for each topic $i$ in the syllabus.

The score relies on three normalized component metrics:
1. $E_i$: **Error Density** $\rightarrow$ (Total wrong attempts on Topic $i$ / Total attempts)
2. $H_i$: **Hesitation Spike** $\rightarrow$ (Average hesitation time on Topic $i$ / Baseline hesitation)
3. $D_i$: **Dropout Correlation** $\rightarrow$ Point Biserial Correlation between failing/skipping Topic $i$ and ultimate student dropout.

**Impact Score Formulation:**
$$ \text{Impact Score}_i = (w_e \cdot E_i) + (w_h \cdot H_i) + (w_d \cdot D_i) $$
*(Example Weights: $w_e = 0.3, w_h = 0.2, w_d = 0.5$)*

An Impact Score approaching 1.0 indicates a critical structural bottleneck.

## 3. Autonomous Recommendation Engine (GenAI)
Once a topic exceeds an Impact Threshold (e.g., $> 0.70$), the system passes the telemetry to a Generative AI layer to suggest structural improvements to the faculty member.

**Mapping Logic:**
- If $E_i$ is extremely high but $H_i$ is low (Students answer quickly but get it wrong) $\rightarrow$ Suggest **Topic Restructuring** or **Extra Practice Material** to clear up misconceptions.
- If $H_i$ is extremely high (Students stare at the page for 10+ minutes) $\rightarrow$ Suggest **Additional Prerequisite Modules** (they are lacking foundational knowledge to even start).
- If all metrics are high and the content volume is dense $\rightarrow$ Suggest **Reduced Content Load** (Cognitive overload is occurring).

## 4. Faculty Dashboard Integration
The module integrates directly into the Instructor/Faculty portal rather than the Student portal.

### UI Panels:
- **The Bottleneck Radar**: A visual heatmap showing the syllabus tree. Nodes glow red corresponding to their `Difficulty Impact Score`.
- **At-Risk Topics List**: A ranked list of the hardest concepts currently hurting cohort retention.
- **AI Syllabus Architect**: A sidebar where the AI presents its parsed structural recommendations.

## 5. Course-Level Analytics Schema
```sql
CREATE TABLE topic_telemetry_aggregations (
    topic_id VARCHAR PRIMARY KEY,
    course_id VARCHAR,
    total_attempts INT,
    total_errors INT,
    avg_hesitation_seconds FLOAT,
    dropout_correlation_coefficient FLOAT,
    last_computed TIMESTAMP
);

CREATE TABLE syllabus_recommendations (
    recommendation_id SERIAL PRIMARY KEY,
    topic_id VARCHAR REFERENCES topic_telemetry_aggregations(topic_id),
    impact_score FLOAT,
    suggested_action VARCHAR,
    generated_rationale TEXT,
    implemented BOOLEAN DEFAULT FALSE
);
```

## 6. Example Recommendation Output
Below is the JSON envelope expected to be rendered in the AI Syllabus Architect sidebar for the instructor:

```json
{
  "topic_name": "Module 4.2: Dynamic Programming - Knapsack Problem",
  "difficulty_score": 0.88,
  "metrics_breakdown": {
    "error_density": 0.76,
    "hesitation_spike": 4.5,
    "dropout_correlation": 0.81
  },
  "suggested_action": "Additional Prerequisite Modules",
  "generated_rationale": "Students are exhibiting a 4.5x spike in hesitation time on this topic compared to the course baseline, and failure here strongly correlates (0.81) with dropping the course. The high hesitation suggests they lack the foundational mental models to begin. We recommend inserting a 15-minute prerequisite refresher on 'Memoization Basics' immediately preceding this module."
}
```

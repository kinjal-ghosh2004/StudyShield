# Dropout Type Classification Model Design

## 1. Overview
Instead of a binary "At Risk" vs "Safe" flag, the **Dropout Type Classification Model** provides an interpretable categorical readout of *why* the student is drifting. By feeding the extracted behavioral indicators through a heuristic mapping (or a trained multi-class classifier like a Random Forest/Gradient Boosted Tree), the system pinpoints the structural reason for the risk.

This classification allows the ReAct Agent and academic advisors to understand the immediate context without parsing raw telemetry arrays.

## 2. Target Classes
The model classifies At-Risk students into five core psychological/behavioral profiles:
1. **Burnout Dropout**: High session volatility, fragmented logins, steady performance decay.
2. **Cognitive Overload Dropout**: High hesitation time, rapid performance decay on complex topics, low completion pace.
3. **Conceptual Confusion Dropout**: Normal pace, normal login patterns, but high error density on specific subjects.
4. **Motivation (Apathy) Dropout**: Uniformly decaying pace, increasing lag between sessions, low volatility (they simply stop showing up).
5. **Time-Management Dropout**: Wildly irregular session timing patterns (e.g., exclusively cramming on Sundays), missing micro-deadlines, but showing bursts of high engagement.

## 3. Feature Mapping & Logic
The logic maps telemetry variables (Pace, Lag, Hesitation, Volatility, Accuracy Decay) into confidence matrices per class.

| Class | Pace | Lag (Absence) | Hesitation | Volatility | Accuracy Decay |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Burnout** | Average | High | Average | **Max** | Medium |
| **Overload** | Low | Medium | **Max** | Low | **Max** |
| **Confusion** | Average | Low | High | Low | **Max** |
| **Motivation** | **Min** | **Max** | Low | Low | Average |
| **Time-Mgt** | High | High | Low | **Max** | Average |

*Note: In the demo, this is implemented as a rule-based weighting algorithm representing the output of a deterministic Multi-class XGBoost model.*

## 4. Dashboard Visualization Method
The UI implements a "Risk Profile Radar" inside the Student Panel.
- **Radar Chart**: A 5-axis visualization mapping the confidence scores of the 5 dropout types. This generates an instantly recognizable 'shape' for the student's struggle.
- **Profile Tag**: The highest-confidence class becomes the "Primary At-Risk Archetype" badge attached to the student's profile.

## 5. Example Output JSON Payload
```json
{
   "dropout_type": "Cognitive Overload Dropout",
   "confidence_score": 0.82,
   "supporting_features": [
      "hesitation_time (+300% vs baseline)",
      "accuracy_decay_rate (rapid)",
      "pace_reduction (-45% vs baseline)"
   ],
   "secondary_risk": "Burnout Dropout (0.41)"
}
```

# Counterfactual Risk Analysis Module Design

## 1. Overview
The Counterfactual Risk Analysis module enhances the Risk Prediction Layer by simulating "what-if" scenarios. Once the ReAct agent selects an intervention strategy, this module estimates the theoretical impact of that specific strategy on the student's dropout probability, providing a highly defensible metric (`risk_reduction_percentage`) for educators and administrators.

## 2. Mathematical Formulation
Let the baseline dropout probability without any intervention be $P(\text{Drop} | X_t)$, where $X_t$ is the current behavioral drift vector.

Let $A = a_k$ be the selected intervention strategy (e.g., $a_1$ = "micro_nudge", $a_2$ = "human_escalation").

We calculate the counterfactual dropout probability as:
$$ P(\text{Drop} | X_t, \text{do}(A = a_k)) = P(\text{Drop} | X_t) \times (1 - \Upsilon(a_k, X_t)) $$

Where $\Upsilon(a_k, X_t)$ is the **Intervention Efficacy Factor** [0, 1]. This factor is continuously approximated by the Contextual Bandit RL Engine's Q-values. For this simulation:
- Lighter interventions (nudge) have smaller maximum efficacy (e.g., 5-15% reduction) but lower cost.
- Heavy interventions (human escalation) have massive efficacy (e.g., 40-70% reduction) but high cost/fatigue.

The **Impact Delta** (\Delta_R) is formulated as:
$$ \Delta_R = \frac{P(\text{Drop} | X_t) - P(\text{Drop} | X_t, \text{do}(A = a_k))}{P(\text{Drop} | X_t)} $$

## 3. Simulation Logic & Model-Based Comparison
Instead of training a dedicated causal inference network (like TarNet or Dragonnet) for the demo, we use a programmatic simulation wrapper around the existing XGBoost mock predictor.

1. **Baseline Inception**: Obtain `risk_without_intervention` natively from the predictor.
2. **Efficacy Table Lookup**: Retrieve the empirically observed (or RL-guided) hazard reduction coefficient for the chosen strategy.
3. **Counterfactual Transformation**: Multiply the baseline risk by the efficacy constraint.
4. **Return Package**: Output the base risk, the post-intervention risk, and the \Delta percentage.

## 4. Visualization Strategy for Dashboard
The UI will render this data inside the **Intervention Action Panel** via a "Risk Abatement Graph":
1. **The 'Do Nothing' Line**: A red dotted line predicting the risk climbing to 100% over the next $T_d$ days.
2. **The 'Post-Intervention' Line**: A green solid line showing the risk sharply pivoting downward immediately after Day $T$, settling at a lower baseline.
3. **The Delta Badges**: Quick stats stating: "Without intervention: 88% chance of dropping. With Micro-Plan: 62% chance. Overall Reduction: 29.5%".

## 5. Output Payload Specification
```json
{
   "strategy_simulated": "schedule_restructure",
   "risk_without_intervention": 0.88,
   "risk_with_intervention": 0.62,
   "risk_reduction_percentage": 29.54
}
```

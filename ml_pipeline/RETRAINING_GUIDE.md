# Continuous Retraining Strategy for OULAD Dropout Predictor

## 1. Trigger Mechanisms
The retraining pipeline is triggered via an Airflow DAG under two conditions:
1. **Time-Based (Scheduled)**: The DAG triggers automatically at the beginning of every new semester (bi-annually).
2. **Drift-Based (Reactive)**: The `evidently` pipeline runs weekly on the incoming studentVle streams. If the `DataDriftPreset` indicates that $>15\%$ of features have drifted significantly (Kolmogorov-Smirnov test $p < 0.05$), an alert is sent to Slack and a retraining DAG is triggered.

## 2. Retraining Workflow
1. **Data Ingestion**: Fetch the latest `studentVle.csv` and `studentInfo.csv`.
2. **Preprocessing**: Run the `data_prep/preprocessing.py` module to generate updated tensors and tabular flattened histories.
3. **Hyperparameter Reboot (Optional)**: If triggered by Drift, run a lightweight Optuna study (e.g., 20 trials) to recenter hyperparameter boundaries.
4. **Model Shadowing**:
   - The newly trained ensemble (`Challenger`) is deployed alongside the existing production ensemble (`Champion`).
   - Via the FastAPI service, 10% of prediction traffic is logged against the Challenger model.
   - Outputs are compared over a 3-week window.
   - **Promotion**: If the Challenger model yields $>2\%$ improvement in PR-AUC on the active stream, MLflow tags the model as `Production` and gracefully re-routes 100% of traffic.

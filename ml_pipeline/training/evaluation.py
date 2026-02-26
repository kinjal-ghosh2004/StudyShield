import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import pandas as pd

def generate_drift_report(reference_data_path, current_data_path, output_html="drift_report.html"):
    """
    Compares the current incoming data stream (e.g., this semester's logs) against 
    the baseline dataset the model was originally trained on.
    """
    print("Generating Evidently AI Drift Report...")
    ref_df = pd.read_csv(reference_data_path)
    cur_df = pd.read_csv(current_data_path)
    
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(output_html)
    print(f"Drift report saved to {output_html}")
    
    return report.as_dict()

def evaluation_ensemble(xgb_probs, lstm_probs, final_labels, weights=(0.6, 0.4)):
    """
    Evaluates the ensemble performance of XGBoost and LSTM over the test set.
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
    import numpy as np
    
    final_probs = (weights[0] * xgb_probs) + (weights[1] * lstm_probs)
    
    # Eval
    roc_auc = roc_auc_score(final_labels, final_probs)
    precision, recall, _ = precision_recall_curve(final_labels, final_probs)
    pr_auc = auc(recall, precision)
    
    # Maximize F1 Threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-5)
    best_f1 = np.max(f1_scores)
    
    print(f"Ensemble ROC-AUC: {roc_auc:.4f}")
    print(f"Ensemble PR-AUC: {pr_auc:.4f}")
    print(f"Max F1-Score: {best_f1:.4f}")
    
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'best_f1': best_f1}

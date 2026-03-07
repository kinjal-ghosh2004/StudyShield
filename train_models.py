"""
Phase 3 Training Script
Reads the pre-augmented OULAD time-series dataset, engineers features,
trains XGBoost + Survival Analysis models, and saves them to ml_pipeline/saved_models/.
Note: LSTM training requires PyTorch and GPU for full runs; here we train a lighter version.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath('.'))

# ─────── Paths ───────
DATA_PATH   = "oulad_augmentation/my_augmented_ts.csv"
SAVE_DIR    = "ml_pipeline/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("PHASE 3 — ML Model Training Pipeline")
print("=" * 60)

# ─────── 1. Load data ───────
print("\n[1/5] Loading augmented OULAD dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows | Columns: {list(df.columns[:8])} ...")

# ─────── 2. Feature engineering ───────
print("\n[2/5] Engineering features...")

# Binary outcome: final_result in ['Withdrawn', 'Fail'] = 1 (at risk), else 0
if 'final_result' in df.columns:
    df['label'] = df['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
elif 'is_collapsed' in df.columns:
    df['label'] = df['is_collapsed'].astype(int)
else:
    df['label'] = (df['dropout_week'].notna()).astype(int)

# Use last observation per student for tabular models
tabular_df = df.sort_values(['id_student', 'week']).groupby('id_student').last().reset_index()
tabular_df = tabular_df.fillna(0)

# Select numeric feature columns
exclude_cols = {'id_student', 'label', 'final_result', 'is_collapsed',
                'dropout_week', 'code_module', 'code_presentation'}
feature_cols = [c for c in tabular_df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(tabular_df[c])]

print(f"  {len(feature_cols)} numeric feature columns | Class distribution: {tabular_df['label'].value_counts().to_dict()}")

X = tabular_df[feature_cols].values
y = tabular_df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f"{SAVE_DIR}/feature_scaler.pkl")
joblib.dump(feature_cols, f"{SAVE_DIR}/feature_columns.pkl")
print(f"  Saved feature scaler and column list.")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train,  y_train, test_size=0.15, stratify=y_train, random_state=42)

# ─────── 3. XGBoost ───────
print("\n[3/5] Training XGBoost model...")
try:
    import xgboost as xgb
    from sklearn.metrics import f1_score, roc_auc_score

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': pos_weight,
        'seed': 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(params, dtrain, num_boost_round=200,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=20, verbose_eval=50)

    proba = bst.predict(dtest)
    auc   = roc_auc_score(y_test, proba)
    f1    = f1_score(y_test, (proba > 0.5).astype(int))
    print(f"  XGBoost Test ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

    bst.save_model(f"{SAVE_DIR}/xgboost_model.json")
    print(f"  Saved → {SAVE_DIR}/xgboost_model.json")

except Exception as e:
    print(f"  XGBoost training failed: {e}")

# ─────── 4. Survival Analysis ───────
print("\n[4/5] Training Survival Analysis model...")
try:
    from lifelines import CoxPHFitter
    survival_features = ['sum_click', 'volatility_idx', 'synthesized_hesitation_sec', 'drift_idx']
    available = [c for c in survival_features if c in tabular_df.columns]

    # Build a simple survival dataframe using the week as duration
    surv_df = tabular_df[['label'] + available].copy()
    # Use max week as a proxy for duration to event
    if 'week' in df.columns:
        max_weeks = df.groupby('id_student')['week'].max().reset_index().rename(columns={'week': 'duration'})
        merged = pd.merge(tabular_df[['id_student', 'label'] + available], max_weeks, on='id_student', how='left')
        surv_df = merged[['duration', 'label'] + available].fillna(0)
    else:
        surv_df['duration'] = 40  # fixed course length fallback

    surv_df = surv_df.rename(columns={'label': 'event'})
    surv_df['duration'] = surv_df['duration'].clip(lower=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(surv_df, duration_col='duration', event_col='event', show_progress=False)
    cph.print_summary()

    joblib.dump(cph, f"{SAVE_DIR}/survival_model.pkl")
    print(f"  Saved → {SAVE_DIR}/survival_model.pkl")

except Exception as e:
    print(f"  Survival model training failed: {e}")

# ─────── 5. LSTM (lightweight) ───────
print("\n[5/5] Training lightweight LSTM model (5 epochs)...")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    SEQ_LEN    = 4
    N_STUDENTS = tabular_df['id_student'].nunique()

    # Build per-student sequences of the last SEQ_LEN weeks
    seq_features = ['sum_click', 'volatility_idx', 'synthesized_hesitation_sec']
    seq_features = [c for c in seq_features if c in df.columns]
    static_features = []  # keep it simple for now

    student_seqs, student_labels = [], []
    for sid, grp in df.sort_values('week').groupby('id_student'):
        seq = grp[seq_features].values[-SEQ_LEN:]
        if len(seq) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(seq), len(seq_features)))
            seq = np.vstack([pad, seq])
        lbl = int(grp['label'].iloc[-1]) if 'label' in grp else 0
        student_seqs.append(seq)
        student_labels.append(lbl)

    X_seq = np.array(student_seqs, dtype=np.float32)
    y_seq = np.array(student_labels, dtype=np.float32)

    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42)

    dataset_train = TensorDataset(torch.tensor(X_seq_train), torch.tensor(y_seq_train))
    loader_train  = DataLoader(dataset_train, batch_size=64, shuffle=True)

    class LightLSTM(nn.Module):
        def __init__(self, in_feats, hidden=32):
            super().__init__()
            self.lstm  = nn.LSTM(in_feats, hidden, batch_first=True)
            self.head  = nn.Linear(hidden, 1)
            self.sig   = nn.Sigmoid()
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.sig(self.head(h[-1]))

    model     = LightLSTM(len(seq_features))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(5):
        model.train()
        total_loss = 0
        for xb, yb in loader_train:
            optimizer.zero_grad()
            out  = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 — Loss: {total_loss/len(loader_train):.4f}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/lstm_model.pt")
    joblib.dump({'seq_features': seq_features, 'seq_len': SEQ_LEN}, f"{SAVE_DIR}/lstm_meta.pkl")
    print(f"  Saved → {SAVE_DIR}/lstm_model.pt")

except Exception as e:
    print(f"  LSTM training failed: {e}")

print("\n" + "=" * 60)
print("Training complete! Models saved to:", SAVE_DIR)
print("=" * 60)

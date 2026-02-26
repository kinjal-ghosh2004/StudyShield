import optuna
from sklearn.model_selection import StratifiedGroupKFold
from .xgboost_model import XGBoostPredictor

def tune_xgboost(X_df, y_series, groups, n_trials=50):
    print("Starting Optuna Hyperparameter Optimization for XGBoost...")
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 15)
        }
        
        cv = StratifiedGroupKFold(n_splits=3)
        pr_aucs = []
        
        for train_idx, val_idx in cv.split(X_df, y_series, groups):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]
            
            xgb = XGBoostPredictor(params=params)
            xgb.train(X_train, y_train, X_val, y_val, num_rounds=300, early_stopping_rounds=20)
            
            res = xgb.evaluate(X_val, y_val)
            pr_aucs.append(res['pr_auc'])
            
        return sum(pr_aucs) / len(pr_aucs)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters found: ", study.best_params)
    return study.best_params

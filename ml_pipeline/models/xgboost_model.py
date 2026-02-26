import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, f1_score

class XGBoostPredictor:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr', # PR-AUC for imbalanced data
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 5 # Handle class imbalance
            }
        else:
            self.params = params
        self.model = None
        
    def train(self, X_train, y_train, X_val, y_val, num_rounds=500, early_stopping_rounds=50):
        print("Training XGBoost Tabular Model...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50
        )
        print(f"Training completed. Best PR-AUC: {self.model.best_score:.4f}")
        
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def evaluate(self, X_test, y_test):
        preds_proba = self.predict_proba(X_test)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, preds_proba)
        pr_auc = auc(recall, precision)
        
        # Evaluate F1 Score at 0.5 threshold (can be tuned later)
        preds_binary = (preds_proba > 0.5).astype(int)
        f1 = f1_score(y_test, preds_binary)
        
        print(f"XGBoost Test PR-AUC: {pr_auc:.4f}")
        print(f"XGBoost Test F1-Score: {f1:.4f}")
        
        return {'pr_auc': pr_auc, 'f1': f1}
    
    def get_feature_importance(self):
        return self.model.get_score(importance_type='gain')

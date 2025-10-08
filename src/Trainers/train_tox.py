import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, precision_recall_curve
import joblib

class LightGBMTrainer:
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50, random_state=42):
        """
        params: dictionary of LightGBM hyperparameters
        num_boost_round: max boosting rounds
        early_stopping_rounds: early stopping for validation
        """
        self.params = params if params else {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'is_unbalance': True,
            'metric': 'binary_logloss',
            'random_state': random_state,
            'n_jobs': -1
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.models = []
        self.thresholds = []

    def train(self, X_train, y_train, X_val=None, y_val=None, save_path="lgbm_model.pkl"):
        """
        X_train, y_train: training features and labels (multi-label one-hot)
        X_val, y_val: optional validation set for metrics
        """
        n_tasks = y_train.shape[1]
        self.models = []
        self.thresholds = []

        for task_idx in range(n_tasks):
            print(f"\nTraining task {task_idx+1}/{n_tasks}")
            train_data = lgb.Dataset(X_train, label=y_train[:, task_idx])

            valid_data = lgb.Dataset(X_val, label=y_val[:, task_idx]) if X_val is not None else None

            callbacks = []
            if X_val is not None and y_val is not None:
              callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True))

            model = lgb.train(
              self.params,
              train_data,
              num_boost_round=self.num_boost_round,
              valid_sets=[valid_data] if valid_data is not None else None,
              callbacks=callbacks
            )
            self.models.append(model)

            if X_val is not None and y_val is not None:
                probs = model.predict(X_val)
                precision, recall, thres = precision_recall_curve(y_val[:, task_idx], probs)
                f1 = 2*precision*recall/(precision+recall+1e-6)
                best_thres = thres[f1.argmax()] if len(thres) > 0 else 0.5
                self.thresholds.append(best_thres)
            else:
                self.thresholds.append(0.5)

        joblib.dump({'models': self.models, 'thresholds': self.thresholds}, save_path)
        print(f"\nAll models saved to {save_path}")

        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            print("\nValidation Metrics:")
            print(classification_report(y_val, y_pred, zero_division=0))

        return self

    def predict(self, X):
        """
        Returns predictions using tuned thresholds
        """
        n_tasks = len(self.models)
        y_pred = np.zeros((X.shape[0], n_tasks))

        for i, model in enumerate(self.models):
            probs = model.predict(X)
            y_pred[:, i] = (probs >= self.thresholds[i]).astype(int)

        return y_pred

    def predict_proba(self, X):
        """
        Returns predicted probabilities for each task
        """
        n_tasks = len(self.models)
        y_proba = np.zeros((X.shape[0], n_tasks))

        for i, model in enumerate(self.models):
            y_proba[:, i] = model.predict(X)

        return y_proba
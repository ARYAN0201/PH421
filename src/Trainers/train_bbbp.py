import os
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

class BBBPExtraTrees:
    def __init__(self, data_dir=".", target_col="p_np", random_state=42):
        """
        Extra Trees model for BBBP classification with Optuna tuning & persistence.
        - Assumes train/valid/test CSVs already prepared with descriptors/fingerprints.
        """
        self.data_dir = data_dir
        self.target_col = target_col
        self.random_state = random_state
        self.model = None
        self.best_params = None

        self.train = pd.read_csv(os.path.join(data_dir, "train_with_desc.csv"))
        self.valid = pd.read_csv(os.path.join(data_dir, "valid_with_desc.csv"))
        self.test = pd.read_csv(os.path.join(data_dir, "test_with_desc.csv"))

        self.X_train, self.y_train = self._split_xy(self.train)
        self.X_valid, self.y_valid = self._split_xy(self.valid)
        self.X_test, self.y_test = self._split_xy(self.test)

    def _split_xy(self, df):
        """Drop SMILES + target, keep descriptors/fingerprints."""
        X = df.drop(columns=[self.target_col, "smiles"], errors="ignore")
        y = df[self.target_col]
        return X, y

    def _objective(self, trial):
        """Optuna objective: maximize ROC-AUC on validation set."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }

        model = ExtraTreesClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        preds = model.predict_proba(self.X_valid)[:, 1]
        return roc_auc_score(self.y_valid, preds)

    def tune(self, n_trials=30):
        """Run Optuna tuning, store best params."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=n_trials)

        self.best_params = study.best_params
        print("Best Params:", self.best_params)
        print("Best ROC-AUC:", study.best_value)
        return self.best_params

    def train_extratrees(self, tuned=True):
        """Train Extra Trees using either tuned or default params."""
        if tuned and self.best_params is None:
            print("⚠️ No tuned params found, running tuning first...")
            self.tune(n_trials=30)

        params = self.best_params if (tuned and self.best_params) else {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }

        self.model = ExtraTreesClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=-1
        )
        # train on train + valid together for final model
        X_final = pd.concat([self.X_train, self.X_valid], axis=0)
        y_final = pd.concat([self.y_train, self.y_valid], axis=0)
        self.model.fit(X_final, y_final)

        # Validation metrics
        val_preds = self.model.predict(self.X_valid)
        val_probs = self.model.predict_proba(self.X_valid)[:, 1]
        val_acc = accuracy_score(self.y_valid, val_preds)
        val_roc = roc_auc_score(self.y_valid, val_probs)
        print(f"✅ Validation ACC: {val_acc:.4f}, ROC-AUC: {val_roc:.4f}")

        # Test metrics
        test_preds = self.model.predict(self.X_test)
        test_probs = self.model.predict_proba(self.X_test)[:, 1]
        test_acc = accuracy_score(self.y_test, test_preds)
        test_roc = roc_auc_score(self.y_test, test_probs)
        print(f"✅ Test ACC: {test_acc:.4f}, ROC-AUC: {test_roc:.4f}")

        return self.model, (val_acc, val_roc), (test_acc, test_roc)

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X_new)

    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict_proba(X_new)

    def save_model(self, filepath="bbbp_extratrees.pkl"):
        if self.model is None:
            raise ValueError("No model trained yet.")
        joblib.dump({"model": self.model, "params": self.best_params}, filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath="bbbp_extratrees.pkl"):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.best_params = data.get("params", None)
        print(f"📂 Model loaded from {filepath}")
        return self.model
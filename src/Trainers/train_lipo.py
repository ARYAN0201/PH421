import os
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.impute import SimpleImputer

class KlipoSVR:
    def __init__(self, data_dir="data", target_col="logP", random_state=42):
        self.data_dir = data_dir
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.model = None
        self.best_params = None

        self.train = pd.read_csv(os.path.join(data_dir, "train_with_desc.csv"))
        self.valid = pd.read_csv(os.path.join(data_dir, "valid_with_desc.csv"))
        self.test = pd.read_csv(os.path.join(data_dir, "test_with_desc.csv"))

        self.X_train, self.y_train = self._split_xy(self.train)
        self.X_valid, self.y_valid = self._split_xy(self.valid)
        self.X_test, self.y_test = self._split_xy(self.test)

        self.X_train = self.imputer.fit_transform(self.X_train)
        self.X_valid = self.imputer.transform(self.X_valid)
        self.X_test = self.imputer.transform(self.X_test)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_valid = self.scaler.transform(self.X_valid)
        self.X_test = self.scaler.transform(self.X_test)

    def _split_xy(self, df):
        """Drop SMILES + target, keep descriptors only"""
        X = df.drop(columns=[self.target_col, "smiles"], errors="ignore")
        y = df[self.target_col]
        return X, y

    def _objective(self, trial):
        """Optuna objective: minimize RMSE on validation set"""
        params = {
            "C": trial.suggest_loguniform("C", 1e-2, 1e3),
            "epsilon": trial.suggest_loguniform("epsilon", 1e-3, 1.0),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        }

        model = SVR(**params)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_valid)
        rmse = np.sqrt(mean_squared_error(self.y_valid, preds))
        return rmse

    def tune(self, n_trials=50):
        """Run Optuna tuning"""
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_params
        print("Best Params:", self.best_params)
        print("Best RMSE:", study.best_value)
        return self.best_params

    def train_svr(self, tuned=True):
        """Train SVR using tuned or default params"""
        if tuned and self.best_params is None:
            print("⚠️ No tuned params found, running tuning first...")
            self.tune(n_trials=50)

        params = self.best_params if (tuned and self.best_params) else {
            "C": 1.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "kernel": "rbf"
        }

        self.model = SVR(**params)
        X_final = pd.concat([pd.DataFrame(self.X_train), pd.DataFrame(self.X_valid)], axis=0).values
        y_final = pd.concat([self.y_train, self.y_valid], axis=0).values
        self.model.fit(X_final, y_final)

        val_preds = self.model.predict(self.X_valid)
        val_rmse = np.sqrt(mean_squared_error(self.y_valid, val_preds))
        val_r2 = r2_score(self.y_valid, val_preds)
        print(f"✅ Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

        test_preds = self.model.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_valid, test_preds))
        test_r2 = r2_score(self.y_test, test_preds)
        print(f"✅ Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        return self.model, (val_rmse, val_r2), (test_rmse, test_r2)

    def predict(self, X_new):
        X_new_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_new_scaled)

    def save_model(self, filepath="klipo_svr.pkl"):
        if self.model is None:
            raise ValueError("No model trained yet.")
        joblib.dump({"model": self.model, "scaler": self.scaler, "params": self.best_params}, filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath="klipo_svr.pkl"):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.best_params = data.get("params", None)
        print(f"📂 Model loaded from {filepath}")
        return self.model
import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import early_stopping, log_evaluation 
import optuna

class ESOLTrainer:
    def __init__(self, data_dir=".", target_col="measured log solubility in mols per litre"):
        self.data_dir = data_dir
        self.target_col = target_col
        self.models = {}
        self.best_params = None

        self.train = pd.read_csv(os.path.join(data_dir, "train_with_desc.csv"))
        self.valid = pd.read_csv(os.path.join(data_dir, "valid_with_desc.csv"))
        self.test = pd.read_csv(os.path.join(data_dir, "test_with_desc.csv"))

        self.X_train, self.y_train = self._split_xy(self.train)
        self.X_valid, self.y_valid = self._split_xy(self.valid)
        self.X_test, self.y_test = self._split_xy(self.test)

    def _split_xy(self, df):
        X = df.drop(columns=[self.target_col, "smiles"], errors="ignore")
        y = df[self.target_col]
        return X, y

    def _objective(self, trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbosity": -1,
            "seed": 42,
        }

        dtrain = lgb.Dataset(self.X_train, self.y_train)
        dvalid = lgb.Dataset(self.X_valid, self.y_valid, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=5000,
            callbacks=[early_stopping(100), log_evaluation(200)]
        )

        preds = model.predict(self.X_valid, num_iteration=model.best_iteration)
        rmse = mean_squared_error(self.y_valid, preds)
        return rmse

    def tune(self, n_trials=30):
        """Run Optuna tuning and save best params."""
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_params
        print("Best params:", self.best_params)
        return self.best_params

    def trainlgb(self, tuned=True):
        """Train LightGBM with either default or tuned params."""
        if tuned and self.best_params is None:
            print("⚠️ No tuned params found, running tuning first...")
            self.tune(n_trials=30)

        params = self.best_params if (tuned and self.best_params) else {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 64,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
        }

        dtrain = lgb.Dataset(self.X_train, self.y_train)
        dvalid = lgb.Dataset(self.X_valid, self.y_valid, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=5000,
            callbacks=[early_stopping(100), log_evaluation(200)]
        )

        self.models["final"] = model

        val_preds = model.predict(self.X_valid, num_iteration=model.best_iteration)
        val_rmse = mean_squared_error(self.y_valid, val_preds)
        val_r2 = r2_score(self.y_valid, val_preds)
        print(f"✅ Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

        test_preds = model.predict(self.X_test, num_iteration=model.best_iteration)
        test_rmse = mean_squared_error(self.y_test, test_preds)
        test_r2 = r2_score(self.y_test, test_preds)
        print(f"✅ Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        return model, (val_rmse, val_r2), (test_rmse, test_r2)

    def save_model(self, filepath="esol_model.txt"):
        """Save the final trained model to disk."""
        if "final" not in self.models:
            raise ValueError("No trained model found. Train a model first with trainlgb().")
        self.models["final"].save_model(filepath)
        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath="esol_model.txt"):
        """Load a trained model from disk into self.models['final']."""
        self.models["final"] = lgb.Booster(model_file=filepath)
        print(f"📂 Model loaded from {filepath}")
        return self.models["final"]
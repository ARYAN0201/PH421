import joblib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import lightgbm as lgb

from .featuriser import DescriptorFeaturizer, FingerprintFeaturizer

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class InferencePipeline:
    def __init__(self, model_paths: Dict[str, Path]):
        """
        model_paths: dict mapping task names -> model file paths
            Example:
            {
                "tox": "models/toxicity_rf.pkl",
                "esol": "models/esol_lgb.txt",
                "lipophilicity": "models/klipo_svr.pkl",
                "bbbp": "models/bbbp_extratrees.pkl"
            }
        """
        self.models = {}
        for task, path in model_paths.items():
            self.models[task] = self._load_model(task, path)

        self.desc_featurizer = DescriptorFeaturizer()
        self.fp_featurizer = FingerprintFeaturizer()

    def _load_model(self, task: str, path: Path):
        """Load models depending on format"""
        if path.suffix == ".txt" and task == "esol":
            model = lgb.Booster(model_file=str(path))
        else:
            try:
                model = joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    model = pickle.load(f)

        if isinstance(model, dict):
            if "model" in model:
                return model["model"]
            else:
                raise ValueError(f"Loaded {path} is a dict but does not contain 'model' key")

        return model

    def _featurize(self, smiles: str, task: str):
        if task == "tox":
            arr = self.fp_featurizer.featurize_smiles(smiles).reshape(1, -1)
        else:
            arr = np.array(self.desc_featurizer.featurize_smiles(smiles)).reshape(1, -1)
        return pd.DataFrame(arr)

    def predict(self, smiles: str) -> Dict[str, Any]:
        """Run inference for all models"""
        results = {}
        for task, model in self.models.items():
            features = self._featurize(smiles, task)

            if task == "esol":
                pred = model.predict(features)
                results[task] = float(pred[0])

            elif task == "lipophilicity":
                pred = model.predict(features)
                results[task] = float(pred[0])

            elif task == "bbbp":
                pred = model.predict(features)
                results[task] = int(pred[0])

            elif task == "tox":
                pred = model.predict(features)
                results[task] = pred[0].tolist()

            else:
                raise ValueError(f"Unknown task: {task}")

        return results
    
if __name__ == "__main__":
    model_paths = {
        "tox": Path("./src/models/toxicity_rf.pkl"),
        "esol": Path("./src/models/esol_lgb.txt"),
        "lipophilicity": Path("./src/models/klipo_xgb.pkl"),
        "bbbp": Path("./src/models/bbbp_extratrees.pkl")
    }

    pipeline = InferencePipeline(model_paths)
    smiles = "CCCCCCCCCCOCC(O)CN"
    results = pipeline.predict(smiles)
    print(results)
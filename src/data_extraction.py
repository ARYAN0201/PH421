import os
import pandas as pd
from deepchem.molnet import load_lipo, load_delaney, load_bbbp, load_tox21


class ADMETDataExtractor:
    """
    A helper class to extract ADMET datasets from DeepChem MolNet,
    convert them to DataFrames, and save as CSV inside property-specific folders.
    """
    def __init__(self, out_dir: str = "../data"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def dc_to_df(dataset):
        """Convert a DeepChem DiskDataset to Pandas DataFrame"""
        return pd.DataFrame({
            "smiles": dataset.ids,
            **{f"label_{i}": dataset.y[:, i] for i in range(dataset.y.shape[1])}
        })

    def save_splits(self, datasets, name: str):
        """Save train/valid/test splits inside ../data/{name}/ folder"""
        property_dir = os.path.join(self.out_dir, name)
        os.makedirs(property_dir, exist_ok=True)

        train, valid, test = datasets
        self.dc_to_df(train).to_csv(os.path.join(property_dir, "train.csv"), index=False)
        self.dc_to_df(valid).to_csv(os.path.join(property_dir, "valid.csv"), index=False)
        self.dc_to_df(test).to_csv(os.path.join(property_dir, "test.csv"), index=False)

        print(f"Saved {name} splits to {property_dir}/")

    def extract_lipo(self):
        _, datasets, _ = load_lipo(featurizer="Raw", splitter="scaffold")
        self.save_splits(datasets, "lipo")

    def extract_delaney(self):
        _, datasets, _ = load_delaney(featurizer="Raw", splitter="scaffold")
        self.save_splits(datasets, "esol")

    def extract_bbbp(self):
        _, datasets, _ = load_bbbp(featurizer="Raw", splitter="scaffold")
        self.save_splits(datasets, "bbbp")

    def extract_tox21(self):
        _, datasets, _ = load_tox21(featurizer="Raw", splitter="scaffold")
        self.save_splits(datasets, "tox21")

    def extract_all(self):
        """Extract and save all chosen ADMET datasets"""
        self.extract_lipo()
        self.extract_delaney()
        self.extract_bbbp()
        self.extract_tox21()
        print("All datasets extracted and saved!")
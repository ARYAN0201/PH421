from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

logger = logging.getLogger("featurizers")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class DescriptorFeaturizer:
    def __init__(self, descriptor_names: Optional[List[str]] = None):
        if descriptor_names is None:
            descriptor_names = [d[0] for d in Descriptors._descList]
        self.descriptor_names = descriptor_names
        self.calc = MolecularDescriptorCalculator(self.descriptor_names)
        logger.info("DescriptorFeaturizer initialized with %d descriptors", len(descriptor_names))

    def featurize_smiles(self, smiles: str) -> List[float]:
        """Return descriptor vector for a single SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(self.descriptor_names)
        return list(self.calc.CalcDescriptors(mol))

    def featurize_dataset(self, input_path: Path, output_path: Path):
        """Read CSV with 'smiles', append descriptors, save new CSV"""
        df = pd.read_csv(input_path)
        if "smiles" not in df.columns:
            raise ValueError(f"No 'smiles' column in {input_path}")

        values = [self.featurize_smiles(s) for s in df["smiles"]]
        desc_df = pd.DataFrame(values, columns=self.descriptor_names, index=df.index)

        df_out = pd.concat([df, desc_df], axis=1)
        df_out.to_csv(output_path, index=False)
        logger.info("✅ Saved with descriptors: %s", output_path)


class FingerprintFeaturizer:
    def __init__(self, radius: int = 2, nbits: int = 1024):
        self.radius = radius
        self.nbits = nbits
        logger.info("FingerprintFeaturizer initialized: r=%d, nbits=%d", radius, nbits)

    def featurize_smiles(self, smiles: str) -> np.ndarray:
        """Return fingerprint vector for a single SMILES using modern RDKit API"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.full((self.nbits,), np.nan, dtype=np.float32)

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.nbits)
        bitvect = gen.GetFingerprint(mol)

        arr = np.zeros((self.nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bitvect, arr)

        return arr.astype(np.float32)

    def featurize_dataset(self, input_path: Path, output_path: Path):
        """Read CSV with 'smiles', append fingerprints, save new CSV"""
        df = pd.read_csv(input_path)
        if "smiles" not in df.columns:
            raise ValueError(f"No 'smiles' column in {input_path}")

        fps = [self.featurize_smiles(s) for s in df["smiles"]]
        fp_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(self.nbits)], index=df.index)

        df_out = pd.concat([df, fp_df], axis=1)
        df_out.to_csv(output_path, index=False)
        logger.info("✅ Saved with fingerprints: %s", output_path)
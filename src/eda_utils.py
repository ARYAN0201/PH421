import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem

class ADMETEDAUtils:
    """
    Parent class for common EDA utilities across ADMET datasets.
    Children classes will handle specific types of data: regression, binary classification, multi-task classification.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Unknown"):
        self.df = df.copy()
        self.name = dataset_name
        self.label_cols = [col for col in df.columns if col.startswith("label")]

    def check_smiles(self):
        """Check validity of SMILES strings."""
        if "smiles" not in self.df.columns:
            print(f"[{self.name}] No SMILES column found.")
            return
        self.df["valid_smiles"] = self.df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        n_valid = self.df["valid_smiles"].sum()
        print(f"[{self.name}] Valid SMILES: {n_valid}/{len(self.df)} ({n_valid/len(self.df)*100:.1f}%)")

    def missing_values(self):
        """Print missing value summary."""
        print(f"\n[{self.name}] Missing values summary:")
        print(self.df.isna().sum())

    def plot_histogram(self, column, bins=30, kde=True):
        """Histogram for a single numeric column."""
        plt.figure(figsize=(6, 4))
        sns.histplot(self.df[column].dropna(), bins=bins, kde=kde)
        plt.title(f"{self.name} - {column} Distribution")
        plt.show()

    def plot_boxplot(self, column):
        """Boxplot for a single numeric column."""
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=self.df[column])
        plt.title(f"{self.name} - {column} Boxplot")
        plt.show()

    def correlation_heatmap(self, columns=None):
        """
        Plot correlation heatmap for numeric columns.
        If columns=None, use label_cols by default.
        """
        if columns is None:
            columns = self.label_cols

        if not columns:
            print(f"[{self.name}] No numeric columns provided for correlation.")
            return

        numeric_df = self.df[columns].dropna()
        corr = numeric_df.corr()
        print(f"\n[{self.name}] Correlation matrix:")
        print(corr)

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title(f"{self.name} Correlation Heatmap")
        plt.show()
        
class RegressionEDA(ADMETEDAUtils):
    """
    EDA class for regression-type ADMET datasets (e.g., Lipo, ESOL)
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Regression Dataset"):
        super().__init__(df, dataset_name)

    def label_analysis(self):
        """Print summary statistics and plot distributions for all regression labels."""
        if not self.label_cols:
            print(f"[{self.name}] No regression labels found.")
            return

        for col in self.label_cols:
            print(f"\n[{self.name}] Statistics for {col}:")
            print(self.df[col].describe())
            self.plot_histogram(col)
            self.plot_boxplot(col)

    def scatterplot_labels(self):
        """Scatterplots between all pairs of regression labels (if multiple labels exist)."""
        if len(self.label_cols) < 2:
            print(f"[{self.name}] Only one label found, skipping scatterplot.")
            return

        for i, col1 in enumerate(self.label_cols):
            for col2 in self.label_cols[i+1:]:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(x=self.df[col1], y=self.df[col2])
                plt.title(f"{self.name} - {col1} vs {col2}")
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.show()

    def run_eda(self, scatter=False):
        """Run full regression EDA."""
        print(f"\n===== Regression EDA Report: {self.name} =====")
        print("Shape:", self.df.shape)
        print("Columns:", self.df.columns.tolist())
        self.check_smiles()
        self.missing_values()
        self.label_analysis()
        self.correlation_heatmap()
        if scatter:
            self.scatterplot_labels()
            
class BinaryClassificationEDA(ADMETEDAUtils):
    """
    EDA class for binary classification-type ADMET datasets (e.g., BBBP)
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Binary Classification Dataset"):
        super().__init__(df, dataset_name)

    def label_analysis(self):
        """Analyze binary labels, print distribution and imbalance warnings, plot countplots."""
        if not self.label_cols:
            print(f"[{self.name}] No label columns found.")
            return

        for col in self.label_cols:
            counts = self.df[col].value_counts(dropna=True)
            total = counts.sum()
            perc = counts / total * 100
            print(f"\n[{self.name}] Label distribution for {col}:")
            for val, p in zip(counts.index, perc):
                print(f"{val}: {counts[val]} ({p:.1f}%)")
            if perc.min() < 20:
                print(f"⚠️ Imbalance detected in {col}")

            plt.figure(figsize=(6, 4))
            sns.countplot(x=self.df[col])
            plt.title(f"{self.name} - {col} Countplot")
            plt.show()

    def run_eda(self):
        """Run full binary classification EDA."""
        print(f"\n===== Binary Classification EDA Report: {self.name} =====")
        print("Shape:", self.df.shape)
        print("Columns:", self.df.columns.tolist())
        self.check_smiles()
        self.missing_values()
        self.label_analysis()
        if len(self.label_cols) > 1:
            self.correlation_heatmap()
            
class MultiTaskBinaryClassificationEDA(ADMETEDAUtils):
    """
    EDA class for multi-task binary classification datasets (e.g., Tox21)
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Multi-Task Binary Classification"):
        super().__init__(df, dataset_name)

    def label_analysis(self):
        """Analyze all binary labels, show imbalance warnings and plot distributions."""
        if not self.label_cols:
            print(f"[{self.name}] No label columns found.")
            return

        n_labels = len(self.label_cols)
        cols = min(4, n_labels)
        rows = (n_labels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.ravel()

        for i, col in enumerate(self.label_cols):
            counts = self.df[col].value_counts(dropna=True)
            total = counts.sum()
            perc = counts / total * 100

            print(f"\n[{self.name}] Label distribution for {col}:")
            for val, p in zip(counts.index, perc):
                print(f"{val}: {counts[val]} ({p:.1f}%)")
            if perc.min() < 20:
                print(f"⚠️ Imbalance detected in {col}")

            sns.countplot(x=self.df[col], ax=axes[i])
            axes[i].set_title(col)

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def run_eda(self):
        """Run full multi-task binary classification EDA."""
        print(f"\n===== Multi-Task Binary Classification EDA Report: {self.name} =====")
        print("Shape:", self.df.shape)
        print("Columns:", self.df.columns.tolist())
        self.check_smiles()
        self.missing_values()
        self.label_analysis()
        if len(self.label_cols) > 1:
            self.correlation_heatmap()
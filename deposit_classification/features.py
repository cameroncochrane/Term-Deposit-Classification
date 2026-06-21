from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from deposit_classification.config import PROCESSED_DATA_DIR, FIGURES_DIR

app = typer.Typer()


@app.command()
def main(
    X_train_path: Path = PROCESSED_DATA_DIR / "X_train.pkl",
    y_train_path: Path = PROCESSED_DATA_DIR / "y_train.pkl",
    figures_directory: Path = FIGURES_DIR):
    print("features.py")

    with open(X_train_path, "rb") as f:
        X_train = pickle.load(f)

    with open(y_train_path, "rb") as f:
        y_train = pickle.load(f)

    train_data = pd.concat([X_train.copy(), y_train.copy()], axis=1, join="inner")

    # 1. Using chi-squared values to test if features and the target are independent.
    # Null hypothesis (H0): Features and the target are independent.
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month']
    target_column = ['y']
    alpha = 0.05
    results = []
    for col in cat_columns:
        table = pd.crosstab(train_data[col], train_data[target_column[0]])
        chi2, p, dof, expected = chi2_contingency(table)
        h0_rejected = p < alpha
        results.append({
            "feature": col,
            "chi_square": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "H0_rejected": "yes" if h0_rejected else "no"
        })
    chi2_results = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    print(chi2_results)

    # 2. Using Cramers V to find the strength of any association
    # Use existing `cat_columns` and target from the notebook
    target = target_column[0]  # 'y'
    def cramers_v(x, y):
        table = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(table)
        n = table.to_numpy().sum()
        r, c = table.shape
        denom = n * (min(r, c) - 1)
        return np.sqrt(chi2 / denom) if denom > 0 else np.nan
    cramers_v_results = pd.DataFrame(
        [{"feature": col, "cramers_v": cramers_v(train_data[col], train_data[target])} for col in cat_columns]
    ).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    print(cramers_v_results)
    

    # 3. Mutual Information:
    X_cat = train_data[cat_columns]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_encoded = encoder.fit_transform(X_cat)
    mi_scores = mutual_info_classif(
        X_encoded,
        train_data['y'],
        discrete_features=True,
        random_state=13
    )
    mi_table = pd.DataFrame({
        "feature": cat_columns,
        "mutual_information": mi_scores
    }).sort_values("mutual_information", ascending=False).reset_index(drop=True)
    print(mi_table)

    # Numerical feature selection:
    numeric_cols = ['age','balance','duration','campaign']
    y = train_data['y']
    corr_df = train_data[numeric_cols].copy()
    corr_df["y_bin"] = y
    corr_matrix = corr_df.corr(method="pearson")
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True
    )
    plt.title("Pearson Correlation Matrix (Numeric Columns and y)")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(figures_directory / "pearson_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == "__main__":
    app()

# Custom feature selection and engineering objects/functions for the model pipeline:

class FeatureDropper(BaseEstimator, TransformerMixin):
    """
    Delete selected dataframe columns.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureDropper requires a pandas DataFrame."
            )

        self.columns_ = list(self.columns or [])

        missing_columns = [
            column
            for column in self.columns_
            if column not in X.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Columns not found: {missing_columns}"
            )

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureDropper requires a pandas DataFrame."
            )

        return X.drop(
            columns=self.columns_
        ).copy()

class CyclicalDateEncoder(BaseEstimator, TransformerMixin):
    """
    Convert day and month into sine/cosine cyclical features.

    Creates:
    - day_sin
    - day_cos
    - month_sin
    - month_cos

    The original day and month columns are then removed.
    """

    def __init__(
        self,
        day_column="day",
        month_column="month",
        drop_original=True
    ):
        self.day_column = day_column
        self.month_column = month_column
        self.drop_original = drop_original

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CyclicalDateEncoder requires a pandas DataFrame."
            )

        required_columns = [
            self.day_column,
            self.month_column
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in X.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing cyclical columns: {missing_columns}"
            )

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "CyclicalDateEncoder requires a pandas DataFrame."
            )

        X = X.copy()

        month_mapping = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12
        }

        day_numeric = pd.to_numeric(
            X[self.day_column],
            errors="coerce"
        )

        if pd.api.types.is_numeric_dtype(
            X[self.month_column]
        ):
            month_numeric = pd.to_numeric(
                X[self.month_column],
                errors="coerce"
            )
        else:
            month_numeric = (
                X[self.month_column]
                .astype("string")
                .str.strip()
                .str.lower()
                .map(month_mapping)
            )

        # Day cycle: 1 to 31
        X["day_sin"] = np.sin(
            2 * np.pi * (day_numeric - 1) / 31
        )

        X["day_cos"] = np.cos(
            2 * np.pi * (day_numeric - 1) / 31
        )

        # Month cycle: 1 to 12
        X["month_sin"] = np.sin(
            2 * np.pi * (month_numeric - 1) / 12
        )

        X["month_cos"] = np.cos(
            2 * np.pi * (month_numeric - 1) / 12
        )

        if self.drop_original:
            X = X.drop(
                columns=[
                    self.day_column,
                    self.month_column
                ]
            )

        return X
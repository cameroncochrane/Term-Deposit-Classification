# The model pipeline uses two custom transformers that need to be located by the unpacked pipeline. Place them here:

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
            columns=self.columns,
            errors="ignore"
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


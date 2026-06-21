from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from deposit_classification.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "term-deposit-marketing-2020.csv",
    X_train_raw_path: Path = INTERIM_DATA_DIR / "X_train_raw.pkl",
    y_train_raw_path: Path = INTERIM_DATA_DIR / "y_train_raw.pkl",
    X_test_raw_path: Path = INTERIM_DATA_DIR / "X_test_raw.pkl",
    y_test_raw_path: Path = INTERIM_DATA_DIR / "y_test_raw.pkl",
    X_train_path: Path = PROCESSED_DATA_DIR / "X_train.pkl",
    y_train_path: Path = PROCESSED_DATA_DIR / "y_train.pkl",
    X_test_path: Path = PROCESSED_DATA_DIR / "X_test.pkl",
    y_test_path: Path = PROCESSED_DATA_DIR / "y_test.pkl",
):
    raw_data = pd.read_csv(input_path)

    # Separate features and target
    X = raw_data.drop(columns='y')
    y = raw_data['y']

    # Train-test split, with stratify='y' to capture the class imbalance of 'y'.
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.3,random_state=13,stratify=y)

    # Map 1/0 onto 'yes'/'no'
    yes_no_map = {"no": 0, "yes": 1}
    y_train_raw = y_train_raw.map(yes_no_map).astype("int8")
    y_test_raw = y_test_raw.map(yes_no_map).astype("int8")

    def clean_and_impute_contact(X, y, unknown_value="unknown"):
        """
        Remove rows with unknown/missing job or education, then impute unknown/missing
        contact using grouped mode based on job and education.

        Imputation hierarchy:
        1. mode(contact) by job + education
        2. mode(contact) by job
        3. mode(contact) by education
        4. global mode(contact)

        Returns
        -------
        X_clean, y_clean
        """

        X = X.copy()
        y = y.copy()

        # Ensure y is a Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must be a Series or a single-column DataFrame.")
            y = y.iloc[:, 0]

        # Check index alignment before concat
        shared_index = X.index.intersection(y.index)

        if len(shared_index) == 0:
            raise ValueError(
                "X and y have no overlapping index values. "
                "This is why concat(join='inner') returns an empty dataframe. "
                "Reset both indexes or align them before calling this function."
            )

        target_col = y.name if y.name is not None else "__target__"

        # Avoid target name collision
        if target_col in X.columns:
            target_col = "__target__"

        data = pd.concat(
            [X, y.rename(target_col)],
            axis=1,
            join="inner"
        )

        if data.empty:
            raise ValueError("Data is empty after concatenating X and y.")

        required_cols = ["job", "education", "contact"]

        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Treat both "unknown" and NaN as missing
        for col in ["job", "education", "contact"]:
            data[col] = data[col].replace(unknown_value, np.nan)

        # Remove rows where job or education is missing
        data = data.dropna(subset=["job", "education"]).copy()

        if data.empty:
            raise ValueError(
                "Data is empty after dropping rows with missing job or education. "
                "Check how many rows contain unknown/NaN in these columns."
            )

        def first_mode(s):
            m = s.dropna().mode()
            return m.iloc[0] if not m.empty else np.nan

        known_contact = data.dropna(subset=["contact"]).copy()

        if known_contact.empty:
            raise ValueError(
                "No known contact values are available to learn imputation rules from."
            )

        # 1. Grouped mode by job + education
        group_mode = (
            known_contact
            .groupby(["job", "education"])["contact"]
            .agg(first_mode)
        )

        # 2. Fallback by job
        job_mode = (
            known_contact
            .groupby("job")["contact"]
            .agg(first_mode)
        )

        # 3. Fallback by education
        edu_mode = (
            known_contact
            .groupby("education")["contact"]
            .agg(first_mode)
        )

        # 4. Global fallback
        global_mode = first_mode(known_contact["contact"])

        unknown_mask = data["contact"].isna()

        if unknown_mask.any():

            keys = pd.Series(
                list(zip(
                    data.loc[unknown_mask, "job"],
                    data.loc[unknown_mask, "education"]
                )),
                index=data.loc[unknown_mask].index
            )

            imputed = keys.map(group_mode)

            # Fallback 1: job mode
            na_idx = imputed[imputed.isna()].index
            if len(na_idx) > 0:
                imputed.loc[na_idx] = data.loc[na_idx, "job"].map(job_mode)

            # Fallback 2: education mode
            na_idx = imputed[imputed.isna()].index
            if len(na_idx) > 0:
                imputed.loc[na_idx] = data.loc[na_idx, "education"].map(edu_mode)

            # Fallback 3: global mode
            imputed = imputed.fillna(global_mode)

            data.loc[unknown_mask, "contact"] = imputed

        X_clean = data.drop(columns=[target_col])
        y_clean = data[target_col]

        if y.name is not None:
            y_clean = y_clean.rename(y.name)

        return X_clean, y_clean
    
    # Apply independently to train and test
    X_train, y_train = clean_and_impute_contact(X_train_raw,y_train_raw)
    X_test, y_test = clean_and_impute_contact(X_test_raw,y_test_raw)

    # Reset indexes after deleting rows
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\nFiltered training shape:", X_train.shape)
    print("Filtered test shape:", X_test.shape)

    print(
        "Training X/y aligned:",
        len(X_train) == len(y_train)
    )

    print(
        "Test X/y aligned:",
        len(X_test) == len(y_test)
    )

    # Save raw train/test data as pkl files (post split, pre imputation)
    with open(X_train_raw_path, "wb") as f:
        pickle.dump(X_train_raw, f)
    
    with open(y_train_raw_path, "wb") as f:
        pickle.dump(y_train_raw, f)
    
    with open(X_test_raw_path, "wb") as f:
        pickle.dump(X_test_raw, f)
    
    with open(y_test_raw_path, "wb") as f:
        pickle.dump(y_test_raw, f)



    # Save processed train/test data as pkl files. These will be fed to later scripts. (post split + imputation)
    with open(X_train_path, "wb") as f:
        pickle.dump(X_train, f)
    
    with open(y_train_path, "wb") as f:
        pickle.dump(y_train, f)
    
    with open(X_test_path, "wb") as f:
        pickle.dump(X_test, f)
    
    with open(y_test_path, "wb") as f:
        pickle.dump(y_test, f)
    
    


if __name__ == "__main__":
    app()

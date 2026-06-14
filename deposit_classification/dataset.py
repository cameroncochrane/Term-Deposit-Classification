from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd

from deposit_classification.config import RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "term-deposit-marketing-2020.csv",
    X_train_path: Path = INTERIM_DATA_DIR / "X_train.pkl",
    y_train_path: Path = INTERIM_DATA_DIR / "y_train.pkl",
    X_test_path: Path = INTERIM_DATA_DIR / "X_test.pkl",
    y_test_path: Path = INTERIM_DATA_DIR / "y_test.pkl",
):
    raw_data = pd.read_csv(input_path)

    # Train test split data 
    # Imputation (deletion) of 'job' and 'education' 'unknown' rows for both the train and test datasets.
    # Conversion of 'yes' / 'no' to binary in y_train and y_test
    # Save to the interim directory (paths defined above)


    


if __name__ == "__main__":
    app()

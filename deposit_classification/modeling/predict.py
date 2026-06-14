from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from deposit_classification.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = INTERIM_DATA_DIR / "xxx", # X_test
    labels_path: Path = INTERIM_DATA_DIR / "xxx", # y_test
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "xxx",
    probabilities_path: Path = PROCESSED_DATA_DIR / "xxx"
    # -----------------------------------------
):
    # Import main model pipeline from 'train.py' here (load pkl) and use .predict functionality.
    

if __name__ == "__main__":
    app()

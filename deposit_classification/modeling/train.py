from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from deposit_classification.config import MODELS_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
        # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
        features_path: Path = INTERIM_DATA_DIR / "features.csv", # X_train
        labels_path: Path = INTERIM_DATA_DIR / "labels.csv", # y-train
        prep_components_path: Path = INTERIM_DATA_DIR / "xxxx", # 'feature selection', 'cyclical_encoding', 'preprocessor', and 'sampler' components written in 'features.py' (If saved as pkl files)
        model_path: Path = MODELS_DIR / "xxxx"
        # -----------------------------------------
    ):
        # Import data preparation components of the overall model pipeline from 'features.py' here
        
        # Define (and use) the overall modelling pipeline here

        # Import the raw + imputed  split data from 'data/interim' here

        # Train

        # Save the 'best_pipeline' as a pkl file in 'models'

        


if __name__ == "__main__":
    app()

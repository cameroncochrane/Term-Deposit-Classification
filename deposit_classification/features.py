from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from deposit_classification.config import INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    output_directory: Path = INTERIM_DATA_DIR / "xxx" # Place the pipeline components here as pkl files.
):
    
    # Here we will define (not use) the 'feature selection', 'cyclical_encoding', 'preprocessor', and 'sampler' components of the overall model pipeline. Make sure they can be exported (either with a standard import in train/predict.py or save as pkl object)

if __name__ == "__main__":
    app()

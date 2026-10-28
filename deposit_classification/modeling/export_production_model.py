import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import typer

from deposit_classification.config import (
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
)
from deposit_classification.features import FeatureDropper, CyclicalDateEncoder

app = typer.Typer()

APP_DIR = PROJ_ROOT / "app"
DEFAULT_OUTPUT_PATH = APP_DIR / "models" / "model_pipeline_2.joblib"

sys.path.insert(0, str(APP_DIR))
import api.transformers as api_transformers  # noqa: E402


@app.command()
def main(
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    alias: str = "production",
    output_path: Path = DEFAULT_OUTPUT_PATH,
):
    """
    Pull the model version currently tagged `alias` (default "production") from the
    MLflow Model Registry and write it into app/models/, re-pointed so the Lambda
    container - which only ships app/, not this package - can unpickle it on its own.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

    # Re-point the pickled transformer classes at app/api/transformers.py (identical
    # class bodies) instead of deposit_classification.features - same fix as phase 2,
    # now reusable instead of a one-off script.
    FeatureDropper.__module__ = "api.transformers"
    CyclicalDateEncoder.__module__ = "api.transformers"
    api_transformers.FeatureDropper = FeatureDropper
    api_transformers.CyclicalDateEncoder = CyclicalDateEncoder

    tmp_path = output_path.with_suffix(".tmp")
    joblib.dump(pipeline, tmp_path)
    tmp_path.replace(output_path)

    print(f"Exported {model_name}@{alias} -> {output_path}")


if __name__ == "__main__":
    app()

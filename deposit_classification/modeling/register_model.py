import json

import mlflow
import typer
from mlflow import MlflowClient

from deposit_classification.config import (
    LATEST_RUN_INFO_PATH,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

app = typer.Typer()


@app.command()
def main(
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
):
    """
    Register the model logged by the most recent train.py run into the MLflow
    Model Registry. New versions start tagged "staging" - promote_model.py decides
    whether a version goes on to "production".
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run_info = json.loads(LATEST_RUN_INFO_PATH.read_text())
    model_uri = run_info["model_uri"]

    model_version = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.set_model_version_tag(
        model_name, model_version.version, "validation_status", "staging"
    )

    print(f"Registered {model_name} version {model_version.version} (run {run_info['run_id']})")
    print("Tagged validation_status=staging")


if __name__ == "__main__":
    app()

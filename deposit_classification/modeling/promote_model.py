import json
from typing import Optional

import mlflow
import typer
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from deposit_classification.config import (
    LATEST_RUN_INFO_PATH,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

app = typer.Typer()

# Promotion criterion: a candidate version only replaces the current "production"
# version if it does not regress on either metric below. Recall matters alongside F1
# here because a false negative (missing a would-be subscriber) is the costlier
# error for this business problem.
PROMOTION_METRICS = ["test_f1_score", "test_recall"]


@app.command()
def main(
    model_name: str = MLFLOW_REGISTERED_MODEL_NAME,
    version: Optional[str] = None,
):
    """
    Promote a registered model version to the "production" alias, but only if it
    does not regress on PROMOTION_METRICS relative to whatever is currently production.
    Defaults to the version register_model.py most recently registered.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if version is None:
        run_info = json.loads(LATEST_RUN_INFO_PATH.read_text())
        candidates = client.search_model_versions(
            f"name='{model_name}' and run_id='{run_info['run_id']}'"
        )
        if not candidates:
            raise typer.BadParameter(
                f"No registered version of {model_name} found for the latest run. "
                "Run register_model.py first, or pass --version explicitly."
            )
        version = candidates[0].version

    candidate_mv = client.get_model_version(model_name, version)
    candidate_metrics = client.get_run(candidate_mv.run_id).data.metrics

    try:
        current_prod_mv = client.get_model_version_by_alias(model_name, "production")
    except MlflowException:
        current_prod_mv = None

    if current_prod_mv is None:
        promote = True
        print("No existing production version - promoting unconditionally.")
    else:
        current_metrics = client.get_run(current_prod_mv.run_id).data.metrics
        regressions = [
            metric
            for metric in PROMOTION_METRICS
            if candidate_metrics.get(metric, float("-inf"))
            < current_metrics.get(metric, float("-inf"))
        ]
        promote = not regressions

        for metric in PROMOTION_METRICS:
            print(
                f"{metric}: candidate={candidate_metrics.get(metric)} "
                f"current_production={current_metrics.get(metric)}"
            )

        if not promote:
            print(f"Refusing to promote version {version}: regression on {regressions}")

    if not promote:
        raise typer.Exit(code=1)

    client.set_registered_model_alias(model_name, "production", version)
    client.set_model_version_tag(model_name, version, "validation_status", "production")

    if current_prod_mv is not None:
        client.set_model_version_tag(
            model_name, current_prod_mv.version, "validation_status", "archived"
        )

    print(f"Promoted {model_name} version {version} to production.")


if __name__ == "__main__":
    app()

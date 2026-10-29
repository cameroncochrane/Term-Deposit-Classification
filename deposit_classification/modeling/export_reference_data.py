from pathlib import Path

import pickle

import typer

from deposit_classification.config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

# The deployed API only ever sees these 5 fields (api/schemas.py's PredictionInput) - everything
# else gets dropped by FeatureDropper before the model ever sees it, so drift monitoring only
# makes sense on this subset.
SERVING_COLUMNS = ["duration", "day", "month", "housing", "job"]

DEFAULT_OUTPUT_PATH = PROJ_ROOT / "app" / "monitoring" / "reference_data.csv"


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_train.pkl",
    output_path: Path = DEFAULT_OUTPUT_PATH,
):
    """
    Export the training distribution's serving-relevant columns as the reference dataset
    drift_report.py compares live traffic against. Rerun this after a meaningfully different
    retrain so the reference stays representative of what the deployed model actually learned.
    """
    with open(features_path, "rb") as f:
        X = pickle.load(f)

    X[SERVING_COLUMNS].to_csv(output_path, index=False)
    print(f"Wrote {len(X)} reference rows to {output_path}")


if __name__ == "__main__":
    app()

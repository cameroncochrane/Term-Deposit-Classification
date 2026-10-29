# Entrypoint for the scheduled drift-monitor workflow: load reference + recent production data,
# run the drift check, write a report artifact, alert on drift, and exit non-zero so the calling
# workflow can branch into the retrain trigger.

from pathlib import Path

import typer

from monitoring.alerting import alert_on_drift
from monitoring.drift_report import (
    DEFAULT_DRIFT_SHARE_THRESHOLD,
    MIN_SAMPLE_SIZE,
    load_production_data,
    load_reference,
    run_drift_check,
)

app = typer.Typer()


@app.command()
def main(
    bucket: str = typer.Option(..., envvar="PREDICTIONS_BUCKET"),
    days: int = 7,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
    report_path: Path = Path("drift_report.html"),
    github_repo: str | None = typer.Option(None, envvar="GITHUB_REPOSITORY"),
    github_token: str | None = typer.Option(None, envvar="GITHUB_TOKEN"),
    run_url: str | None = typer.Option(None, envvar="GITHUB_RUN_URL"),
):
    reference = load_reference()
    current = load_production_data(bucket=bucket, days=days)

    result = run_drift_check(reference, current, drift_share_threshold=drift_share_threshold)

    print(f"Sample size: {result.sample_size}")
    if result.sample_size < MIN_SAMPLE_SIZE:
        print(
            f"Fewer than {MIN_SAMPLE_SIZE} predictions logged in the last {days} days - skipping drift check."
        )
        raise typer.Exit(code=0)

    print(f"Drifted column share: {result.drifted_share:.2f} (threshold: {drift_share_threshold})")
    print(f"Drifted columns: {result.drifted_columns}")
    print(f"Dataset drift detected: {result.dataset_drift}")

    if result.report_html:
        report_path.write_text(result.report_html)
        print(f"Report written to {report_path}")

    if result.dataset_drift and github_repo and github_token:
        alert_on_drift(github_repo, result, github_token, run_url=run_url)
        print(f"Alert posted to {github_repo} issues (label: model-drift)")

    raise typer.Exit(code=1 if result.dataset_drift else 0)


if __name__ == "__main__":
    app()

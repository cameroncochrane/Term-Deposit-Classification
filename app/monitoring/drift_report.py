# Compares live prediction traffic (logged to S3 by app/api/prediction_log.py) against the
# training-time reference distribution (reference_data.csv) using Evidently.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import boto3
from evidently import DataDefinition, Dataset, Report
from evidently.core.report import TestStatus
from evidently.presets import DataDriftPreset
import pandas as pd

NUMERICAL_COLUMNS = ["duration", "day"]
CATEGORICAL_COLUMNS = ["month", "housing", "job"]
SERVING_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

REFERENCE_DATA_PATH = Path(__file__).resolve().parent / "reference_data.csv"
DEFAULT_DRIFT_SHARE_THRESHOLD = 0.4

# Below this many logged predictions, a drift verdict isn't statistically meaningful - treat it
# as "not enough data yet" rather than running Evidently on a handful of noisy rows.
MIN_SAMPLE_SIZE = 30


@dataclass
class DriftResult:
    dataset_drift: bool
    drifted_share: float
    drifted_columns: list[str]
    sample_size: int
    report_html: str | None
    report_dict: dict | None


def load_reference(path: Path = REFERENCE_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_production_data(
    bucket: str,
    days: int = 7,
    max_objects: int = 2000,
    s3_client=None,
) -> pd.DataFrame:
    """
    Pull recent logged prediction inputs from S3 and return them as a DataFrame with just the
    serving columns (ignores prediction/probability/timestamp - drift is checked on inputs).
    """
    s3 = s3_client or boto3.client("s3")
    records: list[dict] = []
    today = datetime.now(UTC).date()

    for offset in range(days):
        if len(records) >= max_objects:
            break
        date_str = (today - timedelta(days=offset)).isoformat()
        prefix = f"predictions/{date_str}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if len(records) >= max_objects:
                    break
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                records.append(json.loads(body))

    if not records:
        return pd.DataFrame(columns=SERVING_COLUMNS)

    return pd.DataFrame(records)[SERVING_COLUMNS]


def run_drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
) -> DriftResult:
    """
    Compare `current` against `reference` on the serving columns and return a verdict. Pure
    function over DataFrames - doesn't touch S3/disk itself, so tests can feed it synthetic data.
    """
    sample_size = len(current)

    if sample_size < MIN_SAMPLE_SIZE:
        return DriftResult(
            dataset_drift=False,
            drifted_share=0.0,
            drifted_columns=[],
            sample_size=sample_size,
            report_html=None,
            report_dict=None,
        )

    data_definition = DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    # include_tests=True gives an explicit pass/fail per metric via TestStatus, which is what we
    # read below - Evidently auto-picks the stat test per column (p-value based for small
    # samples, distance based for large ones), and those two use *opposite* comparison
    # directions against their threshold, so re-deriving pass/fail from the raw metric value
    # ourselves is fragile. Trust Evidently's own verdict instead.
    report = Report(
        metrics=[DataDriftPreset(drift_share=drift_share_threshold)], include_tests=True
    )
    run = report.run(reference_data=reference_dataset, current_data=current_dataset)
    result_dict = run.dict()

    drifted_share = 0.0
    for metric in result_dict["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            drifted_share = metric["value"]["share"]

    dataset_drift = False
    drifted_columns = []
    for test in result_dict.get("tests", []):
        if test["id"] == "lt":
            dataset_drift = test["status"] != TestStatus.SUCCESS
        elif test["id"] == "drift" and test["status"] != TestStatus.SUCCESS:
            column = test["metric_config"]["params"].get("column")
            if column:
                drifted_columns.append(column)

    return DriftResult(
        dataset_drift=dataset_drift,
        drifted_share=drifted_share,
        drifted_columns=drifted_columns,
        sample_size=sample_size,
        report_html=run.get_html_str(as_iframe=False),
        report_dict=result_dict,
    )

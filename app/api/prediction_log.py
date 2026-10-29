# Best-effort logging of live prediction inputs/outputs to S3, feeding drift monitoring
# (app/monitoring/drift_report.py compares this against app/monitoring/reference_data.csv).

from datetime import UTC, datetime
import json
import logging
import os
import uuid

import boto3

from api.schemas import PredictionInput, PredictionOutput

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("PREDICTIONS_BUCKET")
_s3 = boto3.client("s3") if _BUCKET else None


def log_prediction(data: PredictionInput, result: PredictionOutput) -> None:
    """
    Log a prediction's input and output to S3. Never raises - a logging failure must not break
    the actual prediction response. No-op if PREDICTIONS_BUCKET isn't set (e.g. local dev).
    """
    if _s3 is None:
        return

    now = datetime.now(UTC)
    key = f"predictions/{now:%Y-%m-%d}/{now:%H%M%S}-{uuid.uuid4().hex}.json"
    record = {
        **data.model_dump(),
        "prediction": result.prediction,
        "probability": result.probability,
        "timestamp": now.isoformat(),
    }

    try:
        _s3.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(record).encode("utf-8"))
    except Exception:
        logger.warning("Failed to log prediction to S3", exc_info=True)

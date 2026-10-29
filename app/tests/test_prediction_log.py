import json
from unittest.mock import MagicMock

from api import prediction_log
from api.schemas import PredictionInput, PredictionOutput

DATA = PredictionInput(duration=120, day=15, month="may", housing="yes", job="technician")
RESULT = PredictionOutput(prediction=0, probability=0.1)


def test_noop_when_bucket_unset(monkeypatch):
    monkeypatch.setattr(prediction_log, "_s3", None)
    prediction_log.log_prediction(DATA, RESULT)  # must not raise


def test_writes_expected_record(monkeypatch):
    mock_s3 = MagicMock()
    monkeypatch.setattr(prediction_log, "_s3", mock_s3)
    monkeypatch.setattr(prediction_log, "_BUCKET", "test-bucket")

    prediction_log.log_prediction(DATA, RESULT)

    mock_s3.put_object.assert_called_once()
    _, kwargs = mock_s3.put_object.call_args
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"].startswith("predictions/")
    assert kwargs["Key"].endswith(".json")

    body = json.loads(kwargs["Body"])
    assert body["duration"] == 120
    assert body["month"] == "may"
    assert body["prediction"] == 0
    assert body["probability"] == 0.1
    assert "timestamp" in body


def test_s3_errors_are_swallowed(monkeypatch):
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = RuntimeError("boom")
    monkeypatch.setattr(prediction_log, "_s3", mock_s3)
    monkeypatch.setattr(prediction_log, "_BUCKET", "test-bucket")

    prediction_log.log_prediction(DATA, RESULT)  # must not raise

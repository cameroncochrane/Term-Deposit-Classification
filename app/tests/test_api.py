from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "duration": 120,
    "day": 15,
    "month": "may",
    "housing": "yes",
    "job": "technician",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_month():
    payload = {**VALID_PAYLOAD, "month": "not-a-month"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_duration():
    payload = {**VALID_PAYLOAD, "duration": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_day():
    payload = {**VALID_PAYLOAD, "day": 32}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_job():
    payload = {**VALID_PAYLOAD, "job": "astronaut"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "housing"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_field():
    payload = {**VALID_PAYLOAD, "age": 40}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

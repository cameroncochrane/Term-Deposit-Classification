import json

from api.lambda_handler import handler


def _http_api_event(method, path, body=None):
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"content-type": "application/json"},
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "test",
            },
            "domainName": "example.com",
            "stage": "$default",
        },
        "body": json.dumps(body) if body is not None else None,
        "isBase64Encoded": False,
    }


def test_health_via_lambda_handler():
    response = handler(_http_api_event("GET", "/health"), {})
    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def test_predict_via_lambda_handler():
    payload = {
        "duration": 120,
        "day": 15,
        "month": "may",
        "housing": "yes",
        "job": "technician",
    }
    response = handler(_http_api_event("POST", "/predict", payload), {})
    assert response["statusCode"] == 200

    body = json.loads(response["body"])
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0

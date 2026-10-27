FastAPI service exposing the trained model.

- `main.py` — FastAPI app with `GET /health` and `POST /predict`.
- `schemas.py` — Pydantic request/response models (`PredictionInput`, `PredictionOutput`).
- `service.py` — loads `app/models/model_pipeline_2.joblib` and runs inference.
- `transformers.py` — custom sklearn transformers (`FeatureDropper`, `CyclicalDateEncoder`) registered so joblib can unpickle the pipeline.

Run locally from the `app/` directory:

```bash
uvicorn api.main:app --reload
```

Tests live in `app/tests/test_api.py`.

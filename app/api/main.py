# FastAPI routes and application

from fastapi import FastAPI

from api.schemas import PredictionInput, PredictionOutput
from api import service

app = FastAPI(title="Term Deposit Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    return service.predict(data)

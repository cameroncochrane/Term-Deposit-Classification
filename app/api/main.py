# FastAPI routes and application

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import PredictionInput, PredictionOutput
from api.transformers import *  # Register custom transformer classes

app = FastAPI(title="Term Deposit Prediction API")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_pipeline_2.joblib"
)

pipeline = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    X = pd.DataFrame([data.model_dump()])

    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0, 1]

    return PredictionOutput(
        prediction=int(prediction),
        probability=float(probability)
    )
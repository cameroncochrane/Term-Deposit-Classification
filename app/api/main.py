# FastAPI routes and application

from fastapi import FastAPI

from api import prediction_log, service
from api.schemas import PredictionInput, PredictionOutput

app = FastAPI(title="Term Deposit Prediction API")

# FastAPI, in this context, is a fast way to interact with this project's XGBoost pipeline, primarily for inference.
# The major part here is "/predict". It calls the function that uses the model, with 'data' as the payload to make a model prediction.


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    result = service.predict(data)
    prediction_log.log_prediction(data, result)
    return result

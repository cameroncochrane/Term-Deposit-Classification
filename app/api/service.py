# Model loading and inference

from pathlib import Path

import joblib
import pandas as pd

from api.schemas import PredictionInput, PredictionOutput
from api.transformers import *  # noqa: F401,F403 - registers custom transformer classes for unpickling

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_pipeline_2.joblib"
)

pipeline = joblib.load(MODEL_PATH)


def predict(data: PredictionInput) -> PredictionOutput:
    X = pd.DataFrame([data.model_dump()])

    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0, 1]

    return PredictionOutput(
        prediction=int(prediction),
        probability=float(probability)
    )

# Model loading and inference

from pathlib import Path

import joblib
import pandas as pd

from api.schemas import PredictionInput, PredictionOutput
from api.transformers import CyclicalDateEncoder, FeatureDropper  # noqa: F401 - registers custom transformer classes for unpickling developed during model development.

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_pipeline_2.joblib"

pipeline = joblib.load(
    MODEL_PATH
)  # Load the model from the specified path. Make sure the custom column transformers developed during model development are imported.


def predict(data: PredictionInput) -> PredictionOutput:
    X = pd.DataFrame([data.model_dump()])

    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0, 1]

    return PredictionOutput(prediction=int(prediction), probability=float(probability))

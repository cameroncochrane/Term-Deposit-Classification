# Pydantic request and response schemas

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: int = Field(ge=0)
    day: int = Field(ge=1, le=31)

    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    housing: Literal["yes", "no"]
    job: Literal[
        "blue-collar",
        "management",
        "technician",
        "admin",
        "student",
        "entrepreneur",
        "services",
        "retired",
        "housemaid",
        "unemployed",
        "self-employed",
    ]


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
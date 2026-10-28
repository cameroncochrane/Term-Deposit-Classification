# Pydantic request and response schemas

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Here we are defining the strict data types that the model pipeline requires as an input.
# We can also specify what values are expected for discrete data categories e.g months, days, job type


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: int = Field(ge=0)
    day: int = Field(ge=1, le=31)

    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
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

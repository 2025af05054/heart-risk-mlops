"""Request/response models for the prediction API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """One patient record, using the original UCI coding."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }})

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type")
    trestbps: float = Field(..., gt=50, lt=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=80, lt=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., gt=50, lt=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=-3, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured (0-3); optional")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 normal, 6 fixed, 7 reversible")


class PredictRequest(BaseModel):
    """Either a single patient or a batch under `instances`."""

    instances: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["disease", "no_disease"]
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[Prediction]

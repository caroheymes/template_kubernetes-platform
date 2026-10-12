"""
schemas.py — Contrats de données et schémas Pydantic v2 pour FastAPI
"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model_loaded: bool
    database_connected: bool


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        json_schema_extra={"example": {"feature_1": 14.2, "feature_2": 0.75, "feature_3": 110.0, "feature_4": 3.2}},
        description="Dictionnaire clé-valeur des features numériques pour l'inférence XGBoost",
    )


class PredictionResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    prediction: float = Field(..., description="Score ou classe prédite par XGBoost")
    model_name: str
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_processed: int
    total_latency_ms: float


class ExplanationResponse(PredictionResponse):
    feature_contributions: dict[str, float] = Field(
        ...,
        description="Contributions SHAP/Shapash de chaque variable à la décision",
    )
    top_positive_features: list[str] = Field(default_factory=list)
    top_negative_features: list[str] = Field(default_factory=list)


class WebhookTriggerRequest(BaseModel):
    source: str = Field(..., json_schema_extra={"example": "evidently-drift-detector"})
    event_type: str = Field(..., json_schema_extra={"example": "drift_alert"})
    drift_score: float = Field(..., json_schema_extra={"example": 0.38})
    details: dict[str, Any] = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    status: str
    action_taken: str
    event_id: int | None = None

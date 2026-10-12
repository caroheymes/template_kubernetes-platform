"""Tests unitaires pour les contrats de données et schémas Pydantic v2."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    WebhookResponse,
    WebhookTriggerRequest,
)


def test_prediction_request_valid():
    """Vérifie la validation conforme d'une requête de prédiction."""
    payload = {
        "features": {"feature_1": 1.5, "feature_2": -0.4, "feature_3": 100.0, "feature_4": 2.1}
    }
    req = PredictionRequest(**payload)
    assert req.features["feature_1"] == 1.5
    assert len(req.features) == 4


def test_prediction_request_invalid():
    """Vérifie le rejet d'une requête sans features numériques."""
    with pytest.raises(ValidationError):
        PredictionRequest(features="invalid_not_a_dict")


def test_prediction_response_schema():
    """Vérifie la génération d'UUID et les types du contrat de réponse."""
    resp = PredictionResponse(
        prediction=42.1234,
        model_name="test-xgboost",
        model_version="v1",
        latency_ms=12.45,
    )
    assert isinstance(resp.request_id, UUID)
    assert resp.prediction == 42.1234
    assert resp.latency_ms == 12.45


def test_batch_prediction_schema():
    """Vérifie les requêtes et réponses par lots."""
    req = BatchPredictionRequest(
        items=[
            PredictionRequest(features={"feature_1": 1.0}),
            PredictionRequest(features={"feature_1": 2.0}),
        ]
    )
    assert len(req.items) == 2

    resp = BatchPredictionResponse(
        predictions=[
            PredictionResponse(prediction=1.0, model_name="m", model_version="1", latency_ms=1.1),
            PredictionResponse(prediction=2.0, model_name="m", model_version="1", latency_ms=1.2),
        ],
        total_processed=2,
        total_latency_ms=2.3,
    )
    assert resp.total_processed == 2


def test_explanation_response_schema():
    """Vérifie l'héritage de PredictionResponse et les listes de polarités."""
    exp = ExplanationResponse(
        prediction=10.5,
        model_name="xgb",
        model_version="v1",
        latency_ms=5.0,
        feature_contributions={"f1": 0.8, "f2": -0.3},
        top_positive_features=["f1"],
        top_negative_features=["f2"],
    )
    assert exp.prediction == 10.5
    assert exp.feature_contributions["f1"] == 0.8
    assert "f1" in exp.top_positive_features
    assert "f2" in exp.top_negative_features


def test_health_response():
    """Vérifie le schéma de statut santé."""
    health = HealthResponse(status="ready", model_loaded=True, database_connected=True)
    assert health.model_loaded is True
    assert health.database_connected is True


def test_webhook_contracts():
    """Vérifie la structure des requêtes et réponses webhook."""
    req = WebhookTriggerRequest(
        source="evidently",
        event_type="drift_alert",
        drift_score=0.45,
        details={"metric": "ks_test"},
    )
    assert req.drift_score == 0.45

    res = WebhookResponse(status="accepted", action_taken="retraining_triggered", event_id=12)
    assert res.action_taken == "retraining_triggered"
    assert res.event_id == 12

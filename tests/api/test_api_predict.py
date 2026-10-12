"""Tests fonctionnels pour l'API FastAPI d'inférence et de santé."""

from fastapi.testclient import TestClient

from src.serving.main import app
from src.serving.predictor import predictor

client = TestClient(app)


def test_health_live_endpoint():
    """Vérifie la sonde Kubernetes Liveness."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready_endpoint():
    """Vérifie la sonde Kubernetes Readiness (200 si prêt, 503 si DB indisponible)."""
    from unittest.mock import MagicMock, patch

    predictor.load_model()

    # Cas 1 : Base de données disponible -> 200 OK
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    with patch("src.serving.main.get_engine", return_value=mock_engine):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["database_connected"] is True


    # Cas 2 : Base de données inaccessible -> 503 Service Unavailable
    with patch("src.serving.main.get_engine", side_effect=Exception("Database down")):
        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["database_connected"] is False




def test_metrics_endpoint():
    """Vérifie l'exposition des métriques Prometheus."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mlops_predictions_total" in response.text
    assert "mlops_prediction_latency_seconds" in response.text


def test_predict_endpoint_success():
    """Vérifie l'inférence unitaire temps réel via HTTP POST."""
    predictor.load_model()
    payload = {
        "features": {
            "feature_1": 1.5,
            "feature_2": -0.8,
            "feature_3": 12.0,
            "feature_4": 3.4,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "request_id" in data
    assert "latency_ms" in data
    assert isinstance(data["prediction"], float)


def test_predict_batch_endpoint():
    """Vérifie l'inférence par lots vectorisée."""
    predictor.load_model()
    payload = {
        "items": [
            {"features": {"feature_1": 1.0, "feature_2": 0.0, "feature_3": 1.0, "feature_4": 0.0}},
            {"features": {"feature_1": 2.0, "feature_2": 1.0, "feature_3": 2.0, "feature_4": 1.0}},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_predict_explain_endpoint():
    """Vérifie l'inférence augmentée d'explicabilité locale."""
    predictor.load_model()
    payload = {
        "features": {
            "feature_1": 2.0,
            "feature_2": -1.0,
            "feature_3": 0.5,
            "feature_4": 1.5,
        }
    }
    response = client.post("/predict/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "feature_contributions" in data
    assert "top_positive_features" in data
    assert "top_negative_features" in data
    assert len(data["feature_contributions"]) == 4

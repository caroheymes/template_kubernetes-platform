"""Tests fonctionnels pour le gestionnaire de webhooks et la sécurité."""

from fastapi.testclient import TestClient

from src.common.config import webhook_settings
from src.serving.main import app

client = TestClient(app)


def test_webhook_unauthorized_without_secret():
    """Vérifie le rejet HTTP 401 en l'absence de clé secrète webhook."""
    payload = {
        "source": "evidently-monitor",
        "event_type": "drift_alert",
        "drift_score": 0.42,
        "details": {"alert": "distribution_drift"},
    }
    response = client.post("/webhooks/trigger-retrain", json=payload)
    assert response.status_code == 401


def test_webhook_unauthorized_with_wrong_secret():
    """Vérifie le rejet HTTP 401 si le token est falsifié."""
    payload = {
        "source": "evidently-monitor",
        "event_type": "drift_alert",
        "drift_score": 0.42,
    }
    response = client.post(
        "/webhooks/trigger-retrain",
        json=payload,
        headers={"x-webhook-secret": "wrong-token"},
    )
    assert response.status_code == 401


def test_webhook_authorized_drift_retrain_triggered():
    """Vérifie l'acceptation et le déclenchement de réentraînement si dérive critique > 0.30."""
    payload = {
        "source": "evidently-monitor",
        "event_type": "drift_alert",
        "drift_score": 0.45,
        "details": {"metric": "ks_test_pvalue_drop"},
    }
    response = client.post(
        "/webhooks/trigger-retrain",
        json=payload,
        headers={"x-webhook-secret": webhook_settings.secret_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["action_taken"] == "retraining_triggered"


def test_webhook_authorized_minor_drift_logged_only():
    """Vérifie la simple journalisation si la dérive reste en deçà du seuil critique (<= 0.30)."""
    payload = {
        "source": "evidently-monitor",
        "event_type": "drift_alert",
        "drift_score": 0.15,
        "details": {"metric": "stable"},
    }
    response = client.post(
        "/webhooks/trigger-retrain",
        json=payload,
        headers={"x-webhook-secret": webhook_settings.secret_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["action_taken"] == "logged"

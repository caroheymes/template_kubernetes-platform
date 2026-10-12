"""Fixtures globales pour la suite de tests k8s_mlops_platform."""

from unittest.mock import patch

import pytest

from src.serving.predictor import XGBoostPredictor
from src.serving.webhooks import WebhookManager


@pytest.fixture(autouse=True)
def mock_db_logging():
    """Désactive l'accès réseau SQL et l'entraînement lourd en tâche de fond durant les tests API."""
    with (
        patch.object(XGBoostPredictor, "_log_prediction_to_db", return_value=None),
        patch.object(WebhookManager, "_log_event", return_value=1),
        patch.object(WebhookManager, "_trigger_async_retraining", return_value=None),
        patch.object(WebhookManager, "_update_event_status", return_value=None),
    ):
        yield

"""Tests unitaires pour le moteur d'inférence XGBoost."""

from unittest.mock import patch

from src.serving.predictor import XGBoostPredictor
from src.serving.schemas import PredictionRequest


def test_predictor_bootstrap_initialization():
    """Vérifie que le prédicteur initialise un modèle de secours autonome en l'absence de MLflow."""
    pred = XGBoostPredictor()
    assert not pred.is_ready

    # Chargement sans MLflow disponible -> active le bootstrap
    pred.load_model()
    assert pred.is_ready
    assert "bootstrap" in pred.version


def test_predictor_predict_execution():
    """Vérifie le calcul d'une inférence unitaire et le format de sortie."""
    pred = XGBoostPredictor()
    pred.load_model()

    req = PredictionRequest(
        features={"feature_1": 1.2, "feature_2": -0.5, "feature_3": 0.8, "feature_4": 2.1}
    )

    resp = pred.predict(req)
    assert resp.prediction is not None
    assert isinstance(resp.prediction, float)
    assert resp.latency_ms >= 0.0
    assert resp.model_version == pred.version


def test_predictor_handles_missing_features():
    """Vérifie la robustesse face à des colonnes partielles ou manquantes."""
    pred = XGBoostPredictor()
    pred.load_model()

    # Seulement 2 features fournies sur 4
    req = PredictionRequest(features={"feature_1": 3.0, "feature_2": 1.0})
    resp = pred.predict(req)
    assert resp.prediction is not None


def test_predictor_db_logging_resilience():
    """Vérifie que l'échec de la journalisation PostgreSQL ne bloque pas l'inférence."""
    pred = XGBoostPredictor()
    pred.load_model()

    # Dépatcher pour exécuter le vrai corps de _log_prediction_to_db avec déconnexion simulée
    from src.serving.predictor import XGBoostPredictor as RealCls

    with patch.object(
        pred,
        "_log_prediction_to_db",
        lambda *args, **kwargs: RealCls._log_prediction_to_db(pred, *args, **kwargs),
    ):
        with patch("src.serving.predictor.get_engine", side_effect=Exception("Database down")):
            req = PredictionRequest(features={"feature_1": 0.5})
            resp = pred.predict(req)
            assert resp.prediction is not None


def test_predictor_force_hot_reload():
    """Vérifie que force=True recharge le modèle même s'il est déjà instancié."""
    pred = XGBoostPredictor()
    pred.load_model()
    assert pred.is_ready is True

    # Rechargement forcé
    pred.load_model(force=True)
    assert pred.is_ready is True

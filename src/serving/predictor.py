"""
predictor.py — Moteur d'inférence XGBoost en mémoire avec rechargement à chaud
"""

import json
import logging
import time
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import text

from ..common.config import mlflow_settings
from ..common.db import get_engine
from .schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger("k8s_mlops.predictor")


class XGBoostPredictor:
    """Gestionnaire de modèle XGBoost avec chargement paresseux et mise en cache."""

    def __init__(self):
        self._model = None
        self._version = "local-bootstrap"
        self._feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
        self._engine = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    def load_model(self, force: bool = False) -> None:
        """Tente de charger le modèle champion depuis MLflow ou initialise un booster de secours."""
        if self._model is not None and not force:
            return

        if force:
            self._model = None

        is_reachable = True
        if (
            "localhost" in mlflow_settings.tracking_uri
            or "127.0.0.1" in mlflow_settings.tracking_uri
        ):
            import socket
            import urllib.parse

            parsed = urllib.parse.urlparse(mlflow_settings.tracking_uri)
            port = parsed.port or 5000
            host = parsed.hostname or "127.0.0.1"
            try:
                with socket.create_connection((host, port), timeout=0.08):
                    is_reachable = True
            except (OSError, ConnectionRefusedError):
                is_reachable = False

        if not is_reachable:
            self._initialize_bootstrap_model()
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
            model_uri = f"models:/{mlflow_settings.model_name}@{mlflow_settings.model_alias}"
            self._model = mlflow.xgboost.load_model(model_uri)
            self._version = mlflow_settings.model_alias
            logger.info(f"Modèle Champion chargé depuis MLflow : {model_uri}")
        except Exception as e:
            logger.warning(
                f"Impossible de charger depuis MLflow ({e}), initialisation du modèle de bootstrap."
            )
            self._initialize_bootstrap_model()

    def _initialize_bootstrap_model(self) -> None:
        """Génère un modèle XGBoost synthétique minimal pour permettre l'amorçage sans MLflow."""
        import xgboost as xgb

        # Dataset synthétique 100 observations
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
        dtrain = xgb.DMatrix(X, label=y, feature_names=self._feature_names, nthread=1)
        params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror", "nthread": 1}
        self._model = xgb.train(params, dtrain, num_boost_round=10)
        self._version = "bootstrap-v1"
        logger.info("Modèle XGBoost de bootstrap initialisé avec succès.")

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        """Effectue une prédiction unitaire, chronomètre la latence et historise dans PostgreSQL."""
        if not self._model:
            self.load_model()

        start_time = time.perf_counter()
        import xgboost as xgb

        df = pd.DataFrame([req.features])
        # Aligner les colonnes si nécessaire
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self._feature_names]

        dmatrix = xgb.DMatrix(df)
        raw_pred = self._model.predict(dmatrix)
        score = float(raw_pred[0])
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        request_id = uuid4()
        response = PredictionResponse(
            request_id=request_id,
            prediction=round(score, 4),
            model_name=mlflow_settings.model_name,
            model_version=self._version,
            latency_ms=round(latency_ms, 2),
        )

        # Historisation en base (tolérante aux pannes)
        self._log_prediction_to_db(request_id, req.features, score, latency_ms)

        return response

    def _log_prediction_to_db(
        self, request_id: Any, features: dict[str, float], score: float, latency: float
    ) -> None:
        """Enregistre la requête dans predictions.fact_predictions."""
        try:
            if not self._engine:
                self._engine = get_engine()
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                    INSERT INTO predictions.fact_predictions (
                        request_id, features_payload, predicted_value, latency_ms, model_version
                    ) VALUES (
                        :request_id, :payload, :score, :latency, :version
                    );
                """),
                    {
                        "request_id": str(request_id),
                        "payload": json.dumps(features),
                        "score": score,
                        "latency": latency,
                        "version": self._version,
                    },
                )
        except Exception as e:
            logger.debug(f"Journalisation SQL différée : {e}")


predictor = XGBoostPredictor()

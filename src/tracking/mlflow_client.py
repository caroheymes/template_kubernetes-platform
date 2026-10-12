"""
mlflow_client.py — Helper de journalisation MLflow et gestion du Model Registry
"""

import logging
from typing import Any

from ..common.config import mlflow_settings

logger = logging.getLogger("k8s_mlops.tracking")


class MLflowTracker:
    """Interface avec le serveur de Tracking et le Model Registry MLflow."""

    def __init__(self):
        self._uri = mlflow_settings.tracking_uri
        self._client = None

    def _is_server_reachable(self) -> bool:
        """Vérifie rapidement (80ms) si le serveur de tracking est accessible avant de tenter un appel bloquant."""
        if "localhost" in self._uri or "127.0.0.1" in self._uri:
            import socket
            import urllib.parse

            parsed = urllib.parse.urlparse(self._uri)
            port = parsed.port or 5000
            host = parsed.hostname or "127.0.0.1"
            try:
                with socket.create_connection((host, port), timeout=0.08):
                    return True
            except (OSError, ConnectionRefusedError):
                return False
        return True

    def _ensure_client(self):
        if not self._client:
            if not self._is_server_reachable():
                return
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(self._uri)
                self._client = MlflowClient()
            except Exception as e:
                logger.warning(f"Client MLflow non initialisé ({e}).")

    def log_training_run(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        model_name: str,
    ) -> str:
        """Enregistre les hyperparamètres, les métriques et l'artefact XGBoost dans un run."""
        if not self._is_server_reachable():
            logger.info("Serveur MLflow inaccessible, mode mock local.")
            return "local-mock-run-id"
        try:
            import mlflow
            import mlflow.xgboost

            mlflow.set_tracking_uri(self._uri)
            with mlflow.start_run(run_name="xgboost_training_run") as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                # Log du modèle avec signature d'inférence
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                )
                logger.info(f"Modèle et métriques enregistrés dans MLflow run : {run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.warning(f"Échec de l'enregistrement MLflow ({e}). Mode dégradé local.")
            return "local-mock-run-id"

    def set_model_alias(self, version: str, alias: str) -> None:
        """Attribue un alias (ex: 'champion', 'challenger') à une version du modèle."""
        self._ensure_client()
        if self._client:
            try:
                self._client.set_registered_model_alias(
                    name=mlflow_settings.model_name,
                    alias=alias,
                    version=version,
                )
                logger.info(f"Alias '{alias}' assigné à la version {version}.")
            except Exception as e:
                logger.error(f"Erreur d'assignation d'alias MLflow : {e}")

    def get_champion_metadata(self) -> dict[str, Any] | None:
        """Récupère les métadonnées et la version du modèle champion actuel."""
        self._ensure_client()
        if self._client:
            try:
                model_version = self._client.get_model_version_by_alias(
                    name=mlflow_settings.model_name,
                    alias="champion",
                )
                run = self._client.get_run(model_version.run_id)
                return {
                    "version": model_version.version,
                    "run_id": model_version.run_id,
                    "rmse": run.data.metrics.get("rmse", 999.0),
                }
            except Exception:
                return None
        return None

    def get_latest_model_version(self, model_name: str | None = None) -> str:
        """Récupère la dernière version numérique enregistrée pour le modèle."""
        name = model_name or mlflow_settings.model_name
        self._ensure_client()
        if self._client:
            try:
                versions = self._client.search_model_versions(f"name='{name}'")
                if versions:
                    return str(max(versions, key=lambda v: int(v.version)).version)
            except Exception:
                pass
        return "1"


mlflow_tracker = MLflowTracker()

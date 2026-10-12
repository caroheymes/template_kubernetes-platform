"""
mlflow_manager.py — Contrôleur de qualité et gouvernance MLOps (Quality Gate & Model Registry)
Permet d'évaluer, d'enregistrer et de promouvoir automatiquement les modèles avec l'alias
'@champion' uniquement si leurs performances surpassent le modèle actif en production.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from ..common.config import mlflow_settings

logger = logging.getLogger("k8s_mlops.mlflow_manager")


class MLflowQualityGate:
    """
    Contrôleur de gouvernance et d'arbitrage Champion vs Challenger.
    Supporte les métriques de régression (RMSE, MAE : minimisation)
    et de classification (F1, F2, AUPRC, R2 : maximisation).
    """

    LOWER_IS_BETTER = {"rmse", "mae", "mse", "loss", "brier_score"}

    def __init__(
        self,
        model_name: str | None = None,
        metric_target: str = "rmse",
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        min_improvement_pct: float = 2.0,
    ):
        self.model_name = model_name or mlflow_settings.model_name
        self.metric_target = metric_target.lower()
        self.experiment_name = experiment_name or mlflow_settings.experiment_name
        self.tracking_uri = tracking_uri or mlflow_settings.tracking_uri
        self.min_improvement_pct = min_improvement_pct

        self.client = None
        is_reachable = True
        if "localhost" in self.tracking_uri or "127.0.0.1" in self.tracking_uri:
            import socket
            import urllib.parse
            parsed = urllib.parse.urlparse(self.tracking_uri)
            port = parsed.port or 5000
            host = parsed.hostname or "127.0.0.1"
            try:
                with socket.create_connection((host, port), timeout=0.1):
                    is_reachable = True
            except (OSError, ConnectionRefusedError):
                is_reachable = False

        if is_reachable:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.client = MlflowClient(tracking_uri=self.tracking_uri)
            except Exception as e:
                logger.warning(f"MLflow client non initialisé (Mode dégradé/offline) : {e}")
                self.client = None
        else:
            logger.debug("Serveur MLflow local non joignable, initialisation du client ignorée.")


    def get_active_champion_info(self) -> dict[str, Any] | None:
        """Interroge le Model Registry pour extraire les métadonnées et le score du Champion actif."""
        if not self.client:
            return None
        try:
            champ = self.client.get_model_version_by_alias(self.model_name, "champion")
            if not champ:
                return None
            run = self.client.get_run(champ.run_id)
            return {
                "version": str(champ.version),
                "run_id": champ.run_id,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "aliases": champ.aliases,
                "current_score": run.data.metrics.get(self.metric_target),
            }
        except Exception as e:
            logger.debug(f"Aucun Champion actif identifié dans MLflow : {e}")
            return None

    def should_promote(
        self,
        candidate_score: float,
        champion_score: float | None,
    ) -> tuple[bool, float]:
        """
        Détermine si le modèle candidat dépasse le seuil d'amélioration requis.

        Returns:
            Tuple (should_promote: bool, delta_pct: float)
        """
        if champion_score is None:
            # Premier modèle : promotion automatique
            return True, 100.0

        is_minimization = self.metric_target in self.LOWER_IS_BETTER

        if is_minimization:
            # Pour RMSE/MAE : amélioration = réduction du score
            if champion_score == 0:
                delta_pct = 0.0
            else:
                denom = max(abs(champion_score), 1e-6)
                delta_pct = ((champion_score - candidate_score) / denom) * 100.0
            promoted = delta_pct >= self.min_improvement_pct
        else:
            # Pour F1/AUPRC/R2 : amélioration = augmentation du score
            if champion_score == 0:
                delta_pct = 100.0 if candidate_score > 0 else 0.0
            else:
                denom = max(abs(champion_score), 1e-6)
                delta_pct = ((candidate_score - champion_score) / denom) * 100.0
            promoted = delta_pct >= self.min_improvement_pct

        return promoted, round(delta_pct, 2)

    def log_and_evaluate(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        confusion_matrix_dict: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Enregistre le run MLflow, publie le modèle dans le registre et applique le Quality Gate.

        Returns:
            Tuple (is_promoted: bool, target_version: str)
        """
        candidate_score = metrics.get(self.metric_target)
        if candidate_score is None:
            candidate_score = next(iter(metrics.values())) if metrics else 0.0

        if not self.client:
            logger.warning("MLflow non disponible : promotion locale par défaut.")
            return True, "v-offline"

        try:
            with mlflow.start_run(run_name=f"Run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}") as run:
                if params:
                    mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                if tags:
                    mlflow.set_tags(tags)

                if confusion_matrix_dict:
                    import tempfile
                    with tempfile.NamedTemporaryFile("w", suffix="_cm.json", delete=False, encoding="utf-8") as f:
                        json.dump(confusion_matrix_dict, f, indent=2)
                        temp_cm_path = f.name
                    try:
                        mlflow.log_artifact(temp_cm_path, artifact_path="evaluation")
                    finally:
                        if os.path.exists(temp_cm_path):
                            os.remove(temp_cm_path)

                # Priorité au flavor natif mlflow.xgboost pour XGBoostPredictor
                try:
                    import xgboost as xgb
                    if isinstance(model, (xgb.XGBModel, xgb.Booster, xgb.core.Booster)):
                        mlflow.xgboost.log_model(
                            xgb_model=model,
                            artifact_path="model",
                            registered_model_name=self.model_name,
                        )
                    else:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            registered_model_name=self.model_name,
                        )
                except Exception:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=self.model_name,
                    )

                versions = self.client.search_model_versions(f"name='{self.model_name}'")
                target_version = str(max(versions, key=lambda v: int(v.version)).version) if versions else "1"

            # Évaluation Quality Gate
            champ_info = self.get_active_champion_info()
            champ_score = champ_info["current_score"] if champ_info else None

            is_promoted, delta = self.should_promote(candidate_score, champ_score)

            if is_promoted:
                self.client.set_registered_model_alias(self.model_name, "champion", target_version)
                logger.info(f"🏆 Modèle v{target_version} promu @champion (Gain: {delta}% sur {self.metric_target}).")
            else:
                self.client.set_registered_model_alias(self.model_name, "challenger-rejected", target_version)
                logger.info(f"⚠️ Modèle v{target_version} rejeté (Gain insuffisant: {delta}% < {self.min_improvement_pct}%).")

            return is_promoted, target_version
        except Exception as e:
            logger.error(f"Erreur lors de la gouvernance MLflow : {e}")
            return False, "error"

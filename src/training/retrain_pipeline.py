"""
retrain_pipeline.py — Pipeline autonome de réentraînement en boucle fermée (Closed-Loop MLOps)
Intègre la recherche d'hyperparamètres (HPO) sur Ray, l'entraînement XGBoost, l'évaluation et l'auto-promotion.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ..common.config import mlflow_settings
from ..serving.predictor import predictor
from ..tracking.mlflow_client import mlflow_tracker
from .evaluate import evaluate_and_promote
from .hpo_optuna import run_hpo_study
from .train_xgboost import train_model

logger = logging.getLogger("k8s_mlops.retrain_pipeline")


def run_closed_loop_retraining(
    event_details: dict[str, Any] | None = None,
    use_ray: bool = True,
    n_trials: int = 15,
) -> dict[str, Any]:
    """Exécute la chaîne complète : HPO Ray -> Retrain XGBoost -> Évaluation -> Auto-Promotion -> Hot Reload."""
    start_time = datetime.now(timezone.utc).isoformat()
    drift_type = (event_details or {}).get("drift_type", "triggered_manual")
    logger.info(f"🚀 Lancement de la boucle de réentraînement (Cause: {drift_type})...")

    # 1. Optimisation bayésienne distribuée avec Ray et Optuna
    logger.info("Étape 1/4 : Recherche d'hyperparamètres HPO...")
    hpo_results = run_hpo_study(n_trials=n_trials, use_ray=use_ray)
    best_params = hpo_results.get("best_params", {})
    logger.info(f"Hyperparamètres optimaux identifiés : {best_params}")

    # 2. Entraînement du modèle Challenger avec les paramètres optimaux
    logger.info("Étape 2/4 : Entraînement complet du modèle Challenger...")
    booster, metrics = train_model(params=best_params)
    challenger_rmse = metrics["rmse"]
    logger.info(f"Modèle Challenger entraîné : RMSE = {challenger_rmse:.4f}")

    # 3. Évaluation Champion vs Challenger & Auto-promotion
    logger.info("Étape 3/4 : Évaluation et décision d'arbitrage Champion vs Challenger...")
    candidate_version = mlflow_tracker.get_latest_model_version(model_name=mlflow_settings.model_name)
    is_promoted = evaluate_and_promote(
        challenger_rmse=challenger_rmse,
        challenger_version=candidate_version,
        threshold_improvement_pct=2.0,
    )

    # 4. Rechargement à chaud (Hot-Reload) dans le service d'inférence
    if is_promoted:
        logger.info("Étape 4/4 : Rechargement à chaud du nouveau Champion dans le predictor...")
        predictor.load_model(force=True)
    else:
        logger.info("Étape 4/4 : Le Champion en titre est maintenu en production.")


    pipeline_result = {
        "status": "completed",
        "started_at": start_time,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "drift_cause": drift_type,
        "hpo": hpo_results,
        "metrics": metrics,
        "promoted_to_champion": is_promoted,
        "model_version": candidate_version if is_promoted else "previous_champion_retained",
    }
    logger.info(f"✅ Pipeline de réentraînement terminé avec succès (Promu: {is_promoted}).")
    return pipeline_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_closed_loop_retraining(use_ray=False, n_trials=5)


"""
evaluate.py — Évaluation comparative Champion vs Challenger et auto-promotion
"""

import logging

from ..common.config import mlflow_settings
from ..tracking.mlflow_client import mlflow_tracker

logger = logging.getLogger("k8s_mlops.evaluate")


def evaluate_and_promote(
    challenger_rmse: float,
    challenger_version: str,
    threshold_improvement_pct: float = 2.0,
) -> bool:
    """Compare le Challenger au Champion en titre et applique l'alias 'champion' si meilleur."""
    logger.info(f"Évaluation du Challenger (v{challenger_version}) : RMSE = {challenger_rmse:.4f}")

    champion_meta = mlflow_tracker.get_champion_metadata()
    if not champion_meta:
        logger.info("Aucun Champion existant. Promotion automatique du premier modèle.")
        mlflow_tracker.set_model_alias(challenger_version, "champion")
        return True

    champion_rmse = champion_meta.get("rmse", 999.0)
    logger.info(f"Performance Champion actuel : RMSE = {champion_rmse:.4f}")

    # Calcul de l'amélioration relative en pourcentage (sécurisé contre la division par zéro)
    improvement_pct = ((champion_rmse - challenger_rmse) / max(champion_rmse, 1e-6)) * 100.0
    logger.info(f"Gain relatif : {improvement_pct:.2f}% (Seuil requis : {threshold_improvement_pct}%)")


    if improvement_pct >= threshold_improvement_pct:
        logger.info(f"🏆 Challenger accepté ! Promotion de la version {challenger_version} au statut Champion.")
        # Historisation de l'ancien champion en rollback
        mlflow_tracker.set_model_alias(champion_meta["version"], "rollback")
        mlflow_tracker.set_model_alias(challenger_version, "champion")
        return True
    else:
        logger.info(f"❌ Gain insuffisant ({improvement_pct:.2f}% < {threshold_improvement_pct}%). Challenger rejeté.")
        mlflow_tracker.set_model_alias(challenger_version, "challenger-rejected")
        return False

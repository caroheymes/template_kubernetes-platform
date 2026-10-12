"""
drift_detector.py — Détection de dérive statistique de données (Data Drift) et de concept (Concept Drift)
"""

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

from ..common.config import storage_settings
from ..common.storage import storage_client
from ..serving.schemas import WebhookTriggerRequest
from ..serving.webhooks import webhook_manager

logger = logging.getLogger("k8s_mlops.drift")


class StatisticalDriftDetector:
    """Détecte la dérive de distribution des features (Data Drift) et la dégradation de la relation P(Y|X) (Concept Drift)."""

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        drift_share_threshold: float = 0.30,
        concept_drift_ratio_threshold: float = 1.15,
    ):
        self.p_value_threshold = p_value_threshold
        self.drift_share_threshold = drift_share_threshold
        self.concept_drift_ratio_threshold = concept_drift_ratio_threshold

    def calculate_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Exécute un test de Kolmogorov-Smirnov (KS-test) sur chaque variable numérique (Covariate Shift)."""
        cols = features or [
            c
            for c in reference_data.columns
            if c in current_data.columns and c not in ["actual", "prediction"]
        ]
        drifted_features = []
        feature_metrics = {}

        for col in cols:
            ref_s = pd.to_numeric(reference_data[col], errors="coerce").dropna()
            cur_s = pd.to_numeric(current_data[col], errors="coerce").dropna()
            if len(ref_s) < 2 or len(cur_s) < 2:
                feature_metrics[col] = {
                    "ks_statistic": 0.0,
                    "p_value": 1.0,
                    "is_drifted": False,
                    "reason": "Échantillon insuffisant ou non numérique",
                }
                continue

            stat, p_value = ks_2samp(ref_s, cur_s)
            is_drifted = bool(p_value < self.p_value_threshold)
            if is_drifted:
                drifted_features.append(col)

            feature_metrics[col] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 5),
                "is_drifted": is_drifted,
            }

        drift_share = len(drifted_features) / max(len(cols), 1)
        dataset_drifted = bool(drift_share >= self.drift_share_threshold)

        return {
            "drift_share": round(drift_share, 4),
            "dataset_drifted": dataset_drifted,
            "drifted_features_count": len(drifted_features),
            "total_features": len(cols),
            "drifted_features": drifted_features,
            "feature_details": feature_metrics,
        }

    def calculate_concept_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        actual_col: str = "actual",
        pred_col: str = "prediction",
    ) -> dict[str, Any]:
        """Détecte l'altération de la performance prédictive (Concept Drift / P(Y|X)) via analyse des résidus."""
        if (
            actual_col not in reference_data.columns
            or actual_col not in current_data.columns
            or pred_col not in reference_data.columns
            or pred_col not in current_data.columns
        ):
            return {
                "concept_drifted": False,
                "reason": "Colonnes de vérité terrain ou de prédiction absentes",
            }

        ref_clean = reference_data[[actual_col, pred_col]].dropna()
        curr_clean = current_data[[actual_col, pred_col]].dropna()

        if len(ref_clean) < 2 or len(curr_clean) < 2:
            return {
                "concept_drifted": False,
                "reason": "Données insuffisantes après suppression des valeurs manquantes",
            }

        ref_residuals = np.abs(ref_clean[actual_col] - ref_clean[pred_col])
        curr_residuals = np.abs(curr_clean[actual_col] - curr_clean[pred_col])

        ref_rmse = float(np.sqrt(np.mean((ref_clean[actual_col] - ref_clean[pred_col]) ** 2)))
        curr_rmse = float(np.sqrt(np.mean((curr_clean[actual_col] - curr_clean[pred_col]) ** 2)))

        rmse_degradation_ratio = curr_rmse / max(ref_rmse, 1e-6)

        # Test non-paramétrique de Mann-Whitney sur la distribution des erreurs absolues
        try:
            stat, p_value = mannwhitneyu(curr_residuals, ref_residuals, alternative="greater")
        except Exception:
            _stat, p_value = 0.0, 1.0

        is_concept_drifted = bool(
            (rmse_degradation_ratio >= self.concept_drift_ratio_threshold)
            and (p_value < self.p_value_threshold)
        )

        return {
            "concept_drifted": is_concept_drifted,
            "ref_rmse": round(ref_rmse, 4),
            "curr_rmse": round(curr_rmse, 4),
            "rmse_degradation_ratio": round(rmse_degradation_ratio, 3),
            "mannwhitney_p_value": round(float(p_value), 5),
        }

    def calculate_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
        actual_col: str = "actual",
        pred_col: str = "prediction",
    ) -> dict[str, Any]:
        """Analyse complète combinant Data Drift et Concept Drift avec déclenchement d'alerte unifié."""
        data_drift_report = self.calculate_data_drift(reference_data, current_data, features)
        concept_drift_report = self.calculate_concept_drift(
            reference_data, current_data, actual_col, pred_col
        )

        has_data_drift = data_drift_report["dataset_drifted"]
        has_concept_drift = concept_drift_report.get("concept_drifted", False)
        requires_retraining = has_data_drift or has_concept_drift

        drift_type = "none"
        if has_data_drift and has_concept_drift:
            drift_type = "dual_drift"
        elif has_data_drift:
            drift_type = "data_drift"
        elif has_concept_drift:
            drift_type = "concept_drift"

        # Score global synthétique pour l'alerte
        overall_drift_score = max(
            data_drift_report["drift_share"],
            min(1.0, max(0.0, concept_drift_report.get("rmse_degradation_ratio", 1.0) - 1.0)),
        )

        full_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_type": drift_type,
            "requires_retraining": requires_retraining,
            "overall_drift_score": round(overall_drift_score, 4),
            "data_drift": data_drift_report,
            "concept_drift": concept_drift_report,
            # Rétrocompatibilité avec l'ancien schéma
            "drift_share": data_drift_report["drift_share"],
            "dataset_drifted": has_data_drift,
            "drifted_features_count": data_drift_report["drifted_features_count"],
            "total_features": data_drift_report["total_features"],
            "drifted_features": data_drift_report["drifted_features"],
            "details": data_drift_report["feature_details"],
        }

        # Sauvegarde du rapport d'audit sur S3 / stockage local
        key = f"reports/drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        storage_client.put_json(storage_settings.bucket_reports, key, full_report)
        logger.info(f"Rapport de dérive archivé ({key}) - Type: {drift_type}")

        # Déclenchement automatique du webhook si dérive détectée
        if requires_retraining:
            logger.warning(
                f"🚨 Alerte Dérive ({drift_type.upper()}). Score: {overall_drift_score:.2f}. Notification webhook..."
            )
            webhook_req = WebhookTriggerRequest(
                source="evidently-drift-detector",
                event_type="drift_alert",
                drift_score=round(overall_drift_score, 4),
                details=full_report,
            )
            dispatched = False
            try:
                import requests

                from ..common.config import app_settings, webhook_settings

                endpoint = f"http://{app_settings.serving_host}:{app_settings.serving_port}/webhooks/retrain"
                headers = {"X-Webhook-Token": webhook_settings.secret_token}
                resp = requests.post(
                    endpoint, json=webhook_req.model_dump(), headers=headers, timeout=5
                )
                if resp.status_code == 200:
                    dispatched = True
                    logger.info(f"Webhook HTTP envoyé avec succès vers {endpoint}.")
            except Exception as e:
                logger.debug(
                    f"Service serving HTTP non joignable ({e}), passage en mode in-memory."
                )

            if not dispatched:
                webhook_manager.handle_incoming_event(webhook_req)

        return full_report


drift_detector = StatisticalDriftDetector()


def run_drift_check(n_samples: int = 500) -> dict[str, Any]:
    """Point d'entrée pour les Jobs Kubernetes planifiés de détection de dérive multi-niveaux."""
    logger.info("Exécution du contrôle statistique Data Drift & Concept Drift...")
    np.random.seed(42)

    # Référence normale
    ref_df = pd.DataFrame(
        np.random.randn(n_samples, 4),
        columns=["feature_1", "feature_2", "feature_3", "feature_4"],
    )
    ref_y = 2.0 * ref_df["feature_1"] - 1.5 * ref_df["feature_2"] + 0.5 * ref_df["feature_3"]
    ref_df["actual"] = ref_y + np.random.randn(n_samples) * 0.1
    ref_df["prediction"] = ref_y

    # Données actuelles : dérive combinée (features shift + altération de la relation de concept)
    curr_df = pd.DataFrame(
        np.random.randn(n_samples, 4) + np.array([0.05, 0.45, 0.0, 0.8]),
        columns=["feature_1", "feature_2", "feature_3", "feature_4"],
    )
    # Nouvelle relation de concept dégradée (concept drift)
    curr_y = 1.0 * curr_df["feature_1"] + 3.0 * curr_df["feature_2"]  # Changement de coefficients
    curr_df["actual"] = curr_y + np.random.randn(n_samples) * 0.1
    curr_df["prediction"] = 2.0 * curr_df["feature_1"] - 1.5 * curr_df["feature_2"]  # Ancien modèle

    return drift_detector.calculate_drift(ref_df, curr_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_drift_check()

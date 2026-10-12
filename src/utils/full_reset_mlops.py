"""
full_reset_mlops.py — Réinitialisation complète de la plateforme MLOps
Permet une purge saine et contrôlée :
1. Vidage des tables de traçabilité MLflow (runs, metrics, model_versions).
2. Purge des tables opérationnelles PostgreSQL (predictions.fact_predictions, webhooks.event_logs).
3. Vidage des files d'attente locales et rapports de drift.
4. Entraînement d'un modèle Champion initial de référence (Version 1).
5. Rechargement à chaud du serving d'inférence.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from typing import Any

from sqlalchemy import text

from ..common.config import mlflow_settings, storage_settings
from ..common.db import get_engine
from ..training.train_xgboost import train_model
from .api_reloader import reload_serving_api

logger = logging.getLogger("k8s_mlops.full_reset")


def reset_mlflow_database() -> bool:
    """Purge les tables de traçabilité MLflow dans la base PostgreSQL 'mlflow'."""
    logger.info("=== 1. PURGE ET RÉINITIALISATION DE MLFLOW ===")
    try:
        engine = get_engine(db_name="mlflow")
        with engine.connect() as conn:
            logger.info("Purge des tables MLflow (runs, metrics, model_versions)...")
            truncate_query = text("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'runs') THEN
                        TRUNCATE TABLE
                            runs, experiment_tags, latest_metrics, metrics, params, tags,
                            model_version_tags, model_versions, registered_model_aliases,
                            registered_model_tags, registered_models
                        CASCADE;
                    END IF;
                END $$;
            """)
            conn.execute(truncate_query)
            conn.commit()

            # Réactivation / insertion de l'expérience par défaut
            exp_check = conn.execute(
                text("SELECT COUNT(*) FROM experiments WHERE name = :exp_name;"),
                {"exp_name": mlflow_settings.experiment_name},
            ).scalar()

            if not exp_check:
                conn.execute(
                    text("""
                        INSERT INTO experiments (name, artifact_location, lifecycle_stage, creation_time, last_update_time)
                        VALUES (:exp_name, :loc, 'active', :ts, :ts);
                    """),
                    {
                        "exp_name": mlflow_settings.experiment_name,
                        "loc": f"s3://{storage_settings.bucket_models}/artifacts",
                        "ts": int(time.time() * 1000),
                    },
                )
                conn.commit()
                logger.info(
                    f"Expérience '{mlflow_settings.experiment_name}' initialisée avec succès."
                )
            else:
                conn.execute(
                    text(
                        "UPDATE experiments SET lifecycle_stage = 'active' WHERE name = :exp_name;"
                    ),
                    {"exp_name": mlflow_settings.experiment_name},
                )
                conn.commit()

        logger.info("✅ Base MLflow réinitialisée avec succès.")
        return True
    except Exception as e:
        logger.warning(
            f"⚠️ Réinitialisation MLflow non effectuée (PostgreSQL potentiellement offline) : {e}"
        )
        return False


def reset_postgres_operational_data() -> bool:
    """Purge les tables de prédictions et logs d'événements dans la base PostgreSQL 'mlops'."""
    logger.info("=== 2. PURGE DES TABLES OPÉRATIONNELLES POSTGRESQL ===")
    try:
        engine = get_engine(db_name="mlops")
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE predictions.fact_predictions CASCADE;"))
            conn.execute(text("TRUNCATE TABLE webhooks.event_logs CASCADE;"))
            conn.commit()
        logger.info(
            "✅ Tables 'predictions.fact_predictions' et 'webhooks.event_logs' vidées avec succès."
        )
        return True
    except Exception as e:
        logger.warning(f"⚠️ Purge PostgreSQL ignorée (Base inaccessible ou tables absentes) : {e}")
        return False


def reset_queue_and_reports(queue_dir: str = "data/queue", reports_dir: str = "reports") -> None:
    """Supprime les fichiers temporaires dans la file d'attente et le répertoire des rapports."""
    logger.info("=== 3. NETTOYAGE DES FICHIERS LOCAUX (QUEUE & RAPPORTS) ===")
    deleted_files = 0
    for folder in [queue_dir, reports_dir]:
        if os.path.exists(folder):
            for file_path in glob.glob(os.path.join(folder, "*.*")):
                try:
                    os.remove(file_path)
                    deleted_files += 1
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {file_path} : {e}")
    logger.info(f"✅ {deleted_files} fichier(s) temporaire(s) supprimé(s).")


def run_full_reset(
    retrain_baseline: bool = True,
    hot_reload: bool = True,
) -> dict[str, Any]:
    """Exécute la séquence complète de Grand Reset MLOps."""
    logger.info("🚀 Démarrage du Grand Reset MLOps...")

    mlflow_ok = reset_mlflow_database()
    pg_ok = reset_postgres_operational_data()
    reset_queue_and_reports()

    baseline_metrics = None
    if retrain_baseline:
        logger.info("=== 4. ENTRAÎNEMENT D'UN MODÈLE CHAMPION INITIAL ===")
        try:
            _, baseline_metrics = train_model()
            logger.info(f"✅ Modèle Champion initial entraîné : {baseline_metrics}")
            try:
                from mlflow.tracking import MlflowClient

                from ..common.config import mlflow_settings

                client = MlflowClient(tracking_uri=mlflow_settings.tracking_uri)
                versions = client.search_model_versions(f"name='{mlflow_settings.model_name}'")
                if versions:
                    latest = str(max(versions, key=lambda v: int(v.version)).version)
                    client.set_registered_model_alias(
                        mlflow_settings.model_name, "champion", latest
                    )
                    logger.info(f"🏆 Alias @champion réassigné à v{latest} dans le Model Registry.")
            except Exception as e:
                logger.debug(f"Assignation alias @champion ignorée : {e}")
        except Exception as e:
            logger.error(f"❌ Échec de l'entraînement initial : {e}")

    reload_ok = False
    if hot_reload:
        logger.info("=== 5. RECHARGEMENT DU SERVICE D'INFÉRENCE ===")
        reload_ok = reload_serving_api()

    summary = {
        "mlflow_reset": mlflow_ok,
        "postgres_reset": pg_ok,
        "baseline_retrained": baseline_metrics is not None,
        "baseline_metrics": baseline_metrics,
        "hot_reloaded": reload_ok,
    }
    logger.info("🎉 Grand Reset MLOps achevé avec succès.")
    return summary


# Alias pour compatibilité
run_full_mlops_reset = run_full_reset


def main() -> None:
    parser = argparse.ArgumentParser(description="Script d'administration et de Grand Reset MLOps.")
    parser.add_argument(
        "--skip-train", action="store_true", help="Ne pas réentraîner de modèle initial"
    )
    parser.add_argument(
        "--skip-reload", action="store_true", help="Ne pas forcer le rechargement de l'API"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    run_full_reset(retrain_baseline=not args.skip_train, hot_reload=not args.skip_reload)


if __name__ == "__main__":
    main()

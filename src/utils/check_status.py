"""
check_status.py — Diagnostic complet de l'état de la plateforme MLOps
Inspecte le Model Registry MLflow, les métriques en production, les tables PostgreSQL et les files d'attente.
"""

import logging
import os
from typing import Any

from sqlalchemy import text

from ..common.config import app_settings, mlflow_settings
from ..common.db import get_engine

logger = logging.getLogger("k8s_mlops.utils.check_status")


def check_platform_status() -> dict[str, Any]:
    """Exécute un audit complet des sous-systèmes de la plateforme et retourne un résumé structuré."""
    status_report: dict[str, Any] = {
        "mlflow": {},
        "database": {},
        "queue": {},
    }

    print("\n" + "=" * 70)
    print(
        f"📊 DIAGNOSTIC SYSTÈME — {app_settings.app_name.upper()} (Env: {app_settings.environment})"
    )
    print("=" * 70)

    # 1. Vérification MLflow Model Registry
    print("\n[1/3] Inspection du Model Registry MLflow...")
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=mlflow_settings.tracking_uri)
        try:
            rm = client.get_registered_model(mlflow_settings.model_name)
            aliases = rm.aliases or {}
            print(f"  • Modèle enregistré : {mlflow_settings.model_name}")
            print(f"  • Alias actifs      : {aliases}")
            status_report["mlflow"]["aliases"] = aliases

            versions = client.search_model_versions(f"name='{mlflow_settings.model_name}'")
            print(f"  • Total versions    : {len(versions)}")
            version_details = []
            for mv in sorted(versions, key=lambda x: int(x.version), reverse=True)[:5]:
                run = client.get_run(mv.run_id)
                metrics = run.data.metrics
                v_info = {
                    "version": mv.version,
                    "run_id": mv.run_id[:8],
                    "rmse": metrics.get("rmse", None),
                    "r2": metrics.get("r2", None),
                    "status": mv.status,
                }
                version_details.append(v_info)
                print(
                    f"    - v{mv.version:>2} | Run: {mv.run_id[:8]} | RMSE: {metrics.get('rmse', 'N/A')} | R2: {metrics.get('r2', 'N/A')}"
                )
            status_report["mlflow"]["recent_versions"] = version_details
        except Exception as e:
            print(
                f"  ⚠️ Modèle '{mlflow_settings.model_name}' non trouvé ou MLflow inaccessible : {e}"
            )
            status_report["mlflow"]["status"] = "offline_or_not_found"
    except Exception as e:
        print(f"  ⚠️ Module MLflow non disponible : {e}")
        status_report["mlflow"]["status"] = "import_error"

    # 2. Vérification Service FastAPI Serving
    print("\n[2/4] Inspection du Service FastAPI Serving...")
    status_report["fastapi"] = {}
    try:
        import requests

        serving_url = f"http://{app_settings.serving_host}:{app_settings.serving_port}/health/live"
        resp = requests.get(serving_url, timeout=2)
        if resp.status_code == 200:
            print(f"  • FastAPI Serving   : ✅ Opérationnel ({serving_url})")
            status_report["fastapi"]["status"] = "healthy"
        else:
            print(f"  • FastAPI Serving   : ⚠️ Réponse HTTP {resp.status_code}")
            status_report["fastapi"]["status"] = f"http_{resp.status_code}"
    except Exception:
        print("  • FastAPI Serving   : ⚠️ Inaccessible (mode hors-ligne ou conteneur non démarré)")
        status_report["fastapi"]["status"] = "unreachable"

    # 3. Vérification Base de Données PostgreSQL Multi-Bases
    print("\n[3/4] Inspection des Bases PostgreSQL (mlops, mlflow, optuna)...")
    for db_name in ["mlops", "mlflow", "optuna"]:
        try:
            engine = get_engine(db_name)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1;"))
                print(f"  • PostgreSQL [{db_name:<6}] : ✅ Connecté")
                status_report["database"][f"{db_name}_status"] = "connected"

                if db_name == "mlops":
                    pred_count = conn.execute(
                        text("SELECT COUNT(*) FROM predictions.fact_predictions;")
                    ).scalar()
                    wh_count = conn.execute(
                        text("SELECT COUNT(*) FROM webhooks.event_logs;")
                    ).scalar()
                    print(f"    - Prédictions historisées : {pred_count}")
                    print(f"    - Événements Webhooks     : {wh_count}")
                    status_report["database"]["predictions_count"] = pred_count
                    status_report["database"]["webhooks_count"] = wh_count
        except Exception as e:
            print(f"  • PostgreSQL [{db_name:<6}] : ⚠️ Inaccessible ({e})")
            status_report["database"][f"{db_name}_status"] = "offline"

    # 4. File d'attente (Data Queue)
    print("\n[4/4] Inspection de la File d'Attente Fichiers...")
    queue_dir = os.path.abspath("data/queue")
    if os.path.exists(queue_dir):
        files = [f for f in os.listdir(queue_dir) if f.endswith((".csv", ".json"))]
        print(f"  • Fichiers en attente dans data/queue/ : {len(files)}")
        status_report["queue"]["pending_files"] = len(files)
    else:
        print("  • Répertoire data/queue/ inexistant (aucun batch en attente).")
        status_report["queue"]["pending_files"] = 0

    print("=" * 70 + "\n")
    return status_report


if __name__ == "__main__":
    check_platform_status()

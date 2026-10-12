"""
process_queue.py — Traitement par lot de fichiers de file d'attente (Batch Queue Ingestion Worker)
Lit les fichiers tabulaires déposés dans data/queue/, valide le format, transmet les requêtes
vers l'endpoint /predict/batch de FastAPI, et archive ou supprime les fichiers traités avec succès.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from typing import Any

import pandas as pd
import requests

from ..common.config import app_settings

logger = logging.getLogger("k8s_mlops.process_queue")


def process_queue_files(
    queue_dir: str = "data/queue",
    api_url: str | None = None,
    delete_on_success: bool = True,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Parcourt le dossier queue_dir, convertit chaque fichier CSV en lots de prédictions,
    appelle l'API FastAPI et nettoie les fichiers validés.
    """
    endpoint = (
        api_url or f"http://{app_settings.serving_host}:{app_settings.serving_port}/predict/batch"
    )
    os.makedirs(queue_dir, exist_ok=True)
    pattern = os.path.join(queue_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    logger.info(
        f"Traitement de la file d'attente : {len(files)} fichier(s) détecté(s) dans {queue_dir}"
    )

    processed_count = 0
    total_records = 0
    errors = []

    for f_path in files:
        basename = os.path.basename(f_path)
        try:
            df = pd.read_csv(f_path)
            if df.empty:
                logger.warning(f"[{basename}] Fichier vide, ignoré.")
                continue

            # Découpage par lots (chunking) pour éviter les rejets HTTP 413 et limiter la mémoire
            records = df.to_dict(orient="records")
            chunk_size = 500
            file_success = True

            for i in range(0, len(records), chunk_size):
                chunk = records[i : i + chunk_size]
                payload = {"items": [{"features": rec} for rec in chunk]}

                res = requests.post(endpoint, json=payload, timeout=timeout)

                if res.status_code == 200:
                    resp_json = res.json()
                    count = len(resp_json.get("predictions", []))
                    total_records += count
                else:
                    file_success = False
                    err_msg = f"HTTP {res.status_code}: {res.text[:200]}"
                    logger.error(f"[{basename}] ❌ Échec sur lot {i // chunk_size + 1} : {err_msg}")
                    errors.append(
                        {"file": basename, "chunk": i // chunk_size + 1, "error": err_msg}
                    )
                    break

            if file_success:
                processed_count += 1
                logger.info(f"[{basename}] ✅ {len(records)} enregistrements inférés avec succès.")
                if delete_on_success:
                    os.remove(f_path)
        except Exception as e:
            logger.error(f"[{basename}] ❌ Exception lors de l'ingestion : {e}")
            errors.append({"file": basename, "error": str(e)})

    # Laisser le temps à PostgreSQL de persister les enregistrements asynchrones
    time.sleep(1)

    result = {
        "files_processed": processed_count,
        "total_records": total_records,
        "errors_count": len(errors),
        "errors": errors,
    }
    logger.info(
        f"Bilan queue : {processed_count}/{len(files)} traités ({total_records} inférences)."
    )
    return result


# Alias pour compatibilité
process_incoming_queue = process_queue_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker d'ingestion et d'inférence en lot via file d'attente."
    )
    parser.add_argument("--dir", default="data/queue", help="Répertoire de la file d'attente")
    parser.add_argument("--url", default=None, help="URL de l'endpoint FastAPI /predict/batch")
    parser.add_argument(
        "--keep", action="store_true", help="Conserver les fichiers après traitement"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    process_queue_files(queue_dir=args.dir, api_url=args.url, delete_on_success=not args.keep)


if __name__ == "__main__":
    main()

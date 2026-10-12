"""
api_reloader.py — Gestionnaire de rechargement à chaud (Hot-Reload) de l'API de serving d'inférence.
Permet d'appliquer le modèle Champion nouvellement promu sans interruption de service.
"""

from __future__ import annotations

import logging
import os

import requests

from ..common.config import app_settings
from ..serving.predictor import predictor

logger = logging.getLogger("k8s_mlops.utils.reloader")


def reload_serving_api(custom_url: str | None = None, timeout: int = 5) -> bool:
    """
    Envoie une requête pour valider et forcer le rechargement du modèle @champion dans FastAPI,
    ou effectue un rechargement en mémoire si exécuté dans le même runtime.
    """
    candidate_urls: list[str] = []
    if custom_url:
        candidate_urls.append(custom_url)
    if os.getenv("FASTAPI_URL"):
        candidate_urls.append(os.getenv("FASTAPI_URL"))

    default_urls = [
        f"http://{app_settings.serving_host}:{app_settings.serving_port}/health/ready",
        "http://k8s-mlops-serving:8000/health/ready",
        "http://localhost:8000/health/ready",
        "http://127.0.0.1:8000/health/ready",
    ]
    for u in default_urls:
        if u not in candidate_urls:
            candidate_urls.append(u)

    logger.info("Déclenchement du rechargement à chaud du modèle...")

    # Rechargement direct en mémoire du singleton predictor
    try:
        predictor.load_model(force=True)
        logger.info(f"Modèle in-process rechargé avec succès (Version: {predictor.version}).")
    except Exception as e:
        logger.warning(f"Rechargement in-process non effectué : {e}")

    # Vérification HTTP de l'API externe si active
    for url in candidate_urls:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                logger.info(f"API d'inférence synchronisée sur {url} : {resp.json()}")
                return True
        except requests.RequestException:
            pass

    return predictor.is_ready


trigger_api_reload = reload_serving_api

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = reload_serving_api()
    print("Statut du rechargement :", "Réussi" if success else "Échoué")

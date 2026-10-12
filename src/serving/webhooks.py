"""
webhooks.py — Gestionnaire de webhooks entrants et sortants pour k8s_mlops_platform
Intègre le déclenchement asynchrone de la boucle fermée de réentraînement (Ray HPO + XGBoost).
"""

import hmac
import json
import logging
import threading
from typing import Any

import requests
from sqlalchemy import text

from ..common.config import webhook_settings
from ..common.db import get_engine
from .schemas import WebhookResponse, WebhookTriggerRequest

logger = logging.getLogger("k8s_mlops.webhooks")


class WebhookManager:
    """Gère la sécurité, la journalisation et l'émission de webhooks."""

    def __init__(self):
        self._engine = None
        self._retrain_lock = threading.Lock()

    def verify_token(self, token: str | None) -> bool:
        """Valide le token secret du webhook avec protection contre les attaques temporelles."""
        if not token:
            return False
        try:
            return hmac.compare_digest(token.encode("utf-8"), webhook_settings.secret_token.encode("utf-8"))
        except Exception:
            return False

    def handle_incoming_event(self, event: WebhookTriggerRequest) -> WebhookResponse:
        """Traite un webhook entrant (ex: dérive Evidently) et déclenche le réentraînement asynchrone."""
        logger.info(f"Événement webhook reçu : {event.event_type} depuis {event.source}")

        action = "logged"
        drift_type = (event.details or {}).get("drift_type", "")
        is_drift_critical = (
            (event.event_type == "drift_alert" and event.drift_score > 0.30)
            or event.event_type in ("retrain", "drift_detected")
            or "drift_detected" in drift_type
        )

        if is_drift_critical:
            logger.warning(
                f"🚨 Dérive critique détectée (score={event.drift_score:.2f}). Déclenchement du réentraînement."
            )
            action = "retraining_triggered"

        event_id = self._log_event(event.event_type, event.source, event.model_dump(), action)

        # Déclenchement asynchrone protégé par verrou de concurrence
        if action == "retraining_triggered":
            if not self._retrain_lock.acquire(blocking=False):
                logger.warning("Un réentraînement est déjà en cours d'exécution. Requête ignorée pour éviter la concurrence.")
                return WebhookResponse(
                    status="busy",
                    action_taken="skipped_already_running",
                    event_id=event_id,
                )
            self._trigger_async_retraining(event_id, event.details)

        return WebhookResponse(
            status="accepted",
            action_taken=action,
            event_id=event_id,
        )

    def _trigger_async_retraining(self, event_id: int | None, details: dict[str, Any]) -> None:
        """Lance l'orchestration du réentraînement en arrière-plan."""
        def _run_worker():
            try:
                from ..training.retrain_pipeline import run_closed_loop_retraining

                logger.info(f"Démarrage asynchrone de la boucle de réentraînement (Event ID: {event_id})...")
                result = run_closed_loop_retraining(event_details=details, use_ray=True)
                self._update_event_status(event_id, "retraining_completed", result)
                self.dispatch_notification(
                    title="Réentraînement MLOps Complété",
                    message=f"Modèle promu : {result.get('promoted_to_champion')} | Version : {result.get('model_version')}",
                    payload=result,
                )
            except Exception as e:
                logger.error(f"Échec de l'exécution du réentraînement (Event ID: {event_id}) : {e}")
                self._update_event_status(event_id, "retraining_failed", {"error": str(e)})
            finally:
                if self._retrain_lock.locked():
                    self._retrain_lock.release()

        worker_thread = threading.Thread(target=_run_worker, daemon=True)
        worker_thread.start()


    def dispatch_notification(self, title: str, message: str, payload: dict[str, Any] | None = None) -> bool:
        """Émet un webhook sortant vers un système de notification (Slack, Teams, Discord)."""
        if not webhook_settings.notification_url:
            logger.info(f"[Notification Simulée] {title} : {message}")
            return True

        body = {
            "text": f"*{title}*\n{message}",
            "data": payload or {},
        }
        try:
            resp = requests.post(webhook_settings.notification_url, json=body, timeout=5)
            return resp.status_code < 400
        except Exception as e:
            logger.error(f"Échec de l'envoi du webhook sortant : {e}")
            return False

    def _log_event(self, event_type: str, source: str, payload: dict[str, Any], status: str) -> int | None:
        """Enregistre l'événement dans la table webhooks.event_logs."""
        try:
            if not self._engine:
                self._engine = get_engine()
            with self._engine.begin() as conn:
                res = conn.execute(
                    text("""
                    INSERT INTO webhooks.event_logs (event_type, source, payload, status)
                    VALUES (:type, :source, :payload, :status)
                    RETURNING id;
                """),
                    {
                        "type": event_type,
                        "source": source,
                        "payload": json.dumps(payload),
                        "status": status,
                    },
                )
                row = res.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.debug(f"Échec journalisation webhook en base : {e}")
            return None

    def _update_event_status(self, event_id: int | None, status: str, result_payload: dict[str, Any]) -> None:
        """Met à jour le statut et le résultat de l'événement dans webhooks.event_logs."""
        if not event_id:
            return
        try:
            if not self._engine:
                self._engine = get_engine()
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                    UPDATE webhooks.event_logs
                    SET status = :status,
                        payload = payload || :result_payload::jsonb
                    WHERE id = :event_id;
                """),
                    {
                        "status": status,
                        "result_payload": json.dumps({"execution_result": result_payload}),
                        "event_id": event_id,
                    },
                )
        except Exception as e:
            logger.debug(f"Échec mise à jour statut webhook en base : {e}")


webhook_manager = WebhookManager()

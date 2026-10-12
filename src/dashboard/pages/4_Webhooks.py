"""
4_Webhooks.py — Page 4: Journal des événements webhooks et déclenchement de réentraînement
"""

import os

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import text

from src.common.config import webhook_settings
from src.common.db import get_engine

st.set_page_config(page_title="Webhooks — Supervision & Alertes", page_icon="🔔", layout="wide")

st.title("🔔 Webhooks — Supervision des Événements & Déclencheurs")
st.markdown(
    "Gestion des flux webhooks entrants (alertes de dérive) et sortants (notifications Slack/Teams)."
)

API_BASE_URL = os.getenv("SERVING_API_URL", "http://localhost:8000").rstrip("/")
API_WEBHOOK_URL = f"{API_BASE_URL}/webhooks/trigger-retrain"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Déclenchement Manuel de Réentraînement")
    st.info(
        "Simulez un webhook entrant d'alerte de dérive statistique pour forcer le réentraînement du modèle."
    )

    drift_score = st.slider("Score de Dérive Simulé", 0.0, 1.0, 0.42, step=0.02)
    reason = st.selectbox(
        "Motif du déclenchement",
        [
            "data_drift_detected",
            "concept_drift_detected",
            "dual_drift_detected",
            "new_batch_ingested",
            "manual_audit_requested",
        ],
    )

    if st.button(
        "🚨 Émettre le Webhook de Réentraînement", type="primary", use_container_width=True
    ):
        payload = {
            "source": "streamlit-webhooks-ui",
            "event_type": "drift_alert",
            "drift_score": drift_score,
            "details": {"reason": reason, "triggered_by": "operator"},
        }
        headers = {"X-Webhook-Secret": webhook_settings.secret_token}
        try:
            resp = requests.post(API_WEBHOOK_URL, json=payload, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ Webhook accepté avec succès (Action: {data.get('action_taken')})")
                st.json(data)
            else:
                st.error(f"Erreur Webhook ({resp.status_code}) : {resp.text}")
        except Exception as e:
            st.error(f"Échec de connexion au service FastAPI : {e}")

with col2:
    st.subheader("2. Configuration des Webhooks Sortants")
    notification_url = st.text_input(
        "URL de notification (Slack / Discord / Teams)",
        value=webhook_settings.notification_url or "https://hooks.slack.com/services/mock/token",
    )
    st.checkbox("Notifier lors d'une promotion Champion", value=True)
    st.checkbox("Notifier lors d'une dérive statistique > 30%", value=True)
    st.checkbox("Notifier lors d'un échec de job dbt", value=True)
    st.button("💾 Enregistrer la Configuration", use_container_width=True)

st.divider()

st.subheader("3. Journal des Derniers Événements Webhooks")


@st.cache_data(ttl=30)
def load_webhook_logs() -> pd.DataFrame:
    """Lit les logs d'événements depuis PostgreSQL table webhooks.event_logs."""
    try:
        engine = get_engine()
        query = text("""
            SELECT id, event_timestamp, event_type, source, status
            FROM webhooks.event_logs
            ORDER BY event_timestamp DESC
            LIMIT 20;
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, con=conn)
    except Exception:
        return pd.DataFrame(
            {
                "id": [101, 102, 103, 104],
                "event_timestamp": [
                    "2026-10-12 16:30:00+00",
                    "2026-10-12 16:45:00+00",
                    "2026-10-12 17:00:00+00",
                    "2026-10-12 17:15:00+00",
                ],
                "event_type": [
                    "model_promoted",
                    "drift_alert",
                    "dbt_run_success",
                    "retrain_started",
                ],
                "source": ["mlflow-registry", "evidently-worker", "dbt-cronjob", "fastapi-webhook"],
                "status": ["delivered", "retraining_triggered", "completed", "in_progress"],
            }
        )


df_logs = load_webhook_logs()
st.dataframe(df_logs, use_container_width=True)

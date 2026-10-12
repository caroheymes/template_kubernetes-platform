"""
app.py — Point d'entrée principal du Dashboard Streamlit Multi-Pages
"""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="k8s_mlops_platform — Accueil",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 k8s_mlops_platform")
st.markdown("### *Plateforme MLOps Cloud-Native orchestrée sur Kubernetes*")

st.markdown("""
Bienvenue sur l'interface d'administration et de supervision de **`k8s_mlops_platform`**.
Cette plateforme unifie l'inférence temps réel, l'explicabilité, la modélisation analytique dbt et le cycle de vie MLOps.
""")

st.divider()

# Vérification du statut de l'API FastAPI
col1, col2, col3, col4 = st.columns(4)

api_url = os.getenv("SERVING_API_URL", "http://localhost:8000")
is_api_live = False
is_model_ready = False
db_connected = False

try:
    resp = requests.get(f"{api_url}/health/ready", timeout=2)
    if resp.status_code == 200:
        data = resp.json()
        is_api_live = True
        is_model_ready = data.get("model_loaded", False)
        db_connected = data.get("database_connected", False)
except Exception:
    is_api_live = False

with col1:
    st.metric("FastAPI Serving", "En ligne ✅" if is_api_live else "Indisponible ⚠️")

with col2:
    st.metric("Modèle XGBoost", "Champion Chargé 🏆" if is_model_ready else "En attente ⏳")

with col3:
    st.metric("PostgreSQL (PVC)", "Connecté 💾" if db_connected else "Hors ligne ⚠️")

with col4:
    st.metric("Cluster K8s", "k8s-mlops ☸️")

st.divider()

st.subheader("📑 Modules de la Plateforme")

c1, c2 = st.columns(2)
with c1:
    st.info("""
    **1. App (Inférence Directe)**
    Formulaire dynamique pour simuler des prédictions, tester l'API FastAPI et visualiser la latence en temps réel.
    """)
    st.info("""
    **2. XAI (Explicabilité Shapash)**
    Visualisation locale des contributions de features (Waterfall plot) et importance globale des variables du modèle.
    """)

with c2:
    st.success("""
    **3. Analytics (Marts dbt)**
    Tableaux de bord opérationnels branchés directement sur les tables dbt de PostgreSQL (`mart_inference_kpis`, distribution).
    """)
    st.success("""
    **4. Webhooks (Alertes & Triggers)**
    Surveillance des événements de dérive statistique (Evidently) et déclenchement de réentraînement automatisé.
    """)

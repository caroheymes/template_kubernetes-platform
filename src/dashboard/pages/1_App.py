"""
1_App.py — Page 1: Inférence en direct et simulation de scoring XGBoost
"""

import os
import time
import requests
import streamlit as st

st.set_page_config(page_title="App — Inférence Directe", page_icon="⚡", layout="wide")

st.title("⚡ Inférence en Temps Réel — Modèle XGBoost")
st.markdown("Testez en direct le microservice FastAPI Serving orchestré sur Kubernetes.")

API_BASE_URL = os.getenv("SERVING_API_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/predict"

st.sidebar.header("Paramètres de Test")
api_endpoint = st.sidebar.text_input("Endpoint FastAPI", value=API_URL)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Saisie des Features Numériques")
    f1 = st.slider("Feature 1 (Dimension Principale)", -5.0, 5.0, 1.25, step=0.05)
    f2 = st.slider("Feature 2 (Ratio d'Activité)", -5.0, 5.0, -0.45, step=0.05)
    f3 = st.slider("Feature 3 (Indice Cyclique)", -10.0, 10.0, 2.10, step=0.1)
    f4 = st.slider("Feature 4 (Facteur d'Échelle)", -5.0, 5.0, 0.85, step=0.05)

    payload = {
        "features": {
            "feature_1": f1,
            "feature_2": f2,
            "feature_3": f3,
            "feature_4": f4,
        }
    }
    submit_btn = st.button("🚀 Lancer l'Inférence", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Résultat de la Prédiction")
    if submit_btn:
        start_t = time.perf_counter()
        try:
            resp = requests.post(api_endpoint, json=payload, timeout=3)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            if resp.status_code == 200:
                result = resp.json()
                st.success("✅ Inférence réussie")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Score Prédit", f"{result['prediction']:.4f}")
                with m2:
                    st.metric("Latence API", f"{result['latency_ms']:.1f} ms")
                with m3:
                    st.metric("Version Modèle", result.get("model_version", "champion"))

                st.json(result)
            else:
                st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
        except Exception as e:
            st.error(f"Impossible de joindre le service FastAPI : {e}")
            st.info("Vérifiez que le port-forward est actif : `kubectl port-forward svc/k8s-mlops-serving 8000:8000 -n k8s-mlops`")
    else:
        st.info("Ajustez les curseurs et cliquez sur **Lancer l'Inférence** pour tester l'API.")

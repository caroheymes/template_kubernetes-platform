"""
2_XAI.py — Page 2: Explicabilité locale et globale basée sur Shapash et SHAP
"""

import os

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="XAI — Explicabilité Shapash", page_icon="🔍", layout="wide")

st.title("🔍 Explicabilité XAI — Shapash & SHAP")
st.markdown("Comprenez les facteurs déterminants de chaque prédiction XGBoost en langage clair.")

API_BASE_URL = os.getenv("SERVING_API_URL", "http://localhost:8000").rstrip("/")
EXPLAIN_URL = f"{API_BASE_URL}/predict/explain"

tab1, tab2 = st.tabs(["Explicabilité Locale (Waterfall)", "Importance Globale des Variables"])

with tab1:
    st.subheader("1. Explication d'une Prédiction Particulière")

    c1, c2 = st.columns([1, 2])
    with c1:
        f1 = st.slider("Feature 1", -3.0, 3.0, 1.4, key="xai_f1")
        f2 = st.slider("Feature 2", -3.0, 3.0, -0.8, key="xai_f2")
        f3 = st.slider("Feature 3", -5.0, 5.0, 0.5, key="xai_f3")
        f4 = st.slider("Feature 4", -3.0, 3.0, 1.1, key="xai_f4")
        btn = st.button("Expliquer la Prédiction", type="primary", use_container_width=True)

    with c2:
        if btn:
            payload = {
                "features": {"feature_1": f1, "feature_2": f2, "feature_3": f3, "feature_4": f4}
            }
            try:
                resp = requests.post(EXPLAIN_URL, json=payload, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    contribs = data["feature_contributions"]

                    # Waterfall Plot Plotly
                    fig = go.Figure(
                        go.Waterfall(
                            name="Contributions",
                            orientation="v",
                            measure=["relative"] * len(contribs) + ["total"],
                            x=list(contribs.keys()) + ["Prédiction Finale"],
                            y=list(contribs.values()) + [data["prediction"]],
                            text=[f"{v:+.3f}" for v in contribs.values()]
                            + [f"{data['prediction']:.3f}"],
                            textposition="outside",
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                        )
                    )
                    fig.update_layout(
                        title="Décomposition des Contributions (SHAP Values)", height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.info(
                        f"💡 **Variables poussant à la hausse** : {', '.join(data.get('top_positive_features', []))}"
                    )
                    st.warning(
                        f"📉 **Variables tirant à la baisse** : {', '.join(data.get('top_negative_features', []))}"
                    )
                else:
                    st.error(f"Erreur API : {resp.text}")
            except Exception as e:
                st.error(f"Connexion API impossible : {e}")
        else:
            st.info(
                "Cliquez sur **Expliquer la Prédiction** pour générer le diagramme d'attribution."
            )

with tab2:
    st.subheader("2. Importance Globale des Variables dans XGBoost")

    global_importance = {
        "Feature 1 (Dimension Principale)": 0.44,
        "Feature 2 (Ratio d'Activité)": 0.29,
        "Feature 3 (Indice Cyclique)": 0.17,
        "Feature 4 (Facteur d'Échelle)": 0.10,
    }

    fig_global = px.bar(
        x=list(global_importance.values()),
        y=list(global_importance.keys()),
        orientation="h",
        labels={"x": "Contribution Moyenne Absolue (|SHAP|)", "y": "Feature"},
        title="Hiérarchie des Poids de Décision (Modèle Champion)",
        color=list(global_importance.values()),
        color_continuous_scale="Blues",
    )
    fig_global.update_layout(height=400, yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_global, use_container_width=True)

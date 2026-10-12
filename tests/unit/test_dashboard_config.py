"""Tests unitaires pour la configuration et l'interopérabilité du Dashboard Streamlit."""

import os
from unittest.mock import patch


def test_dashboard_serving_api_url_env_resolution():
    """Vérifie que la configuration du dashboard prend en compte la variable SERVING_API_URL."""
    custom_url = "http://k8s-mlops-serving.internal:8000"
    with patch.dict(os.environ, {"SERVING_API_URL": custom_url}):
        api_base = os.getenv("SERVING_API_URL", "http://localhost:8000").rstrip("/")
        predict_url = f"{api_base}/predict"
        explain_url = f"{api_base}/predict/explain"
        webhook_url = f"{api_base}/webhooks/trigger-retrain"

        assert predict_url == "http://k8s-mlops-serving.internal:8000/predict"
        assert explain_url == "http://k8s-mlops-serving.internal:8000/predict/explain"
        assert webhook_url == "http://k8s-mlops-serving.internal:8000/webhooks/trigger-retrain"


def test_dashboard_modules_syntax_and_imports():
    """Vérifie que les scripts Streamlit ne comportent pas d'erreurs de syntaxe ou d'import relatif."""
    # Teste le chargement de bytecode sans lever de SyntaxError ni d'ImportError relatif
    app_path = "src/dashboard/app.py"
    page1_path = "src/dashboard/pages/1_App.py"
    page2_path = "src/dashboard/pages/2_XAI.py"
    page3_path = "src/dashboard/pages/3_Analytics.py"
    page4_path = "src/dashboard/pages/4_Webhooks.py"

    for script in [app_path, page1_path, page2_path, page3_path, page4_path]:
        with open(script, "r", encoding="utf-8") as f:
            code = f.read()
            compile(code, script, "exec")
            assert "from ...common" not in code, f"Import relatif interdit détecté dans {script}"

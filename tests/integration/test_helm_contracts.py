"""Tests d'intégration validant la conformité des templates Helm et les contrats d'infrastructure."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "helm"


def test_helm_chart_structure():
    """Vérifie la présence et la cohérence des fichiers constitutifs du chart Helm."""
    assert (HELM_DIR / "Chart.yaml").exists()
    assert (HELM_DIR / "values.yaml").exists()
    assert (HELM_DIR / "templates" / "ingress.yaml").exists()
    assert (HELM_DIR / "templates" / "serving.yaml").exists()
    assert (HELM_DIR / "templates" / "streamlit.yaml").exists()
    assert (HELM_DIR / "templates" / "secrets.yaml").exists()
    assert (HELM_DIR / "templates" / "configmaps.yaml").exists()
    assert (HELM_DIR / "templates" / "postgres.yaml").exists()
    assert (HELM_DIR / "templates" / "dbt-cronjob.yaml").exists()


@pytest.mark.skipif(shutil.which("helm") is None, reason="Helm CLI non installé sur l'hôte")
def test_helm_lint_validation():
    """Valide l'absence d'erreurs de syntaxe Go-Template et YAML via helm lint."""
    res = subprocess.run(
        ["helm", "lint", str(HELM_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"helm lint a échoué :\n{res.stdout}\n{res.stderr}"
    assert "0 chart(s) failed" in res.stdout


@pytest.mark.skipif(shutil.which("helm") is None, reason="Helm CLI non installé sur l'hôte")
def test_helm_template_contract_verification():
    """Vérifie le rendu complet des manifests avec Ingress activé et contrôle les ports et secrets."""
    res = subprocess.run(
        ["helm", "template", "mlops-test", str(HELM_DIR), "--set", "ingress.enabled=true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"helm template a échoué :\n{res.stderr}"
    rendered = res.stdout

    # 1. Vérification du port Streamlit dans Ingress (8503 et non 8501)
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc is not None]
    ingress_doc = next((d for d in docs if d.get("kind") == "Ingress"), None)
    assert ingress_doc is not None, "Le manifest Ingress n'a pas été rendu"

    streamlit_rule = None
    for rule in ingress_doc["spec"]["rules"]:
        for path_entry in rule["http"]["paths"]:
            if path_entry["backend"]["service"]["name"] == "k8s-mlops-streamlit":
                streamlit_rule = path_entry
                break
    assert streamlit_rule is not None
    assert streamlit_rule["backend"]["service"]["port"]["number"] == 8503

    # 2. Vérification de l'injection des variables d'authentification PostgreSQL dans Serving
    serving_doc = next(
        (
            d
            for d in docs
            if d.get("kind") == "Deployment" and d["metadata"]["name"] == "k8s-mlops-serving"
        ),
        None,
    )
    assert serving_doc is not None
    serving_env = {
        e["name"]: e for e in serving_doc["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert "POSTGRES_USER" in serving_env
    assert "POSTGRES_PASSWORD" in serving_env
    assert "WEBHOOK_SECRET_TOKEN" in serving_env

    # 3. Vérification de l'injection des variables d'authentification PostgreSQL dans Streamlit
    streamlit_doc = next(
        (
            d
            for d in docs
            if d.get("kind") == "Deployment" and d["metadata"]["name"] == "k8s-mlops-streamlit"
        ),
        None,
    )
    assert streamlit_doc is not None
    streamlit_env = {
        e["name"]: e for e in streamlit_doc["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert "POSTGRES_USER" in streamlit_env
    assert "POSTGRES_PASSWORD" in streamlit_env
    assert "WEBHOOK_SECRET_TOKEN" in streamlit_env

    # 4. Vérification de la cible --target prod dans dbt-cronjob
    dbt_doc = next((d for d in docs if d.get("kind") == "CronJob"), None)
    assert dbt_doc is not None
    dbt_args = dbt_doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["args"][
        0
    ]
    assert "--target prod" in dbt_args

    # 5. Vérification du schéma staging dans postgres-init
    pg_init_doc = next(
        (
            d
            for d in docs
            if d.get("kind") == "ConfigMap" and "postgres-init" in d["metadata"]["name"]
        ),
        None,
    )
    assert pg_init_doc is not None
    init_script = pg_init_doc["data"]["init-databases.sh"]
    assert "CREATE SCHEMA IF NOT EXISTS staging;" in init_script

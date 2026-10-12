"""
db.py — Utilitaires de diagnostic, inspection et requêtage pour PostgreSQL multi-bases
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import inspect, text

from ..common.db import get_engine

logger = logging.getLogger("k8s_mlops.utils.db")


def test_database_connections() -> dict[str, bool]:
    """Teste la connectivité individuelle pour les 3 bases isolées : mlops, mlflow, optuna."""
    results = {}
    for db in ["mlops", "mlflow", "optuna"]:
        try:
            eng = get_engine(db_name=db)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1;"))
                results[db] = True
        except Exception as e:
            logger.debug(f"Connexion échouée pour la base '{db}' : {e}")
            results[db] = False
    return results


def inspect_database_tables(db_name: str = "mlops") -> dict[str, list[str]]:
    """Retourne la liste des tables organisées par schéma pour la base spécifiée."""
    tables_by_schema: dict[str, list[str]] = {}
    try:
        eng = get_engine(db_name=db_name)
        inspector = inspect(eng)
        for schema in inspector.get_schema_names():
            if schema not in ["pg_catalog", "information_schema", "pg_toast"]:
                tables_by_schema[schema] = inspector.get_table_names(schema=schema)
    except Exception as e:
        logger.warning(f"Impossible d'inspecter les tables de '{db_name}' : {e}")
    return tables_by_schema


def execute_sql_query(
    query: str, db_name: str = "mlops", params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Exécute une requête SQL SELECT sécurisée et retourne les résultats en liste de dictionnaires."""
    eng = get_engine(db_name=db_name)
    with eng.connect() as conn:
        res = conn.execute(text(query), params or {})
        keys = res.keys()
        return [dict(zip(keys, row)) for row in res.fetchall()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Test de connectivité :", test_database_connections())
    print("Tables détectées :", inspect_database_tables("mlops"))

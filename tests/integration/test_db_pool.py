"""Tests d'intégration pour le pooling de connexions PostgreSQL."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.common.config import db_settings
from src.common.db import get_engine


def test_database_url_generation():
    """Vérifie le formatage des URLs pour l'isolation multi-base."""
    assert "postgresql+psycopg2://" in db_settings.url
    assert "mlops" in db_settings.url

    mlflow_url = db_settings.get_url_for_db("mlflow")
    assert "mlflow" in mlflow_url

    optuna_url = db_settings.get_url_for_db("optuna")
    assert "optuna" in optuna_url


def test_engine_factory():
    """Vérifie l'instanciation de l'Engine SQLAlchemy avec configuration résiliente."""
    engine = get_engine("mlops", pool_size=3, max_overflow=5)
    assert isinstance(engine, Engine)
    assert engine.pool.size() == 3


def test_sqlite_fallback_connectivity():
    """Vérifie le comportement transactionnel avec un moteur SQL local."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE test_tab (id INTEGER PRIMARY KEY, val TEXT);"))
        conn.execute(text("INSERT INTO test_tab (val) VALUES ('mlops');"))
        res = conn.execute(text("SELECT val FROM test_tab;")).scalar()
        assert res == "mlops"

"""
db.py — Usine de connexion PostgreSQL et pooling SQLAlchemy
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from .config import db_settings

logger = logging.getLogger("k8s_mlops.db")


_ENGINES: dict[tuple[str, int, int], Engine] = {}


def get_engine(db_name: str | None = None, pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """Retourne une instance singleton Engine SQLAlchemy mise en cache avec pool_pre_ping."""
    target_db = db_name or db_settings.db_name
    cache_key = (target_db, pool_size, max_overflow)
    if cache_key not in _ENGINES:
        url = db_settings.get_url_for_db(target_db)
        connect_args = {"connect_timeout": 3} if "postgresql" in url else {}
        _ENGINES[cache_key] = create_engine(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args=connect_args,
        )
    return _ENGINES[cache_key]


def dispose_all_engines() -> None:
    """Ferme proprement l'ensemble des pools de connexions lors de l'arrêt."""
    for eng in _ENGINES.values():
        eng.dispose()
    _ENGINES.clear()


@contextmanager
def get_db_connection(db_name: str | None = None) -> Generator[Connection, None, None]:
    """Gestionnaire de contexte pour les transactions de connexion SQL."""
    engine = get_engine(db_name=db_name)
    with engine.begin() as conn:
        yield conn

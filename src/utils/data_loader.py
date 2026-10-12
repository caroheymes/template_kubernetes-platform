"""
data_loader.py — Chargeur et générateur de données universel pour k8s_mlops_platform
Supporte la génération synthétique, l'import depuis CSV/Parquet et l'extraction depuis PostgreSQL.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from ..common.config import db_settings
from ..common.db import get_engine

logger = logging.getLogger("k8s_mlops.utils.data_loader")


def load_or_generate_dataset(
    n_samples: int = 2000,
    test_size: float = 0.2,
    random_state: int = 42,
    csv_path: str | None = None,
    from_db: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Charge ou génère un dataset complet et retourne les splits d'entraînement et de test.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]

    # 1. Tentative d'extraction depuis PostgreSQL si demandé
    if from_db:
        try:
            engine = get_engine("mlops")
            with engine.connect() as conn:
                df_sql = pd.read_sql(
                    text("SELECT features_payload, predicted_value FROM predictions.fact_predictions LIMIT :n;"),
                    conn,
                    params={"n": n_samples},
                )
                if not df_sql.empty and len(df_sql) >= 50:
                    logger.info(f"Chargement de {len(df_sql)} enregistrements depuis PostgreSQL.")
                    extracted = pd.json_normalize(df_sql["features_payload"])
                    for col in feature_names:
                        if col not in extracted.columns:
                            extracted[col] = 0.0
                    X = extracted[feature_names]
                    y = df_sql["predicted_value"]
                    return (*train_test_split(X, y, test_size=test_size, random_state=random_state), feature_names)
        except Exception as e:
            logger.warning(f"Échec de chargement PostgreSQL ({e}), bascule sur le mode standard.")

    # 2. Tentative de lecture depuis fichier CSV
    if csv_path and os.path.exists(csv_path):
        logger.info(f"Lecture du dataset depuis le fichier : {csv_path}")
        df = pd.read_csv(csv_path)
        available_cols = [c for c in feature_names if c in df.columns]
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X = df[available_cols]
        y = df[target_col]
        return (*train_test_split(X, y, test_size=test_size, random_state=random_state), available_cols)

    # 3. Génération synthétique déterministe (fallback par défaut)
    logger.info(f"Génération d'un dataset synthétique ({n_samples} échantillons)...")
    np.random.seed(random_state)
    X_mat = np.random.randn(n_samples, len(feature_names))
    y_vec = (
        3.5 * X_mat[:, 0]
        - 2.0 * X_mat[:, 1]
        + 1.5 * np.sin(X_mat[:, 2])
        + 0.8 * X_mat[:, 3]
        + np.random.randn(n_samples) * 0.25
    )

    df_X = pd.DataFrame(X_mat, columns=feature_names)
    df_y = pd.Series(y_vec, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X_tr, X_te, y_tr, y_te, feats = load_or_generate_dataset(n_samples=500)
    print(f"Dataset prêt : Train={len(X_tr)}, Test={len(X_te)}, Features={feats}")

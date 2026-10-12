"""
train_xgboost.py — Pipeline d'entraînement XGBoost avec traçabilité MLflow et Shapash
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..common.config import mlflow_settings
from ..tracking.mlflow_client import mlflow_tracker
from ..xai.explainer import init_smart_explainer

logger = logging.getLogger("k8s_mlops.train")


def generate_synthetic_data(
    n_samples: int = 2000,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Génère un dataset tabulaire synthétique pour l'entraînement."""
    np.random.seed(42)
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
    X = np.random.randn(n_samples, len(feature_names))
    y = (
        3.5 * X[:, 0]
        - 2.0 * X[:, 1]
        + 1.5 * np.sin(X[:, 2])
        + 0.8 * X[:, 3]
        + np.random.randn(n_samples) * 0.25
    )
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.Series(y, name="target")
    return df_X, df_y, feature_names


def train_model(params: dict[str, Any] | None = None) -> tuple[xgb.Booster, dict[str, float]]:
    """Entraîne un modèle XGBoost et loggue les métriques et artefacts dans MLflow."""
    default_params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 250,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
    }
    final_params = {**default_params, **(params or {})}

    logger.info("Génération des données d'entraînement...")
    X, y, features = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Entraînement du modèle XGBoost...")
    model = xgb.XGBRegressor(**final_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    logger.info(f"Métriques calculées : {metrics}")

    # Enregistrement dans MLflow
    run_id = mlflow_tracker.log_training_run(
        model=model,
        params=final_params,
        metrics=metrics,
        model_name=mlflow_settings.model_name,
    )

    # Entraînement et sauvegarde du SmartExplainer Shapash
    try:
        init_smart_explainer(model, X_test, y_test)
        logger.info(f"SmartExplainer Shapash initialisé avec succès pour le run {run_id}.")
    except Exception as e:
        logger.warning(f"Initialisation Shapash différée : {e}")

    return model.get_booster(), metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()

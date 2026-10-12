"""
hpo_optuna.py — Optimisation distribuée d'hyperparamètres avec Optuna, Ray et PostgreSQL
"""

import logging
import os
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ..common.config import db_settings, mlflow_settings
from .train_xgboost import generate_synthetic_data

logger = logging.getLogger("k8s_mlops.hpo")


def objective(trial: optuna.Trial, X_train, X_val, y_train, y_val) -> float:
    """Fonction objectif pour l'optimisation bayésienne d'hyperparamètres XGBoost."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "reg:squarederror",
        "random_state": 42,
        "nthread": 1,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

    # Support de l'élagage (pruning)
    trial.report(rmse, step=trial.number)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return rmse


def run_hpo_study(
    n_trials: int = 20,
    study_name: str = "xgboost-hpo-study",
    use_ray: bool = True,
    n_workers: int = 2,
) -> dict[str, Any]:
    """Exécute l'étude Optuna distribuée sur cluster Ray avec RDBStorage PostgreSQL."""
    X, y, _ = generate_synthetic_data(n_samples=1000)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    storage_url = db_settings.get_url_for_db("optuna")
    logger.info(f"Connexion RDBStorage Optuna : {db_settings.host}:{db_settings.port}/optuna")

    storage = None
    # Tentative d'utilisation de PostgreSQL si disponible
    if "localhost" not in db_settings.host and "127.0.0.1" not in db_settings.host:
        try:
            storage = optuna.storages.RDBStorage(url=storage_url)
        except Exception as e:
            logger.warning(f"PostgreSQL Optuna inaccessible ({e}), passage en mémoire.")
            storage = None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )

    ray_dispatched = False
    if use_ray:
        try:
            import ray

            ray_address = os.getenv("RAY_ADDRESS", None)
            if not ray.is_initialized():
                ray.init(
                    address=ray_address,
                    ignore_reinit_error=True,
                    logging_level=logging.WARNING,
                )
            logger.info(f"⚡ Ray connecté pour l'orchestration HPO (address={ray_address or 'local'}).")

            @ray.remote
            def _ray_hpo_worker(s_name, s_url, worker_trials, X_tr, X_va, y_tr, y_va):
                import optuna
                stg = optuna.storages.RDBStorage(url=s_url) if s_url else None
                s = optuna.load_study(study_name=s_name, storage=stg)
                s.optimize(lambda t: objective(t, X_tr, X_va, y_tr, y_va), n_trials=worker_trials)
                return True

            trials_per_worker = max(1, n_trials // n_workers)
            futures = [
                _ray_hpo_worker.remote(
                    study_name, storage_url if storage else None, trials_per_worker, X_train, X_val, y_train, y_val
                )
                for _ in range(n_workers)
            ]
            ray.get(futures, timeout=300)
            ray_dispatched = True
            logger.info(f"Distribution de {n_trials} trials complétée avec succès sur {n_workers} workers Ray.")
        except Exception as e:
            logger.warning(f"Exécution Ray ignorée/indisponible ({e}). Exécution Optuna locale.")
            ray_dispatched = False

    if not ray_dispatched:
        logger.info(f"Exécution séquentielle/locale de l'étude Optuna '{study_name}' ({n_trials} trials)...")
        study.optimize(
            lambda t: objective(t, X_train, X_val, y_train, y_val),
            n_trials=n_trials,
            timeout=300,
        )

    # Rechargement des résultats de l'étude
    if storage:
        study = optuna.load_study(study_name=study_name, storage=storage)

    logger.info(f"🏆 Meilleur RMSE obtenu : {study.best_value:.4f}")
    logger.info(f"⚙️ Meilleurs hyperparamètres : {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "total_trials": len(study.trials),
        "ray_distributed": ray_dispatched,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_hpo_study(n_trials=5, use_ray=False)

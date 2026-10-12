"""
Module d'utilitaires transverses pour k8s_mlops_platform.
Regroupe les outils de rechargement d'API, vérification d'état, ingestion, gouvernance MLflow et reset MLOps.
"""

from .api_reloader import reload_serving_api
from .check_status import check_platform_status
from .data_loader import load_or_generate_dataset
from .db import inspect_database_tables, test_database_connections
from .features import compute_cyclical_features, impute_missing_features, scale_numerical_features
from .full_reset_mlops import run_full_mlops_reset, run_full_reset
from .mlflow_manager import MLflowQualityGate
from .process_queue import process_incoming_queue, process_queue_files
from .threshold import evaluate_predictions_and_curves, find_optimal_threshold

__all__ = [
    "reload_serving_api",
    "check_platform_status",
    "load_or_generate_dataset",
    "inspect_database_tables",
    "test_database_connections",
    "compute_cyclical_features",
    "impute_missing_features",
    "scale_numerical_features",
    "run_full_mlops_reset",
    "run_full_reset",
    "MLflowQualityGate",
    "process_incoming_queue",
    "process_queue_files",
    "find_optimal_threshold",
    "evaluate_predictions_and_curves",
]

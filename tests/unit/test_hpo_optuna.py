"""Tests unitaires pour l'optimisation bayésienne d'hyperparamètres Optuna et Ray."""

from src.training.hpo_optuna import run_hpo_study


def test_hpo_optuna_study_execution():
    """Vérifie l'optimisation locale d'hyperparamètres avec échantillonnage TPE."""
    result = run_hpo_study(n_trials=3, study_name="test-study-local", use_ray=False)
    assert "best_params" in result
    assert "best_rmse" in result
    assert result["total_trials"] == 3
    assert result["best_rmse"] > 0.0
    assert "max_depth" in result["best_params"]


def test_hpo_optuna_ray_fallback():
    """Vérifie le repli gracieux et l'achèvement de l'étude quand Ray n'a pas de cluster distant."""
    result = run_hpo_study(n_trials=2, study_name="test-study-ray", use_ray=True)
    assert "best_params" in result
    assert "best_rmse" in result
    assert result["total_trials"] == 2

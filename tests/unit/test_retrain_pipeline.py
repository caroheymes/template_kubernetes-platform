"""Tests unitaires pour l'orchestrateur de réentraînement en boucle fermée."""

from unittest.mock import patch

from src.training.retrain_pipeline import run_closed_loop_retraining


def test_closed_loop_retraining_pipeline():
    """Vérifie l'exécution séquentielle complète : HPO -> Train -> Évaluation -> Résultat."""
    event = {"drift_type": "concept_drift", "drift_score": 0.42}
    with patch("src.serving.predictor.predictor.load_model", return_value=None):
        result = run_closed_loop_retraining(event_details=event, use_ray=False, n_trials=2)

    assert result["status"] == "completed"
    assert result["drift_cause"] == "concept_drift"
    assert "hpo" in result
    assert "metrics" in result
    assert "rmse" in result["metrics"]
    assert isinstance(result["promoted_to_champion"], bool)


def test_evaluate_and_promote_logic():
    """Valide les seuils d'amélioration relative et l'assignation d'alias Champion/Challenger."""
    from src.training.evaluate import evaluate_and_promote

    # Cas 1 : Premier modèle sans champion existant
    with (
        patch("src.tracking.mlflow_client.mlflow_tracker.get_champion_metadata", return_value=None),
        patch("src.tracking.mlflow_client.mlflow_tracker.set_model_alias") as mock_set_alias,
    ):
        promoted = evaluate_and_promote(challenger_rmse=1.0, challenger_version="1")
        assert promoted is True
        mock_set_alias.assert_called_with("1", "champion")

    # Cas 2 : Challenger supérieur (gain > 2%)
    mock_champ = {"version": "1", "rmse": 1.5}
    with (
        patch(
            "src.tracking.mlflow_client.mlflow_tracker.get_champion_metadata",
            return_value=mock_champ,
        ),
        patch("src.tracking.mlflow_client.mlflow_tracker.set_model_alias") as mock_set_alias,
    ):
        promoted = evaluate_and_promote(
            challenger_rmse=1.2, challenger_version="2", threshold_improvement_pct=2.0
        )
        assert promoted is True
        # Ancien champion en rollback, nouveau en champion
        mock_set_alias.assert_any_call("1", "rollback")
        mock_set_alias.assert_any_call("2", "champion")

    # Cas 3 : Challenger insuffisant (gain < 2%)
    with (
        patch(
            "src.tracking.mlflow_client.mlflow_tracker.get_champion_metadata",
            return_value=mock_champ,
        ),
        patch("src.tracking.mlflow_client.mlflow_tracker.set_model_alias") as mock_set_alias,
    ):
        promoted = evaluate_and_promote(
            challenger_rmse=1.49, challenger_version="3", threshold_improvement_pct=2.0
        )
        assert promoted is False
        mock_set_alias.assert_called_with("3", "challenger-rejected")

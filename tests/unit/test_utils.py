"""
Tests unitaires pour les utilitaires transverses src/utils/
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.api_reloader import reload_serving_api
from src.utils.features import compute_cyclical_features, impute_missing_features, scale_numerical_features
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.process_queue import process_incoming_queue, process_queue_files
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probas = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    thresh, score = find_optimal_threshold(y_true, y_probas, metric_target="f1")
    assert 0.05 <= thresh <= 0.95
    assert score > 0.5

    # Test edge case : listes vides ou classe unique
    t_empty, s_empty = find_optimal_threshold([], [])
    assert t_empty == 0.50
    assert s_empty == 0.0

    t_single, s_single = find_optimal_threshold([1, 1, 1], [0.8, 0.9, 0.7])
    assert t_single == 0.50
    assert s_single == 0.0


def test_evaluate_predictions_and_curves():
    y_true = [0, 1, 0, 1]
    y_probas = [0.1, 0.9, 0.2, 0.8]

    metrics, cm = evaluate_predictions_and_curves(y_true, y_probas, threshold=0.5)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert cm["tp"] == 2
    assert cm["tn"] == 2
    assert cm["fp"] == 0
    assert cm["fn"] == 0

    # Test edge case : entrées vides
    empty_metrics, empty_cm = evaluate_predictions_and_curves([], [])
    assert empty_metrics["accuracy"] == 0.0
    assert empty_cm["tp"] == 0


def test_features_utils():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })

    df_cyc = compute_cyclical_features(df, "timestamp")
    assert "hour_sin" in df_cyc.columns
    assert "hour_cos" in df_cyc.columns

    # Test scaling sans statistiques préalables
    df_scaled, stats = scale_numerical_features(df, ["val"])
    assert "val" in df_scaled.columns
    assert "val" in stats
    assert "mean" in stats["val"]
    assert "std" in stats["val"]

    # Test scaling avec statistiques précalculées (inférence temps réel)
    fixed_stats = {"val": {"mean": 5.0, "std": 2.0}}
    df_scaled_fixed, merged_stats = scale_numerical_features(df, ["val"], stats=fixed_stats)
    assert df_scaled_fixed["val"].iloc[0] == pytest.approx((1.0 - 5.0) / 2.0, abs=1e-4)
    assert "val" in merged_stats

    # Test imputation des valeurs manquantes
    df_missing = pd.DataFrame({"feature_1": [np.nan, 2.0], "feature_2": [1.0, np.nan]})
    df_imputed = impute_missing_features(df_missing, defaults={"feature_1": 0.0, "feature_2": 0.0, "feature_3": 10.0})
    assert df_imputed["feature_1"].isna().sum() == 0
    assert "feature_3" in df_imputed.columns


def test_mlflow_quality_gate():
    # Minimisation (RMSE)
    gate_min = MLflowQualityGate(metric_target="rmse", min_improvement_pct=2.0)
    should_promote, delta = gate_min.should_promote(candidate_score=1.0, champion_score=1.2)
    assert should_promote is True
    assert delta > 0

    should_not_promote, _ = gate_min.should_promote(candidate_score=1.19, champion_score=1.2)
    assert should_not_promote is False

    # Maximisation (F1 / R2) y compris scores négatifs
    gate_max = MLflowQualityGate(metric_target="r2", min_improvement_pct=2.0)
    # Champion négatif (-0.5), candidat meilleur (-0.2)
    promoted_neg, delta_neg = gate_max.should_promote(candidate_score=-0.2, champion_score=-0.5)
    assert promoted_neg is True
    assert delta_neg > 0

    # Mode hors-ligne / fallback
    gate_offline = MLflowQualityGate(tracking_uri="http://127.0.0.1:9999")
    assert gate_offline.client is None
    dummy_model = MagicMock()
    is_promoted, version = gate_offline.log_and_evaluate(dummy_model, {"rmse": 0.5})
    assert is_promoted is True
    assert version == "v-offline"


def test_api_reloader():
    from src.serving.predictor import predictor
    with patch.object(predictor, "load_model") as mock_load:
        mock_load.return_value = True
        with patch.object(predictor, "_model", MagicMock()):
            result = reload_serving_api()
            assert result is True
            mock_load.assert_called_with(force=True)


def test_process_queue_batch_format(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    sample_file = queue_dir / "batch_001.csv"
    sample_file.write_text("feature_1,feature_2,feature_3,feature_4\n1.0,2.0,3.0,4.0\n5.0,6.0,7.0,8.0\n")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"predictions": [{"prediction": 0.1}, {"prediction": 0.2}]}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = process_queue_files(queue_dir=str(queue_dir), delete_on_success=True)
        assert res["files_processed"] == 1
        assert res["total_records"] == 2
        assert mock_post.called

        # Vérifier que le payload respecte le schéma BatchPredictionRequest
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert "items" in payload
        assert len(payload["items"]) == 2
        assert "features" in payload["items"][0]

        # Vérifier que le fichier a été supprimé après succès
        assert not sample_file.exists()

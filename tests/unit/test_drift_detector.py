"""Tests unitaires pour la détection de Data Drift et Concept Drift."""

import numpy as np
import pandas as pd

from src.monitoring.drift_detector import StatisticalDriftDetector


def test_data_drift_detection_no_drift():
    """Vérifie l'absence d'alerte lorsque les distributions sont stables."""
    detector = StatisticalDriftDetector()
    np.random.seed(42)
    ref = pd.DataFrame(np.random.randn(300, 3), columns=["f1", "f2", "f3"])
    curr = pd.DataFrame(np.random.randn(300, 3), columns=["f1", "f2", "f3"])

    res = detector.calculate_data_drift(ref, curr)
    assert not res["dataset_drifted"]
    assert res["drifted_features_count"] <= 1


def test_data_drift_detection_with_shift():
    """Vérifie la détection de covariate shift lorsque les variables dévient."""
    detector = StatisticalDriftDetector()
    np.random.seed(42)
    ref = pd.DataFrame(np.random.randn(200, 3), columns=["f1", "f2", "f3"])
    # Dérive artificielle massive sur f1 et f2
    curr = pd.DataFrame(
        np.random.randn(200, 3) + np.array([3.0, 4.0, 0.0]),
        columns=["f1", "f2", "f3"],
    )

    res = detector.calculate_data_drift(ref, curr)
    assert res["dataset_drifted"]
    assert "f1" in res["drifted_features"]
    assert "f2" in res["drifted_features"]


def test_concept_drift_detection():
    """Vérifie la détection de concept drift (dégradation de la performance prédictive P(Y|X))."""
    detector = StatisticalDriftDetector(concept_drift_ratio_threshold=1.15)
    np.random.seed(42)

    # Référence : résidus faibles (RMSE ~ 0.2)
    ref = pd.DataFrame(
        {
            "actual": np.linspace(10, 20, 100),
            "prediction": np.linspace(10, 20, 100) + np.random.randn(100) * 0.2,
        }
    )

    # Données de production avec dérive de concept : fortes erreurs résiduelles (RMSE ~ 2.0)
    curr = pd.DataFrame(
        {
            "actual": np.linspace(10, 20, 100),
            "prediction": np.linspace(10, 20, 100) + np.random.randn(100) * 2.0,
        }
    )

    res = detector.calculate_concept_drift(ref, curr)
    assert res["concept_drifted"]
    assert res["rmse_degradation_ratio"] > 2.0


def test_combined_full_drift_assessment():
    """Vérifie le diagnostic unifié et le drapeau requires_retraining."""
    detector = StatisticalDriftDetector()
    np.random.seed(42)

    ref = pd.DataFrame(
        {
            "f1": np.random.randn(100),
            "actual": np.ones(100),
            "prediction": np.ones(100),
        }
    )
    curr = pd.DataFrame(
        {
            "f1": np.random.randn(100) + 3.0,  # Data drift
            "actual": np.ones(100),
            "prediction": np.ones(100) + np.random.randn(100) * 2.0,  # Concept drift
        }
    )

    res = detector.calculate_drift(ref, curr)
    assert res["requires_retraining"]
    assert res["drift_type"] in ["data_drift", "concept_drift", "dual_drift"]

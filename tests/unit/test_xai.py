"""Tests unitaires pour le composant d'explicabilité XAI Shapash."""

from unittest.mock import patch

import pandas as pd

from src.xai.explainer import ShapashExplainerManager


def test_explainer_fallback_importance():
    """Vérifie que l'explainer fournit une importance par défaut si non compilé."""
    mgr = ShapashExplainerManager()
    importance = mgr.get_global_feature_importance()
    assert isinstance(importance, dict)
    assert "feature_1" in importance
    assert sum(importance.values()) > 0.9  # Somme proche de 1.0


def test_explainer_single_prediction():
    """Vérifie la génération des contributions locales de variables (waterfall)."""
    mgr = ShapashExplainerManager()
    features = {"feature_1": 2.5, "feature_2": -1.0, "feature_3": 0.5, "feature_4": 4.0}
    explanation = mgr.explain_single_prediction(features)

    assert "contributions" in explanation
    assert "top_drivers" in explanation
    assert len(explanation["contributions"]) == 4
    assert len(explanation["top_drivers"]) == 4

    # Vérification que le tri est bien par contribution absolue décroissante
    drivers = explanation["top_drivers"]
    for i in range(len(drivers) - 1):
        assert abs(drivers[i][1]) >= abs(drivers[i + 1][1])


def test_explainer_compile_resilience():
    """Vérifie que l'initialisation avec un modèle non supporté bascule proprement en fallback."""
    mgr = ShapashExplainerManager()
    with patch("shapash.SmartExplainer", side_effect=Exception("Explainer compile failure")):
        mgr.init_explainer("invalid_model", pd.DataFrame({"a": [1, 2]}))
        assert not mgr._is_compiled

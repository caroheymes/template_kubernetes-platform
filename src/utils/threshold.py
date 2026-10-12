"""
threshold.py — Calibration et optimisation du seuil de décision (Threshold Tuning)
Fournit des fonctions universelles pour calibrer le seuil de probabilité selon la métrique
cible (F1, F2, F-beta, AUPRC / PR-AUC, matrice de coût, contrainte de précision/rappel).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("k8s_mlops.threshold")


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series | list[Any],
    y_probas: np.ndarray | pd.Series | list[Any],
    metric_target: str = "f1",
    beta: float = 1.0,
    min_precision: float = 0.50,
    min_recall: float = 0.50,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
) -> tuple[float, float]:
    """
    Recherche le seuil de décision optimal maximisant la métrique choisie sur les probabilités prédites.

    Args:
        y_true: Vérité terrain binaire (0 ou 1).
        y_probas: Probabilités de la classe positive (1).
        metric_target: Métrique cible ('f1', 'f2', 'fbeta', 'auprc', 'recall_at_precision', 'cost_sensitive').
        beta: Paramètre beta pour F-beta (ex: 2.0 pour F2, 1.0 pour F1).
        min_precision: Précision minimale requise pour 'recall_at_precision'.
        min_recall: Rappel minimal requis pour 'precision_at_recall'.
        cost_fp: Coût unitaire d'un faux positif (pour 'cost_sensitive').
        cost_fn: Coût unitaire d'un faux négatif (pour 'cost_sensitive').
        min_thresh: Borne inférieure du seuil de décision (défaut: 0.05).
        max_thresh: Borne supérieure du seuil de décision (défaut: 0.95).

    Returns:
        Tuple (optimal_threshold: float, best_metric_score: float)
    """
    y_true_np = np.asarray(y_true).astype(int)
    y_probas_np = np.asarray(y_probas).astype(float)

    if len(y_true_np) == 0 or len(np.unique(y_true_np)) < 2:
        return 0.50, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_probas_np)

    p = precisions[:-1]
    r = recalls[:-1]
    eps = 1e-10

    metric = metric_target.lower()

    if metric in ["brier", "brier_score"]:
        bs = brier_score_loss(y_true_np, y_probas_np)
        return 0.50, round(float(bs), 4)

    if metric in ["f2", "f2_score"]:
        b = 2.0
        scores = (1 + b**2) * (p * r) / (b**2 * p + r + eps)
    elif metric in ["f1", "f1_score", "auprc", "pr_auc"]:
        scores = 2 * (p * r) / (p + r + eps)
    elif metric == "fbeta":
        b2 = beta**2
        scores = (1 + b2) * (p * r) / (b2 * p + r + eps)
    elif metric == "recall_at_precision":
        valid = p >= min_precision
        scores = np.where(valid, r, -1.0) if np.any(valid) else (p * r)
    elif metric == "precision_at_recall":
        valid = r >= min_recall
        scores = np.where(valid, p, -1.0) if np.any(valid) else (p * r)
    elif metric == "cost_sensitive":
        total_p = np.sum(y_true_np == 1)
        total_n = len(y_true_np) - total_p
        tp = r * total_p
        fp = np.where(p > 0, (1 - p) * tp / (p + eps), total_n)
        fn = total_p - tp
        total_cost = cost_fp * fp + cost_fn * fn
        scores = -total_cost
    else:
        # F1 par défaut
        scores = 2 * (p * r) / (p + r + eps)

    valid_mask = (thresholds >= min_thresh) & (thresholds <= max_thresh)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        best_idx = int(np.argmax(scores))
        raw_thresh = float(thresholds[best_idx])
        optimal_thresh = max(min_thresh, min(max_thresh, raw_thresh))
        best_score = float(scores[best_idx])
    else:
        best_valid_idx = valid_indices[int(np.argmax(scores[valid_indices]))]
        optimal_thresh = float(thresholds[best_valid_idx])
        best_score = float(scores[best_valid_idx])

    return round(optimal_thresh, 4), round(best_score, 4)


def evaluate_predictions_and_curves(
    y_true: np.ndarray | pd.Series | list[Any],
    y_probas: np.ndarray | pd.Series | list[Any],
    threshold: float = 0.50,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Calcule à la fois :
      1. Les métriques globales indépendantes du seuil (PR-AUC / AUPRC, ROC-AUC)
      2. Les métriques opérationnelles au seuil donné (Accuracy, F1, F2, Precision, Recall)
      3. La matrice de confusion (TN, FP, FN, TP)

    Args:
        y_true: Vérité terrain binaire (0 ou 1).
        y_probas: Probabilités de la classe 1 (array 1D).
        threshold: Seuil de décision pour la classification binaire.

    Returns:
        Tuple (metrics_dict: Dict[str, float], confusion_matrix_dict: Dict[str, int])
    """
    y_true_np = np.asarray(y_true).astype(int)
    y_probas_np = np.asarray(y_probas).astype(float)

    if len(y_true_np) == 0:
        empty_metrics = {
            "pr_auc": 0.0,
            "auprc": 0.0,
            "roc_auc": 0.0,
            "brier_score": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "f1_macro": 0.0,
            "recall_macro": 0.0,
            "decision_threshold": float(threshold),
        }
        return empty_metrics, {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    try:
        auprc = float(average_precision_score(y_true_np, y_probas_np))
    except Exception:
        auprc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_np, y_probas_np))
    except Exception:
        roc_auc = 0.0

    try:
        brier = float(brier_score_loss(y_true_np, y_probas_np))
    except Exception:
        brier = 0.0

    y_pred = (y_probas_np >= threshold).astype(int)

    prec_c1 = float(precision_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    rec_c1 = float(recall_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    f1_c1 = float(f1_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    f2_c1 = float(fbeta_score(y_true_np, y_pred, beta=2.0, pos_label=1, zero_division=0))
    f1_glob = float(f1_score(y_true_np, y_pred, average="macro", zero_division=0))
    rec_glob = float(recall_score(y_true_np, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true_np, y_pred))

    metrics = {
        "pr_auc": round(auprc, 4),
        "auprc": round(auprc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec_c1, 4),
        "recall": round(rec_c1, 4),
        "f1": round(f1_c1, 4),
        "f2": round(f2_c1, 4),
        "f1_macro": round(f1_glob, 4),
        "recall_macro": round(rec_glob, 4),
        "decision_threshold": float(threshold),
    }

    try:
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred, labels=[0, 1]).ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    cm_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, cm_dict

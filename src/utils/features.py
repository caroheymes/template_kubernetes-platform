"""
features.py — Boîte à outils de Feature Engineering pour le Train et le Serving (Zero Train-Serving Skew)
Garantit la stricte parité des transformations de données entre l'entraînement et l'inférence temps réel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cyclical_features(
    df: pd.DataFrame,
    datetime_col: str | None = None,
) -> pd.DataFrame:
    """
    Extrait et encode les caractéristiques temporelles en composantes cycliques (sinus et cosinus).
    Gère les valeurs manquantes ou formats invalides sans lever d'erreur bloquante.
    """
    df_out = df.copy()

    if datetime_col and datetime_col in df_out.columns:
        dt = pd.to_datetime(df_out[datetime_col], errors="coerce")
        hour = dt.dt.hour.fillna(0)
        day = dt.dt.dayofweek.fillna(0)
        month = dt.dt.month.fillna(1)

        df_out["hour_sin"] = np.round(np.sin(2 * np.pi * hour / 24.0), 4)
        df_out["hour_cos"] = np.round(np.cos(2 * np.pi * hour / 24.0), 4)
        df_out["day_sin"] = np.round(np.sin(2 * np.pi * day / 7.0), 4)
        df_out["day_cos"] = np.round(np.cos(2 * np.pi * day / 7.0), 4)
        df_out["month_sin"] = np.round(np.sin(2 * np.pi * month / 12.0), 4)
        df_out["month_cos"] = np.round(np.cos(2 * np.pi * month / 12.0), 4)
    elif "hour" in df_out.columns:
        h = pd.to_numeric(df_out["hour"], errors="coerce").fillna(0)
        df_out["hour_sin"] = np.round(np.sin(2 * np.pi * h / 24.0), 4)
        df_out["hour_cos"] = np.round(np.cos(2 * np.pi * h / 24.0), 4)

    return df_out


def scale_numerical_features(
    df: pd.DataFrame,
    columns: list[str],
    stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Normalise les variables numériques (standard z-score: (x - mu) / sigma).
    Retourne le DataFrame transformé et le dictionnaire de statistiques pour l'inférence.
    """
    df_out = df.copy()
    computed_stats: dict[str, dict[str, float]] = {}

    for col in columns:
        if col not in df_out.columns:
            continue
        vals = pd.to_numeric(df_out[col], errors="coerce")
        if stats and col in stats:
            mu = stats[col]["mean"]
            std = max(stats[col]["std"], 1e-6)
        else:
            mu = float(vals.mean()) if not vals.empty else 0.0
            std = float(vals.std()) if len(vals) > 1 and vals.std() > 0 else 1.0
            computed_stats[col] = {"mean": round(mu, 4), "std": round(std, 4)}

        df_out[col] = np.round((vals - mu) / std, 4)

    return df_out, {**(stats or {}), **computed_stats}


def impute_missing_features(df: pd.DataFrame, defaults: dict[str, float] | None = None) -> pd.DataFrame:
    """Remplace les valeurs nulles ou manquantes par les valeurs par défaut spécifiées."""
    df_out = df.copy()
    default_vals = defaults or {"feature_1": 0.0, "feature_2": 0.0, "feature_3": 0.0, "feature_4": 0.0}
    for col, val in default_vals.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(val)
        else:
            df_out[col] = val
    return df_out

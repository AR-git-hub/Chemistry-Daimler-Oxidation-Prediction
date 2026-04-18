"""Leakage-safe per-target component feature selection for DOT Deep Sets."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import SCENARIO_ID, TARGET_COLUMNS
from .data import _coerce_frame_types, _feature_columns

# Always keep process / DOT protocol drivers and mass (weak stats should not drop them).
_PROTECTED_FEATURES: tuple[str, ...] = (
    "Массовая доля, %",
    "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
    "Время испытания | - Daimler Oxidation Test (DOT), ч",
    "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
    "Дозировка катализатора, категория",
)


def _scenario_weighted_means(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """One row per scenario_id: mass-weighted mean of each numeric component feature."""
    mass_col = "Массовая доля, %"
    rows: list[dict] = []
    for _, group in df.groupby(SCENARIO_ID, sort=True):
        weights = group[mass_col].fillna(0.0).to_numpy(dtype=np.float64)
        if float(np.sum(np.abs(weights))) < 1e-12:
            weights = np.ones(len(group), dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1e-12)
        rec: dict = {}
        for f in feature_names:
            v = group[f].fillna(0.0).to_numpy(dtype=np.float64)
            rec[f] = float(np.dot(v, weights))
        for tc in TARGET_COLUMNS:
            rec[tc] = float(group[tc].iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def select_component_features_for_target(
    train_df: pd.DataFrame,
    target_column: str,
    *,
    min_abs_spearman: float = 0.065,
    min_features: int = 50,
) -> List[str]:
    """Drop component columns whose mass-weighted scenario signal is weakest for this target.

    Fit **only** on the provided training frame (e.g. one CV fold's rows) to avoid leakage.

    Uses |Spearman| between scenario-level weighted means and the scenario target. Protected
    protocol/mass columns are always kept; if the threshold removes too many features, we keep
    the top ``min_features`` by |Spearman| among the rest.
    """
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target_column {target_column!r}")

    work = _coerce_frame_types(train_df)
    all_feats = _feature_columns(work)
    if not all_feats:
        return []

    table = _scenario_weighted_means(work, all_feats)
    y = table[target_column].to_numpy(dtype=np.float64)

    abs_corr: dict[str, float] = {}
    for f in all_feats:
        x = table[f].to_numpy(dtype=np.float64)
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            abs_corr[f] = 0.0
            continue
        r, _ = spearmanr(x, y, nan_policy="omit")
        abs_corr[f] = float(abs(r)) if np.isfinite(r) else 0.0

    ranked = sorted(all_feats, key=lambda c: -abs_corr[c])
    keep = {f for f in all_feats if abs_corr[f] >= min_abs_spearman}

    for p in _PROTECTED_FEATURES:
        if p in all_feats:
            keep.add(p)

    if len(keep) < min_features:
        keep = set(ranked[:min_features])

    return [f for f in all_feats if f in keep]

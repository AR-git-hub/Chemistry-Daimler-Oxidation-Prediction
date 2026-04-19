"""Per-component-type (``Компонент``) top-K numeric features by |Spearman| vs one DOT target.

Fit only on the provided training frame (e.g. one CV fold) to avoid leakage."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import TARGET_COLUMNS
from .data import _coerce_frame_types, _feature_columns

COMPONENT_COL = "Компонент"


def _global_top_k_by_spearman(
    work: pd.DataFrame,
    feature_names: List[str],
    target_column: str,
    k: int,
) -> List[str]:
    """Fallback ranking on all train rows vs target."""
    y = work[target_column].to_numpy(dtype=np.float64)
    scores: List[tuple[str, float]] = []
    for f in feature_names:
        x = work[f].fillna(0.0).to_numpy(dtype=np.float64)
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            scores.append((f, 0.0))
            continue
        r, _ = spearmanr(x, y, nan_policy="omit")
        scores.append((f, float(abs(r)) if np.isfinite(r) else 0.0))
    scores.sort(key=lambda t: -t[1])
    return [c for c, _ in scores[:k]]


def top_k_features_per_component_for_target(
    train_df: pd.DataFrame,
    target_column: str,
    *,
    k: int = 3,
    min_rows_per_component: int = 12,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return (sorted union of selected features, map component -> k feature names).

    Rows with ``Компонент`` having fewer than ``min_rows_per_component`` rows use
    a global Spearman ranking on all rows instead of a per-type slice.
    """
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target_column {target_column!r}")
    if k < 1:
        raise ValueError("k must be >= 1")

    work = _coerce_frame_types(train_df)
    if COMPONENT_COL not in work.columns:
        raise ValueError(f"Missing column {COMPONENT_COL!r} for component-aware selection.")

    base_feats = _feature_columns(work)
    if not base_feats:
        return [], {}

    fallback = _global_top_k_by_spearman(work, base_feats, target_column, k)

    comp_map: Dict[str, List[str]] = {}

    for comp, sub in work.groupby(COMPONENT_COL, sort=True):
        comp_s = str(comp)
        if len(sub) < min_rows_per_component:
            comp_map[comp_s] = list(fallback)
            continue
        y = sub[target_column].to_numpy(dtype=np.float64)
        scores: List[tuple[str, float]] = []
        for f in base_feats:
            x = sub[f].fillna(0.0).to_numpy(dtype=np.float64)
            if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                scores.append((f, 0.0))
                continue
            r, _ = spearmanr(x, y, nan_policy="omit")
            scores.append((f, float(abs(r)) if np.isfinite(r) else 0.0))
        scores.sort(key=lambda t: -t[1])
        comp_map[comp_s] = [c for c, _ in scores[:k]]

    union = sorted({f for feats in comp_map.values() for f in feats})
    return union, comp_map

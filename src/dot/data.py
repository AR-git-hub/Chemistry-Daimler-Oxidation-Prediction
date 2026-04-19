"""Data loading and scenario-to-set transformation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import FEATURE_BLOCKLIST, ID_COLUMNS, SCENARIO_ID, TARGET_COLUMNS


BOOL_MAP = {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}


def resolve_typical_fallback(
    measured: pd.DataFrame,
    typical: pd.DataFrame,
    key_columns: Sequence[str],
    value_column: str,
) -> pd.DataFrame:
    """Use measured values first and fill gaps from typical values.

    The current repository already contains merged feature tables, but this helper
    keeps the fallback logic explicit and reusable for raw-data expansion.
    """
    if measured.empty:
        return typical.copy()

    left = measured.copy()
    right = typical.copy()
    merged = left.merge(
        right[key_columns + [value_column]],
        how="left",
        on=list(key_columns),
        suffixes=("", "__typical"),
    )
    merged[value_column] = merged[value_column].fillna(merged[f"{value_column}__typical"])
    return merged.drop(columns=[f"{value_column}__typical"])


@dataclass
class ScenarioSetData:
    """Set-formatted data for Deep Sets training or inference."""

    scenario_ids: List[str]
    components: List[np.ndarray]
    context: np.ndarray
    targets: np.ndarray | None
    feature_columns: List[str]
    context_columns: List[str]
    interaction_feature_columns: List[str]
    target_columns: List[str]


def _coerce_frame_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed dtypes to numeric where possible and fill missing values."""
    work = df.copy()
    for col in work.columns:
        if work[col].dtype == object:
            mapped = work[col].replace(BOOL_MAP)
            try:
                work[col] = pd.to_numeric(mapped)
            except (TypeError, ValueError):
                work[col] = mapped
    for col in work.columns:
        if pd.api.types.is_bool_dtype(work[col]):
            work[col] = work[col].astype(float)
    work = work.replace([np.inf, -np.inf], np.nan)
    return work


def _feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = set(ID_COLUMNS + TARGET_COLUMNS) | set(FEATURE_BLOCKLIST)
    cols = [c for c in df.columns if c not in excluded]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


MASS_INT_PREFIX = "mass_int::"


def _ensure_mass_interaction_columns(work: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    """Add ``mass_int::base`` columns (feat × row mass fraction within scenario) if listed but missing."""
    mass_col = "Массовая доля, %"
    if mass_col not in work.columns:
        return
    for c in feature_cols:
        if not isinstance(c, str) or not c.startswith(MASS_INT_PREFIX):
            continue
        if c in work.columns:
            continue
        base = c.split("::", 1)[1]
        if base not in work.columns:
            raise ValueError(f"mass_int feature {c!r} references missing base column {base!r}")
        parts: list[pd.Series] = []
        for _, g in work.groupby(SCENARIO_ID, sort=True):
            m = g[mass_col].fillna(0.0).to_numpy(dtype=np.float64)
            s = float(np.abs(m).sum())
            if s < 1e-12:
                mn = np.ones_like(m, dtype=np.float64) / max(len(m), 1)
            else:
                mn = np.abs(m) / s
            v = g[base].fillna(0.0).to_numpy(dtype=np.float64)
            parts.append(pd.Series(v * mn, index=g.index))
        work[c] = pd.concat(parts)


def _append_mass_interaction_columns(work: pd.DataFrame, feature_cols: list[str], k: int) -> list[str]:
    """Append top-k variance features as ``mass_int::name`` = value × (|mass| / sum|mass|) per scenario."""
    mass_col = "Массовая доля, %"
    if k <= 0 or mass_col not in work.columns:
        return feature_cols
    candidates = [c for c in feature_cols if c != mass_col and not str(c).startswith(MASS_INT_PREFIX)]
    if not candidates:
        return feature_cols
    slice_df = work[candidates].fillna(0.0).astype(float)
    var = slice_df.var(axis=0).sort_values(ascending=False)
    topk = var.head(min(int(k), len(candidates))).index.tolist()
    new_cols = list(feature_cols)
    for base in topk:
        name = f"{MASS_INT_PREFIX}{base}"
        if name in work.columns:
            new_cols.append(name)
            continue
        parts: list[pd.Series] = []
        for _, g in work.groupby(SCENARIO_ID, sort=True):
            m = g[mass_col].fillna(0.0).to_numpy(dtype=np.float64)
            s = float(np.abs(m).sum())
            if s < 1e-12:
                mn = np.ones_like(m, dtype=np.float64) / max(len(m), 1)
            else:
                mn = np.abs(m) / s
            v = g[base].fillna(0.0).to_numpy(dtype=np.float64)
            parts.append(pd.Series(v * mn, index=g.index))
        work[name] = pd.concat(parts)
        new_cols.append(name)
    return new_cols


def _build_interaction_context(
    group: pd.DataFrame,
    interaction_feature_cols: Sequence[str],
) -> tuple[np.ndarray, List[str]]:
    """Build scenario-level context with robust interaction summaries."""
    mass_col = "Массовая доля, %"
    condition_candidates = [
        "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C",
        "Время испытания | - Daimler Oxidation Test (DOT), ч",
        "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс",
        "Дозировка катализатора, категория",
    ]
    condition_cols = [c for c in condition_candidates if c in group.columns]
    base_cols = [mass_col] + [c for c in condition_cols if c != mass_col]
    base_cols = [c for c in base_cols if c in group.columns]

    weights = group[mass_col].fillna(0.0).to_numpy(dtype=np.float32) if mass_col in group.columns else None
    if weights is None or float(np.sum(np.abs(weights))) < 1e-8:
        weights = np.ones((len(group),), dtype=np.float32)
    w_sum = float(weights.sum())
    weights = weights / max(w_sum, 1e-8)

    top_interaction_cols = list(interaction_feature_cols)

    context_values: List[float] = []
    context_names: List[str] = []

    context_values.append(float(len(group)))
    context_names.append("ctx_num_components")

    if mass_col in group.columns:
        masses = group[mass_col].fillna(0.0).to_numpy(dtype=np.float32)
        context_values.extend([float(masses.sum()), float(masses.mean()), float(masses.std())])
        context_names.extend(["ctx_mass_sum", "ctx_mass_mean", "ctx_mass_std"])

    for col in condition_cols:
        val = float(group[col].iloc[0]) if col in group.columns else 0.0
        context_values.append(val)
        context_names.append(f"ctx_condition::{col}")

    for col in top_interaction_cols:
        vec = group[col].fillna(0.0).to_numpy(dtype=np.float32)
        weighted_mean = float(np.sum(vec * weights))
        centered = vec - weighted_mean
        weighted_std = float(np.sqrt(np.sum((centered**2) * weights)))
        context_values.extend([weighted_mean, weighted_std])
        context_names.extend([f"ctx_wmean::{col}", f"ctx_wstd::{col}"])

    return np.array(context_values, dtype=np.float32), context_names


def frame_to_scenario_sets(
    df: pd.DataFrame,
    is_train: bool,
    feature_columns: Sequence[str] | None = None,
    interaction_feature_columns: Sequence[str] | None = None,
    *,
    mass_interaction_k: int = 0,
) -> ScenarioSetData:
    """Convert row-level component table into per-scenario sets."""
    work = _coerce_frame_types(df)
    if feature_columns is not None:
        feature_cols = list(feature_columns)
        _ensure_mass_interaction_columns(work, feature_cols)
    else:
        feature_cols = _feature_columns(work)
        if int(mass_interaction_k) > 0:
            feature_cols = _append_mass_interaction_columns(work, feature_cols, int(mass_interaction_k))
    if interaction_feature_columns is None:
        global_feature_slice = work[feature_cols].fillna(0.0).astype(float)
        global_variances = global_feature_slice.var(axis=0).sort_values(ascending=False)
        interaction_feature_cols = global_variances.head(min(12, len(global_variances))).index.tolist()
    else:
        interaction_feature_cols = list(interaction_feature_columns)

    scenario_ids: List[str] = []
    components: List[np.ndarray] = []
    context_rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    context_columns: List[str] = []

    grouped = work.groupby(SCENARIO_ID, sort=True)
    for scenario_id, group in grouped:
        x = group[feature_cols].fillna(0.0).astype(float).to_numpy(dtype=np.float32)
        ctx, ctx_cols = _build_interaction_context(group, interaction_feature_cols)
        if not context_columns:
            context_columns = ctx_cols
        scenario_ids.append(str(scenario_id))
        components.append(x)
        context_rows.append(ctx)
        if is_train:
            y = group[TARGET_COLUMNS].iloc[0].to_numpy(dtype=np.float32)
            targets.append(y)

    target_array = np.vstack(targets) if is_train else None
    context_array = np.vstack(context_rows).astype(np.float32)
    return ScenarioSetData(
        scenario_ids=scenario_ids,
        components=components,
        context=context_array,
        targets=target_array,
        feature_columns=feature_cols,
        context_columns=context_columns,
        interaction_feature_columns=interaction_feature_cols,
        target_columns=TARGET_COLUMNS,
    )


def build_feature_scaler_stats(
    components: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute normalization statistics across all component rows."""
    all_rows = np.vstack(components).astype(np.float32)
    mean = all_rows.mean(axis=0)
    std = all_rows.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def apply_feature_scaler(
    components: Sequence[np.ndarray], stats: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    """Apply precomputed scaling to each scenario set."""
    mean = stats["mean"]
    std = stats["std"]
    return [((arr - mean) / std).astype(np.float32) for arr in components]


def normalize_targets(y: np.ndarray) -> Dict[str, np.ndarray]:
    """Normalize target matrix for stable optimization."""
    mean = y.mean(axis=0).astype(np.float32)
    std = y.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    y_norm = (y - mean) / std
    return {"y_norm": y_norm.astype(np.float32), "mean": mean, "std": std}


def build_context_scaler_stats(context: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute normalization stats for scenario-level context features."""
    mean = context.mean(axis=0).astype(np.float32)
    std = context.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def apply_context_scaler(context: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize scenario-level context features."""
    return ((context - stats["mean"]) / stats["std"]).astype(np.float32)


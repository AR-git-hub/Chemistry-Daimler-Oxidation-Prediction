"""Scenario-level HistGradientBoosting on fe_cleaned aggregates (non-neural, orthogonal to DeepSets).

Tuned with KFold CV on train rows (one row per scenario). Intended for aggressive blends with
the wave2 anchor — not for squeezing 0.08886 vs 0.08888.

  PYTHONPATH=src python scripts/scenario_aggregate_hgbr.py \\
    --train-path data/merged/train_fe_cleaned.csv \\
    --test-path data/merged/test_fe_cleaned.csv \\
    --output-path predictions_scenario_agg_hgbr_fe_cleaned.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dot.config import (  # noqa: E402
    FEATURE_BLOCKLIST,
    ID_COLUMNS,
    SCENARIO_ID,
    TARGET_COLUMNS,
)


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS + TARGET_COLUMNS) | set(FEATURE_BLOCKLIST)
    out: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def build_scenario_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(SCENARIO_ID, sort=True)[feature_cols]
    mean = g.mean()
    std = g.std().fillna(0.0)
    mx = g.max()
    mn = g.min()
    sm = g.sum()
    parts = [
        mean.add_suffix("_mean"),
        std.add_suffix("_std"),
        mx.add_suffix("_max"),
        mn.add_suffix("_min"),
        sm.add_suffix("_sum"),
    ]
    X = pd.concat(parts, axis=1)
    X["n_components"] = g.size().astype(float)
    return X


def scenario_targets(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[[SCENARIO_ID, *TARGET_COLUMNS]].drop_duplicates(subset=[SCENARIO_ID], keep="first")
    return sub.set_index(SCENARIO_ID).sort_index()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    p = argparse.ArgumentParser(description="HGBR on per-scenario aggregates (multi-output).")
    p.add_argument("--train-path", type=Path, required=True)
    p.add_argument("--test-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("predictions_scenario_agg_hgbr.csv"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    feat_cols = _numeric_feature_columns(train_df)
    if not feat_cols:
        raise SystemExit("No numeric feature columns after exclusions.")

    X_tr = build_scenario_matrix(train_df, feat_cols)
    y_tr = scenario_targets(train_df).loc[X_tr.index]
    X_te = build_scenario_matrix(test_df, feat_cols)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

    y_np = y_tr.to_numpy(dtype=np.float64)
    kf = KFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))

    # Small grid: 167 scenarios — many configs overfit; keep CV tractable.
    grid = []
    for max_depth, lr, max_iter, l2 in itertools.product(
        (5, 8),
        (0.06, 0.1),
        (180,),
        (1.0, 6.0),
    ):
        grid.append({"max_depth": max_depth, "learning_rate": lr, "max_iter": max_iter, "l2_regularization": l2})

    best_cv = float("inf")
    best_params: dict | None = None

    for params in grid:
        fold_rmses: list[float] = []
        for tr_idx, va_idx in kf.split(X_tr):
            est = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    random_state=int(args.seed),
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=12,
                    **params,
                )
            )
            est.fit(X_tr.iloc[tr_idx], y_np[tr_idx])
            pred = est.predict(X_tr.iloc[va_idx])
            fold_rmses.append(_rmse(y_np[va_idx], pred))
        m = float(np.mean(fold_rmses))
        if m < best_cv:
            best_cv = m
            best_params = params

    assert best_params is not None
    final = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            random_state=int(args.seed),
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=12,
            **best_params,
        )
    )
    final.fit(X_tr, y_np)
    pred_te = final.predict(X_te)
    out = pd.DataFrame({SCENARIO_ID: X_te.index.astype(str)})
    for i, tcol in enumerate(TARGET_COLUMNS):
        out[tcol] = pred_te[:, i]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(
        f"HGBR n_feat={len(feat_cols)} X_dim={X_tr.shape[1]} n_train={len(X_tr)} "
        f"best_params={best_params} cv_rmse_mean={best_cv:.6f} -> {args.output_path}"
    )


if __name__ == "__main__":
    main()

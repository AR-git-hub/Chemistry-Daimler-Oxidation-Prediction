"""Scenario-level tabular baseline: aggregate component rows -> Ridge (other model class than DeepSets).

Uses the same scenario_id / target columns as the neural pipeline. Safe to blend with wave2
predictions CSVs (test row order matches between test_full and test_fe_cleaned).

Example:
  PYTHONPATH=src python scripts/scenario_aggregate_ridge.py \\
    --train-path data/merged/train_fe_cleaned.csv \\
    --test-path data/merged/test_fe_cleaned.csv \\
    --output-path predictions_scenario_agg_ridge_fe_cleaned.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    """One row per scenario_id; mean/std/max/min/sum over component rows."""
    g = df.groupby(SCENARIO_ID, sort=True)[feature_cols]
    mean = g.mean()
    std = g.std().fillna(0.0)
    mx = g.max()
    mn = g.min()
    sm = g.sum()
    parts = [mean.add_suffix("_mean"), std.add_suffix("_std"), mx.add_suffix("_max"), mn.add_suffix("_min"), sm.add_suffix("_sum")]
    X = pd.concat(parts, axis=1)
    X["n_components"] = g.size().astype(float)
    return X


def scenario_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Targets are constant within scenario; take first row per id."""
    sub = df[[SCENARIO_ID, *TARGET_COLUMNS]].drop_duplicates(subset=[SCENARIO_ID], keep="first")
    return sub.set_index(SCENARIO_ID).sort_index()


def main() -> None:
    p = argparse.ArgumentParser(description="Ridge on per-scenario aggregates of numeric features.")
    p.add_argument("--train-path", type=Path, required=True)
    p.add_argument("--test-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("predictions_scenario_agg_ridge.csv"))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--alphas",
        type=str,
        default="0.1,1,4,10,40,100,400,1000",
        help="Comma-separated Ridge alphas for CV selection (multi-output RMSE).",
    )
    args = p.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    feat_cols = _numeric_feature_columns(train_df)
    if not feat_cols:
        raise SystemExit("No numeric feature columns after exclusions.")

    X_tr = build_scenario_matrix(train_df, feat_cols)
    y_tr = scenario_targets(train_df).loc[X_tr.index]
    X_te = build_scenario_matrix(test_df, feat_cols)
    # align columns (rare mismatches)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    kf = KFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))

    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    best_alpha: float | None = None
    best_cv = float("inf")
    y_np = y_tr.to_numpy(dtype=np.float64)
    for alpha in alphas:
        fold_rmses: list[float] = []
        for tr_idx, va_idx in kf.split(X_tr):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, random_state=args.seed)),
                ]
            )
            pipe.fit(X_tr.iloc[tr_idx], y_np[tr_idx])
            pred = pipe.predict(X_tr.iloc[va_idx])
            fold_rmses.append(rmse(y_np[va_idx], pred))
        m = float(np.mean(fold_rmses))
        if m < best_cv:
            best_cv = m
            best_alpha = alpha

    assert best_alpha is not None
    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha, random_state=args.seed)),
        ]
    )
    final.fit(X_tr, y_np)
    pred_te = final.predict(X_te)
    out = pd.DataFrame({SCENARIO_ID: X_te.index.astype(str)})
    for i, tcol in enumerate(TARGET_COLUMNS):
        out[tcol] = pred_te[:, i]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(
        f"features={len(feat_cols)} agg_cols={X_tr.shape[1]} scenarios_train={len(X_tr)} "
        f"best_alpha={best_alpha} cv_rmse_mean={best_cv:.6f} -> {args.output_path}"
    )


if __name__ == "__main__":
    main()

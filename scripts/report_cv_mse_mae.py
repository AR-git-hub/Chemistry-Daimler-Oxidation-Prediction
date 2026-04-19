"""Print CV metrics: neural models from artifacts/*/metadata.json + tabular fe_cleaned OOF (normalized).

Neural: cv_mean_mse_normalized / cv_mean_mae_normalized (mean of per-target metrics on fold-train stats).

Tabular: KFold OOF on scenario aggregates; each fold fits on targets normalized by that fold's train
mean/std (same convention as dot.data.normalize_targets on the train slice).

Usage (from repo root):
  PYTHONPATH=src python scripts/report_cv_mse_mae.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
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

# Root-level metadata only (one json per model run).
NEURAL_KEYS = [
    "artifacts/cmp_train_fe/metadata.json",
    "artifacts/v18_deepsets_trainfe_s42_cmp/metadata.json",
    "artifacts/v20_max_trainfe_s123_t128_m12_rich/metadata.json",
    "artifacts/v16_max_trainfe_s123_m12_rich/metadata.json",
    "artifacts/v7_max_s77_m12/metadata.json",
    "artifacts/v10_max_s123_m12_rich/metadata.json",
    "artifacts/v11_attn_s77_m12_rich/metadata.json",
    "artifacts/v12_max_s123_m12_t1mae_rich/metadata.json",
    "artifacts/v13_max_s123_m12_domloss/metadata.json",
    "artifacts/v14_meanmax_s202_m12_rich/metadata.json",
    "artifacts/v14_sumpool_s311_m12_rich/metadata.json",
]


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS + TARGET_COLUMNS) | set(FEATURE_BLOCKLIST)
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_scenario_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(SCENARIO_ID, sort=True)[feature_cols]
    mean = g.mean()
    std = g.std().fillna(0.0)
    X = pd.concat(
        [
            mean.add_suffix("_mean"),
            std.add_suffix("_std"),
            g.max().add_suffix("_max"),
            g.min().add_suffix("_min"),
            g.sum().add_suffix("_sum"),
        ],
        axis=1,
    )
    X["n_components"] = g.size().astype(float)
    return X


def scenario_y(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[[SCENARIO_ID, *TARGET_COLUMNS]].drop_duplicates(subset=[SCENARIO_ID], keep="first")
    return sub.set_index(SCENARIO_ID).sort_index()


def oof_mean_norm_mse_mae(X: pd.DataFrame, y: np.ndarray, model, n_splits: int = 5, seed: int = 42) -> tuple[float, float]:
    """Mean of per-target MSE/MAE in fold-wise normalized target space (aligned with train.py idea)."""
    n = len(X)
    oof_true = np.zeros((n, 2), dtype=np.float64)
    oof_pred = np.zeros((n, 2), dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        y_tr = y[tr]
        mean = y_tr.mean(axis=0)
        std = y_tr.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        y_tr_n = (y_tr - mean) / std
        y_va = y[va]
        y_va_n = (y_va - mean) / std
        m = model()
        m.fit(X.iloc[tr], y_tr_n)
        oof_true[va] = y_va_n
        oof_pred[va] = m.predict(X.iloc[va])
    mse0 = float(np.mean((oof_pred[:, 0] - oof_true[:, 0]) ** 2))
    mse1 = float(np.mean((oof_pred[:, 1] - oof_true[:, 1]) ** 2))
    mae0 = float(np.mean(np.abs(oof_pred[:, 0] - oof_true[:, 0])))
    mae1 = float(np.mean(np.abs(oof_pred[:, 1] - oof_true[:, 1])))
    return 0.5 * (mse0 + mse1), 0.5 * (mae0 + mae1)


def main() -> None:
    print("=== Neural (root metadata.json): lower is better ===\n")
    rows: list[tuple[str, float, float]] = []
    for rel in NEURAL_KEYS:
        p = ROOT / rel
        if not p.is_file():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        name = rel.split("/")[1]
        rows.append(
            (
                name,
                float(m["cv_mean_mse_normalized"]),
                float(m["cv_mean_mae_normalized"]),
            )
        )
    rows.sort(key=lambda r: r[1])
    w = max(len(r[0]) for r in rows) if rows else 10
    for name, mse, mae in rows:
        print(f"  {name:{w}}  cv_mse_norm={mse:.6f}  cv_mae_norm={mae:.6f}")
    if rows:
        best = rows[0]
        print(f"\n  Best MSE among listed: {best[0]} ({best[1]:.6f})")

    train_path = ROOT / "data/merged/train_fe_cleaned.csv"
    if not train_path.is_file():
        print("\n(no train_fe_cleaned.csv — skip tabular block)")
        return

    df = pd.read_csv(train_path)
    feat = _numeric_feature_columns(df)
    X = build_scenario_matrix(df, feat)
    y = scenario_y(df).loc[X.index].to_numpy(dtype=np.float64)

    def ridge():
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=400.0, random_state=42)),
            ]
        )

    def hgbr():
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_depth=5,
                learning_rate=0.1,
                max_iter=180,
                l2_regularization=6.0,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=12,
            )
        )

    r_mse, r_mae = oof_mean_norm_mse_mae(X, y, ridge)
    h_mse, h_mae = oof_mean_norm_mse_mae(X, y, hgbr)

    print("\n=== Tabular on train_fe_cleaned (scenario aggregates, 5-fold OOF, normalized targets) ===\n")
    print(f"  {'Ridge(scaled)+agg':30}  oof_mse_norm={r_mse:.6f}  oof_mae_norm={r_mae:.6f}")
    print(f"  {'HGBR+agg (fixed hparams)':30}  oof_mse_norm={h_mse:.6f}  oof_mae_norm={h_mae:.6f}")

    if rows:
        ref_mse = float(json.loads((ROOT / "artifacts/v7_max_s77_m12/metadata.json").read_text(encoding="utf-8"))["cv_mean_mse_normalized"])
        print(f"\n  vs v7 MSE: Ridge {r_mse - ref_mse:+.6f}  HGBR {h_mse - ref_mse:+.6f} (negative = better than v7)")
        print(
            "  Note: tabular = scenario rows + KFold; neural = GroupKFold on long rows "
            "(not identical CV), but mse_norm scale is in the same ballpark."
        )


if __name__ == "__main__":
    main()

"""Scenario-level Spearman |rho| between mass-weighted feature means and DOT targets.

Answers: which numeric component features correlate with each target; weak universal
features are candidates for blocklist (noise / degrees-of-freedom for Deep Sets).

Run: PYTHONPATH=src python scripts/analyze_feature_target_signal.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dot.config import SCENARIO_ID, TARGET_COLUMNS, TRAIN_PATH  # noqa: E402
from dot.data import _coerce_frame_types, _feature_columns  # noqa: E402


def _scenario_weighted_means(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    ap.add_argument("--out-csv", type=Path, default=ROOT / "reports" / "feature_target_spearman.csv")
    ap.add_argument(
        "--suggest-weak-max-abs",
        type=float,
        default=0.055,
        help="Features with max_t |rho| below this (both targets) listed as weak-noise candidates.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.train_path)
    work = _coerce_frame_types(df)
    feats = _feature_columns(work)
    table = _scenario_weighted_means(work, feats)

    rows: list[dict] = []
    for tcol in TARGET_COLUMNS:
        y = table[tcol].to_numpy(dtype=np.float64)
        for f in feats:
            x = table[f].to_numpy(dtype=np.float64)
            if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                r = np.nan
            else:
                r, _ = spearmanr(x, y, nan_policy="omit")
            rows.append(
                {
                    "target": tcol,
                    "feature": f,
                    "spearman": float(r) if np.isfinite(r) else np.nan,
                    "abs_spearman": float(abs(r)) if np.isfinite(r) else 0.0,
                }
            )

    rep = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    pivot = rep.pivot_table(index="feature", columns="target", values="abs_spearman", aggfunc="first")
    pivot["max_abs"] = pivot.max(axis=1)
    weak = pivot[pivot["max_abs"] < args.suggest_weak_max_abs].sort_values("max_abs")
    print(f"\nWeak on both targets (max |rho| < {args.suggest_weak_max_abs}): {len(weak)} features")
    for feat, row in weak.head(25).iterrows():
        print(f"  {feat[:70]:70s} max_abs={row['max_abs']:.4f}")
    if len(weak) > 25:
        print(f"  ... and {len(weak) - 25} more")

    print("\nStrongest per target (top 8):")
    for tcol in TARGET_COLUMNS:
        sub = rep[rep["target"] == tcol].nlargest(8, "abs_spearman")
        print(f"\n  {tcol[:60]}...")
        for _, r in sub.iterrows():
            print(f"    {r['feature'][:65]:65s}  rho={r['spearman']:+.3f}")


if __name__ == "__main__":
    main()

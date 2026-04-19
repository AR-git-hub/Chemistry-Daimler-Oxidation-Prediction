"""Compare v13 vs v7 CV and submission CSV deltas vs wave2 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dot.config import SCENARIO_ID, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    v7 = json.loads((ROOT / "artifacts/v7_max_s77_m12/metadata.json").read_text(encoding="utf-8"))
    v13 = json.loads((ROOT / "artifacts/v13_max_s123_m12_domloss/metadata.json").read_text(encoding="utf-8"))
    print("=== Single-model CV (root metadata.json) ===")
    for name, m in [("v7_max_s77", v7), ("v13_domloss", v13)]:
        print(
            f"  {name}:  cv_mean_mse_norm={m['cv_mean_mse_normalized']:.6f}  "
            f"cv_mean_mae_norm={m['cv_mean_mae_normalized']:.6f}"
        )
    dmse = float(v13["cv_mean_mse_normalized"]) - float(v7["cv_mean_mse_normalized"])
    dmae = float(v13["cv_mean_mae_normalized"]) - float(v7["cv_mean_mae_normalized"])
    print(f"  v13 - v7 (MSE): {dmse:+.6f}  (negative = better)")
    print(f"  v13 - v7 (MAE): {dmae:+.6f}  (negative = better)")

    base = pd.read_csv(ROOT / "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv")
    v13s = pd.read_csv(ROOT / "predictions_stacked_k5_v13_dom_dualalpha.csv")
    blend = pd.read_csv(ROOT / "predictions_blend_v13safe_92_wave2_8_dom.csv")

    print("\n=== Test CSV vs wave2 (mean/max abs delta per target) ===")
    for label, df in [
        ("k5_v13_stack", v13s),
        ("blend_92_08", blend),
    ]:
        m = base.merge(df, on=SCENARIO_ID, suffixes=("_b", "_n"))
        parts = []
        for i, t in enumerate(TARGET_COLUMNS):
            d = (m[f"{t}_n"] - m[f"{t}_b"]).abs()
            parts.append(f"t{i} mean|d|={d.mean():.5g} max={d.max():.5g}")
        print(f"  {label}:  " + " | ".join(parts))


if __name__ == "__main__":
    main()

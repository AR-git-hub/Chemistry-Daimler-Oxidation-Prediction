"""Equal-weight (or custom weights) average of several prediction CSVs on scenario_id."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dot.config import SCENARIO_ID, TARGET_COLUMNS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Mean-blend N prediction CSVs (same columns).")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="*", default=None, help="Optional; default equal.")
    p.add_argument("--output-path", type=Path, required=True)
    args = p.parse_args()

    dfs = [pd.read_csv(x) for x in args.inputs]
    need = [SCENARIO_ID, *TARGET_COLUMNS]
    for i, d in enumerate(dfs):
        miss = [c for c in need if c not in d.columns]
        if miss:
            raise SystemExit(f"Missing {miss} in {args.inputs[i]}")

    w = np.array(args.weights if args.weights is not None else [1.0] * len(dfs), dtype=np.float64)
    w = w / w.sum()

    base = dfs[0][need].copy()
    acc = np.zeros((len(base), len(TARGET_COLUMNS)), dtype=np.float64)
    for i, d in enumerate(dfs):
        m = base[[SCENARIO_ID]].merge(d[need], on=SCENARIO_ID, how="left", suffixes=("", f"__{i}"))
        for j, t in enumerate(TARGET_COLUMNS):
            acc[:, j] += w[i] * m[t].to_numpy(dtype=np.float64)
    out = pd.DataFrame({SCENARIO_ID: base[SCENARIO_ID]})
    for j, t in enumerate(TARGET_COLUMNS):
        out[t] = acc[:, j]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(f"Averaged {len(dfs)} files weights={w.tolist()} -> {args.output_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()

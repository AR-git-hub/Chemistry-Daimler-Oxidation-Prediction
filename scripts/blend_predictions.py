"""Blend several predictions.csv files by scenario_id (weighted mean of targets)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dot.config import SCENARIO_ID, TARGET_COLUMNS  # noqa: E402


def blend(paths: list[Path], weights: list[float] | None, output_path: Path) -> None:
    if len(paths) < 2:
        raise SystemExit("Need at least two --inputs CSV files.")
    dfs = [pd.read_csv(p) for p in paths]
    need = [SCENARIO_ID, *TARGET_COLUMNS]
    for i, d in enumerate(dfs):
        miss = [c for c in need if c not in d.columns]
        if miss:
            raise SystemExit(f"Missing {miss} in {paths[i]}")

    merged = dfs[0][need].rename(columns={t: f"{t}__0" for t in TARGET_COLUMNS})
    for i in range(1, len(dfs)):
        part = dfs[i][need].rename(columns={t: f"{t}__{i}" for t in TARGET_COLUMNS})
        merged = merged.merge(part, on=SCENARIO_ID, how="inner")

    n0 = len(dfs[0])
    if len(merged) != n0:
        print(f"Warning: merged {len(merged)} rows vs first file {n0} (ids mismatch?).")

    wts = list(weights) if weights is not None else [1.0 / len(dfs)] * len(dfs)
    if len(wts) != len(dfs):
        raise SystemExit("--weights length must match --inputs")
    sw = sum(wts)
    wts = [w / sw for w in wts]

    out = pd.DataFrame({SCENARIO_ID: merged[SCENARIO_ID]})
    for tcol in TARGET_COLUMNS:
        out[tcol] = sum(wts[i] * merged[f"{tcol}__{i}"].to_numpy(dtype=float) for i in range(len(dfs)))
    out.to_csv(output_path, index=False)
    print(f"Blended {len(paths)} files, weights={wts} -> {output_path} ({len(out)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description="Weighted average of DOT prediction CSVs on scenario_id.")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="*", default=None)
    p.add_argument("--output-path", type=Path, default=Path("predictions.csv"))
    args = p.parse_args()
    blend(args.inputs, args.weights, args.output_path)


if __name__ == "__main__":
    main()

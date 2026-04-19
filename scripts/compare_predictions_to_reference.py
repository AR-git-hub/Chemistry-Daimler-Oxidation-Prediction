"""Compare two prediction CSVs (same scenario_id): mean/median/p90/max |delta| and RMS delta per target."""

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
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True, help="e.g. predictions_submit_blend_97wave2_03sumpoolstack.csv")
    p.add_argument("--candidate", type=Path, required=True)
    args = p.parse_args()

    b = pd.read_csv(args.reference)
    c = pd.read_csv(args.candidate)
    m = b.merge(c, on=SCENARIO_ID, suffixes=("_ref", "_cand"))
    if len(m) != len(b):
        print(f"Warning: merged {len(m)} rows vs reference {len(b)}", flush=True)

    print(f"reference: {args.reference.name}\ncandidate: {args.candidate.name}\n")
    for i, t in enumerate(TARGET_COLUMNS):
        d = (m[f"{t}_cand"] - m[f"{t}_ref"]).to_numpy(dtype=np.float64)
        ad = np.abs(d)
        print(
            f"Target {i}: mean|d|={ad.mean():.6g} median|d|={np.median(ad):.6g} "
            f"p90|d|={np.percentile(ad, 90):.6g} max|d|={ad.max():.6g} rms_d={np.sqrt(np.mean(d**2)):.6g}"
        )


if __name__ == "__main__":
    main()

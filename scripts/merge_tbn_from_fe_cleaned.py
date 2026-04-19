"""Append ``tbn_consolidated`` from fe_cleaned onto train_full / test_full (row-aligned 1:1).

Run once after updating merged CSVs:
  PYTHONPATH=src python scripts/merge_tbn_from_fe_cleaned.py

Writes ``train_full_plus_tbn.csv`` / ``test_full_plus_tbn.csv`` (see dot.config).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dot.config import (  # noqa: E402
    ID_COLUMNS,
    TEST_FE_CLEANED_PATH,
    TEST_PATH,
    TEST_FULL_TBN_PATH,
    TRAIN_FE_CLEANED_PATH,
    TRAIN_PATH,
    TRAIN_FULL_TBN_PATH,
)

COL = "tbn_consolidated"


def _assert_row_aligned(a: pd.DataFrame, b: pd.DataFrame, name: str) -> None:
    if len(a) != len(b):
        raise SystemExit(f"{name}: row count mismatch {len(a)} vs {len(b)}")
    for k in ID_COLUMNS:
        if not (a[k].astype(str).values == b[k].astype(str).values).all():
            raise SystemExit(f"{name}: ID column {k!r} mismatch vs fe_cleaned")


def main() -> None:
    for base_path, fe_path, out_path in (
        (TRAIN_PATH, TRAIN_FE_CLEANED_PATH, TRAIN_FULL_TBN_PATH),
        (TEST_PATH, TEST_FE_CLEANED_PATH, TEST_FULL_TBN_PATH),
    ):
        base = pd.read_csv(base_path).reset_index(drop=True)
        fe = pd.read_csv(fe_path).reset_index(drop=True)
        _assert_row_aligned(base, fe, str(base_path.name))
        if COL not in fe.columns:
            raise SystemExit(f"{fe_path} missing {COL!r}")
        if COL in base.columns:
            out = base.copy()
            out[COL] = fe[COL].values
        else:
            out = base.copy()
            out[COL] = fe[COL].values
        out.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(out)} rows, +{COL})")


if __name__ == "__main__":
    main()

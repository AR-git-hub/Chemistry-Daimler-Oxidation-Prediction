"""Submission format checks for predictions.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import SCENARIO_ID, TARGET_COLUMNS


def validate_predictions(predictions_path: Path, test_path: Path) -> None:
    pred = pd.read_csv(predictions_path)
    test = pd.read_csv(test_path)

    expected_columns = [SCENARIO_ID] + TARGET_COLUMNS
    if list(pred.columns) != expected_columns:
        raise ValueError(
            f"Invalid columns: {list(pred.columns)}. Expected exactly: {expected_columns}"
        )

    expected_ids = sorted(test[SCENARIO_ID].unique().tolist())
    got_ids = pred[SCENARIO_ID].tolist()

    if pred[SCENARIO_ID].duplicated().any():
        duplicates = pred[pred[SCENARIO_ID].duplicated()][SCENARIO_ID].unique().tolist()
        raise ValueError(f"Duplicate scenario_id values in predictions: {duplicates[:10]}")

    got_ids_sorted = sorted(got_ids)
    if got_ids_sorted != expected_ids:
        missing = sorted(set(expected_ids) - set(got_ids_sorted))
        extra = sorted(set(got_ids_sorted) - set(expected_ids))
        raise ValueError(
            f"scenario_id mismatch. Missing: {missing[:10]} Extra: {extra[:10]}"
        )

    if pred[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Target columns contain NaN values.")

    print("predictions.csv validation passed.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate predictions.csv format.")
    parser.add_argument("--predictions-path", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--test-path", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    validate_predictions(args.predictions_path, args.test_path)


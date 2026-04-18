"""Inference pipeline that generates predictions.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ARTIFACTS_DIR, SCENARIO_ID, TARGET_COLUMNS, TEST_PATH
from .data import apply_context_scaler, apply_feature_scaler, frame_to_scenario_sets
from .model import DeepSetsRegressor, ScenarioSetDataset, collate_infer


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.metadata_path)
    feature_columns = metadata["feature_columns"]
    context_columns = metadata["context_columns"]
    target_columns = metadata["target_columns"]

    if target_columns != TARGET_COLUMNS:
        raise ValueError("Artifact target column order does not match task specification.")

    test_df = pd.read_csv(args.test_path)
    scenario_test = frame_to_scenario_sets(test_df, is_train=False)

    if scenario_test.feature_columns != feature_columns:
        missing = set(feature_columns) - set(scenario_test.feature_columns)
        extra = set(scenario_test.feature_columns) - set(feature_columns)
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )
    if scenario_test.context_columns != context_columns:
        missing = set(context_columns) - set(scenario_test.context_columns)
        extra = set(scenario_test.context_columns) - set(context_columns)
        raise ValueError(
            f"Context feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )

    scaler = {
        "mean": np.array(metadata["x_mean"], dtype=np.float32),
        "std": np.array(metadata["x_std"], dtype=np.float32),
    }
    context_scaler = {
        "mean": np.array(metadata["ctx_mean"], dtype=np.float32),
        "std": np.array(metadata["ctx_std"], dtype=np.float32),
    }
    scaled = apply_feature_scaler(scenario_test.components, scaler)
    scaled_context = apply_context_scaler(scenario_test.context, context_scaler)

    model = DeepSetsRegressor(
        input_dim=int(metadata["input_dim"]),
        context_dim=int(metadata["context_dim"]),
        hidden_dim=128,
        output_dim=int(metadata["output_dim"]),
    )
    state_dict = torch.load(args.model_path, map_location=torch.device(args.device))
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    ds = ScenarioSetDataset(scaled, scaled_context, targets=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_infer)

    preds = []
    with torch.no_grad():
        for xb, mask, ctx in loader:
            xb = xb.to(args.device)
            mask = mask.to(args.device)
            ctx = ctx.to(args.device)
            pred_norm = model(xb, mask, ctx).cpu().numpy()
            preds.append(pred_norm)
    pred_norm = np.vstack(preds)

    y_mean = np.array(metadata["y_mean"], dtype=np.float32)
    y_std = np.array(metadata["y_std"], dtype=np.float32)
    pred = pred_norm * y_std + y_mean

    pred_df = pd.DataFrame(
        {
            SCENARIO_ID: scenario_test.scenario_ids,
            TARGET_COLUMNS[0]: pred[:, 0],
            TARGET_COLUMNS[1]: pred[:, 1],
        }
    )
    pred_df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DOT predictions.csv from trained artifacts.")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "metadata.json")
    parser.add_argument("--model-path", type=Path, default=ARTIFACTS_DIR / "deepsets_model.pt")
    parser.add_argument("--output-path", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    infer(build_arg_parser().parse_args())


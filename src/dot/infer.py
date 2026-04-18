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


def _predict_with_artifact(
    test_df: pd.DataFrame,
    artifact_meta: dict,
    model_path: Path,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    feature_columns = artifact_meta["feature_columns"]
    context_columns = artifact_meta["context_columns"]
    interaction_feature_columns = artifact_meta.get("interaction_feature_columns", [])
    target_columns = artifact_meta["target_columns"] if "target_columns" in artifact_meta else TARGET_COLUMNS
    if target_columns != TARGET_COLUMNS:
        raise ValueError("Artifact target column order does not match task specification.")

    scenario_test = frame_to_scenario_sets(
        test_df,
        is_train=False,
        feature_columns=feature_columns,
        interaction_feature_columns=interaction_feature_columns,
    )
    if scenario_test.feature_columns != feature_columns:
        raise ValueError("Feature columns mismatch for inference artifact.")
    if scenario_test.context_columns != context_columns:
        raise ValueError("Context columns mismatch for inference artifact.")

    scaler = {
        "mean": np.array(artifact_meta["x_mean"], dtype=np.float32),
        "std": np.array(artifact_meta["x_std"], dtype=np.float32),
    }
    context_scaler = {
        "mean": np.array(artifact_meta["ctx_mean"], dtype=np.float32),
        "std": np.array(artifact_meta["ctx_std"], dtype=np.float32),
    }
    scaled = apply_feature_scaler(scenario_test.components, scaler)
    scaled_context = apply_context_scaler(scenario_test.context, context_scaler)

    model = DeepSetsRegressor(
        input_dim=int(artifact_meta["input_dim"]),
        context_dim=int(artifact_meta["context_dim"]),
        hidden_dim=int(artifact_meta.get("hidden_dim", 128)),
        output_dim=int(artifact_meta["output_dim"]),
    )
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ds = ScenarioSetDataset(scaled, scaled_context, targets=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_infer)

    preds = []
    with torch.no_grad():
        for xb, mask, ctx in loader:
            xb = xb.to(device)
            mask = mask.to(device)
            ctx = ctx.to(device)
            pred_norm = model(xb, mask, ctx).cpu().numpy()
            preds.append(pred_norm)
    pred_norm = np.vstack(preds)

    y_mean = np.array(artifact_meta["y_mean"], dtype=np.float32)
    y_std = np.array(artifact_meta["y_std"], dtype=np.float32)
    pred = pred_norm * y_std + y_mean
    return scenario_test.scenario_ids, pred


def _infer_fold_ensemble(args: argparse.Namespace, test_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    fold_dirs = sorted([p for p in args.folds_dir.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {args.folds_dir}")

    all_preds = []
    base_ids = None
    for fold_dir in fold_dirs:
        fold_meta_path = fold_dir / "metadata.json"
        fold_model_path = fold_dir / "model.pt"
        if not fold_meta_path.exists() or not fold_model_path.exists():
            raise ValueError(f"Missing fold artifacts in {fold_dir}")
        fold_meta = load_metadata(fold_meta_path)
        ids, pred = _predict_with_artifact(
            test_df=test_df,
            artifact_meta=fold_meta,
            model_path=fold_model_path,
            batch_size=args.batch_size,
            device=args.device,
        )
        if base_ids is None:
            base_ids = ids
        elif ids != base_ids:
            raise ValueError("Fold ensemble scenario ordering mismatch.")
        all_preds.append(pred)
    pred_mean = np.mean(np.stack(all_preds, axis=0), axis=0)
    return base_ids, pred_mean


def infer(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.metadata_path)
    test_df = pd.read_csv(args.test_path)
    use_ensemble = args.use_fold_ensemble and args.folds_dir.exists()
    if use_ensemble:
        scenario_ids, pred = _infer_fold_ensemble(args, test_df)
        print(f"Using fold ensemble inference from: {args.folds_dir}")
    else:
        scenario_ids, pred = _predict_with_artifact(
            test_df=test_df,
            artifact_meta=metadata,
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
        )
        print("Using single final model inference.")

    pred_df = pd.DataFrame(
        {
            SCENARIO_ID: scenario_ids,
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
    parser.add_argument("--folds-dir", type=Path, default=ARTIFACTS_DIR / "folds")
    parser.add_argument("--use-fold-ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-path", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    infer(build_arg_parser().parse_args())


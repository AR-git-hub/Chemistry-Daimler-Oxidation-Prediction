"""Training pipeline for DOT Deep Sets baseline."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .config import ARTIFACTS_DIR, SCENARIO_ID, TARGET_COLUMNS, TRAIN_PATH
from .data import (
    apply_context_scaler,
    apply_feature_scaler,
    build_context_scaler_stats,
    build_feature_scaler_stats,
    frame_to_scenario_sets,
    normalize_targets,
)
from .model import DeepSetsRegressor, ScenarioSetDataset, collate_train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_loader(dataset: ScenarioSetDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_train,
    )


def _evaluate(model: DeepSetsRegressor, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    mae_losses = []
    criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for xb, mask, ctx, yb in loader:
            xb = xb.to(device)
            mask = mask.to(device)
            ctx = ctx.to(device)
            yb = yb.to(device)
            pred = model(xb, mask, ctx)
            loss = criterion(pred, yb)
            mae = mae_criterion(pred, yb)
            losses.append(loss.item())
            mae_losses.append(mae.item())
    mse = float(np.mean(losses)) if losses else float("nan")
    mae = float(np.mean(mae_losses)) if mae_losses else float("nan")
    return {"mse": mse, "mae": mae}


def _train_fold(
    train_dataset: ScenarioSetDataset,
    val_dataset: ScenarioSetDataset,
    input_dim: int,
    context_dim: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    input_noise_std: float,
    device: torch.device,
) -> Tuple[DeepSetsRegressor, Dict[str, float]]:
    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DeepSetsRegressor(
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=128,
        output_dim=output_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience = 20
    stale = 0

    for _ in range(epochs):
        model.train()
        for xb, mask, ctx, yb in train_loader:
            xb = xb.to(device)
            mask = mask.to(device)
            ctx = ctx.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            if input_noise_std > 0:
                xb = xb + torch.randn_like(xb) * input_noise_std
                ctx = ctx + torch.randn_like(ctx) * input_noise_std
            pred = model(xb, mask, ctx)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        val_metrics = _evaluate(model, val_loader, device)
        val_loss = val_metrics["mse"]
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = _evaluate(model, val_loader, device)
    metrics: Dict[str, float] = {
        "val_mse_normalized": final_metrics["mse"],
        "val_mae_normalized": final_metrics["mae"],
    }
    return model, metrics


def _assert_group_split_no_leakage(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = train_groups.intersection(val_groups)
    if overlap:
        raise ValueError(f"Group leakage detected for scenario_id values: {sorted(list(overlap))[:10]}")


def _build_fold_datasets(
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
) -> tuple[ScenarioSetDataset, ScenarioSetDataset, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    fold_train = frame_to_scenario_sets(train_fold_df, is_train=True)
    fold_val = frame_to_scenario_sets(
        val_fold_df,
        is_train=True,
        feature_columns=fold_train.feature_columns,
        interaction_feature_columns=fold_train.interaction_feature_columns,
    )
    if fold_train.targets is None or fold_val.targets is None:
        raise ValueError("Fold targets are missing.")

    x_stats = build_feature_scaler_stats(fold_train.components)
    c_stats = build_context_scaler_stats(fold_train.context)
    y_stats = normalize_targets(fold_train.targets)

    train_x = apply_feature_scaler(fold_train.components, x_stats)
    val_x = apply_feature_scaler(fold_val.components, x_stats)
    train_c = apply_context_scaler(fold_train.context, c_stats)
    val_c = apply_context_scaler(fold_val.context, c_stats)
    train_y = y_stats["y_norm"]
    val_y = ((fold_val.targets - y_stats["mean"]) / y_stats["std"]).astype(np.float32)

    train_dataset = ScenarioSetDataset(train_x, train_c, train_y)
    val_dataset = ScenarioSetDataset(val_x, val_c, val_y)
    meta = {
        "feature_columns": fold_train.feature_columns,
        "context_columns": fold_train.context_columns,
        "interaction_feature_columns": fold_train.interaction_feature_columns,
    }
    return train_dataset, val_dataset, x_stats, c_stats, y_stats, meta


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    folds_dir = ARTIFACTS_DIR / "folds"
    if folds_dir.exists():
        shutil.rmtree(folds_dir)
    folds_dir.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(args.train_path)
    scenario_ids = np.array(sorted(raw_train[SCENARIO_ID].unique().tolist()))
    gkf = GroupKFold(n_splits=args.n_splits)

    fold_metrics = []
    fold_splits = list(gkf.split(np.arange(len(scenario_ids)), scenario_ids, groups=scenario_ids))
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits, start=1):
        _assert_group_split_no_leakage(scenario_ids, tr_idx, val_idx)
        train_ids = set(scenario_ids[tr_idx].tolist())
        val_ids = set(scenario_ids[val_idx].tolist())
        train_fold_df = raw_train[raw_train[SCENARIO_ID].isin(train_ids)].copy()
        val_fold_df = raw_train[raw_train[SCENARIO_ID].isin(val_ids)].copy()

        train_dataset, val_dataset, x_stats, c_stats, y_stats, fold_meta = _build_fold_datasets(train_fold_df, val_fold_df)
        fold_model, fold_metric = _train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_dim=len(fold_meta["feature_columns"]),
            context_dim=len(fold_meta["context_columns"]),
            output_dim=len(TARGET_COLUMNS),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            input_noise_std=args.input_noise_std,
            device=torch.device(args.device),
        )
        fold_metric["fold"] = fold_idx
        fold_metrics.append(fold_metric)

        fold_dir = folds_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_model_path = fold_dir / "model.pt"
        torch.save(fold_model.state_dict(), fold_model_path)

        fold_metadata = {
            "fold": fold_idx,
            "input_dim": len(fold_meta["feature_columns"]),
            "context_dim": len(fold_meta["context_columns"]),
            "output_dim": len(TARGET_COLUMNS),
            "hidden_dim": 128,
            "feature_columns": fold_meta["feature_columns"],
            "context_columns": fold_meta["context_columns"],
            "interaction_feature_columns": fold_meta["interaction_feature_columns"],
            "x_mean": x_stats["mean"].tolist(),
            "x_std": x_stats["std"].tolist(),
            "ctx_mean": c_stats["mean"].tolist(),
            "ctx_std": c_stats["std"].tolist(),
            "y_mean": y_stats["mean"].tolist(),
            "y_std": y_stats["std"].tolist(),
            "model_path": str(fold_model_path),
        }
        with (fold_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(fold_metadata, f, ensure_ascii=False, indent=2)

    # Re-train on full dataset for final inference artifact (no CV leakage concerns).
    grouped_train = frame_to_scenario_sets(raw_train, is_train=True)
    if grouped_train.targets is None:
        raise ValueError("Training targets are missing.")
    scaler_stats = build_feature_scaler_stats(grouped_train.components)
    context_scaler = build_context_scaler_stats(grouped_train.context)
    y_stats = normalize_targets(grouped_train.targets)
    scaled_components = apply_feature_scaler(grouped_train.components, scaler_stats)
    scaled_context = apply_context_scaler(grouped_train.context, context_scaler)
    y_norm = y_stats["y_norm"]
    dataset = ScenarioSetDataset(scaled_components, scaled_context, y_norm)

    full_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_train,
    )
    final_model = DeepSetsRegressor(
        input_dim=len(grouped_train.feature_columns),
        context_dim=len(grouped_train.context_columns),
        hidden_dim=128,
        output_dim=len(TARGET_COLUMNS),
    ).to(torch.device(args.device))
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    final_model.train()
    for _ in range(args.final_epochs):
        for xb, mask, ctx, yb in full_loader:
            xb = xb.to(args.device)
            mask = mask.to(args.device)
            ctx = ctx.to(args.device)
            yb = yb.to(args.device)
            final_optimizer.zero_grad()
            if args.input_noise_std > 0:
                xb = xb + torch.randn_like(xb) * args.input_noise_std
                ctx = ctx + torch.randn_like(ctx) * args.input_noise_std
            pred = final_model(xb, mask, ctx)
            loss = criterion(pred, yb)
            loss.backward()
            final_optimizer.step()

    model_path = ARTIFACTS_DIR / "deepsets_model.pt"
    torch.save(final_model.state_dict(), model_path)

    metadata = {
        "feature_columns": grouped_train.feature_columns,
        "context_columns": grouped_train.context_columns,
        "interaction_feature_columns": grouped_train.interaction_feature_columns,
        "target_columns": grouped_train.target_columns,
        "x_mean": scaler_stats["mean"].tolist(),
        "x_std": scaler_stats["std"].tolist(),
        "ctx_mean": context_scaler["mean"].tolist(),
        "ctx_std": context_scaler["std"].tolist(),
        "y_mean": y_stats["mean"].tolist(),
        "y_std": y_stats["std"].tolist(),
        "input_dim": len(grouped_train.feature_columns),
        "context_dim": len(grouped_train.context_columns),
        "output_dim": len(grouped_train.target_columns),
        "validation_metrics": fold_metrics,
        "cv_mean_mse_normalized": float(np.mean([m["val_mse_normalized"] for m in fold_metrics])),
        "cv_mean_mae_normalized": float(np.mean([m["val_mae_normalized"] for m in fold_metrics])),
        "input_noise_std": float(args.input_noise_std),
        "fold_ensemble_dir": str(folds_dir),
        "fold_count": len(fold_metrics),
        "model_path": str(model_path),
    }
    with (ARTIFACTS_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Artifacts written to: {ARTIFACTS_DIR}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Deep Sets baseline for DOT task.")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--final-epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())


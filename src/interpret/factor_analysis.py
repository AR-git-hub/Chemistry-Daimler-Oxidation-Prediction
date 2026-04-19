"""Permutation-based factor analysis for trained DOT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

from dot.config import ARTIFACTS_DIR, SCENARIO_ID, TARGET_COLUMNS, TRAIN_PATH
from dot.data import (
    apply_context_scaler,
    apply_feature_scaler,
    build_context_scaler_stats,
    build_feature_scaler_stats,
    frame_to_scenario_sets,
    normalize_targets,
)
from dot.model import DeepSetsRegressor, ScenarioSetDataset, collate_train


def _eval_mse(
    models: DeepSetsRegressor | list[DeepSetsRegressor],
    loader: DataLoader,
    device: str,
    *,
    only_target_index: int | None = None,
) -> float:
    if isinstance(models, DeepSetsRegressor):
        models = [models]
    for m in models:
        m.eval()
    losses = []
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for xb, mask, ctx, yb in loader:
            xb = xb.to(device)
            mask = mask.to(device)
            ctx = ctx.to(device)
            yb = yb.to(device)
            if len(models) == 1:
                pred = models[0](xb, mask, ctx)
                if only_target_index is not None:
                    yt = yb[:, only_target_index : only_target_index + 1]
                else:
                    yt = yb
                loss = criterion(pred, yt)
                losses.append(loss.item())
            else:
                acc = 0.0
                for ti, m in enumerate(models):
                    pred = m(xb, mask, ctx)
                    acc += criterion(pred, yb[:, ti : ti + 1]).item()
                losses.append(acc / len(models))
    return float(np.mean(losses))


def run_factor_analysis(args: argparse.Namespace) -> Path:
    with (args.metadata_path).open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_df = pd.read_csv(args.train_path)
    per_target_feats = "feature_columns_t0" in metadata
    if per_target_feats:
        grouped = frame_to_scenario_sets(
            train_df,
            is_train=True,
            feature_columns=metadata["feature_columns_t0"],
            interaction_feature_columns=metadata["interaction_feature_columns_t0"],
        )
    else:
        grouped = frame_to_scenario_sets(train_df, is_train=True)
    if grouped.targets is None:
        raise ValueError("Targets are required for factor analysis.")

    x_stats = build_feature_scaler_stats(grouped.components)
    c_stats = build_context_scaler_stats(grouped.context)
    x_scaled = apply_feature_scaler(grouped.components, x_stats)
    c_scaled = apply_context_scaler(grouped.context, c_stats)
    y_norm = normalize_targets(grouped.targets)["y_norm"]

    dataset = ScenarioSetDataset(x_scaled, c_scaled, y_norm)
    scenario_level = train_df[[SCENARIO_ID]].drop_duplicates().sort_values(SCENARIO_ID)
    groups = scenario_level[SCENARIO_ID].to_numpy()

    splitter = GroupKFold(n_splits=args.n_splits)
    _, val_idx = next(splitter.split(np.arange(len(groups)), y_norm[:, 0], groups=groups))
    val_subset = Subset(dataset, val_idx.tolist())
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, collate_fn=collate_train)

    hidden_dim = int(metadata.get("hidden_dim", 128))
    dropout = float(metadata.get("dropout", 0.0))
    hetero = bool(metadata.get("use_heterogeneity", False))

    if metadata.get("separate_target_models"):
        if per_target_feats:
            mp0 = metadata.get("model_path_t0")
            if not mp0:
                raise ValueError("metadata missing model_path_t0 for factor analysis.")
            m = DeepSetsRegressor(
                input_dim=int(metadata["input_dim_t0"]),
                context_dim=int(metadata["context_dim_t0"]),
                hidden_dim=hidden_dim,
                output_dim=1,
                dropout=dropout,
                use_heterogeneity=hetero,
            )
            m.load_state_dict(torch.load(mp0, map_location=torch.device(args.device)))
            m.to(args.device)
            model_or_list = m
        else:
            models: list[DeepSetsRegressor] = []
            for ti in range(len(TARGET_COLUMNS)):
                mp = metadata.get(f"model_path_t{ti}")
                if not mp:
                    raise ValueError(f"metadata missing model_path_t{ti} for factor analysis.")
                m = DeepSetsRegressor(
                    input_dim=int(metadata["input_dim"]),
                    context_dim=int(metadata["context_dim"]),
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    dropout=dropout,
                    use_heterogeneity=hetero,
                )
                m.load_state_dict(torch.load(mp, map_location=torch.device(args.device)))
                m.to(args.device)
                models.append(m)
            model_or_list = models
    else:
        m = DeepSetsRegressor(
            input_dim=int(metadata["input_dim"]),
            context_dim=int(metadata["context_dim"]),
            hidden_dim=hidden_dim,
            output_dim=int(metadata["output_dim"]),
            dropout=dropout,
            use_heterogeneity=hetero,
        )
        m.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
        m.to(args.device)
        model_or_list = m

    baseline_mse = _eval_mse(
        model_or_list,
        val_loader,
        args.device,
        only_target_index=0 if per_target_feats else None,
    )

    val_context = c_scaled[val_idx].copy()
    rng = np.random.default_rng(args.seed)
    impacts = []
    for feat_idx, feat_name in enumerate(grouped.context_columns):
        perm_context = val_context.copy()
        perm_context[:, feat_idx] = rng.permutation(perm_context[:, feat_idx])
        perm_dataset = ScenarioSetDataset(
            [x_scaled[i] for i in val_idx],
            perm_context,
            y_norm[val_idx],
        )
        perm_loader = DataLoader(perm_dataset, batch_size=64, shuffle=False, collate_fn=collate_train)
        perm_mse = _eval_mse(
            model_or_list,
            perm_loader,
            args.device,
            only_target_index=0 if per_target_feats else None,
        )
        impacts.append(
            {
                "feature": feat_name,
                "baseline_mse": baseline_mse,
                "permuted_mse": perm_mse,
                "delta_mse": perm_mse - baseline_mse,
            }
        )

    impacts = sorted(impacts, key=lambda x: x["delta_mse"], reverse=True)
    out_path = args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"baseline_mse": baseline_mse, "top_context_factors": impacts[: args.top_k]}, f, ensure_ascii=False, indent=2)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run permutation-based factor analysis.")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--metadata-path", type=Path, default=ARTIFACTS_DIR / "metadata.json")
    parser.add_argument("--model-path", type=Path, default=ARTIFACTS_DIR / "deepsets_model.pt")
    parser.add_argument("--output-path", type=Path, default=ARTIFACTS_DIR / "factor_analysis.json")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    output = run_factor_analysis(build_arg_parser().parse_args())
    print(f"Saved factor analysis: {output}")


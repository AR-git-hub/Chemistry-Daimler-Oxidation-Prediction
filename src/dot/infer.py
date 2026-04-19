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
from .model import ScenarioSetDataset, collate_infer
from .set_sequence_models import build_set_regressor


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _optional_positive_int(meta: dict, key: str) -> int | None:
    v = meta.get(key)
    if v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if i > 0 else None


def build_regressor_from_meta(
    artifact_meta: dict,
    *,
    input_dim: int,
    context_dim: int,
    output_dim: int,
) -> torch.nn.Module:
    """Construct set regressor (Deep Sets or attention/Transformer variants) from artifact metadata."""
    arch = str(artifact_meta.get("architecture", "deepsets"))
    return build_set_regressor(
        arch,
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_dim=int(artifact_meta.get("hidden_dim", 128)),
        output_dim=output_dim,
        dropout=float(artifact_meta.get("dropout", 0.0)),
        use_heterogeneity=bool(artifact_meta.get("use_heterogeneity", False)),
        encoder_hidden_dim=_optional_positive_int(artifact_meta, "encoder_hidden_dim"),
        rho_hidden_dim=_optional_positive_int(artifact_meta, "rho_hidden_dim"),
        transformer_layers=int(artifact_meta.get("transformer_layers", 2)),
        transformer_heads=int(artifact_meta.get("transformer_heads", 4)),
        transformer_ffn_mult=int(artifact_meta.get("transformer_ffn_mult", 2)),
    )


def _forward_batches_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    tta_runs: int,
    tta_noise_std: float,
) -> np.ndarray:
    """Monte-Carlo dropout-style noise on inputs at inference (deterministic if runs=1 or noise=0)."""
    if tta_runs < 1:
        tta_runs = 1
    blocks: list[np.ndarray] = []
    for _ in range(tta_runs):
        chunk: list[np.ndarray] = []
        with torch.no_grad():
            for xb, mask, ctx in loader:
                xb = xb.to(device)
                mask = mask.to(device)
                ctx = ctx.to(device)
                if tta_noise_std and tta_noise_std > 0:
                    xb = xb + torch.randn_like(xb) * tta_noise_std
                    ctx = ctx + torch.randn_like(ctx) * tta_noise_std
                pred_norm = model(xb, mask, ctx).cpu().numpy()
                chunk.append(pred_norm)
        blocks.append(np.vstack(chunk))
    return np.mean(np.stack(blocks, axis=0), axis=0)


def _predict_with_artifact(
    test_df: pd.DataFrame,
    artifact_meta: dict,
    model_path: Path | None,
    batch_size: int,
    device: str,
    *,
    tta_runs: int = 1,
    tta_noise_std: float = 0.0,
) -> tuple[list[str], np.ndarray]:
    target_columns = artifact_meta["target_columns"] if "target_columns" in artifact_meta else TARGET_COLUMNS
    if target_columns != TARGET_COLUMNS:
        raise ValueError("Artifact target column order does not match task specification.")

    separate = bool(artifact_meta.get("separate_target_models", False))
    per_target_features = separate and "feature_columns_t0" in artifact_meta

    if separate:
        pred_blocks = []
        scenario_ids_result: list[str] | None = None
        for ti in range(len(TARGET_COLUMNS)):
            mp = artifact_meta.get(f"model_path_t{ti}")
            if not mp:
                raise ValueError(f"Missing model_path_t{ti} for per-target model inference.")
            if per_target_features:
                feature_columns = artifact_meta[f"feature_columns_t{ti}"]
                interaction_feature_columns = artifact_meta[f"interaction_feature_columns_t{ti}"]
                context_columns = artifact_meta[f"context_columns_t{ti}"]
                in_dim = int(artifact_meta[f"input_dim_t{ti}"])
                ctx_dim = int(artifact_meta[f"context_dim_t{ti}"])
                scaler = {
                    "mean": np.array(artifact_meta[f"x_mean_t{ti}"], dtype=np.float32),
                    "std": np.array(artifact_meta[f"x_std_t{ti}"], dtype=np.float32),
                }
                context_scaler = {
                    "mean": np.array(artifact_meta[f"ctx_mean_t{ti}"], dtype=np.float32),
                    "std": np.array(artifact_meta[f"ctx_std_t{ti}"], dtype=np.float32),
                }
            else:
                feature_columns = artifact_meta["feature_columns"]
                interaction_feature_columns = artifact_meta.get("interaction_feature_columns", [])
                context_columns = artifact_meta["context_columns"]
                in_dim = int(artifact_meta["input_dim"])
                ctx_dim = int(artifact_meta["context_dim"])
                scaler = {
                    "mean": np.array(artifact_meta["x_mean"], dtype=np.float32),
                    "std": np.array(artifact_meta["x_std"], dtype=np.float32),
                }
                context_scaler = {
                    "mean": np.array(artifact_meta["ctx_mean"], dtype=np.float32),
                    "std": np.array(artifact_meta["ctx_std"], dtype=np.float32),
                }

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

            scaled = apply_feature_scaler(scenario_test.components, scaler)
            scaled_context = apply_context_scaler(scenario_test.context, context_scaler)
            ds = ScenarioSetDataset(scaled, scaled_context, targets=None)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_infer)

            if scenario_ids_result is None:
                scenario_ids_result = scenario_test.scenario_ids
            elif scenario_ids_result != scenario_test.scenario_ids:
                raise ValueError("Scenario ordering differs across per-target feature sets.")

            model = build_regressor_from_meta(
                artifact_meta,
                input_dim=in_dim,
                context_dim=ctx_dim,
                output_dim=1,
            )
            state_dict = torch.load(mp, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            pred_col = _forward_batches_tta(
                model, loader, device, tta_runs=tta_runs, tta_noise_std=tta_noise_std
            )
            pred_blocks.append(pred_col)
        pred_norm = np.hstack(pred_blocks)
        scenario_ids = scenario_ids_result if scenario_ids_result is not None else []
    elif not separate:
        feature_columns = artifact_meta["feature_columns"]
        context_columns = artifact_meta["context_columns"]
        interaction_feature_columns = artifact_meta.get("interaction_feature_columns", [])
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
        ds = ScenarioSetDataset(scaled, scaled_context, targets=None)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_infer)

        if model_path is None:
            raise ValueError("model_path is required for legacy joint-output checkpoints.")
        model = build_regressor_from_meta(
            artifact_meta,
            input_dim=int(artifact_meta["input_dim"]),
            context_dim=int(artifact_meta["context_dim"]),
            output_dim=int(artifact_meta["output_dim"]),
        )
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        pred_norm = _forward_batches_tta(
            model, loader, device, tta_runs=tta_runs, tta_noise_std=tta_noise_std
        )
        scenario_ids = scenario_test.scenario_ids

    y_mean = np.array(artifact_meta["y_mean"], dtype=np.float32)
    y_std = np.array(artifact_meta["y_std"], dtype=np.float32)
    pred = pred_norm * y_std + y_mean
    return scenario_ids, pred


def _parse_fold_idx(fold_dir: Path) -> int:
    try:
        return int(fold_dir.name.split("_", 1)[1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"Unexpected fold directory name: {fold_dir.name}") from e


def _fold_val_metrics(metadata_path: Path) -> tuple[dict[int, float], dict[int, float]]:
    root = load_metadata(metadata_path)
    vm: dict[int, float] = {}
    va: dict[int, float] = {}
    for m in root.get("validation_metrics", []):
        fn = int(m["fold"])
        vm[fn] = float(m["val_mse_normalized"])
        va[fn] = float(m["val_mae_normalized"])
    return vm, va


def _fold_ensemble_weights(
    metadata_path: Path,
    fold_dirs: list[Path],
    mode: str,
    *,
    mae_coef: float,
) -> list[float]:
    vm, va = _fold_val_metrics(metadata_path)
    weights: list[float] = []
    for fold_dir in fold_dirs:
        fn = _parse_fold_idx(fold_dir)
        mse = vm.get(fn)
        mae = va.get(fn)
        if mse is None or mse <= 0:
            weights.append(1.0)
            continue
        if mode == "inverse_mse":
            weights.append(1.0 / (mse + 1e-8))
        elif mode == "inverse_sqrt_mse":
            weights.append(1.0 / (float(np.sqrt(mse)) + 1e-8))
        elif mode == "inverse_mse_mae":
            mae_t = mae if mae is not None and mae > 0 else 0.0
            weights.append(1.0 / (mse + float(mae_coef) * mae_t + 1e-8))
        else:
            raise ValueError(f"Unknown fold ensemble weighting: {mode}")
    return weights


def _infer_fold_ensemble(args: argparse.Namespace, test_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    fold_dirs = sorted([p for p in args.folds_dir.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {args.folds_dir}")

    fold_weights: list[float] | None = None
    mode = args.fold_ensemble_weighting
    if mode in ("inverse_mse", "inverse_sqrt_mse", "inverse_mse_mae"):
        fold_weights = _fold_ensemble_weights(
            args.metadata_path,
            fold_dirs,
            mode,
            mae_coef=float(args.fold_ensemble_mae_coef),
        )
        sw = sum(fold_weights)
        fold_weights = [w / sw for w in fold_weights]

    all_preds = []
    base_ids = None
    for fold_idx, fold_dir in enumerate(fold_dirs):
        fold_meta_path = fold_dir / "metadata.json"
        if not fold_meta_path.exists():
            raise ValueError(f"Missing fold metadata in {fold_dir}")
        fold_meta = load_metadata(fold_meta_path)
        if fold_meta.get("separate_target_models"):
            for ti in range(len(TARGET_COLUMNS)):
                mp = fold_meta.get(f"model_path_t{ti}")
                if not mp or not Path(mp).is_file():
                    raise ValueError(f"Missing per-target model file for t{ti} in {fold_dir}")
            fold_model_path = None
        else:
            fold_model_path = fold_dir / "model.pt"
            if not fold_model_path.is_file():
                raise ValueError(f"Missing fold model in {fold_dir}")
        ids, pred = _predict_with_artifact(
            test_df=test_df,
            artifact_meta=fold_meta,
            model_path=fold_model_path,
            batch_size=args.batch_size,
            device=args.device,
            tta_runs=args.tta_runs,
            tta_noise_std=args.tta_noise_std,
        )
        if base_ids is None:
            base_ids = ids
        elif ids != base_ids:
            raise ValueError("Fold ensemble scenario ordering mismatch.")
        w = fold_weights[fold_idx] if fold_weights is not None else 1.0
        all_preds.append(pred * w)
    stacked = np.stack(all_preds, axis=0)
    pred_mean = np.sum(stacked, axis=0) if fold_weights is not None else np.mean(stacked, axis=0)
    return base_ids, pred_mean


def infer(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.metadata_path)
    test_df = pd.read_csv(args.test_path)
    use_ensemble = args.use_fold_ensemble and args.folds_dir.exists()
    if use_ensemble:
        scenario_ids, pred = _infer_fold_ensemble(args, test_df)
        wmsg = f" | fold_weighting={args.fold_ensemble_weighting}" if args.fold_ensemble_weighting != "mean" else ""
        print(
            f"Using fold ensemble from: {args.folds_dir}{wmsg} | tta_runs={args.tta_runs} | tta_noise_std={args.tta_noise_std}"
        )
    else:
        mp: Path | None = args.model_path
        if metadata.get("separate_target_models"):
            mp = None
        scenario_ids, pred = _predict_with_artifact(
            test_df=test_df,
            artifact_meta=metadata,
            model_path=mp,
            batch_size=args.batch_size,
            device=args.device,
            tta_runs=args.tta_runs,
            tta_noise_std=args.tta_noise_std,
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
    parser.add_argument(
        "--fold-ensemble-weighting",
        type=str,
        default="inverse_mse",
        choices=["mean", "inverse_mse", "inverse_sqrt_mse", "inverse_mse_mae"],
        help="Fold weights from root metadata validation_metrics: inverse_mse_mae uses 1/(mse+coef*mae) for stabler blend.",
    )
    parser.add_argument(
        "--fold-ensemble-mae-coef",
        type=float,
        default=0.22,
        help="MAE multiplier for inverse_mse_mae weighting (tune 0.15–0.35).",
    )
    parser.add_argument(
        "--tta-runs",
        type=int,
        default=8,
        help="Forward passes averaged at inference (1 for fast dev; 8+ for submit).",
    )
    parser.add_argument("--tta-noise-std", type=float, default=0.01, help="Gaussian noise on scaled x/ctx during TTA.")
    return parser


if __name__ == "__main__":
    infer(build_arg_parser().parse_args())


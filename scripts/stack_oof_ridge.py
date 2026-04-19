"""Stack K fold-trained artifacts with a linear stacker fit on out-of-fold train predictions.

Replays the same GroupKFold split as ``train.py``, runs each fold model on its validation
scenarios, denormalizes with **that fold's** ``y_mean``/``y_std``, then fits per-target
stackers on ``[pred_0, …, pred_{K-1}]`` vs true targets (default: ``Ridge``). Applies the
same map to K test prediction CSVs (original target units).

Stackers (``--stacker``): ``ridge`` (default), ``huber``, ``elasticnet``,
``positive_lr``, ``simplex`` (weights ≥0, sum to 1 per target; use ``--simplex-objective mae``
if MSE collapses redundant models to zero weight).

Example (3 models)::

    PYTHONPATH=src python scripts/stack_oof_ridge.py \\
        --artifacts artifacts/v2_transformer_s123 artifacts/v6_transattn_max_s123 artifacts/deepsets_seed123 \\
        --predictions predictions_v2_transformer_tta20.csv predictions_v6_transattn_max_tta20.csv predictions_deepsets_seed123_tta20.csv \\
        --output-path predictions_stacked_oof_ridge_k3.csv \\
        --ridge-alpha 5.0

Optional: ``--tune-ridge-alphas "1,2,4,6,8,12"`` runs nested GroupKFold on OOF (no stacker
leakage) and picks the best alpha (``ridge`` only). Use ``--tune-metric rmse`` if the LB is
MSE-heavy. ``--ridge-alpha-t0`` / ``--ridge-alpha-t1`` set different L2 per target for ridge.

Other stackers: ``--stacker huber|elasticnet|positive_lr|simplex`` (see CLI flags for Huber / ElasticNet).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dot.config import SCENARIO_ID, TARGET_COLUMNS, TRAIN_PATH  # noqa: E402
from dot.data import apply_context_scaler, apply_feature_scaler, frame_to_scenario_sets  # noqa: E402
from dot.infer import _forward_batches_tta, build_regressor_from_meta  # noqa: E402
from dot.model import ScenarioSetDataset, collate_infer  # noqa: E402


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _oof_one_artifact(
    train_df: pd.DataFrame,
    artifact_root: Path,
    *,
    n_splits: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (oof_preds_orig n×2, y_true n×2, scenario_ids sorted)."""
    scenario_ids = np.array(sorted(train_df[SCENARIO_ID].unique().tolist()))
    sid_to_i = {str(s): i for i, s in enumerate(scenario_ids)}
    n = len(scenario_ids)
    oof = np.full((n, 2), np.nan, dtype=np.float64)
    y_true = np.full((n, 2), np.nan, dtype=np.float64)

    for sid in scenario_ids:
        g = train_df[train_df[SCENARIO_ID] == sid]
        i = sid_to_i[str(sid)]
        y_true[i, 0] = float(g[TARGET_COLUMNS[0]].iloc[0])
        y_true[i, 1] = float(g[TARGET_COLUMNS[1]].iloc[0])

    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(n, dtype=np.int64)
    splits = list(gkf.split(idx, groups=scenario_ids))
    for fold_idx, (_, val_idx) in enumerate(splits, start=1):
        fold_dir = artifact_root / "folds" / f"fold_{fold_idx}"
        meta_path = fold_dir / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        fold_meta = _load_json(meta_path)
        val_ids = set(scenario_ids[val_idx].tolist())
        val_df = train_df[train_df[SCENARIO_ID].isin(val_ids)].copy()

        for ti in range(len(TARGET_COLUMNS)):
            mp = fold_meta.get(f"model_path_t{ti}")
            if not mp or not Path(mp).is_file():
                raise ValueError(f"Missing model for t{ti} in {fold_dir}")
            feature_columns = fold_meta[f"feature_columns_t{ti}"]
            interaction_feature_columns = fold_meta[f"interaction_feature_columns_t{ti}"]
            context_columns = fold_meta[f"context_columns_t{ti}"]
            in_dim = int(fold_meta[f"input_dim_t{ti}"])
            ctx_dim = int(fold_meta[f"context_dim_t{ti}"])
            scaler = {
                "mean": np.array(fold_meta[f"x_mean_t{ti}"], dtype=np.float32),
                "std": np.array(fold_meta[f"x_std_t{ti}"], dtype=np.float32),
            }
            context_scaler = {
                "mean": np.array(fold_meta[f"ctx_mean_t{ti}"], dtype=np.float32),
                "std": np.array(fold_meta[f"ctx_std_t{ti}"], dtype=np.float32),
            }
            y_mean = float(fold_meta["y_mean"][ti])
            y_std = float(fold_meta["y_std"][ti])
            if y_std < 1e-12:
                y_std = 1.0

            scenario_data = frame_to_scenario_sets(
                val_df,
                is_train=False,
                feature_columns=feature_columns,
                interaction_feature_columns=interaction_feature_columns,
            )
            if scenario_data.context_columns != context_columns:
                raise ValueError("Context mismatch in OOF build.")
            scaled = apply_feature_scaler(scenario_data.components, scaler)
            scaled_context = apply_context_scaler(scenario_data.context, context_scaler)
            ds = ScenarioSetDataset(scaled, scaled_context, targets=None)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_infer)

            model = build_regressor_from_meta(
                fold_meta,
                input_dim=in_dim,
                context_dim=ctx_dim,
                output_dim=1,
            )
            state_dict = torch.load(mp, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            pred_norm = _forward_batches_tta(
                model, loader, device, tta_runs=1, tta_noise_std=0.0
            ).reshape(-1)
            pred_orig = pred_norm * y_std + y_mean
            for j, sid in enumerate(scenario_data.scenario_ids):
                oof[sid_to_i[str(sid)], ti] = float(pred_orig[j])

    if np.isnan(oof).any():
        raise RuntimeError("OOF matrix has NaNs — fold coverage mismatch.")
    return oof, y_true, [str(s) for s in scenario_ids]


def _make_stacker(
    stacker: str,
    *,
    ridge_alpha: float,
    huber_epsilon: float,
    huber_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
):
    s = stacker.lower().strip()
    if s == "ridge":
        return Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    if s == "huber":
        return HuberRegressor(
            epsilon=float(huber_epsilon),
            alpha=float(huber_alpha),
            fit_intercept=True,
            max_iter=500,
        )
    if s == "elasticnet":
        return ElasticNet(
            alpha=float(elasticnet_alpha),
            l1_ratio=float(elasticnet_l1_ratio),
            fit_intercept=True,
            max_iter=5000,
            random_state=0,
        )
    if s in ("positive_lr", "positive-linear", "nnls_soft"):
        return LinearRegression(fit_intercept=True, positive=True)
    raise ValueError(f"Unknown stacker: {stacker!r}")


def _softmax(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).ravel()
    w = w - np.max(w)
    e = np.exp(w)
    s = float(np.sum(e)) or 1.0
    return e / s


def _fit_simplex_weights_mse(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 minimizing mean squared error on OOF rows."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    K = X.shape[1]

    def objective(z: np.ndarray) -> float:
        w = _softmax(z)
        r = X @ w - y
        return float(np.mean(r * r))

    z0 = np.zeros(K, dtype=np.float64)
    res = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 500})
    w = _softmax(res.x)
    return w


def _fit_simplex_weights_mae(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Same simplex, minimize mean absolute error (often uses extra correlated models)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    K = X.shape[1]

    def objective(z: np.ndarray) -> float:
        w = _softmax(z)
        return float(np.mean(np.abs(X @ w - y)))

    z0 = np.zeros(K, dtype=np.float64)
    res = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 800, "ftol": 1e-10})
    w = _softmax(res.x)
    return w


def _stacker_coef_str(model, K: int) -> str:
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        parts = [f"c{k}={coef[k]:.4f}" for k in range(min(K, len(coef)))]
        return " ".join(parts)
    return ""


def _nested_stack_cv_score(
    oofs: list[np.ndarray],
    y_true: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    alpha: float,
    n_splits: int,
    metric: str,
) -> tuple[float, float, float]:
    """Fit per-target Ridge with nested GroupKFold on OOF features (no stacker leakage).

    Returns (combined_score, err0, err1) where err* is MAE or RMSE per target and
    combined is their sum after dividing by each target's train std (scale-free).
    """
    n = y_true.shape[0]
    idx = np.arange(n, dtype=np.int64)
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros_like(y_true)
    for tr_idx, va_idx in gkf.split(idx, groups=scenario_ids):
        for ti in range(y_true.shape[1]):
            X_tr = np.column_stack([oof[tr_idx, ti] for oof in oofs])
            y_tr = y_true[tr_idx, ti]
            X_va = np.column_stack([oof[va_idx, ti] for oof in oofs])
            ridge = Ridge(alpha=float(alpha), fit_intercept=True)
            ridge.fit(X_tr, y_tr)
            pred[va_idx, ti] = ridge.predict(X_va)
    std0 = float(np.std(y_true[:, 0])) or 1.0
    std1 = float(np.std(y_true[:, 1])) or 1.0
    if metric == "mae":
        e0 = float(mean_absolute_error(y_true[:, 0], pred[:, 0]))
        e1 = float(mean_absolute_error(y_true[:, 1], pred[:, 1]))
    elif metric == "rmse":
        e0 = float(np.sqrt(mean_squared_error(y_true[:, 0], pred[:, 0])))
        e1 = float(np.sqrt(mean_squared_error(y_true[:, 1], pred[:, 1])))
    else:
        raise ValueError(metric)
    combined = e0 / std0 + e1 / std1
    return combined, e0, e1


def main() -> None:
    ap = argparse.ArgumentParser(description="Ridge stack K>=2 models using OOF train predictions.")
    ap.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    ap.add_argument(
        "--artifacts",
        type=Path,
        nargs="+",
        required=True,
        help="Artifact roots (same fold count / GroupKFold as training).",
    )
    ap.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Test prediction CSVs, same order as --artifacts.",
    )
    ap.add_argument("--output-path", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--ridge-alpha",
        type=float,
        default=4.0,
        help="L2 on Ridge; with more models try ~5–15.",
    )
    ap.add_argument(
        "--ridge-alpha-t0",
        type=float,
        default=None,
        help="Override --ridge-alpha for target 0 only (final fit + test).",
    )
    ap.add_argument(
        "--ridge-alpha-t1",
        type=float,
        default=None,
        help="Override --ridge-alpha for target 1 only (final fit + test).",
    )
    ap.add_argument(
        "--tune-ridge-alphas",
        type=str,
        default=None,
        help=(
            "Comma-separated Ridge alphas to evaluate with nested GroupKFold on OOF "
            "(same groups as training). Prints scores and picks the best for the final fit."
        ),
    )
    ap.add_argument(
        "--tune-metric",
        type=str,
        choices=("mae", "rmse"),
        default="mae",
        help="Objective for --tune-ridge-alphas (default mae). rmse often tracks MSE-heavy leaderboards.",
    )
    ap.add_argument(
        "--stacker",
        type=str,
        default="ridge",
        choices=("ridge", "huber", "elasticnet", "positive_lr", "simplex"),
        help="Second-level regressor (default ridge). tune-ridge-alphas only applies to ridge.",
    )
    ap.add_argument("--huber-epsilon", type=float, default=1.35)
    ap.add_argument(
        "--huber-alpha",
        type=float,
        default=1e-3,
        help="L2-like regularization on HuberRegressor (sklearn).",
    )
    ap.add_argument("--elasticnet-alpha", type=float, default=1e-3)
    ap.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    ap.add_argument(
        "--simplex-objective",
        type=str,
        choices=("mse", "mae"),
        default="mse",
        help="Loss on OOF when fitting --stacker simplex weights (mae often spreads mass across models).",
    )
    args = ap.parse_args()

    if args.tune_ridge_alphas and args.stacker != "ridge":
        raise SystemExit("--tune-ridge-alphas requires --stacker ridge.")
    if args.stacker == "simplex" and (
        args.ridge_alpha_t0 is not None or args.ridge_alpha_t1 is not None
    ):
        raise SystemExit("--ridge-alpha-t0/t1 are not used with --stacker simplex.")

    arts: list[Path] = list(args.artifacts)
    preds_paths: list[Path] = list(args.predictions)
    if len(arts) != len(preds_paths):
        raise SystemExit("--artifacts and --predictions must have the same length.")
    if len(arts) < 2:
        raise SystemExit("Need at least two models to stack.")

    train_df = pd.read_csv(args.train_path)
    oofs: list[np.ndarray] = []
    y_ref: np.ndarray | None = None
    sids_ref: list[str] | None = None

    for root in arts:
        print("Building OOF for:", root, flush=True)
        oof, y_true, sids = _oof_one_artifact(
            train_df, root, n_splits=args.n_splits, device=args.device, batch_size=args.batch_size
        )
        oofs.append(oof)
        if y_ref is None:
            y_ref = y_true
            sids_ref = sids
        else:
            if sids != sids_ref or not np.allclose(y_true, y_ref):
                raise RuntimeError("OOF / target alignment mismatch between artifacts.")

    assert y_ref is not None and sids_ref is not None
    sid_arr = np.array(sids_ref, dtype=object)

    if args.tune_ridge_alphas:
        alphas = [float(x.strip()) for x in args.tune_ridge_alphas.split(",") if x.strip()]
        if len(alphas) < 1:
            raise SystemExit("--tune-ridge-alphas must list at least one value.")
        print(
            f"Nested GroupKFold stacker CV metric={args.tune_metric} (lower combined = better):",
            flush=True,
        )
        best_a: float | None = None
        best_score = float("inf")
        for a in alphas:
            sc, m0, m1 = _nested_stack_cv_score(
                oofs,
                y_ref,
                sid_arr,
                alpha=a,
                n_splits=args.n_splits,
                metric=args.tune_metric,
            )
            label = "MAE" if args.tune_metric == "mae" else "RMSE"
            print(f"  alpha={a:g}  combined={sc:.6f}  {label}_t0={m0:.6f}  {label}_t1={m1:.6f}", flush=True)
            if sc < best_score:
                best_score = sc
                best_a = a
        assert best_a is not None
        print(f"Chosen ridge-alpha from tune: {best_a} (combined={best_score:.6f})", flush=True)
        args.ridge_alpha = float(best_a)

    pred_dfs: list[pd.DataFrame] = []
    for p in preds_paths:
        d = pd.read_csv(p)
        for c in [SCENARIO_ID, *TARGET_COLUMNS]:
            if c not in d.columns:
                raise SystemExit(f"Missing {c} in {p}")
        pred_dfs.append(d)

    merged = pred_dfs[0][[SCENARIO_ID, TARGET_COLUMNS[0], TARGET_COLUMNS[1]]].copy()
    merged = merged.rename(
        columns={
            TARGET_COLUMNS[0]: f"{TARGET_COLUMNS[0]}__0",
            TARGET_COLUMNS[1]: f"{TARGET_COLUMNS[1]}__0",
        }
    )
    for k in range(1, len(pred_dfs)):
        suf = f"__{k}"
        right = pred_dfs[k][[SCENARIO_ID, TARGET_COLUMNS[0], TARGET_COLUMNS[1]]].rename(
            columns={
                TARGET_COLUMNS[0]: f"{TARGET_COLUMNS[0]}{suf}",
                TARGET_COLUMNS[1]: f"{TARGET_COLUMNS[1]}{suf}",
            }
        )
        merged = merged.merge(right, on=SCENARIO_ID, how="inner")

    K = len(arts)
    models: list[object] = []
    simplex_w: list[np.ndarray] | None = None
    alpha_t0 = float(args.ridge_alpha_t0) if args.ridge_alpha_t0 is not None else float(args.ridge_alpha)
    alpha_t1 = float(args.ridge_alpha_t1) if args.ridge_alpha_t1 is not None else float(args.ridge_alpha)
    alphas_per_t = [alpha_t0, alpha_t1]

    if args.stacker == "simplex":
        fit_fn = _fit_simplex_weights_mse if args.simplex_objective == "mse" else _fit_simplex_weights_mae
        simplex_w = []
        for ti, tcol in enumerate(TARGET_COLUMNS):
            X = np.column_stack([oof[:, ti] for oof in oofs])
            y = y_ref[:, ti]
            w = fit_fn(X, y)
            simplex_w.append(w)
            parts = " ".join(f"w{k}={w[k]:.4f}" for k in range(K))
            r = X @ w - y
            mse = float(np.mean(r * r))
            mae = float(mean_absolute_error(y, X @ w))
            print(
                f"Target {ti} ({tcol[:48]}...): stacker=simplex {parts}  OOF_MSE={mse:.4f} OOF_MAE={mae:.4f}",
                flush=True,
            )
    else:
        for ti, tcol in enumerate(TARGET_COLUMNS):
            X = np.column_stack([oof[:, ti] for oof in oofs])
            y = y_ref[:, ti]
            m = _make_stacker(
                args.stacker,
                ridge_alpha=float(alphas_per_t[ti]),
                huber_epsilon=args.huber_epsilon,
                huber_alpha=args.huber_alpha,
                elasticnet_alpha=args.elasticnet_alpha,
                elasticnet_l1_ratio=args.elasticnet_l1_ratio,
            )
            m.fit(X, y)
            models.append(m)
            coef_str = _stacker_coef_str(m, K)
            ic = float(getattr(m, "intercept_", 0.0))
            print(
                f"Target {ti} ({tcol[:48]}...): stacker={args.stacker} {coef_str} intercept={ic:.6f}",
                flush=True,
            )

    col_t0 = [f"{TARGET_COLUMNS[0]}__{k}" for k in range(K)]
    col_t1 = [f"{TARGET_COLUMNS[1]}__{k}" for k in range(K)]

    out_rows: list[dict] = []
    for _, row in merged.iterrows():
        sid = row[SCENARIO_ID]
        feats0 = np.array([float(row[c]) for c in col_t0], dtype=np.float64)
        feats1 = np.array([float(row[c]) for c in col_t1], dtype=np.float64)
        if simplex_w is not None:
            y0 = float(feats0 @ simplex_w[0])
            y1 = float(feats1 @ simplex_w[1])
        else:
            y0 = models[0].predict(feats0.reshape(1, -1))[0]
            y1 = models[1].predict(feats1.reshape(1, -1))[0]
        out_rows.append({SCENARIO_ID: sid, TARGET_COLUMNS[0]: float(y0), TARGET_COLUMNS[1]: float(y1)})

    out = pd.DataFrame(out_rows)
    out.to_csv(args.output_path, index=False)
    print(f"Wrote {args.output_path} ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()

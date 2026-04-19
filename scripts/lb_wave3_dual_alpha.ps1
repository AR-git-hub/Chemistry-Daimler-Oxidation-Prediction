# Wave3: Ridge-стек k5+v7 с разными L2 для ΔKV (t0) и окисления (t1) + вес в nested-CV.
# Слабое место пайплайна: один alpha на обе цели — здесь сетка (a0,a1).

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$arts = @(
    "artifacts/v2_transformer_s123",
    "artifacts/v6_transattn_max_s123",
    "artifacts/deepsets_seed123",
    "artifacts/v3_transformer_s77_m12",
    "artifacts/v7_max_s77_m12"
)
$preds = @(
    "predictions_v2_transformer_tta20.csv",
    "predictions_v6_transattn_max_tta20.csv",
    "predictions_deepsets_seed123_tta20.csv",
    "predictions_v3_transformer_s77_tta20.csv",
    "predictions_v7_max_s77_tta20.csv"
)

Write-Host "=== dual-alpha grid, balanced nested metric (w_t0=0.5) ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts $arts `
    --predictions $preds `
    --device cuda --batch-size 16 `
    --stacker ridge `
    --tune-ridge-per-target-grid "3,6,10,14,20,28|4,8,12,18,26,36" `
    --tune-metric rmse --tune-metric-weight-t0 0.5 `
    --output-path predictions_stacked_k5_v7_ridge_dualalpha_w050.csv

Write-Host "=== dual-alpha, nested metric чуть сильнее t1 (w_t0=0.38) ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts $arts `
    --predictions $preds `
    --device cuda --batch-size 16 `
    --stacker ridge `
    --tune-ridge-per-target-grid "3,6,10,14,20,28|4,8,12,18,26,36" `
    --tune-metric rmse --tune-metric-weight-t0 0.38 `
    --output-path predictions_stacked_k5_v7_ridge_dualalpha_w038_t1heavy.csv

foreach ($f in @(
    "predictions_stacked_k5_v7_ridge_dualalpha_w050.csv",
    "predictions_stacked_k5_v7_ridge_dualalpha_w038_t1heavy.csv"
)) {
    python scripts/validate_submission.py --predictions-path $f --test-path data/merged/test_full.csv
}

Write-Host "Готово." -ForegroundColor Green

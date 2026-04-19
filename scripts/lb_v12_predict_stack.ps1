# После lb_v12_t1_focus_train.ps1: TTA20 + k5 стек (v12 вместо v7), Ridge nested tune.
# Подменяй v7 только если cv_mean_mse / OOF стека лучше эталона.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$V12 = "artifacts/v12_max_s123_m12_t1mae_rich"
$Pred = "predictions_v12_max_s123_t1mae_rich_tta20.csv"

if (-not (Test-Path "$V12/metadata.json")) { Write-Error "Нет $V12 — сначала lb_v12_t1_focus_train.ps1 -Train" }

python scripts/predict.py `
    --metadata-path "$V12/metadata.json" `
    --folds-dir "$V12/folds" `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --output-path $Pred

$arts = @(
    "artifacts/v2_transformer_s123",
    "artifacts/v6_transattn_max_s123",
    "artifacts/deepsets_seed123",
    "artifacts/v3_transformer_s77_m12",
    $V12
)
$preds = @(
    "predictions_v2_transformer_tta20.csv",
    "predictions_v6_transattn_max_tta20.csv",
    "predictions_deepsets_seed123_tta20.csv",
    "predictions_v3_transformer_s77_tta20.csv",
    $Pred
)

python scripts/stack_oof_ridge.py `
    --artifacts $arts `
    --predictions $preds `
    --device cuda --batch-size 16 `
    --stacker ridge `
    --tune-ridge-per-target-grid "3,6,10,14,20,28|4,8,12,18,26,36" `
    --tune-metric rmse --tune-metric-weight-t0 0.5 `
    --output-path predictions_stacked_k5_v12_t1mae_rich_dualalpha.csv

python scripts/validate_submission.py `
    --predictions-path predictions_stacked_k5_v12_t1mae_rich_dualalpha.csv `
    --test-path data/merged/test_full.csv

Write-Host "Файл: predictions_stacked_k5_v12_t1mae_rich_dualalpha.csv" -ForegroundColor Green

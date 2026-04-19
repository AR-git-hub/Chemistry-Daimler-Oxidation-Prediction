# Predict v13 TTA20 -> k5 (v13 вместо v7) Ridge dual-alpha -> validate.
# Плюс консервативный бленд 92%% эталонного стека + 8%% v13-стека (на случай 3 сабмитов).
#
# Сабмит по приоритету (смотри cv v13 vs v7):
#   1) predictions_stacked_k5_v13_dom_dualalpha.csv  — если v13 не хуже v7 по CV MSE
#   2) predictions_blend_v13safe_92_wave2_8_dom.csv — если сомневаешься
#   3) predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv — якорь

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$V13 = "artifacts/v13_max_s123_m12_domloss"
$Pred = "predictions_v13_max_s123_m12_domloss_tta20.csv"
$StackOut = "predictions_stacked_k5_v13_dom_dualalpha.csv"
$BlendOut = "predictions_blend_v13safe_92_wave2_8_dom.csv"
$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"

if (-not (Test-Path "$V13/metadata.json")) { Write-Error "Нет $V13 — сначала lb_v13_train.ps1 -Train" }

python scripts/predict.py `
    --metadata-path "$V13/metadata.json" `
    --folds-dir "$V13/folds" `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --output-path $Pred

$arts = @(
    "artifacts/v2_transformer_s123",
    "artifacts/v6_transattn_max_s123",
    "artifacts/deepsets_seed123",
    "artifacts/v3_transformer_s77_m12",
    $V13
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
    --tune-metric rmse --tune-metric-weight-t0 0.42 `
    --output-path $StackOut

if (-not (Test-Path $Wave2)) {
    Write-Host "Нет $Wave2 — пропуск бленда" -ForegroundColor DarkYellow
} else {
    python scripts/blend_predictions.py `
        --inputs $Wave2 $StackOut `
        --weights 0.92 0.08 `
        --output-path $BlendOut
}

python scripts/validate_submission.py --predictions-path $StackOut --test-path data/merged/test_full.csv
if (Test-Path $BlendOut) {
    python scripts/validate_submission.py --predictions-path $BlendOut --test-path data/merged/test_full.csv
}

Write-Host ""
Write-Host "Файлы:" -ForegroundColor Green
Write-Host "  $StackOut"
if (Test-Path $BlendOut) { Write-Host "  $BlendOut" }

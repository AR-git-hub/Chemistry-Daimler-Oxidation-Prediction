# После lb_wave2_train.ps1 -Train: predict TTA20 + стеки (v7 эталон + замена v7 на v10).
# Без утечки: stack_oof_ridge.py заново считает OOF по fold-моделям и nested GroupKFold для alpha.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$V10Dir = "artifacts/v10_max_s123_m12_rich"
$V10Pred = "predictions_v10_max_s123_m12_rich_tta20.csv"
$V11Dir = "artifacts/v11_attn_s77_m12_rich"
$V11Pred = "predictions_v11_attn_s77_m12_rich_tta20.csv"

function RequireDir($p) {
    if (-not (Test-Path "$p/metadata.json")) { Write-Error "Нет $p — сначала обучение." }
}

if (Test-Path "$V10Dir/metadata.json") {
    Write-Host "=== PREDICT v10 TTA20 ===" -ForegroundColor Cyan
    python scripts/predict.py `
        --metadata-path "$V10Dir/metadata.json" `
        --folds-dir "$V10Dir/folds" `
        --fold-ensemble-weighting inverse_mse_mae `
        --fold-ensemble-mae-coef 0.2 `
        --tta-runs 20 `
        --output-path $V10Pred
} else {
    Write-Host "Пропуск v10: нет $V10Dir" -ForegroundColor DarkYellow
}

if (Test-Path "$V11Dir/metadata.json") {
    Write-Host "=== PREDICT v11 TTA20 ===" -ForegroundColor Cyan
    python scripts/predict.py `
        --metadata-path "$V11Dir/metadata.json" `
        --folds-dir "$V11Dir/folds" `
        --fold-ensemble-weighting inverse_mse_mae `
        --fold-ensemble-mae-coef 0.2 `
        --tta-runs 20 `
        --output-path $V11Pred
} else {
    Write-Host "Пропуск v11: нет $V11Dir" -ForegroundColor DarkYellow
}

$artsK5v7 = @(
    "artifacts/v2_transformer_s123",
    "artifacts/v6_transattn_max_s123",
    "artifacts/deepsets_seed123",
    "artifacts/v3_transformer_s77_m12",
    "artifacts/v7_max_s77_m12"
)
$predsK5v7 = @(
    "predictions_v2_transformer_tta20.csv",
    "predictions_v6_transattn_max_tta20.csv",
    "predictions_deepsets_seed123_tta20.csv",
    "predictions_v3_transformer_s77_tta20.csv",
    "predictions_v7_max_s77_tta20.csv"
)

Write-Host "=== k5 v7: Ridge tune (rmse) — эталонный состав ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts $artsK5v7 `
    --predictions $predsK5v7 `
    --device cuda --batch-size 16 `
    --tune-ridge-alphas "2,4,6,8,10,14,18,24,32" --tune-metric rmse `
    --output-path predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv

Write-Host "=== k5 v7: poly2 t1 + tune ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts $artsK5v7 `
    --predictions $predsK5v7 `
    --device cuda --batch-size 16 `
    --stacker poly2_ridge --poly-only-for t1 --poly-interaction-only `
    --tune-ridge-alphas "40,80,120,180,260,360" --tune-metric rmse `
    --output-path predictions_stacked_k5_v7_poly2t1_tuned_wave2.csv

if (Test-Path $V10Pred) {
    $artsK5v10 = @(
        "artifacts/v2_transformer_s123",
        "artifacts/v6_transattn_max_s123",
        "artifacts/deepsets_seed123",
        "artifacts/v3_transformer_s77_m12",
        $V10Dir
    )
    $predsK5v10 = @(
        "predictions_v2_transformer_tta20.csv",
        "predictions_v6_transattn_max_tta20.csv",
        "predictions_deepsets_seed123_tta20.csv",
        "predictions_v3_transformer_s77_tta20.csv",
        $V10Pred
    )
    Write-Host "=== k5 v10rich: Ridge tune (замена v7) ===" -ForegroundColor Cyan
    python scripts/stack_oof_ridge.py `
        --artifacts $artsK5v10 `
        --predictions $predsK5v10 `
        --device cuda --batch-size 16 `
        --tune-ridge-alphas "2,4,6,8,10,14,18,24,32" --tune-metric rmse `
        --output-path predictions_stacked_k5_v10rich_ridge_tuned.csv

    Write-Host "=== blend 88/12: эталон a6 + v10rich stack ===" -ForegroundColor Cyan
    python scripts/blend_predictions.py `
        --inputs predictions_stacked_oof_k5_v3v7_ridge_a6.csv predictions_stacked_k5_v10rich_ridge_tuned.csv `
        --weights 0.88 0.12 `
        --output-path predictions_blend_k5_a6_88_v10richstack_12.csv
}

$toVal = @(
    "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv",
    "predictions_stacked_k5_v7_poly2t1_tuned_wave2.csv",
    "predictions_stacked_k5_v10rich_ridge_tuned.csv",
    "predictions_blend_k5_a6_88_v10richstack_12.csv"
)
foreach ($f in $toVal) {
    if (Test-Path $f) {
        python scripts/validate_submission.py --predictions-path $f --test-path data/merged/test_full.csv
    }
}

Write-Host ""
Write-Host "Кандидаты на сабмит (по приоритету после проверки OOF/CV):" -ForegroundColor Green
Write-Host "  1) predictions_stacked_k5_v7_poly2t1_tuned_wave2.csv"
Write-Host "  2) predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"
Write-Host "  3) predictions_blend_k5_a6_88_v10richstack_12.csv (если v10 обучен)"
Write-Host "  Запас: predictions_stacked_oof_k5_v3v7_ridge_a6.csv"

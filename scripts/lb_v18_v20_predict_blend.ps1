# После v18 и/или v20: TTA20 + валидация + бленд 94/6 с wave2 (чуть агрессивнее 92/8, если CV лучше cmp).
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TestFe = "data/merged/test_fe.csv"
$TestFull = "data/merged/test_full.csv"
$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"

function DoPred($dir, $outCsv) {
    if (-not (Test-Path "$dir/metadata.json")) { return }
    Write-Host "=== PREDICT $dir -> $outCsv ===" -ForegroundColor Cyan
    python scripts/predict.py `
        --metadata-path "$dir/metadata.json" `
        --folds-dir "$dir/folds" `
        --test-path $TestFe `
        --fold-ensemble-weighting inverse_mse_mae `
        --fold-ensemble-mae-coef 0.2 `
        --tta-runs 20 `
        --device cuda --batch-size 16 `
        --output-path $outCsv
    python scripts/validate_submission.py --predictions-path $outCsv --test-path $TestFull
}

DoPred "artifacts/v18_deepsets_trainfe_s42_cmp" "predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv"
DoPred "artifacts/v20_max_trainfe_s123_t128_m12_rich" "predictions_v20_max_trainfe_s123_t128_m12_rich_tta20.csv"

if (Test-Path $Wave2) {
    foreach ($pair in @(
        @("predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv", "predictions_submit_blend_94wave2_06v18trainfe.csv"),
        @("predictions_v20_max_trainfe_s123_t128_m12_rich_tta20.csv", "predictions_submit_blend_94wave2_06v20trainfe.csv")
    )) {
        $pred = $pair[0]
        $blend = $pair[1]
        if (Test-Path $pred) {
            Write-Host "=== BLEND 94/6 $blend ===" -ForegroundColor Yellow
            python scripts/blend_predictions.py --inputs $Wave2 $pred --weights 0.94 0.06 --output-path $blend
            python scripts/validate_submission.py --predictions-path $blend --test-path $TestFull
        }
    }
}

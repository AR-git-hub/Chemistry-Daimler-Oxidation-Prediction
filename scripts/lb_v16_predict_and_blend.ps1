# После lb_v16_trainfe_wave2.ps1 -Train: TTA20 на test_fe + валидация + консервативный бленд с wave2.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$V16Dir = "artifacts/v16_max_trainfe_s123_m12_rich"
$V16Pred = "predictions_v16_max_trainfe_s123_m12_rich_tta20.csv"
$TestFe = "data/merged/test_fe.csv"
$TestFull = "data/merged/test_full.csv"
$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"

if (-not (Test-Path "$V16Dir/metadata.json")) {
    Write-Error "Нет $V16Dir — сначала .\scripts\lb_v16_trainfe_wave2.ps1 -Train"
}

Write-Host "=== PREDICT v16 TTA20 (test_fe) ===" -ForegroundColor Cyan
python scripts/predict.py `
    --metadata-path "$V16Dir/metadata.json" `
    --folds-dir "$V16Dir/folds" `
    --test-path $TestFe `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --output-path $V16Pred

python scripts/validate_submission.py --predictions-path $V16Pred --test-path $TestFull

if (Test-Path $Wave2) {
    Write-Host "=== Бленд 92/8 wave2 + v16_trainfe ===" -ForegroundColor Cyan
    python scripts/blend_predictions.py `
        --inputs $Wave2 $V16Pred `
        --weights 0.92 0.08 `
        --output-path predictions_submit_blend_92wave2_08v16trainfe.csv
    python scripts/validate_submission.py --predictions-path predictions_submit_blend_92wave2_08v16trainfe.csv --test-path $TestFull
}

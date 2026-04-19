# После lb_v17_train_full_plus_tbn.ps1 -Train: TTA20 на test_full_plus_tbn + валидация + бленд 92/8 с wave2.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$V17Dir = "artifacts/v17_max_fulltbn_s123_m12_rich"
$V17Pred = "predictions_v17_max_fulltbn_s123_m12_rich_tta20.csv"
$TestPlus = "data/merged/test_full_plus_tbn.csv"
$TestFull = "data/merged/test_full.csv"
$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"

if (-not (Test-Path "$V17Dir/metadata.json")) {
    Write-Error "Нет $V17Dir — сначала .\scripts\lb_v17_train_full_plus_tbn.ps1 -Train"
}

Write-Host "=== PREDICT v17 TTA20 (test_full_plus_tbn) ===" -ForegroundColor Cyan
python scripts/predict.py `
    --metadata-path "$V17Dir/metadata.json" `
    --folds-dir "$V17Dir/folds" `
    --test-path $TestPlus `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --device cuda --batch-size 16 `
    --output-path $V17Pred

python scripts/validate_submission.py --predictions-path $V17Pred --test-path $TestFull

if (Test-Path $Wave2) {
    Write-Host "=== Бленд 92/8 wave2 + v17 ===" -ForegroundColor Cyan
    python scripts/blend_predictions.py `
        --inputs $Wave2 $V17Pred `
        --weights 0.92 0.08 `
        --output-path predictions_submit_blend_92wave2_08v17fulltbn.csv
    python scripts/validate_submission.py --predictions-path predictions_submit_blend_92wave2_08v17fulltbn.csv --test-path $TestFull
}

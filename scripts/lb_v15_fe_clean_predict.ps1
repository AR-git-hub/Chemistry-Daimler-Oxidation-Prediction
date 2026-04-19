# После lb_v15_fe_clean_train.ps1: TTA20 на test_fe_cleaned + валидация + быстрый табличный baseline.
#
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$V15Dir = "artifacts/v15_max_fe_clean_s123_m12_rich"
$V15Pred = "predictions_v15_max_fe_clean_s123_m12_rich_tta20.csv"
$TestFe = "data/merged/test_fe_cleaned.csv"
$TestFull = "data/merged/test_full.csv"
$TrainFe = "data/merged/train_fe_cleaned.csv"
$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"

if (Test-Path "$V15Dir/metadata.json") {
    Write-Host "=== PREDICT v15 (fe_cleaned) TTA20 ===" -ForegroundColor Cyan
    python scripts/predict.py `
        --metadata-path "$V15Dir/metadata.json" `
        --folds-dir "$V15Dir/folds" `
        --test-path $TestFe `
        --fold-ensemble-weighting inverse_mse_mae `
        --fold-ensemble-mae-coef 0.2 `
        --tta-runs 20 `
        --output-path $V15Pred
    python scripts/validate_submission.py --predictions-path $V15Pred --test-path $TestFull
} else {
    Write-Host "Пропуск v15: нет $V15Dir (сначала lb_v15_fe_clean_train.ps1 -Train)" -ForegroundColor DarkYellow
}

Write-Host "=== Табличный Ridge на агрегатах fe_cleaned (другой класс модели) ===" -ForegroundColor Cyan
python scripts/scenario_aggregate_ridge.py `
    --train-path $TrainFe `
    --test-path $TestFe `
    --output-path predictions_scenario_agg_ridge_fe_cleaned.csv
python scripts/validate_submission.py --predictions-path predictions_scenario_agg_ridge_fe_cleaned.csv --test-path $TestFull

if ((Test-Path $Wave2) -and (Test-Path $V15Pred)) {
    Write-Host "=== Бленд 97/3 wave2 + v15_fe_clean (по желанию на LB) ===" -ForegroundColor Cyan
    python scripts/blend_predictions.py `
        --inputs $Wave2 $V15Pred `
        --weights 0.97 0.03 `
        --output-path predictions_submit_blend_97wave2_03v15feclean.csv
    python scripts/validate_submission.py --predictions-path predictions_submit_blend_97wave2_03v15feclean.csv --test-path $TestFull
}

if ((Test-Path $Wave2) -and (Test-Path "predictions_scenario_agg_ridge_fe_cleaned.csv")) {
    Write-Host "=== Бленд 97/3 wave2 + scenario Ridge (эксперимент) ===" -ForegroundColor Cyan
    python scripts/blend_predictions.py `
        --inputs $Wave2 predictions_scenario_agg_ridge_fe_cleaned.csv `
        --weights 0.97 0.03 `
        --output-path predictions_submit_blend_97wave2_03scenarioridge.csv
    python scripts/validate_submission.py --predictions-path predictions_submit_blend_97wave2_03scenarioridge.csv --test-path $TestFull
}

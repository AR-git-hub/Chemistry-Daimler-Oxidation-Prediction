# TTA20 для v18 s42/s43/s77 (если артефакты есть), равное усреднение -> predictions_v18_ens3_tta20.csv
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TestFe = "data/merged/test_fe.csv"
$TestFull = "data/merged/test_full.csv"

$runs = @(
    @{ Dir = "artifacts/v18_deepsets_trainfe_s42_cmp"; Out = "predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv" },
    @{ Dir = "artifacts/v18_deepsets_trainfe_s43_cmp"; Out = "predictions_v18_deepsets_trainfe_s43_cmp_tta20.csv" },
    @{ Dir = "artifacts/v18_deepsets_trainfe_s77_cmp"; Out = "predictions_v18_deepsets_trainfe_s77_cmp_tta20.csv" }
)

foreach ($r in $runs) {
    if (-not (Test-Path "$($r.Dir)/metadata.json")) {
        Write-Host "Пропуск: нет $($r.Dir)" -ForegroundColor DarkYellow
        continue
    }
    Write-Host "=== PREDICT $($r.Dir) ===" -ForegroundColor Cyan
    python scripts/predict.py `
        --metadata-path "$($r.Dir)/metadata.json" `
        --folds-dir "$($r.Dir)/folds" `
        --test-path $TestFe `
        --fold-ensemble-weighting inverse_mse_mae `
        --fold-ensemble-mae-coef 0.2 `
        --tta-runs 20 `
        --device cuda --batch-size 16 `
        --output-path $r.Out
    python scripts/validate_submission.py --predictions-path $r.Out --test-path $TestFull
}

$ensInputs = @(
    "predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv",
    "predictions_v18_deepsets_trainfe_s43_cmp_tta20.csv",
    "predictions_v18_deepsets_trainfe_s77_cmp_tta20.csv"
) | Where-Object { Test-Path $_ }

if ($ensInputs.Count -ge 2) {
    Write-Host "=== ENSEMBLE mean ($($ensInputs.Count) files) ===" -ForegroundColor Cyan
    if ($ensInputs.Count -eq 2) {
        python scripts/average_predictions.py --inputs $ensInputs[0] $ensInputs[1] --output-path predictions_v18_ens3_tta20.csv
    } else {
        python scripts/average_predictions.py --inputs $ensInputs[0] $ensInputs[1] $ensInputs[2] --output-path predictions_v18_ens3_tta20.csv
    }
    python scripts/validate_submission.py --predictions-path predictions_v18_ens3_tta20.csv --test-path $TestFull
} else {
    Write-Host "Нужно минимум 2 из 3 предикта для ансамбля — обучи multiseed или оставь только s42." -ForegroundColor Yellow
}

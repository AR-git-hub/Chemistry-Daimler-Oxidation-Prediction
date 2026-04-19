# GO BIG: fe_cleaned + HGBR (деревья на сценарных агрегатах), крупные бленды с wave2.
# Один сабмит — один файл; не слать всё подряд. Смотри OOF/LB и выбери одну долю.
#
# Требует: predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv
#
#   .\scripts\lb_go_big_fe_hgbr.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"
$TestFull = "data/merged/test_full.csv"
$HgbrOut = "predictions_scenario_agg_hgbr_fe_cleaned.csv"

if (-not (Test-Path $Wave2)) { Write-Error "Нет якоря $Wave2" }

Write-Host "=== HGBR на train_fe_cleaned (агрегаты по сценарию) ===" -ForegroundColor Cyan
python scripts/scenario_aggregate_hgbr.py `
    --train-path data/merged/train_fe_cleaned.csv `
    --test-path data/merged/test_fe_cleaned.csv `
    --output-path $HgbrOut
python scripts/validate_submission.py --predictions-path $HgbrOut --test-path $TestFull

$jobs = @(
    @{ w = @(0.80, 0.20); f = "predictions_go_big_80wave2_20hgbr_fe_agg.csv" },
    @{ w = @(0.70, 0.30); f = "predictions_go_big_70wave2_30hgbr_fe_agg.csv" },
    @{ w = @(0.55, 0.45); f = "predictions_go_big_55wave2_45hgbr_fe_agg.csv" }
)
foreach ($j in $jobs) {
    Write-Host "=== Бленд $($j.w[0]) / $($j.w[1]) -> $($j.f) ===" -ForegroundColor Yellow
    python scripts/blend_predictions.py `
        --inputs $Wave2 $HgbrOut `
        --weights $j.w[0] $j.w[1] `
        --output-path $j.f
    python scripts/validate_submission.py --predictions-path $j.f --test-path $TestFull
}

Write-Host ""
Write-Host "Готово. Рекомендация: на LB — максимум ОДИН из go_big_* (самый агрессивный только если готов к просадке)." -ForegroundColor Green

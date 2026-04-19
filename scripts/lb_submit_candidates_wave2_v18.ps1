# Консервативные бленды wave2 + v18 (лучший CV MSE на train_fe). Валидация на test_full.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Wave2 = "predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv"
$TestFull = "data/merged/test_full.csv"

$candidates = @(
    @{ Pred = "predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv"; Name = "submit_blend_97wave2_03v18deepsets" },
    @{ Pred = "predictions_v18_ens3_tta20.csv"; Name = "submit_blend_97wave2_03v18ens3" }
)

if (-not (Test-Path $Wave2)) {
    Write-Error "Нет якоря $Wave2"
}

foreach ($c in $candidates) {
    if (-not (Test-Path $c.Pred)) {
        Write-Host "Skip $($c.Name): missing $($c.Pred)" -ForegroundColor DarkYellow
        continue
    }
    $out = "predictions_$($c.Name).csv"
    Write-Host "=== $out (97/3) ===" -ForegroundColor Cyan
    python scripts/blend_predictions.py --inputs $Wave2 $c.Pred --weights 0.97 0.03 --output-path $out
    python scripts/validate_submission.py --predictions-path $out --test-path $TestFull
}

# Чуть сильнее v18, если осмелишься на LB:
if (Test-Path "predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv") {
    Write-Host "=== predictions_submit_blend_95wave2_05v18deepsets.csv ===" -ForegroundColor Yellow
    python scripts/blend_predictions.py `
        --inputs $Wave2 predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv `
        --weights 0.95 0.05 `
        --output-path predictions_submit_blend_95wave2_05v18deepsets.csv
    python scripts/validate_submission.py --predictions-path predictions_submit_blend_95wave2_05v18deepsets.csv --test-path $TestFull
}

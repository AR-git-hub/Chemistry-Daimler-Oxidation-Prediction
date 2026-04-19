# Predict + k5 стек: v2, v6, deepsets123, v3, и НОВАЯ v14 вместо v7 (v13 на LB хуже — не используем).
#
# Пример:
#   .\scripts\lb_v14_predict_stack.ps1 -ArtifactDir artifacts/v14_meanmax_s202_m12_rich
#
# Выход: predictions_stacked_k5_v14_SLOT_dualalpha.csv (имя по последней папке).

param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactDir
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$meta = Join-Path $ArtifactDir "metadata.json"
if (-not (Test-Path $meta)) { Write-Error "Нет $meta" }

$slug = Split-Path $ArtifactDir -Leaf
# Имя без двойного префикса v14_
$Pred = "predictions_k5slot_$slug`_tta20.csv"
$Out = "predictions_stacked_k5_slot_$slug`_dualalpha.csv"

python scripts/predict.py `
    --metadata-path $meta `
    --folds-dir (Join-Path $ArtifactDir "folds") `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --output-path $Pred

$arts = @(
    "artifacts/v2_transformer_s123",
    "artifacts/v6_transattn_max_s123",
    "artifacts/deepsets_seed123",
    "artifacts/v3_transformer_s77_m12",
    $ArtifactDir
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
    --output-path $Out

python scripts/validate_submission.py --predictions-path $Out --test-path data/merged/test_full.csv
Write-Host "CSV: $Out" -ForegroundColor Green

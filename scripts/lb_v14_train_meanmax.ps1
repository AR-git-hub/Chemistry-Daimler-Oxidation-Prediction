# v14 (A): НОВАЯ архитектура transformer_meanmax — self-attention encoder + mean/max readout,
# БЕЗ cross-attention к контексту (другой граф, чем transformer_attention_max).
#
# НЕ подменяй v7 в стеке, пока cv_mean_mse_normalized не лучше ~0.257 (v7).
# Если хуже — сабмить predictions_stacked_k5_v7_ridge_tuned_rmse_wave2.csv
#
#   .\scripts\lb_v14_train_meanmax.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Out = "artifacts/v14_meanmax_s202_m12_rich"

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v14_train_meanmax.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

python scripts/train.py `
    --device cuda --batch-size 16 `
    --seed 202 `
    --architecture transformer_meanmax `
    --mass-interaction-k 12 `
    --rich-scenario-context `
    --hidden-dim 176 --transformer-heads 8 --transformer-layers 3 --transformer-ffn-mult 3 `
    --dropout 0.12 --optimizer adamw --weight-decay 4e-5 --grad-clip-norm 1.0 `
    --input-noise-std 0.012 `
    --loss-type hybrid --hybrid-mse-weight 0.86 `
    --val-selection-metric combined `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "OK: $Out" -ForegroundColor Green
Write-Host "Дальше: .\scripts\lb_v14_predict_stack.ps1 -ArtifactDir $Out" -ForegroundColor Cyan

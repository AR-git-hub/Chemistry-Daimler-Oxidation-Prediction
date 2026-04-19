# v14 (B): Deep Sets + дополнительный sum-pool (архитектура deepsets_sumpool) + heterogeneity.
# Иной индуктивный сдвиг относительно трансформеров в стеке.
#
#   .\scripts\lb_v14_train_sumpool.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Out = "artifacts/v14_sumpool_s311_m12_rich"

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v14_train_sumpool.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

python scripts/train.py `
    --device cuda --batch-size 16 `
    --seed 311 `
    --architecture deepsets_sumpool `
    --use-heterogeneity `
    --mass-interaction-k 12 `
    --rich-scenario-context `
    --hidden-dim 128 --encoder-hidden-dim 192 --rho-hidden-dim 192 `
    --dropout 0.1 --optimizer adamw --weight-decay 3e-5 --grad-clip-norm 1.0 `
    --input-noise-std 0.01 `
    --loss-type hybrid --hybrid-mse-weight 0.88 `
    --val-selection-metric combined `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "OK: $Out" -ForegroundColor Green
Write-Host "Дальше: .\scripts\lb_v14_predict_stack.ps1 -ArtifactDir $Out" -ForegroundColor Cyan

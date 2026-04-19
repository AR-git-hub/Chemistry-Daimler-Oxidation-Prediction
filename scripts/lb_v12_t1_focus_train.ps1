# v12: та же архитектура/регуляризация, что у сильных max-моделей, но early stopping для
# окисления (t1) по val MAE — меньше гонки за выбросы в нормализованном MSE (без утечки: только fold-val).
#
# Сравни cv_mean_* в metadata с v7; если не лучше — в стек не подмешивай.
#
#   .\scripts\lb_v12_t1_focus_train.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Out = "artifacts/v12_max_s123_m12_t1mae_rich"

if (-not $Train) {
    Write-Host "Запусти с -Train. Потом: predict + stack см. lb_v12_predict_stack.ps1" -ForegroundColor Yellow
    exit 0
}

python scripts/train.py `
    --device cuda --batch-size 16 `
    --seed 123 `
    --architecture transformer_attention_max `
    --mass-interaction-k 12 `
    --rich-scenario-context `
    --hidden-dim 160 --transformer-heads 8 --transformer-layers 2 --transformer-ffn-mult 3 `
    --dropout 0.1 --optimizer adamw --weight-decay 3e-5 --grad-clip-norm 1.0 `
    --input-noise-std 0.01 `
    --val-selection-metric mse `
    --val-selection-metric-t1 mae `
    --loss-type hybrid --hybrid-mse-weight 0.88 `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "Готово: $Out — сравни CV с v7, затем .\scripts\lb_v12_predict_stack.ps1" -ForegroundColor Green

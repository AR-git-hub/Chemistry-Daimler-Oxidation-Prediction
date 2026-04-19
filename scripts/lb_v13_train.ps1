# v13: доменный rich-контекст (расширенный в data.py) + отдельный loss на t1 (logcosh) +
#      отдельная val-метрика на t1 (mae). Без утечки: только fold train / fold val.
#
# После обучения: сравни в metadata.json cv_mean_mse_normalized с v7 (~0.257).
# Если v13 хуже по MSE — в стек лучше не подменять v7; используй бленд из lb_v13_predict_stack.ps1.
#
#   .\scripts\lb_v13_train.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Out = "artifacts/v13_max_s123_m12_domloss"

if (-not $Train) {
    Write-Host "Запусти: .\scripts\lb_v13_train.ps1 -Train" -ForegroundColor Yellow
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
    --loss-type hybrid --hybrid-mse-weight 0.88 `
    --loss-type-t1 logcosh `
    --val-selection-metric mse `
    --val-selection-metric-t1 mae `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "OK: $Out" -ForegroundColor Green
Write-Host "Дальше: .\scripts\lb_v13_predict_stack.ps1" -ForegroundColor Cyan

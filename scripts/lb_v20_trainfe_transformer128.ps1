# v20: «мост» — transformer_attention_max на train_fe, но уже 128 dim (ближе к cmp по ёмкости),
# + rich + m12 + wave2-гиперы. Цель: удержать хороший MAE как у v16, подтянуть MSE к зоне cmp.
#
#   .\scripts\lb_v20_trainfe_transformer128.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TrainCsv = "data/merged/train_fe.csv"
$Out = "artifacts/v20_max_trainfe_s123_t128_m12_rich"

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v20_trainfe_transformer128.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

python scripts/train.py `
    --train-path $TrainCsv `
    --device cuda --batch-size 16 `
    --seed 123 `
    --architecture transformer_attention_max `
    --mass-interaction-k 12 `
    --rich-scenario-context `
    --hidden-dim 128 --transformer-heads 8 --transformer-layers 2 --transformer-ffn-mult 3 `
    --dropout 0.08 --optimizer adamw --weight-decay 3e-5 --grad-clip-norm 1.0 `
    --input-noise-std 0.01 `
    --loss-type hybrid --hybrid-mse-weight 0.86 `
    --val-selection-metric combined `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "OK: $Out" -ForegroundColor Green

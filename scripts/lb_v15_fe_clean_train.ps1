# v15: тот же граф, что wave2 v10 (transformer_attention_max + rich), но train на train_fe_cleaned.csv.
# Тест для predict: test_fe_cleaned.csv (тот же порядок scenario_id, что test_full).
#
# Не смешивать артефакты v15 с k5-стеком wave2 в stack_oof_ridge без общего train_path —
# для стека только нейросети, обученные на том же CSV. Здесь — отдельный слот / бленд с эталоном.
#
#   .\scripts\lb_v15_fe_clean_train.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TrainCsv = "data/merged/train_fe_cleaned.csv"
$Out = "artifacts/v15_max_fe_clean_s123_m12_rich"

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v15_fe_clean_train.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

if (-not (Test-Path $TrainCsv)) { Write-Error "Нет $TrainCsv" }

python scripts/train.py `
    --train-path $TrainCsv `
    --device cuda --batch-size 16 `
    --seed 123 `
    --architecture transformer_attention_max `
    --mass-interaction-k 12 `
    --rich-scenario-context `
    --hidden-dim 160 --transformer-heads 8 --transformer-layers 2 --transformer-ffn-mult 3 `
    --dropout 0.1 --optimizer adamw --weight-decay 3e-5 --grad-clip-norm 1.0 `
    --input-noise-std 0.01 `
    --loss-type hybrid --hybrid-mse-weight 0.86 `
    --val-selection-metric combined `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "OK: $Out" -ForegroundColor Green
Write-Host "Дальше: .\scripts\lb_v15_fe_clean_predict.ps1" -ForegroundColor Cyan

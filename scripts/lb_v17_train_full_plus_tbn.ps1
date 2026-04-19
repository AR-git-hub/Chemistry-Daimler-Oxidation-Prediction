# v17: сначала merge_tbn_from_fe_cleaned.py, затем обучение на train_full + tbn_consolidated (одна новая колонка).
#
#   python scripts/merge_tbn_from_fe_cleaned.py
#   .\scripts\lb_v17_train_full_plus_tbn.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TrainCsv = "data/merged/train_full_plus_tbn.csv"
$Out = "artifacts/v17_max_fulltbn_s123_m12_rich"

if (-not $Train) {
    Write-Host "Сначала: python scripts/merge_tbn_from_fe_cleaned.py" -ForegroundColor Yellow
    Write-Host "Потом:   .\scripts\lb_v17_train_full_plus_tbn.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

if (-not (Test-Path $TrainCsv)) {
    python scripts/merge_tbn_from_fe_cleaned.py
}
if (-not (Test-Path $TrainCsv)) { Write-Error "Нет $TrainCsv после merge" }

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
Write-Host "Predict: use test_full_plus_tbn.csv (same row order as test_full for blends)." -ForegroundColor Cyan

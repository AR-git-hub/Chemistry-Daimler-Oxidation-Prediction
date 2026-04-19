# v18: воспроизведение cmp_train_fe (~cv_mse_norm 0.239) на train_fe.csv:
# DeepSets, hidden 128, dropout 0, Adam, hybrid 0.92, БЕЗ rich context, БЕЗ mass_int — как artifacts/cmp_train_fe.
#
# v16 (transformer+rich+m12) на тех же данных дал ~0.259 MSE — это другая модель; этот трек — твоя «главная находка».
#
#   .\scripts\lb_v18_trainfe_deepsets_cmp.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TrainCsv = "data/merged/train_fe.csv"
$Out = "artifacts/v18_deepsets_trainfe_s42_cmp"

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v18_trainfe_deepsets_cmp.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

python scripts/train.py `
    --train-path $TrainCsv `
    --device cuda --batch-size 16 `
    --seed 42 `
    --architecture deepsets `
    --hidden-dim 128 `
    --dropout 0 `
    --optimizer adam --weight-decay 2e-5 `
    --grad-clip-norm 0 `
    --input-noise-std 0.01 `
    --loss-type hybrid --hybrid-mse-weight 0.92 `
    --val-selection-metric mse `
    --mass-interaction-k 0 `
    --no-rich-scenario-context `
    --artifacts-dir $Out `
    --n-splits 5 --epochs 200

Write-Host "OK: $Out — сравни cv_mean_* с artifacts/cmp_train_fe/metadata.json" -ForegroundColor Green
Write-Host "Predict: python scripts/predict.py --metadata-path $Out/metadata.json --folds-dir $Out/folds --test-path data/merged/test_fe.csv --tta-runs 20 --output-path predictions_v18_deepsets_trainfe_s42_cmp_tta20.csv" -ForegroundColor Cyan

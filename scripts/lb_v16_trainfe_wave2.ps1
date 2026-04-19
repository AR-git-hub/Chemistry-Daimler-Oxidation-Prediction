# v16: wave2-style transformer_attention_max на train_fe.csv (больше фич, чем train_full).
# В репо уже есть cmp_train_fe: cv_mean_mse_norm ~0.239 при старых hparams; здесь — полный wave2 v10 стек гиперпараметров.
#
# predict: test_fe.csv (тот же порядок scenario_id, что test_full).
#
#   .\scripts\lb_v16_trainfe_wave2.ps1 -Train
# После обучения: predict + бленд с wave2 (отдельно, когда будут predictions_v16_*_tta20.csv).

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TrainCsv = "data/merged/train_fe.csv"
$Out = "artifacts/v16_max_trainfe_s123_m12_rich"

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v16_trainfe_wave2.ps1 -Train" -ForegroundColor Yellow
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
Write-Host "Predict: python scripts/predict.py --metadata-path $Out/metadata.json --folds-dir $Out/folds --test-path data/merged/test_fe.csv --tta-runs 20 --output-path predictions_v16_max_trainfe_s123_m12_rich_tta20.csv" -ForegroundColor Cyan

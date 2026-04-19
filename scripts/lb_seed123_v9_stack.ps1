# Seed-123 track: train NEW transformer_attention_max (v9) -> TTA20 -> k5 stack
#   v2(s123) + v6(s123) + DeepSets(s123) + v3_transformer(s77) + v9_max(s123)
# (v7 s77 убираем — вся «ветка» под лучший CV-сид 123.)
#
#   Train (долго):  .\scripts\lb_seed123_v9_stack.ps1 -Train
#   Дальше:        .\scripts\lb_seed123_v9_stack.ps1

param(
    [switch]$Train
)

$ErrorActionPreference = "Stop"
$ProjRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjRoot
$env:PYTHONPATH = "src"

$V9Dir = "artifacts/v9_max_s123_m12"
$V9Pred = "predictions_v9_max_s123_tta20.csv"
$OutRidge = "predictions_stacked_oof_k5_v3v9_s123_ridge_a6.csv"
$OutPoly = "predictions_stacked_k5_v3v9_poly2t1_tuned.csv"

if ($Train) {
    Write-Host "=== TRAIN v9 transformer_attention_max seed=123 ===" -ForegroundColor Cyan
    python scripts/train.py `
        --device cuda `
        --batch-size 16 `
        --seed 123 `
        --architecture transformer_attention_max `
        --mass-interaction-k 12 `
        --hidden-dim 160 `
        --transformer-heads 8 `
        --transformer-layers 2 `
        --transformer-ffn-mult 3 `
        --dropout 0.1 `
        --optimizer adamw `
        --weight-decay 3e-5 `
        --grad-clip-norm 1.0 `
        --input-noise-std 0.01 `
        --val-selection-metric combined `
        --hybrid-mse-weight 0.88 `
        --artifacts-dir $V9Dir `
        --n-splits 5 `
        --epochs 200
}

if (-not (Test-Path "$V9Dir/metadata.json")) {
    Write-Error "Нет $V9Dir — сначала: .\scripts\lb_seed123_v9_stack.ps1 -Train"
}

Write-Host "=== PREDICT v9 TTA20 ===" -ForegroundColor Cyan
python scripts/predict.py `
    --metadata-path "$V9Dir/metadata.json" `
    --folds-dir "$V9Dir/folds" `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --output-path $V9Pred

Write-Host "=== STACK k5 Ridge a=6 (v2+v6+ds+v3+v9, все согласованы с веткой s123 кроме v3) ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts artifacts/v2_transformer_s123 artifacts/v6_transattn_max_s123 artifacts/deepsets_seed123 artifacts/v3_transformer_s77_m12 $V9Dir `
    --predictions predictions_v2_transformer_tta20.csv predictions_v6_transattn_max_tta20.csv predictions_deepsets_seed123_tta20.csv predictions_v3_transformer_s77_tta20.csv $V9Pred `
    --output-path $OutRidge `
    --ridge-alpha 6.0

Write-Host "=== STACK poly2 (только t1) + tune alpha ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts artifacts/v2_transformer_s123 artifacts/v6_transattn_max_s123 artifacts/deepsets_seed123 artifacts/v3_transformer_s77_m12 $V9Dir `
    --predictions predictions_v2_transformer_tta20.csv predictions_v6_transattn_max_tta20.csv predictions_deepsets_seed123_tta20.csv predictions_v3_transformer_s77_tta20.csv $V9Pred `
    --stacker poly2_ridge --poly-only-for t1 --poly-interaction-only `
    --tune-ridge-alphas "40,80,120,180,260,360" --tune-metric rmse `
    --output-path $OutPoly

python scripts/validate_submission.py --predictions-path $OutRidge --test-path data/merged/test_full.csv
python scripts/validate_submission.py --predictions-path $OutPoly --test-path data/merged/test_full.csv

Write-Host ""
Write-Host "ОТПРАВКА: 1) $OutPoly  2) $OutRidge" -ForegroundColor Green

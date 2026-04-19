# v8 re-train (MAE-aware early stopping) -> TTA20 -> k5 stack REPLACING v7 with v8 (NOT k6 — k6 hurt LB).
# From repo root:
#   Train+rest:  .\scripts\lb_pipeline_v8.ps1 -Train
#   Stack only:  .\scripts\lb_pipeline_v8.ps1

param(
    [switch]$Train
)

$ErrorActionPreference = "Stop"
$ProjRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjRoot
$env:PYTHONPATH = "src"

# New artifact dir so you do not overwrite old v8 runs.
$V8Dir = "artifacts/v8_max_s456_d01_m12_maecombo"
$V8Pred = "predictions_v8_maecombo_tta20.csv"
$StackOut = "predictions_stacked_oof_k5_v3v8_ridge_a6.csv"

if ($Train) {
    Write-Host "=== TRAIN v8 (val metric = combined, seed 456, dropout 0.1) ===" -ForegroundColor Cyan
    python scripts/train.py `
        --device cuda `
        --batch-size 16 `
        --seed 456 `
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
        --artifacts-dir $V8Dir `
        --n-splits 5 `
        --epochs 200
}

if (-not (Test-Path "$V8Dir/metadata.json")) {
    Write-Error "Missing $V8Dir/metadata.json — run: .\scripts\lb_pipeline_v8.ps1 -Train"
}

Write-Host "=== PREDICT v8 TTA20 ===" -ForegroundColor Cyan
python scripts/predict.py `
    --metadata-path "$V8Dir/metadata.json" `
    --folds-dir "$V8Dir/folds" `
    --fold-ensemble-weighting inverse_mse_mae `
    --fold-ensemble-mae-coef 0.2 `
    --tta-runs 20 `
    --output-path $V8Pred

Write-Host "=== STACK k5: v2+v6+ds+v3+v8 (replace v7), Ridge a=6 ===" -ForegroundColor Cyan
python scripts/stack_oof_ridge.py `
    --artifacts artifacts/v2_transformer_s123 artifacts/v6_transattn_max_s123 artifacts/deepsets_seed123 artifacts/v3_transformer_s77_m12 $V8Dir `
    --predictions predictions_v2_transformer_tta20.csv predictions_v6_transattn_max_tta20.csv predictions_deepsets_seed123_tta20.csv predictions_v3_transformer_s77_tta20.csv $V8Pred `
    --output-path $StackOut `
    --ridge-alpha 6.0

Write-Host "=== OPTIONAL blend: 97% old best k5(v3+v7) + 3% k5(v3+v8) ===" -ForegroundColor Cyan
python -c @"
import pandas as pd
from dot.config import TARGET_COLUMNS, SCENARIO_ID
old = pd.read_csv('predictions_stacked_oof_k5_v3v7_ridge_a6.csv').sort_values(SCENARIO_ID)
new = pd.read_csv('$StackOut').sort_values(SCENARIO_ID)
assert (old[SCENARIO_ID].values == new[SCENARIO_ID].values).all()
p = 0.97
out = old.copy()
for c in TARGET_COLUMNS:
    out[c] = p * old[c].values + (1-p) * new[c].values
out.to_csv('predictions_blend_97oldk5v3v7_03k5v3v8.csv', index=False)
print('wrote predictions_blend_97oldk5v3v7_03k5v3v8.csv')
"@

python scripts/validate_submission.py --predictions-path $StackOut --test-path data/merged/test_full.csv
python scripts/validate_submission.py --predictions-path predictions_blend_97oldk5v3v7_03k5v3v8.csv --test-path data/merged/test_full.csv

Write-Host ""
Write-Host "SUBMIT (in order): 1) $StackOut  2) predictions_blend_97oldk5v3v7_03k5v3v8.csv" -ForegroundColor Green
Write-Host "Do NOT use k6 stacks; your best remains k5 v3+v7 @ 0.088858." -ForegroundColor Yellow

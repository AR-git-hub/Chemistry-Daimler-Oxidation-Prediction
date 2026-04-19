# Wave 2: новые базовые модели без утечки между сценариями:
#   --rich-scenario-context (только внутрисценарные агрегаты в data.py)
#   GroupKFold / cap full-data как в train.py (не трогаем).
#
# Запуск обучения (долго, по очереди):
#   .\scripts\lb_wave2_train.ps1 -Train
#
# После этого: .\scripts\lb_wave2_predict_stack.ps1

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$Common = @(
    "--device", "cuda",
    "--batch-size", "16",
    "--mass-interaction-k", "12",
    "--rich-scenario-context",
    "--hidden-dim", "160",
    "--transformer-heads", "8",
    "--transformer-layers", "2",
    "--transformer-ffn-mult", "3",
    "--dropout", "0.1",
    "--optimizer", "adamw",
    "--weight-decay", "3e-5",
    "--grad-clip-norm", "1.0",
    "--input-noise-std", "0.01",
    "--val-selection-metric", "combined",
    "--n-splits", "5",
    "--epochs", "200"
)

if (-not $Train) {
    Write-Host "Укажи -Train чтобы запустить два прогона train.py (v10 + v11)." -ForegroundColor Yellow
    exit 0
}

Write-Host "=== v10: transformer_attention_max seed 123 + rich context ===" -ForegroundColor Cyan
python scripts/train.py @Common `
    --seed 123 `
    --architecture transformer_attention_max `
    --hybrid-mse-weight 0.86 `
    --artifacts-dir artifacts/v10_max_s123_m12_rich

Write-Host "=== v11: transformer_attention (cross) seed 77 + rich + log-cosh ===" -ForegroundColor Cyan
python scripts/train.py @Common `
    --seed 77 `
    --architecture transformer_attention `
    --loss-type logcosh `
    --val-selection-metric mae `
    --artifacts-dir artifacts/v11_attn_s77_m12_rich

Write-Host "Готово. Дальше: .\scripts\lb_wave2_predict_stack.ps1" -ForegroundColor Green

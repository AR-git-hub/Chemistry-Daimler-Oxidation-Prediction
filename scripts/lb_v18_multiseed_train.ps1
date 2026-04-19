# Доп. сиды для ансамбля v18 (тот же граф, что v18_deepsets_trainfe_s42_cmp — cmp-реплика на train_fe).
# После обучения: lb_v18_three_seed_average.ps1
#
#   .\scripts\lb_v18_multiseed_train.ps1 -Train

param([switch]$Train)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "src"

$TrainCsv = "data/merged/train_fe.csv"

$jobs = @(
    @{ Seed = 43; Out = "artifacts/v18_deepsets_trainfe_s43_cmp" },
    @{ Seed = 77; Out = "artifacts/v18_deepsets_trainfe_s77_cmp" }
)

if (-not $Train) {
    Write-Host "Запуск: .\scripts\lb_v18_multiseed_train.ps1 -Train" -ForegroundColor Yellow
    exit 0
}

foreach ($j in $jobs) {
    Write-Host "=== TRAIN seed=$($j.Seed) -> $($j.Out) ===" -ForegroundColor Cyan
    python scripts/train.py `
        --train-path $TrainCsv `
        --device cuda --batch-size 16 `
        --seed $j.Seed `
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
        --artifacts-dir $j.Out `
        --n-splits 5 --epochs 200
}

Write-Host "OK. Дальше: .\scripts\lb_v18_three_seed_average.ps1" -ForegroundColor Green

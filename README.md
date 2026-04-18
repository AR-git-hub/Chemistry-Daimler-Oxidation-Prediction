# Chemistry-Daimler-Oxidation-Prediction

End-to-end baseline for the Daimler Oxidation Test (DOT) hackathon task.
The solution follows the technical specification constraints:
- multi-target regression for two DOT outputs;
- variable number of components per scenario;
- no tree-based methods in the final predictor;
- reproducible inference and submission artifacts.

## Repository structure

- `data/merged/` - prepared train/test tables from the organizer.
- `docs/` - fixed acceptance contract from the technical specification.
- `src/dot/` - core implementation:
  - `data.py` - scenario-level set construction and normalization;
  - `model.py` - Deep Sets model for variable-size component sets;
  - `train.py` - leakage-safe GroupKFold training + fold artifact export for ensembling;
  - `infer.py` - deterministic inference (`fold ensemble` by default, single-model fallback);
  - `validate.py` - strict output format validation.
- `src/data/`, `src/features/`, `src/models/`, `src/train/`, `src/infer/`, `src/interpret/` - structured module layout matching project plan.
- `scripts/` - CLI entrypoints.
- `artifacts/` - saved model and metadata after training.
- `reports/` - literature hypothesis table and factor interpretation summaries.
- `notebooks/inference.ipynb` - notebook required by platform.

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train model

Default training uses a **hybrid loss** (mostly MSE + small SmoothL1 term) with `--hybrid-mse-weight 0.92`, which in GroupKFold CV typically **lowers mean normalized MSE** versus pure `mse` at the cost of a slightly higher mean MAE on validation. For the original pure-MSE baseline, pass `--loss-type mse`.

Training uses **two separate Deep Sets** (one output per DOT target). **EMA weight smoothing** (`--ema-decay`, default `0.998`) improves CV vs off in quick sweeps; set `--ema-decay 0` to disable. Optional **per-target Spearman screening** (`--feature-selection`) is off by default. Reproduce cheap hparam sweeps: `PYTHONPATH=src python scripts/quick_sweep.py` (writes `artifacts/quick_sweep/summary.json`).

```bash
set PYTHONPATH=src
python scripts/train.py --epochs 200 --final-epochs 140 --batch-size 32 --input-noise-std 0.01
```

This writes:
- `artifacts/deepsets_model_t0.pt` and `artifacts/deepsets_model_t1.pt` (one Deep Sets per DOT target)
- `artifacts/metadata.json` (lists `model_path_t0` / `model_path_t1` and `separate_target_models: true`)

### 3) Run inference

```bash
set PYTHONPATH=src
python scripts/predict.py --output-path predictions.csv
```

By default inference uses **fold ensembling** with **inverse-MSE weights** (reads `validation_metrics` from `artifacts/metadata.json`) and **TTA** (several noisy forwards per batch). Tune e.g. `python scripts/predict.py --tta-runs 8 --fold-ensemble-weighting mean` if you want to compare.

To force single-model inference:

```bash
set PYTHONPATH=src
python scripts/predict.py --no-use-fold-ensemble --output-path predictions.csv
```

### 4) Validate submission format

```bash
set PYTHONPATH=src
python scripts/validate_submission.py --predictions-path predictions.csv --test-path data/merged/test_full.csv
```

### 5) Run factor analysis (interpretability)

```bash
set PYTHONPATH=src
python scripts/analyze_factors.py --top-k 20
```

### 6) Package submission zip

```bash
python scripts/package_submission.py --predictions-path predictions.csv --notebook-path notebooks/inference.ipynb --output-zip artifacts/submission.zip
```

### 7) Dry-run in isolated folder

```bash
set PYTHONPATH=src
python scripts/dry_run_clean.py
```

## Submission requirements alignment

This baseline produces:
- `predictions.csv` with required columns and one row per `scenario_id`.
- `notebooks/inference.ipynb` that reproduces prediction generation.
- container run files (`Dockerfile`, `docker-compose.yml`).
- `artifacts/submission.zip` with required platform files.

For expert review package:
- permutation factor analysis in `artifacts/factor_analysis.json`;
- summary report in `reports/factor_analysis_summary.md`;
- hypothesis mapping table in `reports/hypothesis_table.md`.

## Container run

Build and run:

```bash
docker build -t dot-infer .
docker run --rm -v ${PWD}:/app dot-infer
```

Or:

```bash
docker compose up --build
```

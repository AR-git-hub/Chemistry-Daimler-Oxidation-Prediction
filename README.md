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

Default training uses a **hybrid loss** (mostly MSE + log-cosh on residuals) with `--hybrid-mse-weight 0.92`. Degenerate / leakage-prone numeric columns are listed in `dot.config.FEATURE_BLOCKLIST` (e.g. constant viscosity at -30°C). For pure MSE, pass `--loss-type mse`; for pure log-cosh, `--loss-type logcosh`.

Training uses **two separate Deep Sets** (one output per DOT target). **EMA** (`--ema-decay`, default `0.998`) smooths weights for checkpoints and export; set `--ema-decay 0` to disable. **Anti–over-fit defaults:** **CV-based cap** on full-data epochs (`--final-epochs-from-cv`), a **scenario holdout probe** (`--final-holdout-fraction`, default `0.12`; `0` disables): a short train on an ~88%/12% split estimates `best_epoch`, then full-data training runs `min(CV cap, best_epoch + slack)` epochs (`final_epochs_effective_t*`). Optional gradient clip (`--grad-clip-norm`, default `0`) and dropout (`--dropout`, default `0`). Each CV fold logs `train_*` vs `val_*`. Optional **Spearman screening** (`--feature-selection`) is off by default. Cheap sweeps: `PYTHONPATH=src python scripts/quick_sweep.py`.

```bash
set PYTHONPATH=src
python scripts/train.py --epochs 200 --final-epochs 140 --batch-size 32 --input-noise-std 0.01
# Full-data epochs are auto-capped from CV unless --no-final-epochs-from-cv
```

This writes:
- `artifacts/deepsets_model_t0.pt` and `artifacts/deepsets_model_t1.pt` (one Deep Sets per DOT target)
- `artifacts/metadata.json` (lists `model_path_t0` / `model_path_t1` and `separate_target_models: true`)

### 3) Run inference

```bash
set PYTHONPATH=src
python scripts/predict.py --output-path predictions.csv
```

By default inference uses **fold ensembling** with **`inverse_mse`** weights (from `metadata.json`) and **TTA** (`--tta-runs` default 8). On the public LB, try `--fold-ensemble-weighting inverse_mse_mae` or `inverse_sqrt_mse`, and `--tta-runs 12`, without touching the train split.

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

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
  - `train.py` - model training and artifact export;
  - `infer.py` - deterministic inference and `predictions.csv` generation;
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

```bash
set PYTHONPATH=src
python scripts/train.py --epochs 200 --final-epochs 140 --batch-size 32 --input-noise-std 0.01
```

This writes:
- `artifacts/deepsets_model.pt`
- `artifacts/metadata.json`

### 3) Run inference

```bash
set PYTHONPATH=src
python scripts/predict.py --output-path predictions.csv
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

# DOT Acceptance Contract

This document fixes the delivery contract from the hackathon specification.

## Mandatory submission archive content

- `predictions.csv`
- `inference.ipynb`

Both files must be packed into a single `.zip` archive for platform upload.

## `predictions.csv` strict format

- Columns (exact order):
  1. `scenario_id`
  2. `Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %`
  3. `Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm`
- Exactly one row per test `scenario_id`.
- No missing test ids.
- No extra ids absent in test.
- No duplicated `scenario_id`.
- No empty values in target columns.

## Modeling constraints

- Final predictor must not use tree-based methods:
  - Gradient Boosting family (including XGBoost/LightGBM/CatBoost),
  - Random Forest / Extra Trees / related tree ensembles.
- Internal exploratory experiments may use them, but not final inference pipeline.

## Reproducibility and runtime expectations

- `inference.ipynb` runs without manual edits.
- Inference reads test input and produces valid `predictions.csv`.
- Docker-based reproducible run is provided (`Dockerfile`, optional compose).
- Solution should run in a clean environment without local hidden artifacts.

## Expert review expectations

- Variable-size multi-component handling.
- Feature engineering and feature selection rationale.
- Factor interpretation tied to chemistry/physics.
- Literature-backed hypothesis table:
  - `Factor -> DOT mechanism -> Implementation in model/features`.


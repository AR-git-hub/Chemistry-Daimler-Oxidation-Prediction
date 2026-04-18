# Hypothesis Table: Factor -> Mechanism -> Model Mapping

| Factor | Mechanism affecting DOT | How mapped to features | How mapped to architecture |
|---|---|---|---|
| Test temperature | Higher temperature accelerates oxidation chain reactions and viscosity growth | `ctx_condition::Температура испытания...` + component-level numeric columns | Deep Sets combines component embeddings with scenario context encoder |
| Test duration | Longer exposure increases oxidation products accumulation | `ctx_condition::Время испытания...` | Context branch conditions global pooled representation |
| Catalyst dosage category | Higher catalyst load can speed oxidation and additive depletion | `ctx_condition::Дозировка катализатора, категория` | Dedicated context features merged into prediction head |
| Biofuel fraction | Biofuel changes oxidation stability and acid formation pathways | `ctx_condition::Количество биотоплива...` | Context encoder + nonlinear head for interactions with component pool |
| Low-temperature rheology spread | CCS variability reflects base/additive composition diversity and stability behavior | `ctx_wstd::Динамическая вязкость CCS ...` | Weighted interaction summaries injected as context vector |
| Weighted mean component rheology | Aggregate viscosity behavior influences oxidation/viscosity outcomes | `ctx_wmean::Динамическая вязкость CCS ...` | Interaction-aware context + permutation-invariant set pooling |
| Hydrophobic tail mass spread | Surfactant/detergent architecture can modify oxidation inhibition dynamics | `ctx_wmean/wstd::Масса гидрофобного хвоста...` | Set encoder captures composition; context captures scenario-level dispersion |
| Alkalinity reserve | Neutralization reserve can delay acidic oxidation effects | `ctx_wmean::Щелочное число, ГОСТ 11362` | Nonlinear fusion in `rho` head over pooled set + context |

## Usage in final presentation

- Use this table as the bridge between domain assumptions and model design choices.
- Pair each row with feature-importance evidence from `reports/factor_analysis_summary.md`.


# DOT Factor Analysis Summary

Source: `artifacts/factor_analysis.json` (permutation analysis on validation fold).

## Key observations

- Baseline normalized MSE on validation split: `0.4219`.
- Most sensitive factors are scenario-level interaction summaries built from component properties.
- DOT condition controls (`temperature`, `time`, `biofuel share`, `catalyst category`) are consistently in top factors.

## Top factors by delta MSE (permutation)

| Rank | Feature | Delta MSE |
|---|---|---:|
| 1 | `ctx_wstd::Динамическая вязкость CCS -20°C, ASTM D5293` | 0.0137 |
| 2 | `ctx_condition::Количество биотоплива | - Daimler Oxidation Test (DOT), % масс` | 0.0122 |
| 3 | `ctx_wstd::Динамическая вязкость CCS -35°C, ASTM D5293` | 0.0112 |
| 4 | `ctx_wstd::Динамическая вязкость CCS -15°C, ASTM D5293` | 0.0109 |
| 5 | `ctx_wmean::Динамическая вязкость CCS -30°C, ASTM D5293` | 0.0100 |
| 6 | `ctx_condition::Время испытания | - Daimler Oxidation Test (DOT), ч` | 0.0096 |
| 7 | `ctx_wmean::Динамическая вязкость CCS -35°C, ASTM D5293` | 0.0090 |
| 8 | `ctx_condition::Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C` | 0.0071 |
| 9 | `ctx_wmean::Динамическая вязкость CCS -25°C, ASTM D5293` | 0.0068 |
| 10 | `ctx_condition::Дозировка катализатора, категория` | 0.0052 |

## Interpretation note

The model currently emphasizes viscosity- and condition-related factors, which is consistent with DOT stress conditions and oxidation-response dependence on base oil/additive rheology.


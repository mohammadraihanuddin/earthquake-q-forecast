# Results

All numbers are on test issue dates 2021–2023 unless marked *judge* (held-out window 2025–2026, scored once).

## Figures

| File | What it shows |
|---|---|
| `01_model_comparison.png` | Information gain of every model on the same 767,745 test rows |
| `02_external_drivers.png` | Change in IG from adding geomagnetic, solar, tidal or GPS features, paired 95% intervals |
| `03_per_horizon.png` | IG, AUC and calibration error for each of the 15 lead windows |
| `04_reliability.png` | Reliability diagram and distribution of forecast probabilities |
| `05_training_curve.png` | Log loss over the 120 boosting iterations |
| `06_data_distribution.png` | Catalog activity, magnitudes, event rate by year, split sizes |
| `07_feature_importance.png` | Permutation importance on held-out rows |
| `08_forecast_2025.png` | 2025 forecast map and grid-average probability by window |
| `09_forecast_alignment_fix.png` | The row-alignment bug in our first forecast file, before and after the fix |
| `10_test_vs_judge.png` | Test vs *judge*: IG, AUC, ECE |
| `11_quantum_paired_test.png` | Quantum features: paired 95% intervals for ΔIG and ΔAUC |
| `12_entry_thresholds.png` | Entry cut-off trade-off: cells cleared vs events missed |

## Tables (`tables/`)

`01_model_comparison_test` · `02_external_drivers_test` · `03_test_vs_judge` · `04_per_horizon_test` ·
`05_per_window_judge` · `06_quantum_paired_test` · `07_model_parameters` · `08_data_splits` ·
`09_risk_bands_test` · `10_entry_thresholds_test` · `11_feature_importance_test`

## Reports (`reports/`)

| File | Contents |
|---|---|
| `final_run.log` | Full log of the final run: environment, data hashes, parameters, training curve, calibration, metrics |
| `SUBMISSION_CHECK.md`, `verify.log`, `verification.json` | Reproducibility checks and findings |
| `judge_check.md`, `judge_metrics.json` | The single held-out judge score |
| `quantum_paired_test.json` | Quantum feature-map experiment and bootstrap intervals |
| `stakeholder.md` | 6-month look-ahead validation, risk bands, entry thresholds |

## Forecast

`forecast_2025.csv`: 23,265 rows (`lat, lon, window_start, mask_id, probability`), one per cell × 30-day
window from 2025-01-01, probabilities clipped to [0.0001, 0.9999].

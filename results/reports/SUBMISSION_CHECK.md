# Submission check — 2026-09-27

Everything behind the recorded results was rerun from the raw catalogs into
`rerun/`, then checked by `verify_results.py` (18/19 checks pass). The recorded
artifacts in `mohammad/` were not modified.

## Verdict

**The recorded results are reproducible and correct. The recorded forecast
file is not.** Do not submit `../frozen_provisional_forecast/predictions.csv`.
Use `rerun/frozen_forecast_fixed/predictions.csv` for engineering work, and
regenerate with `run_frozen_forecast_fixed.py` once the organizers supply the
official mask, cell order and window table. Submission is still blocked on
those files.

## What was rerun

| Stage | Result | Time |
|---|---|---:|
| Full-grid examples + development comparison | bit-identical examples, identical metrics | 479 s |
| ECE / early-middle-late / spatial prior | identical | 3 s |
| Matched external-driver comparison | identical metrics and block intervals | 775 s |
| Matched ETAS comparison | identical | 57 s |
| Day-7 QML ablation | identical | 9 s |
| Corrected 2025 forecast | 23,265 rows, validated | 29 s |
| Synthetic contract test | OK | 0 s |

## Checks

| Check | Outcome |
|---|---|
| Historical examples regenerate bit-identically | PASS |
| Headline IG 0.14582367854116265 / log loss 0.16876916 / Brier 0.04962805 | PASS (exact) |
| ECE 0.007146 (early 0.007435, middle 0.007892, late 0.006189) < 0.02 | PASS |
| Spatial-prior control 0.141048 bits | PASS |
| External drivers: all five reproduce, none reliably improves seismic-only | PASS |
| ETAS 0.145321 bits, Δ −0.000502, 95% CI [−0.001649, +0.000625] | PASS |
| QML ablation −0.001418 vs classical −0.001176 | PASS |
| Headline pipeline scripts never read the judge catalog | PASS (3 documented audit/self-check scripts do) |
| Candidate beats climatology under fit-rate and test-rate references | PASS (0.1458 / 0.1423 bits) |
| Corrected forecast: contract order, keys, clipping, 200 rows recomputed independently | PASS |
| **Recorded forecast rows aligned** | **FAIL — 22,163 of 23,265 rows on the wrong cell/window** |
| Purged-split sensitivity | PASS with a caveat (below) |

## Findings to carry into the presentation

1. **Forecast row bug (fixed here).** The original script paired a
   horizon-major probability vector with cell-major contract rows. The recorded
   file equals the horizon-major vector exactly. After the fix, per-row
   probabilities correlate with each cell's 2024 activity (Spearman 0.79 vs
   −0.04 for the recorded file).
2. **Label windows cross split boundaries.** Fit labels from late-2017/2018
   origins reach into 2019–2020, and calibration labels from 2020 origins reach
   into 2021–2022 (the test period). Dropping those rows gives **0.1417 bits**
   (−0.0041). The candidate remains well ahead of climatology, but gaps between
   close models (ETAS −0.0005, geomagnetic −0.0008) are smaller than this effect.
3. **Early stopping is active by default.** All 120 iterations ran, but
   scikit-learn held out a random 10% of fit rows for the stopping check, so the
   model trains on 90% of fit rows. State this in the model card.
4. **The QML ablation is uninformative, not negative.** 97.8% of its scaled
   input is zero and trees see the same information either way.
5. **Docs to update:** FINAL_RESULTS.md still says no quantum comparison was
   run; the last Study Deck audit section in the experiment log lists gaps
   already closed above it; driver intervals use 3-origin blocks and the ETAS
   interval uses single origins.

## Files

- `final_run/final_run.log` — full log of the final run (environment, hashes,
  parameters, training curve, calibration, metrics, importance, forecast)
- `final_run/final_summary.json` — the same, machine-readable
- `figures/01–09_*.png` — presentation figures
- `verification.json`, `verify.log` — check results
- `rerun/` — all regenerated artifacts and per-stage logs

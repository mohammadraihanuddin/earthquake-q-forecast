#!/bin/sh
# Rebuild every result from the raw CSVs in data/raw/ (about 30 minutes).
set -e
cd "$(dirname "$0")/pipeline/analysis"
python3 rerun_all.py            # examples, model comparison, drivers, ETAS, QML, forecast
python3 final_run.py            # final model with full log -> final_run/final_run.log
python3 judge_check.py          # held-out judge score (reads the judge CSV)
python3 quantum_paired_test.py  # quantum feature map + paired block bootstrap
python3 stakeholder.py          # 6-month look-ahead, risk bands, entry thresholds
python3 make_figures.py
python3 make_final_figures.py
python3 build_map.py            # interactive map -> stakeholder/risk_map.html
echo "Done. Forecast: pipeline/analysis/rerun/frozen_forecast_fixed/predictions.csv"

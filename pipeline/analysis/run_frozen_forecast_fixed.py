#!/usr/bin/env python3
"""Corrected frozen-candidate forecast on the provisional contract.

Same model, data, and calibration as ``../run_frozen_provisional_forecast.py``.
The only change is row alignment: probabilities are computed on a
(cell, horizon) grid and then looked up for every contract row by
(mask position, chronological rank of the row's window).  The original script
assigned a horizon-major probability vector to cell-major contract rows.

Also writes ``probability_matrix.npy`` (cells x horizons, mask order) so the
verifier can cross-check the CSV independently.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from official_grid_contract import validate_predictions as validate_official  # noqa: E402
from run_contract_interface import contract_rows, load_mask, load_windows, validate_predictions  # noqa: E402
from run_full_grid_development_comparison import ACCEL, HORIZONS, catalog, snapshot  # noqa: E402

CONTRACT = ROOT / "provisional_contract"
EXAMPLES = ROOT / "full_grid_development_comparison" / "historical_examples.npz"
OUT = HERE / "rerun" / "frozen_forecast_fixed"
SEED = 42
CUTOFF = pd.Timestamp("2025-01-01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    mask = load_mask(CONTRACT / "provisional_mask.csv")
    windows = load_windows(CONTRACT / "provisional_windows.csv")
    n_cells = len(mask)
    if n_cells != 1551 or len(windows) != HORIZONS:
        raise ValueError("Provisional contract dimensions do not match 1551 x 15")
    # Horizon of each supplied window = its chronological rank, independent of
    # the order in which the window table lists them.
    window_horizon = windows["window_start"].rank(method="first").astype(int).to_numpy() - 1
    first = windows["window_start"].min()
    if first != CUTOFF:
        raise ValueError(f"First window {first} does not start at the feature cutoff {CUTOFF}")

    z = np.load(EXAMPLES)
    x_train, y_train, h_train = z["x_train"], z["y_train"], z["h_train"]
    x_cal, y_cal, h_cal = z["x_cal"], z["y_cal"], z["h_cal"]
    if x_train.shape[1] != 12:
        raise ValueError(f"Unexpected seismic feature width: {x_train.shape[1]}")

    events = catalog()
    cells = mask[["lat", "lon"]].to_numpy(float)
    x_snapshot = snapshot(events, CUTOFF, cells)[:, ACCEL]

    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED)
    model = HistGradientBoostingClassifier(**args).fit(
        np.column_stack([x_train[:, ACCEL], h_train]), y_train
    )
    raw_cal = model.predict_proba(np.column_stack([x_cal[:, ACCEL], h_cal]))[:, 1]

    # matrix[i, h] = calibrated probability for mask row i at horizon h.
    matrix = np.empty((n_cells, HORIZONS), dtype=float)
    calibration_details = []
    for h in range(HORIZONS):
        h_value = np.float32(h / (HORIZONS - 1))
        cal = h_cal == h_value
        raw = model.predict_proba(
            np.column_stack([x_snapshot, np.full(n_cells, h_value, dtype=np.float32)])
        )[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal[cal], y_cal[cal])
        matrix[:, h] = np.clip(calibrator.predict(raw), 1e-4, 1.0 - 1e-4)
        calibration_details.append({
            "horizon": h,
            "calibration_rows": int(cal.sum()),
            "calibration_positives": int(y_cal[cal].sum()),
        })

    rows = contract_rows(mask, windows)  # mask order x window input order
    cell_pos = np.repeat(np.arange(n_cells), len(windows))
    horizon_pos = np.tile(window_horizon, n_cells)
    output = rows.assign(probability=matrix[cell_pos, horizon_pos])

    validate_predictions(output, rows)
    validate_official(output, mask, pd.DatetimeIndex(windows["window_start"]).sort_values())

    OUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT / "predictions.csv", index=False)
    np.save(OUT / "probability_matrix.npy", matrix)
    metadata = {
        "official": False,
        "status": "frozen_candidate_engineering_forecast_row_alignment_fixed",
        "fix": "probabilities looked up by (mask position, chronological window rank); "
               "original script paired horizon-major probabilities with cell-major rows",
        "judge_loaded": False,
        "mask_sha256": sha256(CONTRACT / "provisional_mask.csv"),
        "windows_sha256": sha256(CONTRACT / "provisional_windows.csv"),
        "cells": n_cells,
        "windows": len(windows),
        "rows": len(output),
        "feature_layout": "ACCEL plus normalized numeric horizon_index",
        "model": args,
        "calibration": "independent isotonic fit per horizon on calibration origins",
        "forecast_cutoff": str(CUTOFF.date()),
        "probability_range": [float(matrix.min()), float(matrix.max())],
        "calibration_details": calibration_details,
        "warning": "Provisional mask/order/windows; do not submit as official.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "calibration_details"}, indent=2))


if __name__ == "__main__":
    main()

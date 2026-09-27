#!/usr/bin/env python3
"""Validate the official Earthquake-Q grid/output contract.

The repository materials state the required dimensions but do not ship the
authoritative 1,551-row mask or an output-order table. This validator requires
that artifact explicitly and never reconstructs it from geographic bounds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CELLS = 1551
EXPECTED_WINDOWS = 15
EXPECTED_ROWS = EXPECTED_CELLS * EXPECTED_WINDOWS


def load_mask(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "cell_lat": "lat",
        "latitude": "lat",
        "cell_lon": "lon",
        "longitude": "lon",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items()})
    required = {"lat", "lon"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Official mask is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_CELLS:
        raise ValueError(f"Official mask has {len(frame)} rows; expected {EXPECTED_CELLS}")
    if frame[["lat", "lon"]].duplicated().any():
        raise ValueError("Official mask contains duplicate cell coordinates")
    if not np.allclose((frame.lat * 2) % 1, 0) or not np.allclose((frame.lon * 2) % 1, 0):
        raise ValueError("Official mask coordinates are not on a 0.5-degree grid")
    return frame.reset_index(drop=True)


def validate_predictions(
    predictions: pd.DataFrame, mask: pd.DataFrame, window_starts: pd.DatetimeIndex
) -> dict[str, int | bool]:
    required = {"lat", "lon", "window_start", "probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if len(predictions) != EXPECTED_ROWS:
        raise ValueError(f"Predictions have {len(predictions)} rows; expected {EXPECTED_ROWS}")
    if predictions[["lat", "lon", "window_start"]].duplicated().any():
        raise ValueError("Predictions contain duplicate cell-window rows")
    if not predictions.probability.between(0.0, 1.0).all():
        raise ValueError("Predictions contain probabilities outside [0, 1]")
    mask_keys = list(zip(mask.lat.round(6), mask.lon.round(6)))
    pred_keys = list(zip(predictions.lat.round(6), predictions.lon.round(6)))
    if set(pred_keys) != set(mask_keys):
        raise ValueError("Prediction cells do not exactly match the official mask")
    starts = pd.DatetimeIndex(
        pd.to_datetime(predictions.window_start).drop_duplicates().sort_values()
    )
    if not starts.equals(pd.DatetimeIndex(window_starts)):
        raise ValueError("Prediction windows do not match the supplied official window table")
    return {"valid": True, "cells": EXPECTED_CELLS, "windows": EXPECTED_WINDOWS, "rows": EXPECTED_ROWS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    mask = load_mask(args.mask)
    windows = pd.read_csv(args.windows)
    if "window_start" not in windows or len(windows) != EXPECTED_WINDOWS:
        raise ValueError("Official window table must contain exactly 15 window_start rows")
    window_starts = pd.to_datetime(windows.window_start).sort_values().reset_index(drop=True)
    if window_starts.duplicated().any() or window_starts.iloc[0] != pd.Timestamp("2025-01-01"):
        raise ValueError("Official window table must start at 2025-01-01 with unique dates")
    if not np.all(np.diff(window_starts.to_numpy()).astype("timedelta64[D]") == np.timedelta64(30, "D")):
        raise ValueError("Official windows must be consecutive 30-day starts")
    result = {
        "mask_valid": True,
        "cells": len(mask),
        "window_starts": [date.strftime("%Y-%m-%d") for date in window_starts],
        "expected_rows": EXPECTED_ROWS,
    }
    if args.predictions:
        result.update(validate_predictions(pd.read_csv(args.predictions), mask, window_starts))
    print(result)


if __name__ == "__main__":
    main()

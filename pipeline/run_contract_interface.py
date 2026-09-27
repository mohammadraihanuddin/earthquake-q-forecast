#!/usr/bin/env python3
"""Generate an ordered, mask-driven prediction grid.

This is deliberately independent of the exploratory experiment's ``main``.
The supplied mask and windows are authoritative: rows are emitted in
``mask row order × window row order`` and are never sorted.  Feature history
is restricted to timestamps strictly before each window start.  Any
preprocessing and model fitting performed by a caller must likewise use only
the fit period; this module does not load judge data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from run_multiscale_experiment import feature_names, make_examples


def load_mask(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    aliases = {"cell_lat": "lat", "latitude": "lat", "cell_lon": "lon",
               "longitude": "lon", "id": "mask_id", "cell_id": "mask_id"}
    frame = frame.rename(columns={k: v for k, v in aliases.items()})
    missing = {"lat", "lon"} - set(frame)
    if missing:
        raise ValueError(f"Mask is missing columns: {sorted(missing)}")
    if "mask_id" in frame and frame["mask_id"].duplicated().any():
        raise ValueError("Mask contains duplicate mask IDs")
    if "mask_id" in frame and frame["mask_id"].isna().any():
        raise ValueError("Mask contains missing mask IDs")
    if frame[["lat", "lon"]].isna().any().any():
        raise ValueError("Mask contains missing coordinates")
    if not np.isfinite(frame[["lat", "lon"]].to_numpy(dtype=float)).all():
        raise ValueError("Mask contains non-finite coordinates")
    if frame[["lat", "lon"]].duplicated().any():
        raise ValueError("Mask contains duplicate lat/lon coordinates")
    return frame.reset_index(drop=True)


def load_windows(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    if "window_start" not in frame:
        raise ValueError("Windows are missing the window_start column")
    frame = frame.copy()
    frame["window_start"] = pd.to_datetime(frame["window_start"], errors="raise")
    if frame["window_start"].duplicated().any():
        raise ValueError("Windows contain duplicate window_start values")
    return frame.reset_index(drop=True)


def contract_rows(mask: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Return the exact cell-major cross-product without sorting either input."""
    rows = []
    for cell in mask.itertuples(index=False):
        for window in windows.itertuples(index=False):
            row = {"lat": float(cell.lat), "lon": float(cell.lon),
                   "window_start": window.window_start}
            if hasattr(cell, "mask_id"):
                row["mask_id"] = cell.mask_id
            rows.append(row)
    return pd.DataFrame(rows)


def generate_examples(
    history: pd.DataFrame,
    mask: pd.DataFrame,
    windows: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    freeze_features_at: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate features and optional labels in contract row order.

    When ``freeze_features_at`` is supplied, one feature snapshot is computed
    at that cutoff and reused for every forecast window; labels remain
    window-specific.
    """
    required = {"time", "latitude", "longitude", "magnitude"}
    if not required.issubset(history.columns):
        raise ValueError(f"History is missing columns: {sorted(required - set(history))}")
    history = history.copy()
    history["time"] = pd.to_datetime(history["time"], errors="raise")
    if labels is None:
        labels = pd.DataFrame(columns=["time", "latitude", "longitude", "magnitude"])
    else:
        labels = labels.copy()
        if not required.issubset(labels.columns):
            raise ValueError(f"Labels are missing columns: {sorted(required - set(labels))}")
        labels["time"] = pd.to_datetime(labels["time"], errors="raise")
    cells = mask[["lat", "lon"]].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(windows["window_start"])
    n_cells, n_windows = len(mask), len(windows)
    # make_examples is date-major; reshape explicitly, then transpose to the
    # contract's cell-major ordering.
    if freeze_features_at is None:
        x_date, y_date = make_examples(history, labels, dates, cells)
    else:
        cutoff = pd.Timestamp(freeze_features_at)
        x_snapshot, _ = make_examples(
            history, pd.DataFrame(columns=labels.columns), pd.DatetimeIndex([cutoff]), cells
        )
        x_date = np.tile(x_snapshot, (n_windows, 1))
        _, y_date = make_examples(history, labels, dates, cells)
    x = x_date.reshape(n_windows, n_cells, -1).transpose(1, 0, 2).reshape(-1, x_date.shape[1])
    y = y_date.reshape(n_windows, n_cells).transpose(1, 0).reshape(-1)
    rows = contract_rows(mask, windows)
    return rows, x, y


def validate_predictions(predictions: pd.DataFrame, expected: pd.DataFrame) -> None:
    required = {"lat", "lon", "window_start", "probability"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if len(predictions) != len(expected):
        raise ValueError(f"Predictions have {len(predictions)} rows; expected {len(expected)}")
    actual = predictions[["lat", "lon", "window_start"]].copy()
    actual["window_start"] = pd.to_datetime(actual["window_start"], errors="raise")
    wanted = expected[["lat", "lon", "window_start"]].copy()
    wanted["window_start"] = pd.to_datetime(wanted["window_start"])
    if actual.reset_index(drop=True).equals(wanted.reset_index(drop=True)) is False:
        raise ValueError("Predictions do not match the ordered cells×windows cross-product")
    probabilities = pd.to_numeric(predictions["probability"], errors="coerce")
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError("Predictions contain invalid probabilities")


def run_synthetic_test() -> None:
    mask = pd.DataFrame({"mask_id": ["b", "a"], "lat": [1.0, 0.0], "lon": [2.0, 1.0]})
    windows = pd.DataFrame({"window_start": pd.to_datetime(["2020-02-01", "2020-01-01"])})
    events = pd.DataFrame({"time": pd.to_datetime(["2019-12-01", "2020-01-15"]),
        "latitude": [1.0, 0.0], "longitude": [2.0, 1.0], "magnitude": [2.5, 2.5]})
    rows, x, y = generate_examples(events, mask, windows, events)
    assert len(rows) == 4 and x.shape == (4, len(feature_names()))
    assert list(rows.mask_id) == ["b", "b", "a", "a"]
    assert list(rows.window_start.dt.strftime("%Y-%m-%d")) == [
        "2020-02-01", "2020-01-01", "2020-02-01", "2020-01-01"]
    probabilities = np.arange(4, dtype=float) / 10
    output = rows.assign(probability=probabilities)
    validate_predictions(output, rows)
    try:
        load_mask(pd.DataFrame({"mask_id": ["x", "x"], "lat": [0, 1], "lon": [0, 1]}))
    except (ValueError, FileNotFoundError):
        pass
    else:
        raise AssertionError("duplicate mask IDs were not rejected")
    try:
        load_windows(pd.DataFrame({"not_window_start": [1]}))
    except ValueError:
        pass
    else:
        raise AssertionError("missing window_start was not rejected")
    print("synthetic contract test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask")
    parser.add_argument("--windows")
    parser.add_argument("--catalog", help="fit-period event catalog CSV")
    parser.add_argument("--labels", help="optional labels catalog CSV")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_test:
        run_synthetic_test()
        return
    if not (args.mask and args.windows and args.catalog and args.predictions):
        parser.error("--mask, --windows, --catalog, and --predictions are required")
    mask, windows = load_mask(args.mask), load_windows(args.windows)
    history = pd.read_csv(args.catalog)
    labels = pd.read_csv(args.labels) if args.labels else None
    rows, x, _ = generate_examples(history, mask, windows, labels)
    if not 0 <= args.probability <= 1:
        parser.error("--probability must be between 0 and 1")
    output = rows.assign(probability=float(args.probability))
    validate_predictions(output, rows)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    output[["lat", "lon", "window_start", "probability"]].to_csv(args.predictions, index=False)
    if args.features:
        pd.DataFrame(x, columns=feature_names()).to_csv(args.features, index=False)
    print(f"wrote {len(output)} ordered predictions to {args.predictions}")


if __name__ == "__main__":
    main()

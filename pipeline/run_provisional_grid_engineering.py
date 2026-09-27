#!/usr/bin/env python3
"""Create and exercise a non-official 1,551-cell contract.

This artifact is for interface/performance testing only. It must never be used
as an official score or submission because the organizer mask/order is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_contract_interface import load_mask, load_windows, validate_predictions

WORK = Path(__file__).resolve().parent
DATA = WORK.parent / "data/raw"
OUT = WORK / "provisional_contract"
N_CELLS = 1551


def main() -> None:
    raw = pd.concat(
        [pd.read_csv(DATA / f"earthquakeq_{split}.csv", parse_dates=["time"])
         for split in ("train", "test")],
        ignore_index=True,
    )
    seismic = raw.loc[
        raw.source.eq("seismic"),
        ["time", "latitude", "longitude", "magnitude"],
    ].dropna()
    step = 0.5
    lats = np.arange(25.0, 50.0, step)
    lons = np.arange(-90.0, -65.0, step)
    grid = pd.DataFrame(
        [(lat, lon) for lat in lats for lon in lons],
        columns=["lat", "lon"],
    )
    event_cells = (
        seismic.assign(
            lat=np.floor(seismic.latitude / step) * step,
            lon=np.floor(seismic.longitude / step) * step,
        )
        .groupby(["lat", "lon"])
        .size()
        .rename("event_count")
        .reset_index()
    )
    grid = grid.merge(event_cells, on=["lat", "lon"], how="left").fillna({"event_count": 0})
    # Engineering-only deterministic expansion: active cells first, then
    # inactive cells in row-major order. This is explicitly not official.
    grid["active"] = grid.event_count > 0
    grid = grid.sort_values(["active", "event_count", "lat", "lon"],
                            ascending=[False, False, True, True]).head(N_CELLS)
    mask = grid[["lat", "lon"]].copy()
    mask.insert(0, "mask_id", [f"provisional_{i:04d}" for i in range(len(mask))])
    windows = pd.DataFrame({
        "window_start": pd.date_range("2025-01-01", periods=15, freq="30D")
    })
    OUT.mkdir(exist_ok=True)
    mask.to_csv(OUT / "provisional_mask.csv", index=False)
    windows.to_csv(OUT / "provisional_windows.csv", index=False)
    seismic.to_csv(OUT / "history_catalog.csv", index=False)
    # No judge labels are loaded; constant probabilities exercise only shape,
    # ordering, and feature-generation plumbing.
    from run_contract_interface import generate_examples, contract_rows
    rows, features, _ = generate_examples(seismic, mask, windows)
    output = rows.assign(probability=0.5)
    validate_predictions(output, rows)
    output.to_csv(OUT / "placeholder_predictions.csv", index=False)
    pd.DataFrame(features).to_csv(OUT / "placeholder_features.csv", index=False)
    metadata = {
        "official": False,
        "purpose": "engineering-only contract and performance test",
        "selection_rule": "active rectangle cells ranked by observed event count, then row-major tie-break",
        "cells": len(mask),
        "windows": len(windows),
        "rows": len(output),
        "judge_loaded": False,
        "warning": "Do not use for official scoring or submission.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

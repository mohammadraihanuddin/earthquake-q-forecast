#!/usr/bin/env python3
"""Stakeholder products from the corrected forecast, validated on test data.

1. 6-month (repair) look-ahead: P(>=1 event in windows 1-6) = 1 - prod(1 - p_h).
   Independence across windows gives an upper bound when events cluster
   (positive dependence), so it errs on the cautious side; checked on test.
2. Safety-threshold evidence: for each 30-day cut-off, the share of test
   cell-windows it would clear and the event rate actually observed in them.
3. Risk bands with observed test rates, and per-cell data for the map.

Uses only fit/calibration/test data and the corrected 2025 forecast.
Judge data is not read.  Outputs: stakeholder/stakeholder.json, .md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
RERUN = HERE / "rerun"
OUT = HERE / "stakeholder"
H, N = 15, 1551
SIX = 6
BANDS = [(0, 0.01, "Low"), (0.01, 0.05, "Guarded"), (0.05, 0.20, "Elevated"),
         (0.20, 0.50, "High"), (0.50, 1.01, "Very high")]


def ig_bits(y, p, base):
    return float((log_loss(y, np.full(len(y), base)) - log_loss(y, p)) / np.log(2))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pred = np.load(RERUN / "full_grid_development_comparison" / "test_predictions.npz")
    ex = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    p = pred["pooled_numeric_horizon_index"].reshape(-1, H, N)   # origin, horizon, cell
    y = pred["y"].reshape(-1, H, N)

    # --- 6-month look-ahead, validated on test origins -------------------
    p6 = np.clip(1 - np.prod(1 - p[:, :SIX, :], axis=1), 1e-4, 1 - 1e-4).ravel()
    y6 = y[:, :SIX, :].max(axis=1).ravel()
    y6_fit = ex["y_train"].reshape(-1, H, N)[:, :SIX, :].max(axis=1)
    base6 = float(y6_fit.mean())
    edges = np.quantile(p6, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, p6, side="right") - 1, 0, 9)
    rel6 = [{"mean_forecast": float(p6[idx == b].mean()), "observed": float(y6[idx == b].mean()),
             "n": int((idx == b).sum())} for b in range(10) if (idx == b).any()]
    six = {"test_rows": int(len(y6)), "observed_rate": float(y6.mean()), "mean_forecast": float(p6.mean()),
           "ig_bits_vs_fit_6m_rate": ig_bits(y6, p6, base6), "fit_6m_base_rate": base6,
           "reliability_decile": rel6}

    # --- 30-day bands and safety thresholds on test ----------------------
    p30, y30 = p.ravel(), y.ravel()
    bands = []
    for lo, hi, name in BANDS:
        m = (p30 >= lo) & (p30 < hi)
        bands.append({"band": name, "range": [lo, min(hi, 1.0)], "share_of_cell_windows": float(m.mean()),
                      "mean_forecast": float(p30[m].mean()), "observed_rate": float(y30[m].mean()),
                      "n": int(m.sum())})
    thresholds = []
    for t in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        clear = p30 < t
        thresholds.append({"threshold": t, "share_cleared": float(clear.mean()),
                           "observed_rate_in_cleared": float(y30[clear].mean()),
                           "share_of_all_events_in_cleared": float(y30[clear].sum() / y30.sum())})
    six_bands = []
    for lo, hi, name in BANDS:
        m = (p6 >= lo) & (p6 < hi)
        if m.any():
            six_bands.append({"band": name, "share": float(m.mean()), "mean_forecast": float(p6[m].mean()),
                              "observed_rate": float(y6[m].mean()), "n": int(m.sum())})

    # --- 2025 forecast per cell ------------------------------------------
    matrix = np.load(RERUN / "frozen_forecast_fixed" / "probability_matrix.npy")
    fc = pd.read_csv(RERUN / "frozen_forecast_fixed" / "predictions.csv")
    cells = fc.drop_duplicates("mask_id")[["mask_id", "lat", "lon"]].reset_index(drop=True)
    windows = sorted(fc.window_start.unique())
    six_2025 = 1 - np.prod(1 - matrix[:, :SIX], axis=1)
    cell_rows = [{"id": r.mask_id.replace("provisional_", ""), "lat": r.lat, "lon": r.lon,
                  "p": [round(float(v), 5) for v in matrix[i]], "p6": round(float(six_2025[i]), 5)}
                 for i, r in cells.iterrows()]

    result = {"six_month_test_validation": six, "bands_30d_test": bands, "bands_6m_test": six_bands,
              "thresholds_30d_test": thresholds, "windows": windows, "cells": cell_rows,
              "six_month_windows": [windows[0], windows[SIX - 1]]}
    (OUT / "stakeholder.json").write_text(json.dumps(result) + "\n")

    lines = ["# Stakeholder products — validated on 2021–2023 test origins", "",
             "## 6-month repair look-ahead", "",
             f"P(≥1 M≥2.0 within 100 km in the next 6 windows) = 1 − Π(1 − p_h), h = 1..6.",
             f"On {six['test_rows']:,} test cell-origins: mean forecast {six['mean_forecast']:.4f}, "
             f"observed {six['observed_rate']:.4f}, IG {six['ig_bits_vs_fit_6m_rate']:.4f} bits vs the fit-period 6-month rate.",
             "", "| Forecast decile mean | Observed | n |", "|---:|---:|---:|"]
    lines += [f"| {r['mean_forecast']:.4f} | {r['observed']:.4f} | {r['n']:,} |" for r in rel6]
    lines += ["", "## 30-day risk bands (test)", "", "| Band | Range | Share of cell-windows | Mean forecast | Observed rate |",
              "|---|---|---:|---:|---:|"]
    lines += [f"| {b['band']} | {b['range'][0]:.0%}–{b['range'][1]:.0%} | {b['share_of_cell_windows']:.1%} | "
              f"{b['mean_forecast']:.4f} | {b['observed_rate']:.4f} |" for b in bands]
    lines += ["", "## Entry thresholds (test)", "", "| Clear cells below | Share cleared | Event rate in cleared | Share of all events in cleared |",
              "|---:|---:|---:|---:|"]
    lines += [f"| {t['threshold']:.1%} | {t['share_cleared']:.1%} | {t['observed_rate_in_cleared']:.4f} | "
              f"{t['share_of_all_events_in_cleared']:.2%} |" for t in thresholds]
    (OUT / "stakeholder.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

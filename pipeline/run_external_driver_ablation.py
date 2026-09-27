#!/usr/bin/env python3
"""Run 021 full-grid, cutoff-safe external-driver ablation.

Only train/test long-format driver rows are read.  The provisional mask is
used in its supplied order, and every mask cell (including inactive cells) is
included in the 15-horizon historical protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from run_contract_interface import load_mask, load_windows
from run_full_grid_development_comparison import (
    ACCEL, HORIZONS, CONTRACT, DATA, SEED, catalog, make_split, metric,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "external_driver_ablation"
DRIVERS = ("geomagnetic", "solar", "tidal_ephemeris", "gps")
WINDOWS = (7, 30, 90, 365)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_driver_daily() -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        f = pd.read_csv(
            DATA / f"earthquakeq_{split}.csv",
            usecols=["source", "time", "station_id", "parameter", "value"],
            parse_dates=["time"],
        )
        frames.append(f[f.source.isin(DRIVERS)])
    raw = pd.concat(frames, ignore_index=True)
    raw["day"] = raw.time.dt.floor("D")
    gps = raw[raw.source.eq("gps")].copy()
    if not gps.empty:
        gps = (gps.groupby(["day", "source", "parameter", "station_id"],
                           as_index=False).value.mean())
        raw = pd.concat([raw[~raw.source.eq("gps")], gps], ignore_index=True)
    return raw.groupby(["day", "source", "parameter"], as_index=False).value.mean()


def external_features(daily: pd.DataFrame, dates: pd.DatetimeIndex):
    days = pd.date_range(daily.day.min(), daily.day.max(), freq="D")
    columns, names = [], []
    for source in DRIVERS:
        params = sorted(daily.loc[daily.source.eq(source), "parameter"].dropna().unique())
        for parameter in params:
            s = (daily[(daily.source == source) & (daily.parameter == parameter)]
                 .set_index("day").value.reindex(days).astype(float).ffill())
            prior = s.shift(1)  # strictly before the monthly origin
            for window in WINDOWS:
                columns += [prior.rolling(window, min_periods=1).mean().to_numpy(),
                            prior.rolling(window, min_periods=1).std().fillna(0).to_numpy()]
                names += [f"{source}_{parameter}_mean_{window}d",
                          f"{source}_{parameter}_std_{window}d"]
            columns.append((prior - prior.shift(30)).to_numpy())
            names.append(f"{source}_{parameter}_delta_30d")
    frame = pd.DataFrame(dict(zip(names, columns)), index=days)
    return frame.reindex(pd.DatetimeIndex(dates)).ffill().bfill().to_numpy(float), names


def paired_interval(y, challenger, reference, groups, reps=1000):
    """Paired log-likelihood improvement, resampling complete origin blocks."""
    challenger = np.clip(challenger, 1e-4, 1 - 1e-4)
    reference = np.clip(reference, 1e-4, 1 - 1e-4)
    delta = y * np.log(challenger) + (1-y) * np.log1p(-challenger)
    delta -= y * np.log(reference) + (1-y) * np.log1p(-reference)
    keys = np.unique(groups)
    blocks = [delta[groups == k] for k in keys]
    rng = np.random.default_rng(SEED)
    draws = np.array([np.concatenate([blocks[i] for i in
                         rng.integers(0, len(blocks), len(blocks))]).mean()
                      for _ in range(reps)])
    q = np.quantile(draws, [0.025, 0.975])
    return {"mean_log_loss_improvement": float(delta.mean()),
            "ci95_low": float(q[0]), "ci95_high": float(q[1]),
            "ci95_ig_bits_low": float(q[0] / np.log(2)),
            "ci95_ig_bits_high": float(q[1] / np.log(2))}


def main():
    OUT.mkdir(exist_ok=True)
    mask = load_mask(CONTRACT / "provisional_mask.csv")
    windows = load_windows(CONTRACT / "provisional_windows.csv")
    events = catalog()
    cells = mask[["lat", "lon"]].to_numpy(float)
    train_o = pd.date_range("2001-01-01", "2018-12-01", freq="MS")
    cal_o = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    test_o = pd.date_range("2021-01-01", "2023-09-01", freq="MS")
    xtr, ytr, htr, otr = make_split(events, train_o, cells)
    xca, yca, hca, oca = make_split(events, cal_o, cells)
    xte, yte, hte, ote = make_split(events, test_o, cells)
    daily = load_driver_daily()
    ext, names = external_features(daily, train_o.append(cal_o).append(test_o))
    n = len(cells)
    rows_per_origin = HORIZONS * n
    xtr = np.hstack([xtr, np.repeat(ext[:len(train_o)], rows_per_origin, axis=0)])
    j = len(train_o)
    xca = np.hstack([xca, np.repeat(ext[j:j+len(cal_o)], rows_per_origin, axis=0)])
    j += len(cal_o)
    xte = np.hstack([xte, np.repeat(ext[j:], rows_per_origin, axis=0)])
    base = list(ACCEL)
    base_count = xtr.shape[1] - ext.shape[1]
    source_indices = {s: [i for i, name in enumerate(names)
                          if name.startswith(s + "_")] for s in DRIVERS}
    sets = {"seismic_only": [], **source_indices,
            "all_external": list(range(len(names)))}
    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED)
    preds = {}
    for name, eidx in sets.items():
        idx = base + [base_count + i for i in eidx]
        model = HistGradientBoostingClassifier(**args).fit(xtr[:, idx], ytr)
        cal_raw = model.predict_proba(xca[:, idx])[:, 1]
        test_raw = model.predict_proba(xte[:, idx])[:, 1]
        preds[name] = np.clip(IsotonicRegression(out_of_bounds="clip")
                               .fit(cal_raw, yca).predict(test_raw), 1e-4, 1-1e-4)
    base_rate = float(ytr.mean())
    overall = [metric(k, yte, v, base_rate) for k, v in preds.items()]
    per_horizon = {str(h): [metric(k, yte[hte == h/(HORIZONS-1)],
                                    v[hte == h/(HORIZONS-1)], base_rate)
                            for k, v in preds.items()] for h in range(HORIZONS)}
    paired = {}
    origin_horizon = np.array([f"{o}:{h}" for o, h in zip(ote, hte)])
    for name, p in preds.items():
        paired[name] = {"reference": True} if name == "seismic_only" else {
            "origin": paired_interval(yte, p, preds["seismic_only"], ote),
            "origin_horizon_block": paired_interval(yte, p, preds["seismic_only"], origin_horizon),
        }
    result = {"protocol": {"run_reference": "Run 021 full-grid development",
        "official": False, "judge_loaded": False, "mask": str(CONTRACT/"provisional_mask.csv"),
        "mask_sha256": sha256(CONTRACT/"provisional_mask.csv"), "windows": str(CONTRACT/"provisional_windows.csv"),
        "windows_sha256": sha256(CONTRACT/"provisional_windows.csv"), "cells": len(mask),
        "inactive_cells_included": True, "contract_windows": len(windows), "horizons": HORIZONS,
        "train_origins": [str(train_o[0].date()), str(train_o[-1].date())],
        "calibration_origins": [str(cal_o[0].date()), str(cal_o[-1].date())],
        "test_origins": [str(test_o[0].date()), str(test_o[-1].date())],
        "feature_freeze": "strictly before each monthly origin",
        "external_input": "train/test only; daily station aggregation; prior-day shift",
        "model": args, "calibration": "single isotonic fit on calibration block",
        "seed": SEED}, "overall": overall, "per_horizon": per_horizon,
        "paired_origin_block_intervals": paired}
    (OUT/"external_driver_ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT/"test_predictions.npz", y=yte, horizon=hte,
                        origin=ote, **preds)
    print(json.dumps({"shapes": {k: list(v.shape) for k, v in
                                 np.load(OUT/"test_predictions.npz").items()},
                      "overall": overall}, indent=2))


if __name__ == "__main__":
    main()

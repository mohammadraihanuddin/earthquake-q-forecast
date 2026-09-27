#!/usr/bin/env python3
"""Contract-driven full-grid historical development comparison.

The provisional mask/window files are used only as an engineering contract.
No judge file is read.  Historical examples use frozen feature snapshots at
each monthly origin and strictly chronological fit/calibration/test blocks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from run_contract_interface import load_mask, load_windows
from run_multiscale_experiment import MC, TARGET_MAG, RADIUS_KM, LOOKBACK_DAYS, haversine_km

ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "provisional_contract"
DATA = ROOT.parent / "data/raw"
OUT = ROOT / "full_grid_development_comparison"
SEED = 42
HORIZONS = 15
# Compact snapshot layout: lat/lon, 100-km counts (7/30/90/365d),
# recency/max/mean/log-recency, then the two acceleration ratios.
STARTER = [0, 1, 2, 3, 4, 5, 9]
ACCEL = STARTER + [10, 11]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def catalog() -> pd.DataFrame:
    # Deliberately read only train and test; never judge.
    frames = []
    for split in ("train", "test"):
        f = pd.read_csv(DATA / f"earthquakeq_{split}.csv", usecols=["source", "time", "latitude", "longitude", "magnitude"], parse_dates=["time"])
        frames.append(f.loc[f.source.eq("seismic"), ["time", "latitude", "longitude", "magnitude"]])
    f = pd.concat(frames, ignore_index=True)
    for c in ("latitude", "longitude", "magnitude"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    return f.dropna().sort_values("time").reset_index(drop=True)


def snapshot(events: pd.DataFrame, origin: pd.Timestamp, cells: np.ndarray) -> np.ndarray:
    e = events.loc[(events.time < origin) & (events.magnitude >= MC)]
    lat = e.latitude.to_numpy(float); lon = e.longitude.to_numpy(float)
    mag = e.magnitude.to_numpy(float)
    t = e.time.to_numpy(dtype="datetime64[s]").astype("int64")
    cutoff = origin.to_datetime64().astype("datetime64[s]").astype("int64")
    ages = (cutoff - t) / 86400.0
    out = np.empty((len(cells), 10), dtype=np.float32)
    # Chunking keeps the distance matrix bounded while retaining vectorized math.
    for start in range(0, len(cells), 128):
        stop = min(start + 128, len(cells))
        clat, clon = cells[start:stop, 0], cells[start:stop, 1]
        p1 = np.radians(clat[:, None]); p2 = np.radians(lat[None, :])
        a = np.sin(np.radians(lat[None, :] - clat[:, None]) / 2) ** 2
        a += np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon[None, :] - clon[:, None]) / 2) ** 2
        dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        vals = np.zeros((stop - start, 10), dtype=np.float32)
        vals[:, :2] = cells[start:stop]
        counts = {}
        for j, radius in enumerate(RADIUS_KM):
            for k, days in enumerate(LOOKBACK_DAYS):
                counts[(radius, days)] = np.sum((dist <= radius) & (ages[None, :] <= days), axis=1)
        vals[:, 2:6] = np.column_stack([counts[(100.0, d)] for d in LOOKBACK_DAYS])
        near = dist <= max(RADIUS_KM)
        for j in range(stop - start):
            lm = mag[near[j]]
            la = ages[near[j]]
            vals[j, 6] = la.min() if la.size else 999.0
            vals[j, 7] = lm.max() if lm.size else MC
            vals[j, 8] = lm.mean() if lm.size else MC
            vals[j, 9] = np.log1p(vals[j, 6])
        out[start:stop] = vals
    ratios = np.column_stack([
        out[:, 2] / (out[:, 3] + 1e-6),
        out[:, 3] / (out[:, 5] + 1e-6),
    ])
    return np.column_stack([out, ratios]).astype(np.float32)


def labels(events: pd.DataFrame, origin: pd.Timestamp, horizon: int, cells: np.ndarray) -> np.ndarray:
    start = origin + pd.Timedelta(days=30 * horizon)
    end = start + pd.Timedelta(days=30)
    f = events.loc[(events.time > start) & (events.time <= end) & (events.magnitude >= TARGET_MAG)]
    if f.empty:
        return np.zeros(len(cells), dtype=np.int8)
    result = np.zeros(len(cells), dtype=np.int8)
    for i in range(0, len(cells), 256):
        d = np.stack([haversine_km(a, b, f.latitude.to_numpy(float), f.longitude.to_numpy(float)) for a, b in cells[i:i+256]])
        result[i:i+len(d)] = np.any(d <= 100.0, axis=1)
    return result


def make_split(events, origins, cells):
    xs, ys, hs, os = [], [], [], []
    for origin in origins:
        x0 = snapshot(events, origin, cells)
        for h in range(HORIZONS):
            xs.append(x0)
            ys.append(labels(events, origin, h, cells))
            hs.append(np.full(len(cells), h / (HORIZONS - 1), dtype=np.float32))
            os.append(np.full(len(cells), origin.value, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(hs), np.concatenate(os)


def calibrate(raw, y_cal, raw_test, h_cal, h_test):
    p = np.empty(len(raw_test), float)
    for h in range(HORIZONS):
        a, b = h_cal == h / (HORIZONS - 1), h_test == h / (HORIZONS - 1)
        if np.unique(y_cal[a]).size > 1:
            p[b] = IsotonicRegression(out_of_bounds="clip").fit(raw[a], y_cal[a]).predict(raw_test[b])
        else:
            p[b] = raw_test[b]
    return np.clip(p, 1e-4, 1 - 1e-4)


def metric(name, y, p, base):
    return {"model": name, "n": int(len(y)), "positives": int(y.sum()), "log_loss": float(log_loss(y, p)),
            "ig_bits": float((log_loss(y, np.full(len(y), base)) - log_loss(y, p)) / np.log(2)),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def paired_bootstrap(y, challenger, reference, origins, reps=1000):
    deltas = -(y * np.log(reference) + (1-y) * np.log1p(-reference)) + (y * np.log(challenger) + (1-y) * np.log1p(-challenger))
    unique = np.unique(origins); grouped = [deltas[origins == x] for x in unique]
    rng = np.random.default_rng(SEED)
    means = [np.concatenate([grouped[i] for i in rng.integers(0, len(grouped), len(grouped))]).mean() for _ in range(reps)]
    q = np.quantile(means, [0.025, 0.975])
    return {"mean_log_loss_improvement": float(deltas.mean()), "ci95_low": float(q[0]), "ci95_high": float(q[1]),
            "ci95_ig_bits_low": float(q[0] / np.log(2)), "ci95_ig_bits_high": float(q[1] / np.log(2))}


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
    np.savez_compressed(OUT / "historical_examples.npz", x_train=xtr, y_train=ytr, h_train=htr, x_cal=xca, y_cal=yca, h_cal=hca, x_test=xte, y_test=yte, h_test=hte, origin_test=ote)
    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
    preds = {}
    # The first two are horizon-specific models (the direct historical
    # baseline); pooled variants share one model across all horizons.
    for name, idx in [("starter", STARTER), ("starter_acceleration", ACCEL)]:
        p = np.empty(len(yte))
        for h in range(HORIZONS):
            a, b = htr == h / (HORIZONS - 1), hca == h / (HORIZONS - 1)
            c = hte == h / (HORIZONS - 1)
            model = HistGradientBoostingClassifier(**args).fit(xtr[a][:, idx], ytr[a])
            cal_raw = model.predict_proba(xca[b][:, idx])[:, 1]
            test_raw = model.predict_proba(xte[c][:, idx])[:, 1]
            p[c] = calibrate(cal_raw, yca[b], test_raw, hca[b], hte[c])
        preds[name] = p
    for name, idx, with_h in [("pooled_no_horizon", ACCEL, False), ("pooled_numeric_horizon_index", ACCEL, True)]:
        model = HistGradientBoostingClassifier(**args).fit(xtr[:, idx] if not with_h else np.column_stack([xtr[:, idx], htr]), ytr)
        cal_raw = model.predict_proba(xca[:, idx] if not with_h else np.column_stack([xca[:, idx], hca]))[:, 1]
        test_raw = model.predict_proba(xte[:, idx] if not with_h else np.column_stack([xte[:, idx], hte]))[:, 1]
        preds[name] = calibrate(cal_raw, yca, test_raw, hca, hte)
    base = float(ytr.mean()); preds["climatology"] = np.full(len(yte), base)
    metrics = [metric(n, yte, preds[n], base) for n in ["climatology", "starter", "starter_acceleration", "pooled_no_horizon", "pooled_numeric_horizon_index"]]
    per_horizon = {str(h): [metric(n, yte[hte == h/(HORIZONS-1)], preds[n][hte == h/(HORIZONS-1)], base) for n in preds] for h in range(HORIZONS)}
    paired = {"starter_vs_climatology": paired_bootstrap(yte, preds["starter"], preds["climatology"], ote),
              "starter_acceleration_vs_climatology": paired_bootstrap(yte, preds["starter_acceleration"], preds["climatology"], ote),
              "pooled_numeric_horizon_index_vs_no_horizon": paired_bootstrap(yte, preds["pooled_numeric_horizon_index"], preds["pooled_no_horizon"], ote)}
    result = {"protocol": {"official": False, "judge_loaded": False, "mask": str(CONTRACT / "provisional_mask.csv"), "mask_sha256": sha256(CONTRACT / "provisional_mask.csv"), "windows": str(CONTRACT / "provisional_windows.csv"), "windows_sha256": sha256(CONTRACT / "provisional_windows.csv"), "cells": len(mask), "inactive_cells_included": True, "contract_windows": len(windows), "horizons": HORIZONS, "train_origins": [str(train_o[0].date()), str(train_o[-1].date())], "calibration_origins": [str(cal_o[0].date()), str(cal_o[-1].date())], "test_origins": [str(test_o[0].date()), str(test_o[-1].date())], "feature_freeze": "strictly before each monthly origin", "seed": SEED}, "overall": metrics, "per_horizon": per_horizon, "paired_origin_bootstrap": paired}
    (OUT / "comparison_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "test_predictions.npz", y=yte, horizon=hte, origin=ote, **preds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

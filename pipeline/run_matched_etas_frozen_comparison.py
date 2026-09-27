#!/usr/bin/env python3
"""Matched full-grid ETAS versus the frozen provisional candidate.

This is a development diagnostic on the provisional 1,551-cell contract.
Both models use the exact historical rows in
``full_grid_development_comparison/historical_examples.npz`` and independent
per-horizon isotonic calibration fit only on the calibration rows.  The judge
catalog is intentionally never read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from run_contract_interface import load_mask, load_windows
from run_full_grid_development_comparison import ACCEL, HORIZONS, catalog

ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "provisional_contract"
EXAMPLES = ROOT / "full_grid_development_comparison" / "historical_examples.npz"
ETAS_RESULTS = ROOT / "etas_experiment" / "results.json"
OUT = ROOT / "matched_etas_frozen_comparison"
DAY = 86400.0
RADIUS = 100.0
MC = 1.75
TARGET_SCALE = float(json.loads(ETAS_RESULTS.read_text())["target_scale"])
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hav(lat, lon, lats, lons):
    a = np.sin(np.radians(lats - lat) / 2) ** 2
    a += np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(np.radians(lons - lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def omori(age, horizon, c, p):
    if np.isclose(p, 1.0):
        return np.log((age + horizon + c) / (age + c))
    return ((age + c) ** (1 - p) - (age + horizon + c) ** (1 - p)) / (p - 1)


def etas_raw(events, cells, origins, theta):
    """Return origin-major, horizon-major, cell-minor probabilities."""
    _, K, alpha, c, p = theta
    e = events.loc[events.magnitude >= MC]
    et = e.time.to_numpy(dtype="datetime64[s]").astype("int64") / DAY
    ela, elo = e.latitude.to_numpy(float), e.longitude.to_numpy(float)
    emag = e.magnitude.to_numpy(float)
    # Background uses the same fitted 2001--2019 activity-share convention.
    fit = e[(e.time >= "2001-01-01") & (e.time < "2020-01-01")]
    flat_t = fit.time.to_numpy(dtype="datetime64[s]").astype("int64") / DAY
    flat_la, flat_lo = fit.latitude.to_numpy(float), fit.longitude.to_numpy(float)
    share = np.array([np.sum(hav(a, b, flat_la, flat_lo) <= RADIUS) / len(fit)
                      for a, b in cells])
    background = theta[0] * share * 30.0
    out = []
    for origin in origins:
        cut = origin.value / 1e9 / DAY
        past = et < cut
        ages = cut - et[past]
        weights = K * np.exp(alpha * (emag[past] - MC))
        latp, lonp = ela[past], elo[past]
        triggers = np.zeros((len(cells), HORIZONS), dtype=float)
        for start in range(0, len(cells), 128):
            stop = min(start + 128, len(cells))
            dist = np.stack([hav(a, b, latp, lonp) for a, b in cells[start:stop]])
            use = dist <= RADIUS
            for h in range(HORIZONS):
                age = ages + h * 30.0
                triggers[start:stop, h] = np.sum(
                    np.where(use, weights * omori(age, 30.0, c, p), 0.0), axis=1
                )
        expected = (background[:, None] + triggers) * TARGET_SCALE
        out.append((1.0 - np.exp(-np.maximum(expected, 0))).T.reshape(-1))
    return np.concatenate(out)


def calibrate(raw_cal, y_cal, raw_test, h_cal, h_test):
    result = np.empty(len(raw_test), dtype=float)
    for h in range(HORIZONS):
        a, b = h_cal == h / (HORIZONS - 1), h_test == h / (HORIZONS - 1)
        result[b] = IsotonicRegression(out_of_bounds="clip").fit(
            raw_cal[a], y_cal[a]
        ).predict(raw_test[b])
    return np.clip(result, 1e-4, 1 - 1e-4)


def metric(name, y, p, base):
    return {
        "model": name, "n": int(len(y)), "positives": int(y.sum()),
        "log_loss": float(log_loss(y, p)),
        "ig_bits": float((log_loss(y, np.full(len(y), base)) - log_loss(y, p)) / np.log(2)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def paired_bootstrap(y, challenger, reference, origins, reps=1000):
    delta = -(y * np.log(reference) + (1 - y) * np.log1p(-reference))
    delta += y * np.log(challenger) + (1 - y) * np.log1p(-challenger)
    groups = [delta[origins == x] for x in np.unique(origins)]
    rng = np.random.default_rng(SEED)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(reps)]
    q = np.quantile(means, [0.025, 0.975])
    return {"mean_log_loss_improvement": float(delta.mean()),
            "ci95_ig_bits_low": float(q[0] / np.log(2)),
            "ci95_ig_bits_high": float(q[1] / np.log(2))}


def main():
    mask = load_mask(CONTRACT / "provisional_mask.csv")
    windows = load_windows(CONTRACT / "provisional_windows.csv")
    z = np.load(EXAMPLES)
    events = catalog()
    cells = mask[["lat", "lon"]].to_numpy(float)
    cal_h, test_h = z["h_cal"], z["h_test"]
    origins = z["origin_test"]
    params = json.loads(ETAS_RESULTS.read_text())["parameters"]
    theta = np.array([params["mu_global_day"], params["K"], params["alpha"], params["c_days"], params["p"]])

    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED)
    model = HistGradientBoostingClassifier(**args).fit(
        np.column_stack([z["x_train"][:, ACCEL], z["h_train"]]), z["y_train"])
    frozen_cal_raw = model.predict_proba(np.column_stack([z["x_cal"][:, ACCEL], z["h_cal"]]))[:, 1]
    frozen_test_raw = model.predict_proba(np.column_stack([z["x_test"][:, ACCEL], z["h_test"]]))[:, 1]
    frozen_cal = calibrate(frozen_cal_raw, z["y_cal"], frozen_cal_raw, z["h_cal"], z["h_cal"])
    frozen_test = calibrate(frozen_cal_raw, z["y_cal"], frozen_test_raw, z["h_cal"], z["h_test"])

    cal_origins = np.unique(z["origin_test"])  # overwritten below from the exact example blocks
    # historical_examples stores no calibration origin array; infer monthly ordering.
    ncell = len(mask)
    cal_origins = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    test_origins = pd.date_range("2021-01-01", "2023-09-01", freq="MS")
    etas_cal_raw = etas_raw(events, cells, cal_origins, theta)
    etas_test_raw = etas_raw(events, cells, test_origins, theta)
    etas_cal = calibrate(etas_cal_raw, z["y_cal"], etas_cal_raw, z["h_cal"], z["h_cal"])
    etas_test = calibrate(etas_cal_raw, z["y_cal"], etas_test_raw, z["h_cal"], z["h_test"])

    base = float(z["y_train"].mean())
    metrics = {
        "frozen_candidate": metric("frozen_candidate", z["y_test"], frozen_test, base),
        "etas": metric("etas", z["y_test"], etas_test, base),
    }
    comparison = paired_bootstrap(z["y_test"], etas_test, frozen_test, origins)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "predictions.npz", y=z["y_test"], origin=origins,
                        horizon=z["h_test"], frozen_candidate=frozen_test, etas=etas_test,
                        etas_raw=etas_test_raw)
    report = {
        "protocol": {
            "official": False, "judge_loaded": False, "cells": len(mask),
            "windows": len(windows), "rows": int(len(z["y_test"])),
            "calibration": "independent isotonic regression per horizon on 2019-01..2020-12",
            "test": "same 1,551-cell, 15-horizon historical examples as frozen candidate",
            "fit_origins": "2001-01..2018-12", "test_origins": "2021-01..2023-09",
            "mask_sha256": sha256(CONTRACT / "provisional_mask.csv"),
            "windows_sha256": sha256(CONTRACT / "provisional_windows.csv"),
        },
        "metrics": metrics, "etas_vs_frozen_paired_origin_bootstrap": comparison,
        "etas_approximation": {
            "parameters_source": str(ETAS_RESULTS), "background": "fitted global mu times local pre-2020 M>=1.75 activity share",
            "kernel": "hard 100-km parents; unnormalized Omori analytically integrated over each 30-day horizon",
            "magnitude_scale": "M>=1.75 forecast scaled to M>=2.0 using fitted regional b=0.846",
            "limitation": "ETAS parameters were fitted on the earlier 644-cell diagnostic and reused; no spatial likelihood refit or exact organizer mask exists",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    md = f"""# Matched ETAS vs frozen candidate

Development-only comparison on the provisional **1,551-cell × 15-horizon** contract ({len(z['y_test']):,} identical historical rows). Judge data was not read. Both methods use the same fit/calibration/test origins and independent per-horizon isotonic calibration.

| Model | IG (bits) | Log loss | Brier | AUC |
|---|---:|---:|---:|---:|
| Frozen candidate | {metrics['frozen_candidate']['ig_bits']:.6f} | {metrics['frozen_candidate']['log_loss']:.6f} | {metrics['frozen_candidate']['brier']:.6f} | {metrics['frozen_candidate']['auc']:.6f} |
| ETAS approximation | {metrics['etas']['ig_bits']:.6f} | {metrics['etas']['log_loss']:.6f} | {metrics['etas']['brier']:.6f} | {metrics['etas']['auc']:.6f} |

ETAS minus frozen candidate paired-origin bootstrap: **{comparison['mean_log_loss_improvement'] / np.log(2):+.6f} IG bits**, 95% interval [{comparison['ci95_ig_bits_low']:+.6f}, {comparison['ci95_ig_bits_high']:+.6f}].

The ETAS parameters are reused from `etas_experiment/results.json` (the earlier 644-cell fit). The approximation uses a hard 100-km parent filter, the unnormalized analytically integrated Omori kernel, local empirical background shares, and the recorded Gutenberg–Richter scaling. It is not exact organizer ETAS parity: the authoritative mask/order/window artifacts are absent, and parameters were not refit to the provisional full grid. Raw catalogs and the frozen model were not modified.
"""
    (OUT / "report.md").write_text(md)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

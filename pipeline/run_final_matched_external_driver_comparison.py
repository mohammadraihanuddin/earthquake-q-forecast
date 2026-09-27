#!/usr/bin/env python3
"""Final matched external-driver comparison for the numeric-horizon candidate.

This reruns the six pooled models on the exact Run 021 rows, preserving the
numeric horizon feature and changing only the calibration from the earlier
external-driver artifact to one isotonic fit per horizon. Test uncertainty is
paired and resamples contiguous three-origin blocks; every horizon remains in
its origin block.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from run_contract_interface import load_mask, load_windows
from run_external_driver_ablation import (
    DRIVERS, WINDOWS, external_features, load_driver_daily,
)
from run_full_grid_development_comparison import (
    ACCEL, HORIZONS, SEED, catalog, make_split, metric,
)

ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "provisional_contract"
DATA = ROOT.parent / "data/raw"
EXAMPLES = ROOT / "full_grid_development_comparison/historical_examples.npz"
OUT = ROOT / "final_matched_external_driver_comparison"
REPS = 2000
BLOCK_SIZE_ORIGINS = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def per_horizon_isotonic(raw_cal, y_cal, h_cal, raw_test, h_test):
    """Calibrate each horizon using calibration rows from that horizon only."""
    calibrated = np.empty(len(raw_test), dtype=float)
    details = []
    for h in range(HORIZONS):
        cal = h_cal == h / (HORIZONS - 1)
        test = h_test == h / (HORIZONS - 1)
        if np.unique(y_cal[cal]).size > 1:
            calibrated[test] = IsotonicRegression(
                out_of_bounds="clip"
            ).fit(raw_cal[cal], y_cal[cal]).predict(raw_test[test])
            method = "isotonic"
        else:
            calibrated[test] = raw_test[test]
            method = "raw_fallback_single_class"
        details.append({"horizon": h, "calibration_rows": int(cal.sum()),
                        "calibration_positives": int(y_cal[cal].sum()),
                        "method": method})
    return np.clip(calibrated, 1e-4, 1 - 1e-4), details


def contiguous_origin_bootstrap(y, challenger, reference, origins):
    """Paired LL improvement, with all 15 horizons kept inside each origin."""
    challenger = np.clip(challenger, 1e-4, 1 - 1e-4)
    reference = np.clip(reference, 1e-4, 1 - 1e-4)
    row_delta = (
        y * np.log(challenger) + (1 - y) * np.log1p(-challenger)
        - y * np.log(reference) - (1 - y) * np.log1p(-reference)
    )
    unique = np.unique(origins)
    # One value per origin is the mean over cells and all horizons.
    origin_delta = np.asarray(
        [row_delta[origins == origin].mean() for origin in unique]
    )
    blocks = [
        origin_delta[i:i + BLOCK_SIZE_ORIGINS]
        for i in range(0, len(origin_delta), BLOCK_SIZE_ORIGINS)
    ]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(REPS):
        selected = rng.integers(0, len(blocks), int(np.ceil(len(origin_delta) / BLOCK_SIZE_ORIGINS)))
        draws.append(float(np.concatenate([blocks[i] for i in selected])[:len(origin_delta)].mean()))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "origins": int(len(unique)),
        "rows_per_origin": int(np.sum(origins == unique[0])),
        "horizons_per_origin": HORIZONS,
        "block_size_origins": BLOCK_SIZE_ORIGINS,
        "repetitions": REPS,
        "mean_log_loss_improvement_nats": float(origin_delta.mean()),
        "mean_ig_improvement_bits": float(origin_delta.mean() / np.log(2)),
        "ci95_log_loss_improvement_nats_low": float(low),
        "ci95_log_loss_improvement_nats_high": float(high),
        "ci95_ig_improvement_bits_low": float(low / np.log(2)),
        "ci95_ig_improvement_bits_high": float(high / np.log(2)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    mask = load_mask(CONTRACT / "provisional_mask.csv")
    windows = load_windows(CONTRACT / "provisional_windows.csv")
    if len(mask) != 1551 or len(windows) != HORIZONS:
        raise ValueError("Run 021 contract dimensions changed")
    events = catalog()
    cells = mask[["lat", "lon"]].to_numpy(float)
    train_o = np.arange(np.datetime64("2001-01"), np.datetime64("2019-01"),
                         np.timedelta64(1, "M")).astype("datetime64[ns]")
    cal_o = np.arange(np.datetime64("2019-01"), np.datetime64("2021-01"),
                      np.timedelta64(1, "M")).astype("datetime64[ns]")
    test_o = np.arange(np.datetime64("2021-01"), np.datetime64("2023-10"),
                       np.timedelta64(1, "M")).astype("datetime64[ns]")
    train_o, cal_o, test_o = map(pd.DatetimeIndex, (train_o, cal_o, test_o))

    # Reuse the already generated Run 021 seismic rows; this is read-only and
    # also makes exact row identity auditable against the prior artifact.
    z = np.load(EXAMPLES)
    xtr, ytr, htr = z["x_train"], z["y_train"], z["h_train"]
    xca, yca, hca = z["x_cal"], z["y_cal"], z["h_cal"]
    xte, yte, hte, ote = z["x_test"], z["y_test"], z["h_test"], z["origin_test"]
    expected_tr = len(train_o) * HORIZONS * len(mask)
    expected_ca = len(cal_o) * HORIZONS * len(mask)
    expected_te = len(test_o) * HORIZONS * len(mask)
    if (len(ytr), len(yca), len(yte)) != (expected_tr, expected_ca, expected_te):
        raise ValueError("Historical examples do not match Run 021 row dimensions")
    daily = load_driver_daily()
    dates = train_o.append(cal_o).append(test_o)
    ext, names = external_features(daily, dates)
    rows_per_origin = HORIZONS * len(mask)
    xtr = np.hstack([xtr, np.repeat(ext[:len(train_o)], rows_per_origin, axis=0)])
    j = len(train_o)
    xca = np.hstack([xca, np.repeat(ext[j:j + len(cal_o)], rows_per_origin, axis=0)])
    j += len(cal_o)
    xte = np.hstack([xte, np.repeat(ext[j:], rows_per_origin, axis=0)])

    base_count = xtr.shape[1] - ext.shape[1]
    source_indices = {
        source: [i for i, name in enumerate(names) if name.startswith(source + "_")]
        for source in DRIVERS
    }
    feature_sets = {"seismic_only": [], **source_indices,
                    "all_external": list(range(len(names)))}
    model_args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                      l2_regularization=1.0, random_state=SEED)
    predictions, calibration = {}, {}
    for name, external_idx in feature_sets.items():
        idx = list(ACCEL) + [base_count + i for i in external_idx]
        train_features = np.column_stack([xtr[:, idx], htr])
        calibration_features = np.column_stack([xca[:, idx], hca])
        test_features = np.column_stack([xte[:, idx], hte])
        model = HistGradientBoostingClassifier(**model_args).fit(
            train_features, ytr
        )
        raw_cal = model.predict_proba(calibration_features)[:, 1]
        raw_test = model.predict_proba(test_features)[:, 1]
        predictions[name], calibration[name] = per_horizon_isotonic(
            raw_cal, yca, hca, raw_test, hte
        )

    base_rate = float(ytr.mean())
    names_order = list(feature_sets)
    overall = [metric(name, yte, predictions[name], base_rate) for name in names_order]
    per_horizon = {
        str(h): [
            metric(name, yte[hte == h / (HORIZONS - 1)],
                   predictions[name][hte == h / (HORIZONS - 1)], base_rate)
            for name in names_order
        ]
        for h in range(HORIZONS)
    }
    paired = {
        name: {"reference": True} if name == "seismic_only" else
        contiguous_origin_bootstrap(
            yte, predictions[name], predictions["seismic_only"], ote
        )
        for name in names_order
    }
    protocol = {
        "run": "021",
        "official": False,
        "judge_loaded": False,
        "raw_inputs_read": ["earthquakeq_train.csv", "earthquakeq_test.csv"],
        "mask": str(CONTRACT / "provisional_mask.csv"),
        "mask_sha256": sha256(CONTRACT / "provisional_mask.csv"),
        "mask_order": "supplied provisional row order",
        "windows": str(CONTRACT / "provisional_windows.csv"),
        "windows_sha256": sha256(CONTRACT / "provisional_windows.csv"),
        "cells": len(mask), "inactive_cells_included": True,
        "horizons": HORIZONS, "rows_per_split": {
            "train": int(len(ytr)), "calibration": int(len(yca)), "test": int(len(yte))
        },
        "train_origins": [str(train_o[0].date()), str(train_o[-1].date())],
        "calibration_origins": [str(cal_o[0].date()), str(cal_o[-1].date())],
        "test_origins": [str(test_o[0].date()), str(test_o[-1].date())],
        "feature_freeze": "strictly before each monthly origin",
        "seismic_features": "ACCEL (starter + two acceleration ratios) plus numeric horizon_index",
        "external_features": "daily source/parameter means; prior-day shift; 7/30/90/365d mean/std and 30d delta",
        "external_sources": list(DRIVERS),
        "model": model_args, "calibration": "independent isotonic fit per horizon on calibration origins only",
        "bootstrap": "paired contiguous 3-origin blocks; all cells and all 15 horizons together within each origin",
        "seed": SEED,
    }
    result = {"protocol": protocol, "overall": overall,
              "per_horizon": per_horizon,
              "paired_contiguous_origin_block_bootstrap": paired,
              "calibration_details": calibration}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "test_predictions.npz", y=yte, horizon=hte,
                        origin=ote, **predictions)
    print(json.dumps({"artifact": str(OUT), "overall": overall,
                      "paired": paired}, indent=2))


if __name__ == "__main__":
    main()

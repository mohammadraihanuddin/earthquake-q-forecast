#!/usr/bin/env python3
"""Compare starter and multi-scale seismic features on a chronological split.

This is an exploratory development run on the starter's 644 active cells. It
does not produce the official 1,551-cell submission.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR.parent / "data/raw"
OUT_DIR = WORK_DIR / "multiscale_experiment"
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
MC = 1.75
TARGET_MAG = 2.0
RADIUS_KM = (50.0, 100.0, 200.0)
LOOKBACK_DAYS = (7.0, 30.0, 90.0, 365.0)
DAY_SECONDS = 86400.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def haversine_km(lat: float, lon: float, event_lat: np.ndarray, event_lon: np.ndarray) -> np.ndarray:
    radius = 6371.0
    p1 = np.radians(lat)
    p2 = np.radians(event_lat)
    dphi = np.radians(event_lat - lat)
    dlambda = np.radians(event_lon - lon)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_catalog() -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        frame = pd.read_csv(DATA_DIR / f"earthquakeq_{split}.csv", parse_dates=["time"])
        seismic = frame.loc[
            frame["source"] == "seismic",
            ["time", "latitude", "longitude", "depth_km", "magnitude"],
        ].copy()
        frames.append(seismic)
    catalog = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    return catalog


def active_cells(catalog: pd.DataFrame) -> np.ndarray:
    cell_lat = np.floor(catalog["latitude"].to_numpy() / 0.5) * 0.5
    cell_lon = np.floor(catalog["longitude"].to_numpy() / 0.5) * 0.5
    cells = sorted(set(zip(np.round(cell_lat, 3), np.round(cell_lon, 3))))
    return np.asarray(cells, dtype=float)


def issue_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS")


def make_examples(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    event_times = history["time"].to_numpy(dtype="datetime64[s]").astype("int64")
    event_lat = history["latitude"].to_numpy(dtype=float)
    event_lon = history["longitude"].to_numpy(dtype=float)
    event_mag = history["magnitude"].to_numpy(dtype=float)
    label_times = labels["time"].to_numpy(dtype="datetime64[s]").astype("int64")
    label_lat = labels["latitude"].to_numpy(dtype=float)
    label_lon = labels["longitude"].to_numpy(dtype=float)
    label_mag = labels["magnitude"].to_numpy(dtype=float)

    rows: list[list[float]] = []
    targets: list[int] = []
    for date in dates:
        cutoff = np.datetime64(date).astype("datetime64[s]").astype("int64")
        past = event_times < cutoff
        past_age = (cutoff - event_times[past]) / DAY_SECONDS
        past_distances = [
            haversine_km(lat, lon, event_lat[past], event_lon[past])
            for lat, lon in cells
        ]
        future = (label_times > cutoff) & (
            label_times <= cutoff + int(30 * DAY_SECONDS)
        )
        future_distances = [
            haversine_km(lat, lon, label_lat[future], label_lon[future])
            for lat, lon in cells
        ]

        for index, (lat, lon) in enumerate(cells):
            distances = past_distances[index]
            near = distances <= np.max(RADIUS_KM)
            ages = past_age[near]
            magnitudes = event_mag[past][near]
            values: list[float] = [lat, lon]
            counts: dict[tuple[float, float], int] = {}
            for radius in RADIUS_KM:
                for lookback in LOOKBACK_DAYS:
                    count = int(np.sum((distances <= radius) & (past_age <= lookback)))
                    counts[(radius, lookback)] = count
                    values.append(float(count))
            values.extend(
                [
                    float(np.min(ages)) if ages.size else 999.0,
                    float(np.max(magnitudes)) if magnitudes.size else MC,
                    float(np.mean(magnitudes)) if magnitudes.size else MC,
                    float(np.log1p(np.min(ages))) if ages.size else np.log1p(999.0),
                ]
            )
            local_30 = counts[(100.0, 30.0)] / 30.0
            local_365 = counts[(100.0, 365.0)] / 365.0
            local_7 = counts[(100.0, 7.0)] / 7.0
            values.extend(
                [
                    local_7 / (local_30 + 1e-6),
                    local_30 / (local_365 + 1e-6),
                    np.log10(np.e) / (np.mean(magnitudes) - MC)
                    if magnitudes.size and np.mean(magnitudes) > MC
                    else 0.0,
                ]
            )
            rows.append(values)
            future_near = future_distances[index] <= 100.0
            targets.append(int(np.any(future_near & (label_mag[future] >= TARGET_MAG))))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=int)


def feature_names() -> list[str]:
    names = ["cell_lat", "cell_lon"]
    names.extend(
        f"count_{int(radius)}km_{int(days)}d"
        for radius in RADIUS_KM
        for days in LOOKBACK_DAYS
    )
    return names + [
        "recency_days",
        "max_magnitude",
        "mean_magnitude",
        "log_recency",
        "rate_ratio_7d_30d",
        "rate_ratio_30d_365d",
        "b_value",
    ]


def information_gain(y: np.ndarray, probabilities: np.ndarray, base_rate: float) -> float:
    return float((log_loss(y, np.full(len(y), base_rate)) - log_loss(y, probabilities)) / np.log(2.0))


def ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right)
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(probabilities[mask].mean()))
    return float(total)


def score(name: str, y: np.ndarray, probabilities: np.ndarray, base_rate: float) -> dict[str, float | int | str]:
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    return {
        "split": name,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(base_rate),
        "ig_bits": information_gain(y, probabilities, base_rate),
        "auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "ece": ece(y, probabilities),
    }


def main() -> None:
    catalog = load_catalog()
    complete_catalog = catalog[catalog["magnitude"] >= MC].reset_index(drop=True)
    cells = active_cells(catalog[catalog["time"] < "2022-01-01"])
    train_catalog = complete_catalog[complete_catalog["time"] < "2020-01-01"]
    test_catalog = complete_catalog[complete_catalog["time"] < "2025-01-01"]
    train_dates = issue_dates("2001-01-01", "2019-12-01")
    calibration_dates = issue_dates("2020-01-01", "2021-12-01")
    test_dates = issue_dates("2022-01-01", "2024-11-01")

    # Labels are deliberately restricted to each block's observed future catalog.
    x_train, y_train = make_examples(train_catalog, catalog, train_dates, cells)
    x_calibration, y_calibration = make_examples(
        complete_catalog[complete_catalog["time"] < "2022-01-01"],
        catalog,
        calibration_dates,
        cells,
    )
    x_test, y_test = make_examples(test_catalog, catalog, test_dates, cells)

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    )
    model.fit(x_train, y_train)
    raw_calibration = model.predict_proba(x_calibration)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_calibration, y_calibration)
    p_calibration = np.clip(calibrator.predict(raw_calibration), 1e-4, 1.0 - 1e-4)
    p_test = np.clip(calibrator.predict(raw_test), 1e-4, 1.0 - 1e-4)
    base_rate = float(y_train.mean())

    starter_indices = [0, 1, 6, 7, 8, 9, 17]
    starter_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    )
    starter_model.fit(x_train[:, starter_indices], y_train)
    starter_calibration = starter_model.predict_proba(x_calibration[:, starter_indices])[:, 1]
    starter_test = starter_model.predict_proba(x_test[:, starter_indices])[:, 1]
    starter_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        starter_calibration, y_calibration
    )

    report = {
        "config": {
            "seed": SEED,
            "grid": "starter active cells (exploratory, not official 1551-cell mask)",
            "radii_km": RADIUS_KM,
            "lookbacks_days": LOOKBACK_DAYS,
            "fit_dates": ["2001-01-01", "2019-12-01"],
            "calibration_dates": ["2020-01-01", "2021-12-01"],
            "test_dates": ["2022-01-01", "2024-11-01"],
            "target_magnitude": TARGET_MAG,
            "feature_completeness_magnitude": MC,
        },
        "inputs": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (DATA_DIR / "earthquakeq_train.csv", DATA_DIR / "earthquakeq_test.csv")
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dimensions": {
            "cells": int(len(cells)),
            "features": int(x_train.shape[1]),
            "train": list(x_train.shape),
            "calibration": list(x_calibration.shape),
            "test": list(x_test.shape),
        },
        "feature_names": feature_names(),
        "metrics": [
            score("calibration", y_calibration, p_calibration, base_rate),
            score("test", y_test, p_test, base_rate),
            score(
                "test_starter_features",
                y_test,
                np.clip(
                    starter_calibrator.predict(starter_test), 1e-4, 1.0 - 1e-4
                ),
                base_rate,
            ),
        ],
    }
    (OUT_DIR / "config_and_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUT_DIR / "test_predictions.npz",
        y=y_test,
        probability=p_test,
    )
    print(json.dumps(report["dimensions"], indent=2))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()

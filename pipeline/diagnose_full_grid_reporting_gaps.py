#!/usr/bin/env python3
"""Compute the missing Study Deck diagnostics on the frozen provisional grid.

This script consumes only the saved historical full-grid artifacts.  It never
loads a judge file, refits the selected model, or modifies raw data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
ART = ROOT / "full_grid_development_comparison"
PREDICTIONS = ART / "test_predictions.npz"
EXAMPLES = ART / "historical_examples.npz"
OUT_JSON = ART / "reporting_gap_diagnostics.json"
OUT_MD = ART / "reporting_gap_diagnostics.md"
N_CELLS, N_HORIZONS = 1551, 15
EPS = 1e-4


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def metrics(y: np.ndarray, p: np.ndarray, base: float) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    out = {
        "n": int(y.size),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p)),
        "ig_bits": float((log_loss(y, np.full(y.size, base)) - log_loss(y, p)) / np.log(2)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10_equal_width": ece(y, p),
    }
    out["auc"] = float(roc_auc_score(y, p)) if np.unique(y).size > 1 else None
    out["pr_auc"] = float(average_precision_score(y, p)) if np.unique(y).size > 1 else None
    return out


def main() -> None:
    pred = np.load(PREDICTIONS, allow_pickle=False)
    ex = np.load(EXAMPLES, allow_pickle=False)
    y = pred["y"].astype(np.int8)
    h = np.rint(pred["horizon"] * (N_HORIZONS - 1)).astype(int)
    origin = pred["origin"]
    if y.size != N_CELLS * N_HORIZONS * 33:
        raise ValueError(f"unexpected test row count: {y.size}")
    if not np.all(np.bincount(h, minlength=N_HORIZONS) == y.size // N_HORIZONS):
        raise ValueError("test horizons are not balanced")
    # The saved examples are ordered origin, horizon, cell.  Fit a deliberately
    # simple spatial prior from fit origins only, with Laplace smoothing.
    yh = ex["y_train"].astype(np.int8).reshape(-1, N_HORIZONS, N_CELLS)
    spatial_rate = (yh.sum(axis=0) + 1.0) / (yh.shape[0] + 2.0)
    spatial_prior = spatial_rate[h, np.arange(y.size) % N_CELLS]
    global_rate = float(ex["y_train"].mean())
    models = {
        "candidate_pooled_numeric_horizon_index": pred["pooled_numeric_horizon_index"],
        "climatology_saved": pred["climatology"],
        "spatial_prior_train_cell_horizon": spatial_prior,
    }
    groups = {"early": list(range(0, 5)), "middle": list(range(5, 10)), "late": list(range(10, 15))}
    result = {
        "protocol": {
            "predictions": str(PREDICTIONS),
            "historical_examples": str(EXAMPLES),
            "judge_loaded": False,
            "raw_data_modified": False,
            "cells": N_CELLS,
            "horizons": N_HORIZONS,
            "test_origins": int(np.unique(origin).size),
            "ece_definition": "10 equal-width probability bins, weighted by bin mass",
            "spatial_prior": "Laplace-smoothed positive rate per cell and horizon, fit on x_train/y_train only",
        },
        "overall": {},
        "per_horizon": {},
        "horizon_groups": {},
    }
    for name, p in models.items():
        result["overall"][name] = metrics(y, p, global_rate)
        result["per_horizon"][name] = {
            str(i): metrics(y[h == i], p[h == i], float(y[h == i].mean()))
            for i in range(N_HORIZONS)
        }
        result["horizon_groups"][name] = {
            group: metrics(y[np.isin(h, hs)], p[np.isin(h, hs)], float(y[np.isin(h, hs)].mean()))
            for group, hs in groups.items()
        }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Full-grid Study Deck reporting-gap diagnostics",
        "",
        "Computed from the saved 1,551-cell provisional historical test artifact. "
        "No judge data was loaded, model selection was not changed, and raw data was not modified.",
        "",
        f"- Test rows: **{y.size:,}** ({N_CELLS} cells × {N_HORIZONS} horizons × {np.unique(origin).size} origins)",
        "- ECE: 10 equal-width bins, weighted by bin mass.",
        "- Skill groups: early horizons 0–4, middle 5–9, late 10–14.",
        "- Spatial prior: Laplace-smoothed cell-and-horizon event rate fit only on `y_train`.",
        "",
        "## Overall",
        "",
        "| Model | Log loss | IG (bits) | Brier | ECE | AUC | PR AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in result["overall"].items():
        lines.append(f"| {name} | {m['log_loss']:.6f} | {m['ig_bits']:.6f} | {m['brier']:.6f} | {m['ece_10_equal_width']:.6f} | {m['auc'] if m['auc'] is not None else 'n/a'} | {m['pr_auc'] if m['pr_auc'] is not None else 'n/a'} |")
    lines += ["", "## Early / middle / late skill and calibration", ""]
    for group in groups:
        lines += [f"### {group}", "", "| Model | Log loss | IG (bits) | Brier | ECE |", "|---|---:|---:|---:|---:|"]
        for name, vals in result["horizon_groups"].items():
            m = vals[group]
            lines.append(f"| {name} | {m['log_loss']:.6f} | {m['ig_bits']:.6f} | {m['brier']:.6f} | {m['ece_10_equal_width']:.6f} |")
        lines.append("")
    lines += ["## Per-horizon ECE (candidate)", "", "| Horizon | ECE | Log loss | IG (bits) |", "|---:|---:|---:|---:|"]
    for i in range(N_HORIZONS):
        m = result["per_horizon"]["candidate_pooled_numeric_horizon_index"][str(i)]
        lines.append(f"| {i} | {m['ece_10_equal_width']:.6f} | {m['log_loss']:.6f} | {m['ig_bits']:.6f} |")
    lines += ["", "Machine-readable output: `" + str(OUT_JSON) + "`.", ""]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"json": str(OUT_JSON), "markdown": str(OUT_MD), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()

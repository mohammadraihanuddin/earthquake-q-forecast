#!/usr/bin/env python3
"""One-time judge self-check of the frozen candidate (corrected forecast).

The forecast was produced before this script ran and is only read here.
Labels: any M>=2.0 judge-catalog event within 100 km of the cell centre with
window_start < time <= window_start + 30 days (same rule as training labels).
Reference rate for IG = fit-period base rate, as for the test headline.
Outputs: judge_check/judge_metrics.json, judge_check.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RERUN = HERE / "rerun"
OUT = HERE / "judge_check"
DATA = ROOT.parent / "data/raw"
H, N = 15, 1551


def hav(lat, lon, lats, lons):
    a = np.sin(np.radians(lats - lat) / 2) ** 2
    a += np.cos(np.radians(lat)) * np.cos(np.radians(lats)) * np.sin(np.radians(lons - lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean()) for b in range(bins) if (i == b).any()))


def metrics(y, p, base):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"n": int(len(y)), "positives": int(y.sum()), "positive_rate": float(y.mean()),
            "ig_bits": float((log_loss(y, np.full(len(y), base), labels=[0, 1]) - log_loss(y, p, labels=[0, 1])) / np.log(2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else None, "ece": ece(y, p)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fc = pd.read_csv(RERUN / "frozen_forecast_fixed" / "predictions.csv", parse_dates=["window_start"])
    old_path = ROOT / "frozen_provisional_forecast" / "predictions.csv"   # original (misaligned) file, if present
    old = pd.read_csv(old_path, parse_dates=["window_start"]) if old_path.exists() else None
    j = pd.read_csv(DATA / "earthquakeq_judge.csv", usecols=["source", "time", "latitude", "longitude", "magnitude"],
                    parse_dates=["time"])
    j = j[(j.source == "seismic") & (j.magnitude >= 2.0)]
    starts = sorted(fc.window_start.unique())
    cells = fc.drop_duplicates("mask_id")[["mask_id", "lat", "lon"]].reset_index(drop=True)
    lab = {}
    for s in starts:
        e = j[(j.time > s) & (j.time <= s + pd.Timedelta(days=30))]
        la, lo = e.latitude.to_numpy(float), e.longitude.to_numpy(float)
        hit = np.array([bool(len(la)) and (hav(a, b, la, lo) <= 100).any() for a, b in zip(cells.lat, cells.lon)])
        lab.update({(m, s): int(h) for m, h in zip(cells.mask_id, hit)})
    y = np.array([lab[(m, s)] for m, s in zip(fc.mask_id, fc.window_start)])
    h = np.tile(np.arange(H), N)

    ex = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    base = float(ex["y_train"].mean())
    yh = ex["y_train"].reshape(-1, H, N)
    spatial = ((yh.sum(axis=0) + 1) / (yh.shape[0] + 2))[h, np.repeat(np.arange(N), H)]

    res = {"run_at": time.strftime("%Y-%m-%d %H:%M %Z"), "use": "single final judge self-check of the frozen candidate",
           "judge_m2_events": int(len(j)), "windows": [str(s.date()) for s in starts], "fit_base_rate": base,
           "frozen_candidate_corrected": metrics(y, fc.probability.to_numpy(), base),
           "recorded_file_misaligned": metrics(y, old.probability.to_numpy(), base) if old is not None else None,
           "spatial_prior": metrics(y, spatial, base),
           "per_window": [metrics(y[h == i], fc.probability.to_numpy()[h == i], base) for i in range(H)],
           "test_reference": {"ig_bits": 0.14582367854116265, "auc": 0.9061, "ece": 0.0071}}
    c = res["frozen_candidate_corrected"]
    res["ig_vs_judge_rate"] = float((log_loss(y, np.full(len(y), y.mean())) - c["log_loss"]) / np.log(2))
    p6 = 1 - np.prod(1 - fc.probability.to_numpy().reshape(N, H)[:, :6], axis=1)
    y6 = y.reshape(N, H)[:, :6].max(axis=1)
    base6 = float(yh[:, :6, :].max(axis=1).mean())
    res["six_month"] = metrics(y6, p6, base6)
    (OUT / "judge_metrics.json").write_text(json.dumps(res, indent=2) + "\n")

    r, o, sp = c, res["recorded_file_misaligned"], res["spatial_prior"]
    md = [f"# Judge self-check — run once, {res['run_at']}", "",
          f"{c['n']:,} cell-windows, {c['positives']:,} positive ({c['positive_rate']:.2%}); {res['judge_m2_events']} M≥2.0 judge events.", "",
          "| Forecast | IG (bits) | Log loss | Brier | AUC | ECE |", "|---|---:|---:|---:|---:|---:|",
          f"| Frozen candidate (corrected) | {r['ig_bits']:.4f} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['auc']:.4f} | {r['ece']:.4f} |",
          f"| Spatial prior | {sp['ig_bits']:.4f} | {sp['log_loss']:.4f} | {sp['brier']:.4f} | {sp['auc']:.4f} | {sp['ece']:.4f} |",
          *([f"| Recorded file (misaligned) | {o['ig_bits']:.4f} | {o['log_loss']:.4f} | {o['brier']:.4f} | {o['auc']:.4f} | {o['ece']:.4f} |"] if o else []),
          "", f"Test (2021–2023) for comparison: IG 0.1458, AUC 0.906, ECE 0.0071.",
          f"IG vs the judge period's own rate: {res['ig_vs_judge_rate']:.4f} bits.",
          f"6-month (windows 1–6): IG {res['six_month']['ig_bits']:.4f} bits, forecast mean vs observed "
          f"{p6.mean():.4f} / {y6.mean():.4f}.", "", "| Window | IG | AUC | ECE | Positives |", "|---:|---:|---:|---:|---:|"]
    md += [f"| {i + 1} | {m['ig_bits']:.4f} | {m['auc'] if m['auc'] is None else round(m['auc'], 4)} | {m['ece']:.4f} | {m['positives']} |"
           for i, m in enumerate(res["per_window"])]
    (OUT / "judge_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

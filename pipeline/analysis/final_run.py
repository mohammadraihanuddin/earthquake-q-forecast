#!/usr/bin/env python3
"""Final run of the frozen pipeline with a full presentation log.

Reads the rerun historical examples (regenerated from raw catalogs by
``rerun_all.py``), refits the frozen model once, and records everything a
presenter needs: environment, data provenance and distributions, model
hyper-parameters and learned size, training curve, calibration, test metrics,
reliability, feature importance, and the corrected 2025 forecast summary.

Outputs (in ``final_run/``): final_run.log, final_summary.json.
Judge data is never read.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RERUN = HERE / "rerun"
OUT = HERE / "final_run"
sys.path.insert(0, str(ROOT))

from run_full_grid_development_comparison import ACCEL, DATA, catalog  # noqa: E402

H = 15
N_CELLS = 1551
SEED = 42
FEATURES = [
    "lat", "lon", "count_7d_100km", "count_30d_100km", "count_90d_100km",
    "count_365d_100km", "log1p_days_since_last", "ratio_7d_30d", "ratio_30d_365d",
    "horizon_index",
]
MODEL_ARGS = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                  l2_regularization=1.0, random_state=SEED)
SPLITS = {
    "fit": ("2001-01-01", "2018-12-01"),
    "calibration": ("2019-01-01", "2020-12-01"),
    "test": ("2021-01-01", "2023-09-01"),
}

log = logging.getLogger("final_run")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out, total = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
            out.append({"bin": b, "n": int(m.sum()), "mean_p": float(p[m].mean()),
                        "observed": float(y[m].mean())})
    return float(total), out


def metrics(y, p, base):
    return {
        "n": int(len(y)), "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "ig_bits": float((log_loss(y, np.full(len(y), base)) - log_loss(y, p)) / np.log(2)),
        "log_loss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "ece": ece(y, p)[0],
    }


def origins(split):
    return pd.date_range(*SPLITS[split], freq="MS")


def section(title):
    log.info("")
    log.info("=" * 72)
    log.info(title)
    log.info("=" * 72)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.FileHandler(OUT / "final_run.log", mode="w"), logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    summary: dict = {}

    section("Environment")
    env = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "python": platform.python_version(),
           "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "seed": SEED}
    for k, v in env.items():
        log.info(f"{k:>14}: {v}")
    summary["environment"] = env

    section("Raw data provenance (judge file hashed only, never parsed)")
    provenance = {p.name: sha256(p) for p in sorted(DATA.glob("earthquakeq_*.csv"))}
    for k, v in provenance.items():
        log.info(f"{k:>24}  sha256 {v}")
    summary["raw_sha256"] = provenance

    events = catalog()
    ev = events.assign(year=events.time.dt.year)
    by_year = ev.groupby("year").agg(events=("magnitude", "size"),
                                     m2_plus=("magnitude", lambda m: int((m >= 2.0).sum())))
    mags = events.magnitude.to_numpy()
    hist_edges = np.arange(np.floor(mags.min() * 2) / 2, np.ceil(mags.max() * 2) / 2 + 0.5, 0.25)
    summary["catalog"] = {
        "seismic_events": int(len(events)),
        "time_range": [str(events.time.min()), str(events.time.max())],
        "magnitude_range": [float(mags.min()), float(mags.max())],
        "events_by_year": by_year.reset_index().to_dict(orient="records"),
        "magnitude_hist": {"edges": hist_edges.tolist(),
                           "counts": np.histogram(mags, hist_edges)[0].tolist()},
        "lat_lon": events[["latitude", "longitude", "magnitude"]].round(3).to_numpy().tolist(),
    }
    log.info(f"Seismic events (train+test catalogs): {len(events):,}  "
             f"{events.time.min():%Y-%m-%d} .. {events.time.max():%Y-%m-%d}  "
             f"M {mags.min():.2f}..{mags.max():.2f}")

    section("Dataset: cell x origin x horizon examples")
    z = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    data = {s: (z[f"x_{k}"], z[f"y_{k}"], z[f"h_{k}"]) for s, k in
            [("fit", "train"), ("calibration", "cal"), ("test", "test")]}
    dist = {}
    for s, (x, y, h) in data.items():
        o = origins(s)
        hi = np.rint(h * 14).astype(int)
        per_h = [float(y[hi == i].mean()) for i in range(H)]
        yo = y.reshape(len(o), H, N_CELLS)
        per_year = pd.Series(yo.mean(axis=(1, 2)), index=o.year).groupby(level=0).mean()
        dist[s] = {"origins": [str(o[0].date()), str(o[-1].date())], "n_origins": len(o),
                   "rows": int(len(y)), "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
                   "positive_rate": float(y.mean()), "positive_rate_by_horizon": per_h,
                   "positive_rate_by_origin_year": {int(k): float(v) for k, v in per_year.items()}}
        log.info(f"{s:>11}: {o[0]:%Y-%m}..{o[-1]:%Y-%m} ({len(o)} origins) rows={len(y):>9,} "
                 f"positives={int(y.sum()):>7,} rate={y.mean():.4f} "
                 f"(h0 {per_h[0]:.4f} .. h14 {per_h[-1]:.4f})")
    xtr = z["x_train"][:, ACCEL]
    feat_stats = {}
    for j, name in enumerate(FEATURES[:-1]):
        col = xtr[:, j]
        feat_stats[name] = {q: float(np.percentile(col, v)) for q, v in
                            [("p01", 1), ("p25", 25), ("median", 50), ("p75", 75), ("p99", 99)]}
        feat_stats[name]["mean"] = float(col.mean())
        feat_stats[name]["zero_share"] = float((col == 0).mean())
        log.info(f"  {name:>24}: median {feat_stats[name]['median']:.3g}  p99 "
                 f"{feat_stats[name]['p99']:.3g}  zero-share {feat_stats[name]['zero_share']:.2f}")
    summary["dataset"] = {"cells": N_CELLS, "horizons": H, "target": "any M>=2.0 within 100 km in the 30-day window",
                          "splits": dist, "fit_feature_stats": feat_stats}

    section("Model: frozen pipeline")
    fit_x = np.column_stack([xtr, z["h_train"]])
    model = HistGradientBoostingClassifier(**MODEL_ARGS).fit(fit_x, z["y_train"])
    params = model.get_params()
    trees = [p[0] for p in model._predictors]
    leaves = [int(t.nodes["is_leaf"].sum()) for t in trees]
    nodes = [len(t.nodes) for t in trees]
    depth = [int(t.nodes["depth"].max()) for t in trees]
    train_curve = (-model.train_score_).tolist()        # scoring='loss' stores negative loss
    val_curve = (-model.validation_score_).tolist() if model.do_early_stopping_ else []
    size = {"boosting_iterations_run": int(model.n_iter_), "max_iter_configured": MODEL_ARGS["max_iter"],
            "early_stopping_active": bool(model.do_early_stopping_),
            "trees": len(trees), "leaves_total": int(sum(leaves)), "nodes_total": int(sum(nodes)),
            "split_nodes_total": int(sum(nodes) - sum(leaves)),
            "learned_parameters_approx": int(sum(nodes) - sum(leaves)) * 2 + int(sum(leaves)),
            "max_depth_reached": int(max(depth)), "mean_leaves_per_tree": float(np.mean(leaves)),
            "input_features": FEATURES}
    for k in ("loss", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "max_depth",
              "l2_regularization", "max_bins", "early_stopping", "validation_fraction",
              "n_iter_no_change", "tol", "scoring", "random_state"):
        log.info(f"{k:>22}: {params[k]}")
    for k, v in size.items():
        if k != "input_features":
            log.info(f"{k:>30}: {v}")
    log.info("Training curve (log loss, nats): iter  train  early-stopping-validation")
    for i in list(range(0, len(train_curve), 10)) + [len(train_curve) - 1]:
        v = f"{val_curve[i]:.5f}" if val_curve else "n/a"
        log.info(f"    {i:>4}  {train_curve[i]:.5f}  {v}")
    summary["model"] = {"algorithm": "sklearn HistGradientBoostingClassifier (pooled over 15 horizons)",
                        "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                            for k, v in params.items()},
                        "learned_size": size,
                        "training_curve": {"train_log_loss": train_curve, "validation_log_loss": val_curve},
                        "batching": "none: histogram gradient boosting uses the full fit set each iteration "
                                    "(no mini-batches, no epochs); 1 boosting iteration = 1 tree"}

    section("Calibration: one isotonic regression per horizon (calibration origins only)")
    raw_cal = model.predict_proba(np.column_stack([z["x_cal"][:, ACCEL], z["h_cal"]]))[:, 1]
    raw_te = model.predict_proba(np.column_stack([z["x_test"][:, ACCEL], z["h_test"]]))[:, 1]
    hc, ht = np.rint(z["h_cal"] * 14).astype(int), np.rint(z["h_test"] * 14).astype(int)
    p = np.empty(len(raw_te))
    cal_info = []
    for h in range(H):
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal[hc == h], z["y_cal"][hc == h])
        p[ht == h] = iso.predict(raw_te[ht == h])
        cal_info.append({"horizon": h, "rows": int((hc == h).sum()), "positives": int(z["y_cal"][hc == h].sum()),
                         "breakpoints": int(len(iso.X_thresholds_))})
        log.info(f"  h{h:>2}: rows {cal_info[-1]['rows']:,} positives {cal_info[-1]['positives']:,} "
                 f"breakpoints {cal_info[-1]['breakpoints']}")
    p = np.clip(p, 1e-4, 1 - 1e-4)
    summary["calibration"] = cal_info

    section("Test results (2021-01..2023-09 origins; reference = fit-period base rate)")
    yte = z["y_test"]
    base = float(z["y_train"].mean())
    headline = metrics(yte, p, base)
    headline["ig_bits_vs_test_rate"] = float(
        (log_loss(yte, np.full(len(yte), yte.mean())) - log_loss(yte, p)) / np.log(2))
    for k, v in headline.items():
        log.info(f"{k:>22}: {v}")
    recorded = 0.14582367854116265
    log.info(f"Matches recorded headline IG {recorded}: {abs(headline['ig_bits'] - recorded) < 1e-9}")
    per_h = [metrics(yte[ht == h], p[ht == h], base) for h in range(H)]
    log.info("Per horizon: h  IG(bits)  AUC  ECE  positive-rate")
    for h, m in enumerate(per_h):
        log.info(f"  {h:>2}  {m['ig_bits']:.4f}  {m['auc']:.4f}  {m['ece']:.4f}  {m['positive_rate']:.4f}")
    reliability = ece(yte, p, bins=10)[1]
    # Finer quantile reliability for the plot.
    qs = np.quantile(p, np.linspace(0, 1, 21))
    qi = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, 19)
    rel_q = [{"mean_p": float(p[qi == b].mean()), "observed": float(yte[qi == b].mean()), "n": int((qi == b).sum())}
             for b in range(20) if (qi == b).any()]
    summary["test"] = {"headline": headline, "per_horizon": per_h, "reliability_10bin": reliability,
                       "reliability_quantile20": rel_q,
                       "probability_hist": np.histogram(p, np.linspace(0, 1, 41))[0].tolist()}
    preds = np.load(RERUN / "full_grid_development_comparison" / "test_predictions.npz")
    same = np.allclose(preds["pooled_numeric_horizon_index"], p)
    log.info(f"Identical to rerun comparison predictions: {same}")

    section("Permutation importance (test, 200k random rows, 3 repeats, log-loss increase)")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(yte), 200_000, replace=False)
    xs = np.column_stack([z["x_test"][sub][:, ACCEL], z["h_test"][sub]])
    ys = yte[sub]
    def ll(xm):
        r = model.predict_proba(xm)[:, 1]
        q = np.empty(len(r))
        hs = np.rint(xm[:, -1] * 14).astype(int)
        for h in range(H):
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal[hc == h], z["y_cal"][hc == h])
            q[hs == h] = iso.predict(r[hs == h])
        return log_loss(ys, np.clip(q, 1e-4, 1 - 1e-4))
    base_ll = ll(xs)
    importance = {}
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(3):
            xp = xs.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            drops.append((ll(xp) - base_ll) / np.log(2))
        importance[name] = {"mean_bits": float(np.mean(drops)), "std_bits": float(np.std(drops))}
        log.info(f"  {name:>24}: +{importance[name]['mean_bits']:.4f} bits")
    summary["permutation_importance_bits"] = importance

    for extra in (comparisons, forecast):
        try:
            extra(summary)
        except FileNotFoundError as exc:
            log.info(f"SKIPPED {extra.__name__}: rerun artifact not ready ({exc.filename})")

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    log.info("")
    log.info(f"Done in {summary['runtime_seconds']} s -> {OUT}")


def comparisons(summary: dict) -> None:
    section("Comparison models (from rerun artifacts)")
    comp ={r["model"]: r for r in json.loads(
        (RERUN / "full_grid_development_comparison" / "comparison_metrics.json").read_text())["overall"]}
    ext = json.loads((RERUN / "final_matched_external_driver_comparison" / "metrics.json").read_text())
    etas = json.loads((RERUN / "matched_etas_frozen_comparison" / "metrics.json").read_text())
    gaps = json.loads((RERUN / "full_grid_development_comparison" / "reporting_gap_diagnostics.json").read_text())
    qml = json.loads((RERUN / "day7_qml_ablation" / "metrics.json").read_text())
    table = [
        ("Frozen candidate (pooled HGB + horizon)", comp["pooled_numeric_horizon_index"]["ig_bits"], None),
        ("Pooled HGB, no horizon feature", comp["pooled_no_horizon"]["ig_bits"],
         json.loads((RERUN / "full_grid_development_comparison" / "comparison_metrics.json").read_text())
         ["paired_origin_bootstrap"]["pooled_numeric_horizon_index_vs_no_horizon"]),
        ("15 per-horizon HGB (starter + accel)", comp["starter_acceleration"]["ig_bits"], None),
        ("15 per-horizon HGB (starter)", comp["starter"]["ig_bits"], None),
        ("ETAS approximation", etas["metrics"]["etas"]["ig_bits"], etas["etas_vs_frozen_paired_origin_bootstrap"]),
        ("Spatial prior (cell x horizon rate)", gaps["overall"]["spatial_prior_train_cell_horizon"]["ig_bits"], None),
        ("Climatology (fit base rate)", comp["climatology"]["ig_bits"], None),
    ]
    for name, ig, _ in table:
        log.info(f"  {name:<42} {ig:.6f} bits")
    driver_rows = {r["model"]: r for r in ext["overall"]}
    paired = ext["paired_contiguous_origin_block_bootstrap"]
    log.info("External drivers (IG bits; paired 3-origin block 95% CI of difference vs seismic-only):")
    for name, r in driver_rows.items():
        ci = "" if name == "seismic_only" else (f"[{paired[name]['ci95_ig_improvement_bits_low']:+.5f}, "
                                                 f"{paired[name]['ci95_ig_improvement_bits_high']:+.5f}]")
        log.info(f"  {name:<16} {r['ig_bits']:.6f} {ci}")
    log.info(f"Day-7 QML sample ablation IG: quantum {qml['qml_quantum_features']['ig_bits']:.6f}, "
             f"classical control {qml['classical_two_input_control']['ig_bits']:.6f} (24k-row sample)")
    summary["comparison"] = {
        "models": [{"model": n, "ig_bits": ig,
                    "paired": ({"low": c.get("ci95_ig_bits_low"), "high": c.get("ci95_ig_bits_high")} if c else None)}
                   for n, ig, c in table],
        "external": [{"model": n, "ig_bits": r["ig_bits"], "log_loss": r["log_loss"], "brier": r["brier"],
                      "ci_low": None if n == "seismic_only" else paired[n]["ci95_ig_improvement_bits_low"],
                      "ci_high": None if n == "seismic_only" else paired[n]["ci95_ig_improvement_bits_high"]}
                     for n, r in driver_rows.items()],
        "qml": qml,
    }


def forecast(summary: dict) -> None:
    section("2025 forecast (corrected row alignment, provisional contract)")
    fc = pd.read_csv(RERUN / "frozen_forecast_fixed" / "predictions.csv")
    matrix = np.load(RERUN / "frozen_forecast_fixed" / "probability_matrix.npy")
    by_window = fc.groupby("window_start").probability.agg(["mean", "median", "max"])
    for w, r in by_window.iterrows():
        log.info(f"  {w}: mean {r['mean']:.4f} median {r['median']:.4f} max {r['max']:.4f}")
    cells = fc.drop_duplicates("mask_id")[["mask_id", "lat", "lon"]].reset_index(drop=True)
    cells["mean_probability"] = matrix.mean(axis=1)
    top = cells.sort_values("mean_probability", ascending=False).head(10)
    log.info("  Top-10 cells by mean 15-window probability:")
    for _, r in top.iterrows():
        log.info(f"    {r.mask_id} ({r.lat:.1f}, {r.lon:.1f})  {r.mean_probability:.4f}")
    summary["forecast"] = {"rows": len(fc), "expected_events_sum": float(fc.probability.sum()),
                           "by_window": by_window.reset_index().astype({"window_start": str}).to_dict(orient="records"),
                           "cells": cells.to_dict(orient="records"),
                           "top10": top.to_dict(orient="records")}


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Presentation figures from ``final_run/final_summary.json`` and rerun artifacts.

Writes PNGs (200 dpi) to ``figures/``.  Colors follow the reference data-viz
palette: slot 1 blue for the candidate / single series, slots 2-3 only when a
second or third series is needed, gray for context.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
S = json.loads((HERE / "final_run" / "final_summary.json").read_text())

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, CONTEXT = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 12, "legend.frameon": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
})


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.03, note, fontsize=8.5, color=MUTED, ha="left", va="top")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def bars(ax, labels, values, colors, fmt="{:.4f}", err=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62, xerr=err,
            error_kw=dict(ecolor=INK2, lw=1, capsize=0))
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", colors=INK, labelsize=10)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, lw=0.8)
    span = max(values) - min(0, min(values))
    for yi, v in zip(y, values):
        ax.text(v + span * 0.012 if v >= 0 else span * 0.012, yi, fmt.format(v),
                va="center", ha="left", fontsize=9.5, color=INK2) if abs(v) >= 5e-5 else ax.text(
                span * 0.012, yi, "≈ 0", va="center", ha="left", fontsize=9.5, color=INK2)
    ax.set_xlim(min(0, min(values)), max(values) + span * 0.16)


# 1. Model comparison -------------------------------------------------------
def fig_models():
    rows = S["comparison"]["models"]
    rows = sorted(rows, key=lambda r: -r["ig_bits"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [BLUE if r["model"].startswith("Frozen") else CONTEXT for r in rows]
    bars(ax, [r["model"] for r in rows], [r["ig_bits"] for r in rows], colors)
    ax.set_xlabel("Information gain over climatology (bits per cell-window, higher is better)")
    ax.set_title("Frozen candidate scores highest; ETAS is statistically tied (Δ −0.0005 bits, 95% CI spans 0)")
    save(fig, "01_model_comparison.png",
         "767,745 test rows (1,551 cells × 15 horizons × 33 monthly origins). Same rows and per-horizon "
         "isotonic calibration for every model.")


# 2. External drivers -------------------------------------------------------
def fig_drivers():
    names = {"geomagnetic": "Geomagnetic", "solar": "Solar", "tidal_ephemeris": "Tidal", "gps": "GPS",
             "all_external": "All external"}
    rows = [r for r in S["comparison"]["external"] if r["model"] != "seismic_only"]
    rows = sorted(rows, key=lambda r: -(r["ci_low"] + r["ci_high"]) / 2)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        mid = (r["ci_low"] + r["ci_high"]) / 2
        crosses = r["ci_low"] < 0 < r["ci_high"]
        c = MUTED if crosses else ORANGE
        ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=c, lw=3, solid_capstyle="round")
        ax.plot(mid, yi, "o", ms=8, color=c, mec=SURFACE, mew=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.text(0, len(rows) - 0.35, " no change vs seismic-only", color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(y, [names.get(r["model"], r["model"]) for r in rows])
    ax.tick_params(axis="y", colors=INK, labelsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Change in information gain vs seismic-only (bits), paired 95% interval")
    ax.set_title("No external driver improves the seismic-only model")
    save(fig, "02_external_drivers.png",
         "Paired bootstrap over contiguous 3-origin blocks, 2,000 repetitions. Gray = interval includes zero; "
         "orange = reliably worse.")


# 3. Per-horizon skill ------------------------------------------------------
def fig_horizon():
    ph = S["test"]["per_horizon"]
    h = np.arange(len(ph))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title, fmt in [
        (axes[0], "ig_bits", "Information gain (bits)", "{:.3f}"),
        (axes[1], "auc", "ROC AUC", "{:.3f}"),
        (axes[2], "ece", "Calibration error (ECE)", "{:.4f}"),
    ]:
        v = [m[key] for m in ph]
        ax.plot(h, v, color=BLUE, marker="o", ms=5, mec=SURFACE, mew=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xlabel("Horizon (30-day window index after origin)")
        for i in (0, len(v) - 1):
            ax.annotate(fmt.format(v[i]), (h[i], v[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=INK2)
        if key == "ece":
            ax.axhline(0.02, color=MUTED, lw=1)
            ax.text(14, 0.02, "Deck reference 0.02", ha="right", va="bottom", fontsize=8.5, color=MUTED)
            ax.set_ylim(0, 0.024)
        if key == "ig_bits":
            ax.set_ylim(0, 0.18)
        elif key == "auc":
            ax.set_ylim(0.5, 1.0)
    fig.suptitle("Skill is steady across all 15 forecast windows", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "03_per_horizon.png", "Frozen candidate, 2021–2023 test origins. Each point = 51,183 rows.")


# 4. Reliability ------------------------------------------------------------
def fig_reliability():
    rq = S["test"]["reliability_quantile20"]
    mp = np.array([r["mean_p"] for r in rq]); ob = np.array([r["observed"] for r in rq])
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    lo = max(min(mp.min(), ob[ob > 0].min()) * 0.7, 1e-4)
    ax.plot([lo, 1], [lo, 1], color=AXIS, lw=1)
    ax.text(0.5, 0.33, "perfect calibration", rotation=37, color=MUTED, fontsize=8.5, ha="center")
    ax.plot(mp, np.maximum(ob, lo), color=BLUE, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, 1); ax.set_ylim(lo, 1)
    ax.set_xlabel("Forecast probability (bin mean)"); ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability: forecasts match observed rates", fontsize=11.5)
    hist = np.array(S["test"]["probability_hist"]); edges = np.linspace(0, 1, 41)
    axh.bar(edges[:-1], hist / hist.sum(), width=edges[1] - edges[0] - 0.004, align="edge", color=BLUE)
    axh.set_yscale("log")
    axh.set_xlabel("Forecast probability"); axh.set_ylabel("Share of test rows (log)")
    axh.set_title("Most cells get low probabilities; a few are near-certain", fontsize=11.5)
    axh.grid(axis="x", visible=False)
    ece = S["test"]["headline"]["ece"]
    fig.tight_layout()
    save(fig, "04_reliability.png",
         f"Left: 20 equal-count bins on test rows, log–log axes. Overall ECE (10 equal-width bins) = {ece:.4f}.")


# 5. Training curve ---------------------------------------------------------
def fig_training():
    tc = S["model"]["training_curve"]
    tr, va = np.array(tc["train_log_loss"]), np.array(tc["validation_log_loss"])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    it = np.arange(len(tr))
    ax.plot(it, tr, color=BLUE)
    if len(va):
        ax.plot(it, va, color=ORANGE, lw=1.5)
        ax.annotate(f"final: fit {tr[-1]:.4f}, holdout {va[-1]:.4f} nats", (it[-1], va[-1]),
                    xytext=(-10, 28), textcoords="offset points", ha="right", fontsize=9.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    size = S["model"]["learned_size"]
    ax.set_xlim(0, len(tr) + 2)
    ax.set_ylim(min(tr.min(), va.min() if len(va) else 1) * 0.97, tr[1] * 1.02)
    ax.set_xlabel("Boosting iteration (1 iteration = 1 tree)")
    ax.set_ylabel("Log loss (nats)")
    stop = "stopped early" if size["boosting_iterations_run"] < size["max_iter_configured"] else "no early stop"
    ax.set_title(f"Training: {size['boosting_iterations_run']} of {size['max_iter_configured']} "
                 f"boosting iterations ({stop})")
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, label="Fit rows (90%)"),
                       plt.Line2D([], [], color=ORANGE, label="Early-stopping holdout (10%, random)")],
              loc="upper right", fontsize=9)
    save(fig, "05_training_curve.png",
         "Iteration 0 is the constant base-rate model and is off the top of the axis. "
         "Rows: 5,025,240 fit examples (2001–2018 origins).")


# 6. Data distribution ------------------------------------------------------
def fig_data():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    cat = pd.DataFrame(S["catalog"]["events_by_year"])
    ax = axes[0, 0]
    ax.bar(cat.year, cat.m2_plus, color=BLUE, width=0.75)
    ax.set_title("Seismic catalog: M≥2.0 events per year (shaded = calibration years)", fontsize=11)
    ax.grid(axis="x", visible=False)
    for s, (a, b) in {"fit": (2001, 2018), "cal": (2019, 2020), "test": (2021, 2023)}.items():
        ax.axvspan(a - 0.5, b + 0.5, color=GRID, alpha=0.45 if s == "cal" else 0.0, lw=0)
    ax.set_xlabel("Year")

    ax = axes[0, 1]
    mh = S["catalog"]["magnitude_hist"]; e = np.array(mh["edges"])
    ax.bar(e[:-1], mh["counts"], width=np.diff(e) - 0.03, align="edge", color=BLUE)
    ax.axvline(2.0, color=INK2, lw=1)
    ax.text(2.03, max(mh["counts"]) * 0.95, "target threshold M 2.0", color=INK2, fontsize=9, va="top")
    ax.set_yscale("log")
    ax.set_title("Magnitude distribution (all catalog events)", fontsize=11)
    ax.set_xlabel("Magnitude"); ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    sp = S["dataset"]["splits"]
    for name, color in [("fit", BLUE), ("calibration", ORANGE), ("test", AQUA)]:
        d = sp[name]["positive_rate_by_origin_year"]
        yrs = np.array([int(k) for k in d]); v = np.array(list(d.values()))
        ax.plot(yrs, v * 100, color=color, marker="o", ms=4, mec=SURFACE, mew=1)
        top = int(np.argmax(v))
        ax.annotate(name, (yrs[top], v[top] * 100), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    ax.set_title("Positive rate by origin year (share of cell-windows with an event)", fontsize=11)
    ax.set_ylabel("Positive rate (%)"); ax.set_xlabel("Origin year")
    ax.set_ylim(3, 8.8)
    ax.set_xlim(2000, 2026)

    ax = axes[1, 1]
    labels = ["Fit\n2001–2018", "Calibration\n2019–2020", "Test\n2021–2023"]
    rows = [sp[k]["rows"] for k in ("fit", "calibration", "test")]
    pos = [sp[k]["positives"] for k in ("fit", "calibration", "test")]
    x = np.arange(3)
    ax.bar(x, [r / 1e6 for r in rows], color=CONTEXT, width=0.55, label="All rows")
    ax.bar(x, [p / 1e6 for p in pos], color=BLUE, width=0.55, label="Positive rows")
    for xi, r, p in zip(x, rows, pos):
        ax.text(xi, r / 1e6, f"{r:,} rows\n{p / r:.1%} positive", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(x, labels); ax.tick_params(axis="x", colors=INK)
    ax.set_ylabel("Rows (millions)"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(rows) / 1e6 * 1.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Chronological split sizes and class balance", fontsize=11)
    fig.suptitle("Data: seismic catalog (1990–2024) → 1,551 cells × 15 windows per monthly origin", x=0.01,
                 ha="left", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "06_data_distribution.png",
         "Target: at least one M≥2.0 event within 100 km of the cell centre in the 30-day window. "
         "Judge catalog not read.")


# 7. Feature importance -----------------------------------------------------
def fig_importance():
    imp = S["permutation_importance_bits"]
    pretty = {"lat": "Latitude", "lon": "Longitude", "count_7d_100km": "Events, last 7 d",
              "count_30d_100km": "Events, last 30 d", "count_90d_100km": "Events, last 90 d",
              "count_365d_100km": "Events, last 365 d", "log1p_days_since_last": "Days since last event (log)",
              "ratio_7d_30d": "Acceleration 7 d / 30 d", "ratio_30d_365d": "Acceleration 30 d / 365 d",
              "horizon_index": "Horizon index"}
    items = sorted(imp.items(), key=lambda kv: -kv[1]["mean_bits"])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars(ax, [pretty[k] for k, _ in items], [v["mean_bits"] for _, v in items],
         [BLUE] * len(items), fmt="{:.4f}")
    ax.set_xlabel("Log-loss increase when the feature is shuffled (bits)")
    ax.set_title("What the model relies on: long-run activity and location")
    save(fig, "07_feature_importance.png",
         "Permutation importance on 200,000 random test rows, 3 shuffles each; calibration held fixed.")


# 8. Forecast map -----------------------------------------------------------
def fig_forecast():
    cells = pd.DataFrame(S["forecast"]["cells"])
    bw = pd.DataFrame(S["forecast"]["by_window"])
    fig, (ax, axw) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    ev = np.array(S["catalog"]["lat_lon"])
    sc = ax.scatter(cells.lon, cells.lat, c=cells.mean_probability, cmap=SEQ,
                    norm=LogNorm(vmin=1e-3, vmax=1), s=16, marker="s", lw=0)
    ax.set_aspect(1 / np.cos(np.radians(cells.lat.mean())))
    ax.grid(False)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean probability over 15 windows (log scale)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("2025 forecast: probability of an M≥2.0 event within 100 km", fontsize=11.5)
    del ev
    w = pd.to_datetime(bw.window_start)
    axw.plot(w, bw["mean"], color=BLUE, marker="o", ms=5, mec=SURFACE, mew=1.5)
    axw.text(w.iloc[-1], bw["mean"].iloc[-1], "  mean", color=INK2, fontsize=9, va="center")
    axw.plot(w, bw["median"], color=ORANGE, marker="o", ms=5, mec=SURFACE, mew=1.5)
    axw.text(w.iloc[-1], bw["median"].iloc[-1], "  median", color=INK2, fontsize=9, va="center")
    axw.set_ylim(0, max(bw["mean"]) * 1.3)
    axw.set_title("Grid-average probability by window", fontsize=11.5)
    axw.set_ylabel("Probability")
    axw.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    save(fig, "08_forecast_2025.png",
         "Provisional 1,551-cell mask and windows (not official). Features frozen at 2025-01-01; "
         "row alignment corrected.")


# 9. Alignment fix ----------------------------------------------------------
def fig_alignment():
    old = pd.read_csv(ROOT / "frozen_provisional_forecast" / "predictions.csv")
    new = pd.read_csv(HERE / "rerun" / "frozen_forecast_fixed" / "predictions.csv")
    cells = pd.DataFrame(S["forecast"]["cells"]).sort_values("mean_probability")
    picks = [cells.iloc[int(len(cells) * q)].mask_id for q in (0.55, 0.8, 0.95)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, cid in zip(axes, picks):
        o = old[old.mask_id == cid].probability.to_numpy()
        n = new[new.mask_id == cid].probability.to_numpy()
        x = np.arange(1, 16)
        ax.plot(x, o, color=ORANGE, marker="o", ms=4, mec=SURFACE, mew=1)
        ax.plot(x, n, color=BLUE, marker="o", ms=4, mec=SURFACE, mew=1)
        ax.set_title(cid.replace("provisional_", "cell "), fontsize=10.5)
        ax.set_xlabel("Window (2025-01 → 2026-02)")
        ax.set_yscale("log"); ax.set_ylim(5e-5, 1.5)
    axes[0].set_ylabel("Probability (log)")
    axes[-1].legend(handles=[plt.Line2D([], [], color=ORANGE, marker="o", label="Recorded file (misaligned)"),
                             plt.Line2D([], [], color=BLUE, marker="o", label="Corrected file")],
                    loc="lower right", fontsize=8.5)
    fig.suptitle("Forecast file fix: recorded rows mixed up cells and windows", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "09_forecast_alignment_fix.png",
         "Features are one 2025-01-01 snapshot, so a cell's probability should change only smoothly with horizon.")


if __name__ == "__main__":
    for f in (fig_models, fig_drivers, fig_horizon, fig_reliability, fig_training, fig_data,
              fig_importance, fig_forecast, fig_alignment):
        try:
            f()
        except (KeyError, FileNotFoundError) as exc:
            print(f"skipped {f.__name__}: input not ready ({exc})")

#!/usr/bin/env python3
"""Extra presentation figures: test vs judge, quantum paired test, entry thresholds."""
from __future__ import annotations

import json

import numpy as np

import make_figures as mf
from make_figures import AXIS, BLUE, CONTEXT, INK, INK2, MUTED, ORANGE, SURFACE, plt, save

HERE = mf.HERE
J = json.loads((HERE / "judge_check" / "judge_metrics.json").read_text())
ST = json.loads((HERE / "stakeholder" / "stakeholder.json").read_text())


def fig_test_vs_judge():
    fs = mf.S
    test = fs["test"]["headline"]
    judge = J["frozen_candidate_corrected"]
    sp_test = fs["comparison"]["models"]
    sp_test = next(m["ig_bits"] for m in sp_test if m["model"].startswith("Spatial"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    groups = [("Information gain (bits)", [test["ig_bits"], judge["ig_bits"]], [sp_test, J["spatial_prior"]["ig_bits"]], "{:.3f}"),
              ("ROC AUC", [test["auc"], judge["auc"]], None, "{:.3f}"),
              ("Calibration error (ECE)", [test["ece"], judge["ece"]], None, "{:.4f}")]
    for ax, (title, v, ref, fmt) in zip(axes, groups):
        x = np.arange(2)
        ax.bar(x, v, width=0.55, color=[BLUE, "#0d366b"])
        for xi, vi in zip(x, v):
            ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=10, color=INK2)
        if ref:
            ax.plot(x, ref, "o", color=ORANGE, ms=8, mec=SURFACE, mew=2)
            ax.text(1.32, ref[1], "location-only\nbaseline", color=INK2, fontsize=8.5, va="center")
        ax.set_xticks(x, ["Test\n2021–2023", "Judge\n2025–2026"]); ax.tick_params(axis="x", colors=INK)
        ax.set_title(title, fontsize=11); ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, 1.9)
        if title.startswith("ROC"):
            ax.set_ylim(0.5, 1.0)
        if title.startswith("Calib"):
            ax.axhline(0.02, color=MUTED, lw=1); ax.text(1.85, 0.02, "0.02 target", ha="right", va="bottom", fontsize=8.5, color=MUTED)
            ax.set_ylim(0, 0.024)
    fig.suptitle("Held-out check: the judge score is higher than the test score, so no sign of leakage", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "10_test_vs_judge.png", "Judge period used once (2026-09-27 09:44), with the corrected forecast. "
         "Judge: 23,265 cell-windows, 1,838 positive. Test: 767,745 cell-windows.")


def fig_quantum():
    Q = json.loads((HERE / "quantum_test" / "quantum_paired_test.json").read_text())
    b = Q["paired_block_bootstrap_quantum_minus_classical"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, key, ci, title, deck in [
        (axes[0], "delta_ig_bits", "ig_ci95", "Δ information gain, bits (95% CI)", None),
        (axes[1], "delta_auc", "auc_ci95", "Δ ROC AUC (95% CI)", (0.0011, [-0.0004, 0.0026])),
    ]:
        lo, hi = b[ci]; mid = b[key]
        rows = [("Ours: classical + quantum − classical", mid, lo, hi, BLUE)]
        if deck:
            rows.append(("Study Deck reference (blind window)", deck[0], deck[1][0], deck[1][1], CONTEXT))
        for i, (lab, m, l, h, c) in enumerate(rows):
            yv = len(rows) - 1 - i
            ax.plot([l, h], [yv, yv], color=c, lw=3, solid_capstyle="round")
            ax.plot(m, yv, "o", color=c, ms=9, mec=SURFACE, mew=2)
        ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows]); ax.tick_params(axis="y", colors=INK, labelsize=9.5)
        ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        span = max([abs(lo), abs(hi), 1e-4] + ([abs(deck[1][0]), abs(deck[1][1])] if deck else [])) * 1.25
        ax.set_xlim(-span, span); ax.set_title(title, fontsize=11)
    verdict = "CI spans 0: no reliable quantum lift" if b["ig_ci95"][0] < 0 < b["ig_ci95"][1] else (
        "CI entirely below 0: quantum features hurt" if b["ig_ci95"][1] < 0 else "CI entirely above 0: quantum lift")
    fig.suptitle("Quantum features are used by the model but give no lift: IG slightly worse, AUC unchanged", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.06)
    fig.tight_layout()
    save(fig, "11_quantum_paired_test.png",
         f"4-qubit angle-encoded circuit, 8 Pauli Z/ZZ features appended to the frozen model. Paired contiguous 3-origin "
         f"block bootstrap ({b['ig_reps']} reps IG, {b['auc_reps']} reps AUC). Trees split on quantum features "
         f"{Q['quantum_feature_split_share']:.0%} of the time.")


def fig_thresholds():
    t = ST["thresholds_30d_test"]
    x = np.array([r["threshold"] for r in t]) * 100
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.plot(x, [r["share_cleared"] * 100 for r in t], color=BLUE, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.plot(x, [r["share_of_all_events_in_cleared"] * 100 for r in t], color=ORANGE, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.text(x[-1] + 0.4, t[-1]["share_cleared"] * 100, "cell-windows cleared", color=INK2, fontsize=9.5, va="center")
    ax.text(x[-1] + 0.4, t[-1]["share_of_all_events_in_cleared"] * 100, "events that happened\nin cleared cells", color=INK2, fontsize=9.5, va="center")
    ax.set_xscale("log"); ax.set_xticks(x, [f"{v:g}%" for v in x]); ax.minorticks_off()
    ax.set_xlim(x[0] * 0.8, x[-1] * 2.6); ax.set_ylim(0, 100)
    ax.set_xlabel("Entry cut-off on the 30-day forecast"); ax.set_ylabel("Percent")
    ax.set_title("Entry threshold trade-off (test 2021–2023): a 1% cut-off clears 53% and misses 1.3% of events")
    save(fig, "12_entry_thresholds.png", "The acceptable cut-off is a decision for emergency managers; this shows the evidence for each choice.")


if __name__ == "__main__":
    for f in (fig_test_vs_judge, fig_thresholds, fig_quantum):
        try:
            f()
        except FileNotFoundError as exc:
            print("skipped", f.__name__, exc)

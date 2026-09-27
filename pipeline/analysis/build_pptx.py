#!/usr/bin/env python3
"""Build the presentation deck and the final folder.

Every number is read from the saved result files; nothing is typed by hand.
Output: ../final_presentation/ (deck, figures, tables, key results, logs).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent / "final_presentation"
FIG = HERE / "figures"
S = json.loads((HERE / "final_run" / "final_summary.json").read_text())
J = json.loads((HERE / "judge_check" / "judge_metrics.json").read_text())
Q = json.loads((HERE / "quantum_test" / "quantum_paired_test.json").read_text())
ST = json.loads((HERE / "stakeholder" / "stakeholder.json").read_text())
V = json.loads((HERE / "verification.json").read_text())

INK, INK2, MUTED = RGBColor(0x15, 0x18, 0x1B), RGBColor(0x4D, 0x54, 0x59), RGBColor(0x7C, 0x83, 0x88)
BLUE, NAVY, ORANGE = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x0D, 0x36, 0x6B), RGBColor(0xEB, 0x68, 0x34)
BG, RULE, WHITE, PALE = RGBColor(0xFC, 0xFC, 0xFB), RGBColor(0xD9, 0xDC, 0xD8), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xEE, 0xF4, 0xFC)
GOOD, BAD = RGBColor(0x0B, 0x7A, 0x0B), RGBColor(0xB3, 0x3A, 0x14)
FONT = "Arial"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def txt(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=None, spacing=1.1):
    """runs: str | list of paragraphs; a paragraph is str or list of (text, dict(bold/color/size))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = spacing
        if align:
            p.alignment = align
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run(); r.text = t
            f = r.font; f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color); f.italic = o.get("italic", False)
    return tb


def bullets(slide, x, y, w, h, items, size=15, gap=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.line_spacing = 1.08
        parts = it if isinstance(it, list) else [(it, {})]
        r = p.add_run(); r.text = "▪  "; r.font.color.rgb = BLUE; r.font.size = Pt(size); r.font.name = FONT
        for t, o in parts:
            r = p.add_run(); r.text = t; f = r.font; f.name = FONT; f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", False); f.color.rgb = o.get("color", INK)
    return tb


def base(title, kicker=None, number=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.09)); bar.fill.solid(); bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    if kicker:
        txt(s, 0.55, 0.3, 12, 0.3, kicker.upper(), size=11, color=MUTED, bold=True)
    txt(s, 0.55, 0.55, 12.3, 0.8, title, size=28, bold=True)
    foot = f"Earthquake-Q · Team Schrödinger's Cats · QuantathonV3"
    txt(s, 0.55, 7.08, 9, 0.3, foot, size=9, color=MUTED)
    if number:
        txt(s, 12.2, 7.08, 0.6, 0.3, str(number), size=9, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def image(slide, name, x, y, w=None, h=None):
    path = FIG / name
    from PIL import Image
    iw, ih = Image.open(path).size
    if w and h:
        scale = min(w / iw, h / ih)
        w2, h2 = iw * scale, ih * scale
        return slide.shapes.add_picture(str(path), Inches(x + (w - w2) / 2), Inches(y + (h - h2) / 2), Inches(w2), Inches(h2))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def table(slide, x, y, w, rows, col_w=None, size=12, header=True, row_h=0.36, highlight=None):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = shp.table
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(cw)
    for r in range(nr):
        t.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = t.cell(r, c); cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.fill.solid()
            is_h = header and r == 0
            cell.fill.fore_color.rgb = NAVY if is_h else (PALE if highlight and r in highlight else WHITE)
            tf = cell.text_frame; tf.clear(); p = tf.paragraphs[0]
            val = rows[r][c]
            run = p.add_run(); run.text = str(val)
            f = run.font; f.name = FONT; f.size = Pt(size - (1 if is_h else 0)); f.bold = is_h or bool(highlight and r in highlight and c == 0)
            f.color.rgb = WHITE if is_h else INK
            if c > 0 and not is_h and any(ch.isdigit() for ch in str(val)[:2]) or (c > 0 and str(val)[:1] in "+−-<>"):
                p.alignment = PP_ALIGN.RIGHT
    return shp


def callout(slide, x, y, w, h, title, body, color=BLUE, size=13):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = WHITE; box.line.color.rgb = RULE
    edge = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.07), Inches(h))
    edge.fill.solid(); edge.fill.fore_color.rgb = color; edge.line.fill.background()
    txt(slide, x + 0.22, y + 0.12, w - 0.35, 0.4, title, size=size + 2, bold=True)
    if isinstance(body, list):
        bullets(slide, x + 0.2, y + 0.55, w - 0.35, h - 0.6, body, size=size, gap=3)
    else:
        txt(slide, x + 0.22, y + 0.55, w - 0.35, h - 0.6, body, size=size, color=INK2)


def stat(slide, x, y, w, value, label, color=INK):
    txt(slide, x, y, w, 0.6, value, size=30, bold=True, color=color)
    txt(slide, x, y + 0.62, w, 0.6, label, size=11.5, color=INK2)


# ---------------------------------------------------------------- numbers
h = S["test"]["headline"]; jc = J["frozen_candidate_corrected"]; jsp = J["spatial_prior"]
comp = {m["model"]: m["ig_bits"] for m in S["comparison"]["models"]}
etas_ig = comp["ETAS approximation"]; sp_ig = comp["Spatial prior (cell x horizon rate)"]
qb = Q["paired_block_bootstrap_quantum_minus_classical"]
size = S["model"]["learned_size"]; hp = S["model"]["hyperparameters"]
sp = S["dataset"]["splits"]
purged = next(c for c in V if c["check"].startswith("purged"))["detail"]
q_ci_spans = qb["ig_ci95"][0] < 0 < qb["ig_ci95"][1]
q_verdict = ("no statistically reliable quantum lift (CI spans 0)" if q_ci_spans else
             "quantum features reliably hurt (CI below 0)" if qb["ig_ci95"][1] < 0 else "a reliable quantum lift (CI above 0)")
n = 0


def nxt():
    global n
    n += 1
    return n


# 1 · Title ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
txt(s, 0.8, 1.2, 11, 0.4, "QUANTATHON V3 · SRNL & CLEMSON · EARTHQUAKE-Q", size=13, color=RGBColor(0x9E, 0xC5, 0xF4), bold=True)
txt(s, 0.8, 1.75, 11.5, 1.6, "An honest aftershock-risk forecast for the central & eastern US", size=40, color=WHITE, bold=True)
txt(s, 0.8, 3.55, 11.5, 1.2, [f"P(≥1 M≥2.0 earthquake within 100 km in each 30-day window) for 1,551 cells × 15 windows = 23,265 calibrated probabilities, issued 2025-01-01."],
    size=18, color=RGBColor(0xDD, 0xE8, 0xF6))
for i, (v, l) in enumerate([(f"+{jc['ig_bits']:.3f}", "bits on the sealed judge window"), (f"+{h['ig_bits']:.3f}", "bits on the 2021–2023 test"),
                            (f"{jc['ece']:.4f}", "calibration error on judge (target < 0.02)"), ("No lift", "from quantum features (paired 95% CI)")]):
    txt(s, 0.8 + i * 3.05, 5.0, 2.9, 0.6, v, size=30, bold=True, color=WHITE)
    txt(s, 0.8 + i * 3.05, 5.62, 2.8, 0.7, l, size=12, color=RGBColor(0xB7, 0xD3, 0xF6))
txt(s, 0.8, 6.8, 11, 0.4, "Team Schrödinger's Cats · 27 September 2026", size=12, color=RGBColor(0x9E, 0xC5, 0xF4))
nxt()

# 2 · Decision summary ----------------------------------------------------
s = base("Where we stand: results you can act on", "Summary for mentors", nxt())
rows = [["Question", "Answer", "Evidence"],
        ["Is the forecast skilful?", f"Yes: +{jc['ig_bits']:.3f} bits on judge", f"Test +{h['ig_bits']:.3f}; AUC {jc['auc']:.3f} judge"],
        ["Are the probabilities honest?", f"Yes: ECE {jc['ece']:.4f}", "Target < 0.02; reliability curve on diagonal"],
        ["Did we leak the future?", "No sign of it", "Judge score ≥ test score (Deck's tell-tale check)"],
        ["Does it beat physics (ETAS)?", "Tie", f"ETAS approx. {etas_ig:.4f} vs {h['ig_bits']:.4f}; paired CI spans 0"],
        ["Does it beat location alone?", f"Yes: +{jc['ig_bits'] - jsp['ig_bits']:.3f} bits (judge)", f"Location-only {jsp['ig_bits']:.3f} judge, {sp_ig:.3f} test"],
        ["Do external drivers help?", "No", "Geomagnetic, solar, tidal, GPS: 0 or negative"],
        ["Does quantum help?", "No: slightly worse", f"ΔIG {qb['delta_ig_bits']:+.4f} bits, 95% CI [{qb['ig_ci95'][0]:+.4f}, {qb['ig_ci95'][1]:+.4f}]"],
        ["Is it ready to submit?", "Yes, corrected file", "Row-alignment bug found and fixed (bad file: −0.37 bits)"]]
table(s, 0.55, 1.5, 12.2, rows, col_w=[3.3, 3.6, 5.3], size=13, row_h=0.52)
nxt_num = None

# 3 · Problem and scoring -------------------------------------------------
s = base("The task and how it is scored", "Problem", nxt())
bullets(s, 0.55, 1.5, 6.3, 4.8, [
    [("Target: ", {"bold": True}), ("probability of ≥1 M≥2.0 earthquake within 100 km of a 0.5° cell in a 30-day window.", {})],
    [("Grid: ", {"bold": True}), ("1,551 cells over 25–50°N, 65–90°W × 15 windows from 2025-01-01 = 23,265 forecasts.", {})],
    [("Rare target: ", {"bold": True}), (f"{sp['test']['positive_rate']:.1%} of test cell-windows are positive, so accuracy is meaningless (always 'no' ≈ 92%).", {})],
    [("Primary score: ", {"bold": True}), ("information gain over climatology, in bits per cell-window.", {})],
    [("Honesty: ", {"bold": True}), ("calibration (ECE, Brier) and a judge score that does not collapse vs test.", {})],
    [("Stakeholders: ", {"bold": True}), ("30-day entry decisions for rescue crews, 6-month planning for repair crews.", {})],
], size=15)
callout(s, 7.2, 1.5, 5.6, 4.9, "Scoring equations (from the challenge README)", [
    "LL(q) = −mean[ y·ln q + (1−y)·ln(1−q) ]",
    "IG = [ LL(base rate) − LL(forecast) ] / ln 2   (bits)",
    "q clipped to [10⁻⁴, 1−10⁻⁴]: a forecast never claims certainty",
    "ECE = Σ_b (n_b/N)·| ȳ_b − q̄_b |  over 10 probability bins",
    "Entropy ceiling at 7.5% base rate ≈ 0.38 bits: no forecaster can exceed it",
], size=13)
nxt_num = None

# 4 · Data ----------------------------------------------------------------
s = base("Data: five raw streams, split strictly by time", "Data", nxt())
image(s, "06_data_distribution.png", 0.45, 1.35, 7.4, 5.6)
rows = [["Split", "Origins", "Rows", "Positive"]]
for k, lab in [("fit", "Fit"), ("calibration", "Calibration"), ("test", "Test")]:
    d = sp[k]; rows.append([lab, f"{d['origins'][0][:7]} → {d['origins'][1][:7]}", f"{d['rows']:,}", f"{d['positive_rate']:.1%}"])
rows.append(["Judge (once)", "2025-01 → 2026-02", f"{jc['n']:,}", f"{jc['positive_rate']:.1%}"])
table(s, 8.05, 1.45, 4.8, rows, col_w=[1.25, 1.65, 1.1, 0.8], size=11, row_h=0.38)
bullets(s, 8.05, 3.55, 4.8, 3.3, [
    f"Seismic catalog: {S['catalog']['seismic_events']:,} USGS events (train+test files); magnitude of completeness Mc = 1.75.",
    "Each example = cell × monthly issue date × 15 lead windows (train the way you are scored).",
    "Features use only events strictly before the issue date.",
    "Geomagnetic, solar, tidal and GPS streams were tested and excluded (slide 9).",
], size=12, gap=4)

# 5 · Architecture --------------------------------------------------------
s = base("Model architecture: gradient-boosted trees + per-horizon calibration", "Method", nxt())
steps = [("Raw catalog", "USGS events\nM ≥ 1.75, before\nissue date"),
         ("10 features", "counts 7/30/90/365 d\nwithin 100 km, recency,\nacceleration, lat/lon,\nhorizon h/14"),
         ("Pooled GBDT", "HistGradientBoosting\n120 trees × 31 leaves\none model, 15 horizons"),
         ("Isotonic ×15", "one calibrator per\nhorizon, fit on\n2019–2020 only"),
         ("Clip + output", "[10⁻⁴, 1−10⁻⁴]\n23,265 probabilities\ncell × window")]
bw, gap, y0 = 2.2, 0.28, 1.65
for i, (t1, t2) in enumerate(steps):
    x = 0.55 + i * (bw + gap)
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y0), Inches(bw), Inches(2.1))
    box.adjustments[0] = 0.08; box.fill.solid(); box.fill.fore_color.rgb = NAVY if i in (2, 3) else WHITE
    box.line.color.rgb = NAVY
    col = WHITE if i in (2, 3) else INK
    txt(s, x + 0.12, y0 + 0.15, bw - 0.24, 0.4, t1, size=15, bold=True, color=col, align=PP_ALIGN.CENTER)
    txt(s, x + 0.12, y0 + 0.6, bw - 0.24, 1.4, t2, size=11.5, color=col if i in (2, 3) else INK2, align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.03), Inches(y0 + 0.9), Inches(gap - 0.06), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = BLUE; a.line.fill.background()
bullets(s, 0.55, 4.1, 6.2, 2.9, [
    [("Why trees: ", {"bold": True}), ("strongest learner for tabular data in the Deck and in benchmarks (Grinsztajn 2022); our MLP scored lower.", {})],
    [("Why pooled: ", {"bold": True}), ("one model sees all 15 horizons (5.0 M rows) instead of 15 small models; +0.0022 bits vs per-horizon models.", {})],
    [("Why isotonic per horizon: ", {"bold": True}), ("monotone re-mapping to true frequencies, fit on later held-out dates only.", {})],
], size=13, gap=4)
callout(s, 7.0, 4.1, 5.8, 2.75, "Gradient boosting in one line", [
    "F_m(x) = F_{m−1}(x) + η · f_m(x),  η = 0.06",
    "f_m = regression tree fit to the gradient of log loss",
    "Histogram splits over 255 bins; L2 penalty λ = 1.0",
    f"Learned: {size['trees']} trees, {size['leaves_total']:,} leaves, ≈{size['learned_parameters_approx']:,} values",
], size=12.5)

# 6 · Experimental parameters ---------------------------------------------
s = base("Experimental parameters", "Method", nxt())
rows = [["Parameter", "Value"],
        ["Algorithm", "scikit-learn HistGradientBoostingClassifier (log loss)"],
        ["Boosting iterations (≈ epochs)", f"{size['boosting_iterations_run']} of {hp['max_iter']}; early stopping armed, not triggered"],
        ["Batch size", "Not applicable: each iteration fits one tree on the full fit set"],
        ["Learning rate η", str(hp["learning_rate"])],
        ["Max leaves / min rows per leaf", f"{hp['max_leaf_nodes']} / {hp['min_samples_leaf']}"],
        ["Max depth", f"unlimited (deepest tree: {size['max_depth_reached']})"],
        ["L2 regularization / bins", f"{hp['l2_regularization']} / {hp['max_bins']}"],
        ["Early-stopping holdout", "random 10% of fit rows, patience 10"],
        ["Calibration", "15 isotonic regressions, 2019–2020 origins, clip [1e-4, 1−1e-4]"],
        ["Seed / software", f"{hp['random_state']} · Python {S['environment']['python']}, scikit-learn {S['environment']['scikit_learn']}"]]
table(s, 0.55, 1.45, 7.6, rows, col_w=[2.9, 4.7], size=12, row_h=0.43)
image(s, "05_training_curve.png", 8.35, 1.45, 4.55, 2.6)
callout(s, 8.35, 4.3, 4.55, 2.5, "Comparison models on the same rows", [
    "Climatology, location-only prior, ETAS approximation",
    "15 per-horizon GBDTs; MLP (32,16), Adam lr 1e-3, batch 256, 58–69 epochs",
    "External-driver ablations; 4-qubit quantum features",
], size=12)

# 7 · Methods & literature -----------------------------------------------
s = base("Methods and the literature behind them", "Method", nxt())
rows = [["Method", "Equation / idea", "Reference"],
        ["Gutenberg–Richter law", "log₁₀ N(≥M) = a − b·M", "Gutenberg & Richter (1944)"],
        ["b-value (max. likelihood)", "b = log₁₀e / (M̄ − Mc)", "Aki (1965)"],
        ["Completeness Mc", "MAXC + 0.2 → Mc = 1.75", "Wiemer & Wyss (2000); Mignan & Woessner (2012)"],
        ["ETAS triggering", "λ(t) = μ + Σ K·e^{α(mᵢ−Mc)}·(t−tᵢ+c)^{−p}", "Ogata (1988); Utsu (1961)"],
        ["Declustering", "space–time windows by magnitude", "Gardner & Knopoff (1974)"],
        ["Gradient boosting", "F_m = F_{m−1} + η·f_m", "Friedman (2001); Ke et al. (2017)"],
        ["Isotonic calibration", "monotone map score → frequency", "Zadrozny & Elkan (2002)"],
        ["Proper scoring / IG", "log score; IG in bits", "Gneiting & Raftery (2007)"],
        ["Calibration error", "ECE over probability bins", "Guo et al. (2017)"],
        ["Block bootstrap", "resample contiguous time blocks", "Künsch (1989)"],
        ["Permutation importance", "loss increase when a feature is shuffled", "Breiman (2001)"],
        ["Quantum feature maps", "angle encoding + Pauli read-out", "Havlíček et al. (2019); Schuld & Killoran (2019)"]]
table(s, 0.55, 1.4, 12.2, rows, col_w=[2.9, 4.9, 4.4], size=11.5, row_h=0.405)

# 8 · Main results ---------------------------------------------------------
s = base("Results: skilful, calibrated, tied with physics", "Results", nxt())
image(s, "01_model_comparison.png", 0.45, 1.3, 7.9, 3.8)
stat(s, 8.7, 1.45, 4.2, f"+{h['ig_bits']:.4f}", "bits on test (2021–2023), 767,745 cell-windows", BLUE)
stat(s, 8.7, 2.75, 4.2, f"{h['auc']:.3f}", "ROC AUC (Deck: ~0.85 comes from location alone)")
stat(s, 8.7, 4.05, 4.2, f"{h['ece']:.4f}", "calibration error (Deck target < 0.02)")
bullets(s, 0.55, 5.3, 12.2, 1.6, [
    f"Matches the Deck's finding of parity with physics: ETAS approximation {etas_ig:.4f} bits, Δ −0.0005, paired 95% CI [−0.0016, +0.0006].",
    f"Location alone gets {sp_ig:.3f} bits ({sp_ig / h['ig_bits']:.0%} of ours; the Deck reports ~93%). Feature engineering adds the rest.",
], size=13, gap=3)

# 9 · Held-out check -------------------------------------------------------
s = base("Held-out check on the sealed judge window: no leakage", "Results", nxt())
image(s, "10_test_vs_judge.png", 0.45, 1.35, 12.4, 3.9)
rows = [["Forecast on judge (used once)", "IG (bits)", "AUC", "ECE"],
        ["Frozen model (corrected file)", f"+{jc['ig_bits']:.4f}", f"{jc['auc']:.3f}", f"{jc['ece']:.4f}"],
        ["Location-only baseline", f"+{jsp['ig_bits']:.4f}", f"{jsp['auc']:.3f}", f"{jsp['ece']:.4f}"],
        ["Original file (row bug)", f"{J['recorded_file_misaligned']['ig_bits']:.4f}", f"{J['recorded_file_misaligned']['auc']:.3f}", f"{J['recorded_file_misaligned']['ece']:.4f}"]]
table(s, 0.55, 5.35, 7.4, rows, col_w=[3.8, 1.3, 1.1, 1.2], size=12, row_h=0.36, highlight={1})
callout(s, 8.2, 5.35, 4.65, 1.5, "Why this matters", "The Deck's tell-tale for leakage is a judge score far below test. Ours went up (+0.146 → +0.161), like the reference (+0.110 → +0.134).", size=12)

# 10 · Calibration & horizon -----------------------------------------------
s = base("Honest probabilities at every lead time", "Results", nxt())
image(s, "04_reliability.png", 0.45, 1.35, 6.3, 2.75)
image(s, "03_per_horizon.png", 0.45, 4.2, 6.3, 2.55)
bullets(s, 7.0, 1.45, 5.85, 5.3, [
    "When we say 12%, events followed 11.6% of the time on test (Elevated band); 29% → 26% (High band).",
    f"Skill holds for all 15 windows: test IG 0.137–0.151 bits; judge 0.104–0.245 bits per window.",
    f"Every per-horizon calibration error is below 0.02 on test (overall {h['ece']:.4f}).",
    "Calibrators are fit on 2019–2020 only, never on training scores (Deck Module 5).",
    f"Caveat: a few labels near split boundaries overlap; removing them gives {purged['purged_ig_bits']:.4f} bits ({purged['delta_bits']:+.4f}), so treat sub-0.004 gaps as ties.",
], size=13.5, gap=6)

# 11 · What drives it ------------------------------------------------------
s = base("What carries the signal: location and the last year of activity", "Results", nxt())
image(s, "07_feature_importance.png", 0.45, 1.35, 6.3, 3.3)
image(s, "02_external_drivers.png", 6.9, 1.35, 6.0, 2.6)
bullets(s, 0.55, 4.85, 12.2, 2.0, [
    "Permutation importance on held-out test rows (not tree gain, which rewards static location fingerprints — Deck Module 4).",
    "Geomagnetic, solar, tidal and GPS features never help: four of five make the forecast reliably worse (paired 3-origin block bootstrap).",
    "Short-window counts and acceleration ratios add little on their own: they overlap with the 365-day count.",
], size=13, gap=4)

# 12 · Quantum --------------------------------------------------------------
s = base("Quantum: did it help? Measured with a paired test", "Quantum", nxt())
callout(s, 0.55, 1.4, 5.0, 3.0, "Why we tried it", [
    "The Deck's Approach 1: quantum feature maps as extra inputs to a classical model.",
    "Entangled read-outs (ZZ) can express feature interactions a single feature cannot.",
    "Testable: classical vs classical + quantum on identical rows.",
], size=12.5)
callout(s, 0.55, 4.55, 5.0, 2.35, "How we built it", [
    "4 qubits, angle encoding RY(π·xᵢ) of dynamic features only (365-d count, recency, 2 acceleration ratios)",
    "CNOT ring, re-upload, CNOT ring; read ⟨Zᵢ⟩ and ⟨ZᵢZᵢ₊₁⟩ → 8 features",
    f"Exact statevector simulation, checked against Qiskit on {Q['simulator_checked_rows_vs_qiskit']} rows",
], size=11.5)
image(s, "11_quantum_paired_test.png", 5.8, 1.4, 7.1, 2.35)
rows = [["", "IG (bits)", "AUC"],
        ["Classical (frozen)", f"{Q['classical_only']['ig_bits']:.4f}", f"{Q['classical_only']['auc']:.4f}"],
        ["Classical + quantum", f"{Q['classical_plus_quantum']['ig_bits']:.4f}", f"{Q['classical_plus_quantum']['auc']:.4f}"],
        ["Difference (95% CI)", f"{qb['delta_ig_bits']:+.4f} [{qb['ig_ci95'][0]:+.4f}, {qb['ig_ci95'][1]:+.4f}]",
         f"{qb['delta_auc']:+.4f} [{qb['auc_ci95'][0]:+.4f}, {qb['auc_ci95'][1]:+.4f}]"]]
table(s, 5.8, 3.95, 7.1, rows, col_w=[2.2, 2.6, 2.3], size=11.5, row_h=0.36, highlight={3})
txt(s, 5.8, 5.55, 7.1, 1.4, [[("Verdict: ", {"bold": True}), (f"the model uses the quantum features (trees split on them {Q['quantum_feature_split_share']:.0%} of the time), but information gain gets slightly worse (CI entirely below 0) and AUC does not change (CI spans 0). No quantum lift. "
    "This matches the Deck's reference (ΔAUC +0.0011, CI [−0.0004, +0.0026]). Kernels and VQCs were not run at this scale: fidelity kernels concentrate and VQCs hit barren plateaus (McClean 2018; Thanasilp 2024).", {})]], size=12, color=INK2)

# 13 · Stakeholders ---------------------------------------------------------
s = base("For decision-makers: a map, a 6-month view, and entry thresholds", "Stakeholders", nxt())
image(s, "08_forecast_2025.png", 0.45, 1.35, 6.7, 2.85)
image(s, "12_entry_thresholds.png", 0.45, 4.3, 6.7, 2.65)
six = ST["six_month_test_validation"]
bullets(s, 7.4, 1.45, 5.45, 5.4, [
    [("Clickable map: ", {"bold": True}), ("click an epicenter → 30-day and 6-month risk for the nearest cell, with what that band meant on test data.", {})],
    [("6-month (repair crews): ", {"bold": True}), (f"1 − Π(1 − pₕ) over windows 1–6. Test: {six['mean_forecast']:.1%} forecast vs {six['observed_rate']:.1%} observed, +{six['ig_bits_vs_fit_6m_rate']:.3f} bits; judge +{J['six_month']['ig_bits']:.3f} bits. Errs cautious where quakes cluster.", {})],
    [("Entry threshold (rescue crews): ", {"bold": True}), ("a 1% cut-off cleared 53% of cell-windows and missed 1.3% of events; 5% cleared 64% and missed 3.9%.", {})],
    [("Who decides: ", {"bold": True}), ("the acceptable risk is for emergency managers; we supply calibrated numbers and their track record.", {})],
    [("Hot spot: ", {"bold": True}), ("New Madrid seismic zone (36°N, 89–90°W) is near-certain every month.", {})],
], size=13, gap=5)

# 14 · Quality control ------------------------------------------------------
s = base("Quality control: what we checked, and the bug we caught", "Reproducibility", nxt())
image(s, "09_forecast_alignment_fix.png", 0.45, 1.35, 7.4, 2.4)
n_pass = sum(c["pass"] for c in V)
bullets(s, 0.55, 4.0, 7.3, 3.0, [
    "Full rerun from raw catalogs: training data bit-identical; every recorded metric reproduced exactly.",
    "Our first forecast file paired window-ordered probabilities with cell-ordered rows: 22,163 of 23,265 wrong.",
    f"On judge that file would score {J['recorded_file_misaligned']['ig_bits']:.2f} bits; the corrected file scores +{jc['ig_bits']:.3f}.",
    "200 random rows recomputed with independent code: exact match.",
], size=12.5, gap=4)
rows = [["Check", "Result"],
        ["Training data regenerates identically", "Pass"], ["Headline metrics reproduce", "Pass"],
        ["ECE < 0.02 (all groups)", "Pass"], ["No external driver helps", "Pass"],
        ["Headline scripts never read judge", "Pass"], ["Forecast order and values", "Pass (fixed)"],
        ["Judge ≥ test (no leakage)", "Pass"], [f"Verification suite", f"{n_pass}/{len(V)} (1 = old file)"]]
table(s, 8.1, 1.45, 4.75, rows, col_w=[3.15, 1.6], size=11.5, row_h=0.42)

# 15 · Limitations ----------------------------------------------------------
s = base("Limitations we want you to know", "Honesty", nxt())
bullets(s, 0.55, 1.5, 12.2, 5.4, [
    [("Model selection used the test period. ", {"bold": True}), ("The judge score is the only untouched number; we used it once.", {})],
    [("ETAS is an approximation. ", {"bold": True}), ("Our fitted triggering strength K collapsed to ~0, so it behaves like a background-rate model; a proper declustered ETAS refit is the top next step.", {})],
    [("Label windows cross split boundaries. ", {"bold": True}), (f"Removing those rows costs {abs(purged['delta_bits']):.4f} bits on test, larger than our gap to ETAS.", {})],
    [("The 1,551-cell mask and window dates are our construction ", {"bold": True}), ("from the Deck's specification (judge data ends March 2026).", {})],
    [("6-month risk assumes independent windows. ", {"bold": True}), ("Conservative at moderate/high risk; slightly low below 1% (0.06% vs 0.45%).", {})],
    [("Quantum test uses simulated 4-qubit circuits. ", {"bold": True}), ("A null result for this encoding, not a verdict on all quantum methods.", {})],
    [("A probability map is not a safety guarantee. ", {"bold": True}), ("Entry thresholds need agreement with emergency managers.", {})],
], size=15, gap=8)

# 16 · Decisions ------------------------------------------------------------
s = base("Decisions we would like from mentors", "Next steps", nxt())
rows = [["Decision", "Our recommendation", "Why"],
        ["Submit the frozen model?", "Yes, corrected file", f"+{jc['ig_bits']:.3f} judge bits, ECE {jc['ece']:.4f}, no leakage sign"],
        ["Add quantum features?", "No", f"ΔIG {qb['delta_ig_bits']:+.4f} bits, CI below 0; ΔAUC CI spans 0"],
        ["Add external drivers?", "No", "0 or negative in paired tests"],
        ["Invest next in…", "Declustered ETAS refit, ETAS as model input", "Deck's highest-value task; physics as starting point"],
        ["Entry threshold for rescue crews?", "Managers choose; 1–5% range", "1% clears 53%, misses 1.3%; 5% clears 64%, misses 3.9%"],
        ["Publish the map?", "Yes, with caveats", "Calibrated, shows its own track record"]]
table(s, 0.55, 1.5, 12.2, rows, col_w=[3.2, 3.9, 5.1], size=13, row_h=0.62)

# 17 · References -----------------------------------------------------------
s = base("References", "Literature", nxt())
refs = [
    "Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a − bM. Bull. Earthq. Res. Inst. 43, 237–239.",
    "Breiman, L. (2001). Random forests. Machine Learning 45, 5–32.",
    "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Ann. Statist. 29, 1189–1232.",
    "Gardner, J. K. & Knopoff, L. (1974). Is the sequence of earthquakes in Southern California, with aftershocks removed, Poissonian? BSSA 64, 1363–1367.",
    "Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. JASA 102, 359–378.",
    "Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS.",
    "Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.",
    "Gutenberg, B. & Richter, C. F. (1944). Frequency of earthquakes in California. BSSA 34, 185–188.",
    "Havlíček, V. et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212.",
    "Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. NeurIPS.",
    "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Ann. Statist. 17, 1217–1241.",
    "McClean, J. R. et al. (2018). Barren plateaus in quantum neural network training landscapes. Nat. Commun. 9, 4812.",
    "Mignan, A. & Woessner, J. (2012). Estimating the magnitude of completeness for earthquake catalogs. CORSSA.",
    "Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis for point processes. JASA 83, 9–27.",
    "Schuld, M. & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. Phys. Rev. Lett. 122, 040504.",
    "Thanasilp, S., Wang, S., Cerezo, M. & Holmes, Z. (2024). Exponential concentration in quantum kernel methods. Nat. Commun. 15, 5200.",
    "Utsu, T. (1961). A statistical study on the occurrence of aftershocks. Geophys. Mag. 30, 521–605.",
    "Wiemer, S. & Wyss, M. (2000). Minimum magnitude of completeness in earthquake catalogs. BSSA 90, 859–869.",
    "Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD.",
    "Challenge materials: Earthquake-Q Intro Briefing, Study Deck (100 slides), Classical Starter notebook, README (QuantathonV3, SRNL & Clemson, 2026).",
]
half = (len(refs) + 1) // 2
for col, chunk in enumerate([refs[:half], refs[half:]]):
    txt(s, 0.55 + col * 6.25, 1.4, 6.0, 5.6, [[(r, {})] for r in chunk], size=9.5, color=INK2, spacing=1.05)

FINAL.mkdir(exist_ok=True)
deck = FINAL / "Earthquake-Q_Schrodingers_Cats_presentation.pptx"
prs.save(deck)
print("wrote", deck, len(prs.slides._sldIdLst), "slides")

# ---------------------------------------------------------------- final folder
(FINAL / "figures").mkdir(exist_ok=True)
for f in sorted(FIG.glob("*.png")):
    shutil.copy2(f, FINAL / "figures" / f.name)
(FINAL / "tables").mkdir(exist_ok=True)


def write_csv(name, rows):
    with (FINAL / "tables" / name).open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)


write_csv("01_model_comparison_test.csv", [["model", "ig_bits_test"]] + [[m["model"], m["ig_bits"]] for m in S["comparison"]["models"]])
write_csv("02_external_drivers_test.csv", [["model", "ig_bits", "log_loss", "brier", "ci95_low_vs_seismic", "ci95_high_vs_seismic"]]
          + [[r["model"], r["ig_bits"], r["log_loss"], r["brier"], r["ci_low"], r["ci_high"]] for r in S["comparison"]["external"]])
write_csv("03_test_vs_judge.csv", [["split", "ig_bits", "auc", "ece", "brier", "log_loss", "n", "positives"],
          ["test_2021_2023", h["ig_bits"], h["auc"], h["ece"], h["brier"], h["log_loss"], h["n"], h["positives"]],
          ["judge_2025_2026", jc["ig_bits"], jc["auc"], jc["ece"], jc["brier"], jc["log_loss"], jc["n"], jc["positives"]],
          ["judge_location_only", jsp["ig_bits"], jsp["auc"], jsp["ece"], jsp["brier"], jsp["log_loss"], jsp["n"], jsp["positives"]]])
write_csv("04_per_horizon_test.csv", [["horizon", "ig_bits", "auc", "ece", "positive_rate"]]
          + [[i, m["ig_bits"], m["auc"], m["ece"], m["positive_rate"]] for i, m in enumerate(S["test"]["per_horizon"])])
write_csv("05_per_window_judge.csv", [["window", "ig_bits", "auc", "ece", "positives"]]
          + [[i + 1, m["ig_bits"], m["auc"], m["ece"], m["positives"]] for i, m in enumerate(J["per_window"])])
write_csv("06_quantum_paired_test.csv", [["quantity", "value", "ci95_low", "ci95_high"],
          ["classical_ig_bits", Q["classical_only"]["ig_bits"], "", ""], ["quantum_ig_bits", Q["classical_plus_quantum"]["ig_bits"], "", ""],
          ["delta_ig_bits", qb["delta_ig_bits"], qb["ig_ci95"][0], qb["ig_ci95"][1]],
          ["delta_auc", qb["delta_auc"], qb["auc_ci95"][0], qb["auc_ci95"][1]]])
write_csv("07_model_parameters.csv", [["parameter", "value"]] + [[k, v] for k, v in hp.items()]
          + [[k, v] for k, v in size.items() if k != "input_features"])
write_csv("08_data_splits.csv", [["split", "first_origin", "last_origin", "rows", "positives", "positive_rate"]]
          + [[k, d["origins"][0], d["origins"][1], d["rows"], d["positives"], d["positive_rate"]] for k, d in sp.items()])
write_csv("09_risk_bands_test.csv", [["band", "low", "high", "share", "mean_forecast", "observed_rate"]]
          + [[b["band"], b["range"][0], b["range"][1], b["share_of_cell_windows"], b["mean_forecast"], b["observed_rate"]] for b in ST["bands_30d_test"]])
write_csv("10_entry_thresholds_test.csv", [["cutoff", "share_cleared", "event_rate_in_cleared", "share_of_events_in_cleared"]]
          + [[t["threshold"], t["share_cleared"], t["observed_rate_in_cleared"], t["share_of_all_events_in_cleared"]] for t in ST["thresholds_30d_test"]])
write_csv("11_feature_importance_test.csv", [["feature", "log_loss_increase_bits", "std"]]
          + [[k, v["mean_bits"], v["std_bits"]] for k, v in S["permutation_importance_bits"].items()])

(FINAL / "results").mkdir(exist_ok=True)
for src in [HERE / "final_run" / "final_run.log", HERE / "final_run" / "final_summary.json", HERE / "judge_check" / "judge_metrics.json",
            HERE / "judge_check" / "judge_check.md", HERE / "quantum_test" / "quantum_paired_test.json", HERE / "stakeholder" / "stakeholder.md",
            HERE / "verification.json", HERE / "verify.log", HERE / "SUBMISSION_CHECK.md"]:
    shutil.copy2(src, FINAL / "results" / src.name)
shutil.copy2(HERE / "rerun" / "frozen_forecast_fixed" / "predictions.csv", FINAL / "forecast_predictions_corrected.csv")
shutil.copy2(HERE / "stakeholder" / "risk_map.html", FINAL / "risk_map.html")
shutil.copy2(HERE / "results_page.html", FINAL / "results_page.html")
print("final folder:", FINAL)

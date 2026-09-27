#!/usr/bin/env python3
"""Build the self-contained presentation page ``results_page.html``."""
from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "final_run" / "final_summary.json").read_text())
V = json.loads((HERE / "verification.json").read_text())


def img(name: str, alt: str) -> str:
    data = base64.b64encode((HERE / "figures" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{escape(alt)}" loading="lazy">'


def fig(name, alt, caption):
    return f'<figure class="plate">{img(name, alt)}<figcaption>{caption}</figcaption></figure>'


h = S["test"]["headline"]
m = S["model"]
size = m["learned_size"]
hp = m["hyperparameters"]
sp = S["dataset"]["splits"]
purged = next(c for c in V if c["check"].startswith("purged"))["detail"]

params = [
    ("Algorithm", "HistGradientBoostingClassifier, one model pooled over 15 horizons"),
    ("Loss", hp["loss"]),
    ("Learning rate", hp["learning_rate"]),
    ("Boosting iterations", f"{size['boosting_iterations_run']} of {hp['max_iter']} (early stopping armed, not triggered)"),
    ("Max leaves per tree", hp["max_leaf_nodes"]),
    ("Min rows per leaf", hp["min_samples_leaf"]),
    ("Max depth", f"unlimited (deepest tree reached {size['max_depth_reached']})"),
    ("L2 regularization", hp["l2_regularization"]),
    ("Feature bins", hp["max_bins"]),
    ("Early-stopping holdout", f"random {int(hp['validation_fraction'] * 100)}% of fit rows, patience {hp['n_iter_no_change']}"),
    ("Epochs / batch size", "not applicable: each iteration fits one tree on the full fit set"),
    ("Learned size", f"{size['trees']} trees, {size['leaves_total']:,} leaves, {size['split_nodes_total']:,} splits ≈ {size['learned_parameters_approx']:,} values"),
    ("Calibration", "15 isotonic regressions, one per horizon, fit on 2019–2020 origins only"),
    ("Probability clip", "[0.0001, 0.9999]"),
    ("Seed", hp["random_state"]),
]
param_rows = "".join(f"<tr><th scope='row'>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in params)

features = ["Latitude, longitude", "Event counts within 100 km: 7, 30, 90, 365 days",
            "Days since last event within 100 km (log1p)", "Acceleration ratios 7 d / 30 d and 30 d / 365 d",
            "Horizon index h / 14"]

split_rows = "".join(
    f"<tr><th scope='row'>{name.title()}</th><td>{d['origins'][0][:7]} → {d['origins'][1][:7]}</td>"
    f"<td class='num'>{d['n_origins']}</td><td class='num'>{d['rows']:,}</td>"
    f"<td class='num'>{d['positives']:,}</td><td class='num'>{d['positive_rate']:.2%}</td></tr>"
    for name, d in sp.items())

comp_rows = "".join(
    f"<tr{' class=\"lead\"' if r['model'].startswith('Frozen') else ''}><th scope='row'>{escape(r['model'])}</th>"
    f"<td class='num'>{r['ig_bits']:.4f}</td></tr>"
    for r in sorted(S["comparison"]["models"], key=lambda r: -r["ig_bits"]))

check_rows = "".join(
    f"<tr><td><span class='chip {'ok' if c['pass'] else 'bad'}'>{'Pass' if c['pass'] else 'Fail'}</span></td>"
    f"<td>{escape(c['check'])}</td></tr>" for c in V)
n_pass = sum(c["pass"] for c in V)

top = S["forecast"]["top10"][:5]
top_rows = "".join(f"<tr><td>{t['mask_id'].replace('provisional_', '')}</td><td class='num'>{t['lat']:.1f}°N</td>"
                   f"<td class='num'>{abs(t['lon']):.1f}°W</td><td class='num'>{t['mean_probability']:.4f}</td></tr>"
                   for t in top)

page = f"""<title>Earthquake-Q Frozen Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
:root {{
  --bg: #f3f4f2; --surface: #fcfcfb; --ink: #15181b; --ink-2: #4d5459; --muted: #7c8388;
  --rule: #d9dcd8; --accent: #2a78d6; --accent-ink: #1c5cab; --ok: #0b7a0b; --ok-bg: #e3f2e1;
  --bad: #b33a14; --bad-bg: #fbe7de; --warn-bg: #fff4d6; --warn: #7a5600;
  --plate-ring: rgba(21,24,27,0.08);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #111416; --surface: #1a1d20; --ink: #e8ebe7; --ink-2: #b3b9bd; --muted: #8b9296;
    --rule: #2c3134; --accent: #6aa6ef; --accent-ink: #8dbcf3; --ok: #5cc75c; --ok-bg: #16301a;
    --bad: #f08a66; --bad-bg: #3a1d13; --warn-bg: #33290f; --warn: #f0c75e;
    --plate-ring: rgba(255,255,255,0.10);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #111416; --surface: #1a1d20; --ink: #e8ebe7; --ink-2: #b3b9bd; --muted: #8b9296;
  --rule: #2c3134; --accent: #6aa6ef; --accent-ink: #8dbcf3; --ok: #5cc75c; --ok-bg: #16301a;
  --bad: #f08a66; --bad-bg: #3a1d13; --warn-bg: #33290f; --warn: #f0c75e;
  --plate-ring: rgba(255,255,255,0.10);
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 400 16px/1.6 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  padding-inline: 20px; padding-block: 0 64px; }}
main {{ max-width: 1040px; margin: 0 auto; display: grid; gap: 56px; }}
h1, h2, h3 {{ font-family: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(2rem, 5vw, 3rem); line-height: 1.05; font-weight: 700; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.6rem; font-weight: 600; line-height: 1.2; }}
h3 {{ font-size: 1.1rem; font-weight: 600; }}
p {{ margin: 0; max-width: 68ch; }}
.eyebrow {{ font: 500 0.75rem/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
header {{ padding-block: 48px 0; display: grid; gap: 16px; border-bottom: 1px solid var(--rule); padding-bottom: 32px; }}
.lede {{ font-size: 1.15rem; color: var(--ink-2); }}
.status {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font: 500 0.8rem/1.4 "IBM Plex Sans", sans-serif; white-space: nowrap; }}
.chip.ok {{ background: var(--ok-bg); color: var(--ok); }}
.chip.bad {{ background: var(--bad-bg); color: var(--bad); }}
.chip.warn {{ background: var(--warn-bg); color: var(--warn); }}
.chip::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }}
.metric {{ background: var(--surface); padding: 16px 18px; display: grid; gap: 4px; align-content: start; }}
.metric .v {{ font: 500 1.7rem/1.1 "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.metric .v.lead {{ color: var(--accent-ink); }}
.metric .k {{ font-size: 0.85rem; color: var(--ink-2); }}
section {{ display: grid; gap: 20px; }}
.section-head {{ display: grid; gap: 8px; }}
.plate {{ margin: 0; background: #fcfcfb; border-radius: 6px; box-shadow: 0 0 0 1px var(--plate-ring); overflow: hidden; }}
.plate img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
.plate figcaption {{ background: var(--surface); color: var(--ink-2); font-size: 0.9rem; padding: 12px 16px; border-top: 1px solid var(--rule); }}
.grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 24px; align-items: start; }}
.table-wrap {{ overflow-x: auto; background: var(--surface); border-radius: 6px; box-shadow: 0 0 0 1px var(--plate-ring); }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
th, td {{ text-align: left; padding: 9px 14px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
tr:last-child > * {{ border-bottom: 0; }}
thead th {{ font: 500 0.72rem/1.3 "IBM Plex Mono", monospace; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
tbody th {{ font-weight: 500; color: var(--ink); }}
td.num, th.num {{ text-align: right; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }}
tr.lead th, tr.lead td {{ color: var(--accent-ink); font-weight: 600; }}
.callout {{ background: var(--surface); border-radius: 6px; box-shadow: 0 0 0 1px var(--plate-ring); padding: 20px 22px; display: grid; gap: 10px; }}
.callout.bad {{ box-shadow: inset 3px 0 0 var(--bad), 0 0 0 1px var(--plate-ring); }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; max-width: 72ch; }}
code {{ font: 0.88em "IBM Plex Mono", monospace; background: var(--bg); padding: 1px 5px; border-radius: 3px; }}
footer {{ color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--rule); padding-top: 20px; }}
</style>

<main>
<header>
  <span class="eyebrow">SC Quantathon v3 · SRNL challenge · Earthquake-Q</span>
  <h1>Earthquake-Q Frozen Forecast</h1>
  <p class="lede">Probability of at least one M≥2.0 earthquake within 100 km of each of 1,551 grid cells,
  for 15 consecutive 30-day windows starting 2025-01-01. Every recorded result was rerun from the raw
  catalogs on 2026-09-27 and matched to the last digit.</p>
  <div class="status">
    <span class="chip ok">Results reproduce exactly</span>
    <span class="chip ok">{n_pass} of {len(V)} checks pass</span>
    <span class="chip bad">Recorded forecast file misaligned (fixed)</span>
    <span class="chip warn">Official mask and windows still missing</span>
  </div>
</header>

<section>
  <div class="section-head">
    <span class="eyebrow">Test period · 2021-01 to 2023-09 origins · 767,745 cell-windows</span>
    <h2>Headline result</h2>
  </div>
  <div class="metrics">
    <div class="metric"><span class="v lead">{h['ig_bits']:.4f}</span><span class="k">Information gain, bits per cell-window</span></div>
    <div class="metric"><span class="v">{h['auc']:.3f}</span><span class="k">ROC AUC</span></div>
    <div class="metric"><span class="v">{h['ece']:.4f}</span><span class="k">Calibration error (ECE); reference 0.02</span></div>
    <div class="metric"><span class="v">{h['log_loss']:.4f}</span><span class="k">Log loss, nats</span></div>
    <div class="metric"><span class="v">{h['brier']:.4f}</span><span class="k">Brier score</span></div>
  </div>
  {fig("01_model_comparison.png", "Bar chart of information gain by model",
       "Every model is scored on the same rows with the same per-horizon calibration. The candidate and the ETAS approximation are within noise of each other; both clearly beat the spatial prior and climatology.")}
  {fig("02_external_drivers.png", "Interval chart of external driver effects",
       "Adding geomagnetic, solar, tidal or GPS features never helps. Four of five make the forecast reliably worse, so the frozen model uses seismic history only.")}
</section>

<section>
  <div class="section-head">
    <span class="eyebrow">Stability and calibration</span>
    <h2>Skill holds across all 15 windows</h2>
    <p>Information gain stays between 0.137 and 0.151 bits for every window, and calibration error stays well below the 0.02 reference at every horizon.</p>
  </div>
  {fig("03_per_horizon.png", "Per-horizon information gain, AUC and ECE", "Each point is 51,183 test rows (1,551 cells × 33 origins).")}
  {fig("04_reliability.png", "Reliability diagram and forecast histogram",
       "Forecasts above 1% track observed rates closely. Below 1% the bins contain very few events, so the curve is noisy there.")}
</section>

<section>
  <div class="section-head">
    <span class="eyebrow">Model card</span>
    <h2>Model and training</h2>
    <p>Gradient-boosted trees have no epochs or mini-batches. Each boosting iteration adds one tree fit on the full fit set, so the iteration count is the closest equivalent to epochs.</p>
  </div>
  <div class="grid-2">
    <div class="table-wrap"><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>{param_rows}</tbody></table></div>
    <div style="display:grid;gap:16px">
      <div class="callout"><h3>Inputs (10 features)</h3><ul>{''.join(f'<li>{escape(f)}</li>' for f in features)}</ul>
      <p style="color:var(--ink-2);font-size:.9rem">Features use only events strictly before each origin. One snapshot per origin is shared by all 15 horizons.</p></div>
      <div class="callout"><h3>Environment</h3>
      <p style="font-size:.92rem">Python {S['environment']['python']} · scikit-learn {S['environment']['scikit_learn']} · NumPy {S['environment']['numpy']} · pandas {S['environment']['pandas']}. Final run took {S['runtime_seconds']:.0f} s; full rerun about 23 min.</p></div>
    </div>
  </div>
  {fig("05_training_curve.png", "Training loss curve", "Fit and holdout loss track each other throughout, so the model is not overfitting. Loss was still falling slowly at iteration 120.")}
  {fig("07_feature_importance.png", "Permutation feature importance",
       "Location and long-run activity carry the forecast. Short-window counts and the acceleration ratios add almost nothing when shuffled on their own, because they overlap with the 365-day count.")}
</section>

<section>
  <div class="section-head">
    <span class="eyebrow">Data</span>
    <h2>Data and splits</h2>
    <p>Examples are cell × monthly origin × horizon, split strictly by time. The event rate rises over the period, from about 4% of cell-windows in 2001 to about 7.5% after 2015.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Split</th><th>Origins</th><th class="num">Months</th><th class="num">Rows</th><th class="num">Positives</th><th class="num">Rate</th></tr></thead>
    <tbody>{split_rows}</tbody></table></div>
  {fig("06_data_distribution.png", "Catalog and label distributions",
       f"Seismic catalog: {S['catalog']['seismic_events']:,} events from the train and test files. The judge catalog was never read by the headline pipeline.")}
</section>

<section>
  <div class="section-head">
    <span class="eyebrow">2025-01-01 to 2026-03-27 · provisional contract</span>
    <h2>The 2025 forecast</h2>
    <p>The highest probabilities sit on the New Madrid seismic zone around 36°N 89–90°W, where the model gives near-certainty of an M≥2.0 event each month.</p>
  </div>
  {fig("08_forecast_2025.png", "Map of 2025 forecast probabilities", "23,265 probabilities (1,551 cells × 15 windows). Grid-average probability stays near 8% in every window.")}
  <div class="grid-2">
    <div class="table-wrap"><table><thead><tr><th>Cell</th><th class="num">Lat</th><th class="num">Lon</th><th class="num">Mean p</th></tr></thead><tbody>{top_rows}</tbody></table></div>
    <div class="callout bad"><h3>Forecast file fix</h3>
      <p>The recorded <code>predictions.csv</code> put 22,163 of 23,265 probabilities on the wrong cell or window. Probabilities were computed window by window but written cell by cell. The corrected file tracks each cell's 2024 activity (Spearman 0.79, against −0.04 for the recorded file).</p></div>
  </div>
  {fig("09_forecast_alignment_fix.png", "Recorded versus corrected forecast for three cells",
       "With one feature snapshot, a cell's probability should vary smoothly across windows. The recorded file jumps by orders of magnitude.")}
</section>

<section>
  <div class="section-head">
    <span class="eyebrow">Verification</span>
    <h2>Is it ready to submit?</h2>
    <p>The model and every reported number are ready. The submission file is not, for two reasons: the organizers have not supplied the official cell mask, order and window table, and the recorded forecast must be replaced by the corrected one.</p>
  </div>
  <div class="grid-2">
    <div class="table-wrap"><table><thead><tr><th>Status</th><th>Check</th></tr></thead><tbody>{check_rows}</tbody></table></div>
    <div style="display:grid;gap:16px">
      <div class="callout"><h3>Caveats to state when presenting</h3><ul>
        <li>The test period was used to choose between models, so 0.1458 bits is a development score, not an untouched holdout.</li>
        <li>Some training and calibration labels extend past the end of their split. Removing those rows gives {purged['purged_ig_bits']:.4f} bits ({purged['delta_bits']:+.4f}). That is larger than the gap to ETAS, so treat close rankings as ties.</li>
        <li>Early stopping held out a random 10% of fit rows, so the model trained on 90% of them.</li>
        <li>The quantum ablation used two inputs, almost all zero. It shows no difference either way and is not evidence against quantum methods.</li>
        <li>ETAS parameters were reused from an earlier 644-cell fit.</li>
      </ul></div>
      <div class="callout"><h3>Before submitting</h3><ul>
        <li>Receive the official mask, cell order, windows and schema, and record their hashes.</li>
        <li>Regenerate with <code>run_frozen_forecast_fixed.py</code> on those files.</li>
        <li>Validate 23,265 rows in the supplied order.</li>
        <li>Update FINAL_RESULTS.md, the manifest and the experiment log with these findings.</li>
      </ul></div>
    </div>
  </div>
</section>

<footer>Generated from <code>pipeline/analysis</code>: <code>final_run/final_run.log</code>, <code>verification.json</code>, <code>figures/</code>. Test data 2021–2023; forecast contract provisional and not official.</footer>
</main>
"""
(HERE / "results_page.html").write_text(page)
print("wrote", HERE / "results_page.html", f"{len(page) / 1e6:.2f} MB")

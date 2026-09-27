#!/usr/bin/env python3
"""Build the clickable aftershock-risk map ``stakeholder/risk_map.html``."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "stakeholder" / "stakeholder.json").read_text())
payload = {
    "cells": [[c["lat"], c["lon"], c["id"], c["p"], c["p6"]] for c in D["cells"]],
    "windows": D["windows"],
    "bands30": D["bands_30d_test"],
    "bands6": D["bands_6m_test"],
    "thresholds": D["thresholds_30d_test"],
    "six": {k: v for k, v in D["six_month_test_validation"].items() if k != "reliability_decile"},
}

threshold_rows = "".join(
    f"<tr><td class='num'>{t['threshold']:.1%}</td><td class='num'>{t['share_cleared']:.0%}</td>"
    f"<td class='num'>{t['observed_rate_in_cleared']:.2%}</td><td class='num'>{t['share_of_all_events_in_cleared']:.1%}</td></tr>"
    for t in D["thresholds_30d_test"])
band_rows = "".join(
    f"<tr><td><span class='sw' data-band='{b['band']}'></span>{b['band']}</td><td class='num'>{b['range'][0]:.0%}–{b['range'][1]:.0%}</td>"
    f"<td class='num'>{b['mean_forecast']:.1%}</td><td class='num'>{b['observed_rate']:.1%}</td></tr>"
    for b in D["bands_30d_test"])
six = D["six_month_test_validation"]

page = """<title>Aftershock Risk Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #f3f4f2; --surface: #fcfcfb; --ink: #15181b; --ink-2: #4d5459; --muted: #7c8388;
  --rule: #d9dcd8; --accent: #1c5cab; --ring: rgba(21,24,27,0.10); --focus: #2a78d6;
  --city: #15181b; --empty: #e7e9e5;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #111416; --surface: #1a1d20; --ink: #e8ebe7; --ink-2: #b3b9bd; --muted: #8b9296;
    --rule: #2c3134; --accent: #8dbcf3; --ring: rgba(255,255,255,0.12); --focus: #6aa6ef;
    --city: #f3f4f2; --empty: #23272a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #111416; --surface: #1a1d20; --ink: #e8ebe7; --ink-2: #b3b9bd; --muted: #8b9296;
  --rule: #2c3134; --accent: #8dbcf3; --ring: rgba(255,255,255,0.12); --focus: #6aa6ef;
  --city: #f3f4f2; --empty: #23272a;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 400 15px/1.55 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  padding-inline: 18px; padding-block: 0 48px; }
main { max-width: 1180px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { font-family: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif; margin: 0; text-wrap: balance; }
h1 { font-size: clamp(1.8rem, 4vw, 2.5rem); line-height: 1.05; }
h2 { font-size: 1.3rem; }
p { margin: 0; max-width: 70ch; }
.eyebrow { font: 500 0.72rem/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
header { padding-top: 36px; display: grid; gap: 10px; }
.lede { color: var(--ink-2); font-size: 1.05rem; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: end; }
.controls label { display: grid; gap: 4px; font-size: 0.8rem; color: var(--ink-2); }
select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 6px 10px; min-width: 200px; }
select:focus-visible, canvas:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.layout { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(280px, 1fr); gap: 20px; align-items: start; }
@media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }
.card { background: var(--surface); border-radius: 8px; box-shadow: 0 0 0 1px var(--ring); }
.mapwrap { padding: 10px; display: grid; gap: 8px; }
canvas { width: 100%; height: auto; display: block; cursor: crosshair; border-radius: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 0.8rem; color: var(--ink-2); padding: 2px 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 2px; display: inline-block; flex: none; margin-right: 6px; vertical-align: -1px; }
.legend .sw { margin-right: 0; }
.panel { padding: 20px; display: grid; gap: 16px; }
.panel .place { font: 600 1.15rem/1.2 "IBM Plex Sans Condensed", sans-serif; }
.panel .coords { font: 0.8rem "IBM Plex Mono", monospace; color: var(--muted); }
.big { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--rule); border-radius: 6px; overflow: hidden; }
.big > div { background: var(--surface); padding: 12px 14px; display: grid; gap: 2px; }
.big .v { font: 500 1.7rem/1.1 "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.big .k { font-size: 0.8rem; color: var(--ink-2); }
.pill { justify-self: start; display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
  background: var(--bg); box-shadow: 0 0 0 1px var(--ring); }
.note { font-size: 0.85rem; color: var(--ink-2); }
.spark { width: 100%; height: 120px; display: block; }
.tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); gap: 20px; }
.tables .card { padding: 18px; display: grid; gap: 10px; align-content: start; }
.tw { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--rule); }
thead th { font: 500 0.7rem/1.3 "IBM Plex Mono", monospace; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
footer { font-size: 0.8rem; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
</style>

<main>
<header>
  <span class="eyebrow">Earthquake-Q · forecast issued 2025-01-01 · provisional 1,551-cell grid</span>
  <h1>Aftershock Risk Map</h1>
  <p class="lede">Click any point, such as an epicenter, to see the chance of at least one M≥2.0 earthquake within 100 km:
  per 30-day window for emergency crews, and over the next six months for repair crews.</p>
</header>

<div class="controls">
  <label for="view">Map shows
    <select id="view"></select></label>
  <label for="city">Jump to a place
    <select id="city"><option value="">Choose a place…</option></select></label>
</div>

<div class="layout">
  <div class="card mapwrap">
    <canvas id="map" width="1000" height="760" tabindex="0" aria-label="Forecast map. Click a location to see its risk."></canvas>
    <div class="legend" id="legend"></div>
  </div>
  <aside class="card panel" id="panel" aria-live="polite"></aside>
</div>

<div class="tables">
  <div class="card">
    <h2>What each band meant in practice</h2>
    <p class="note">30-day forecasts on 767,745 test cell-windows (2021–2023). Observed rate is how often an M≥2.0 event actually followed.</p>
    <div class="tw"><table><thead><tr><th>Band</th><th class="num">Forecast</th><th class="num">Avg forecast</th><th class="num">Observed</th></tr></thead>
    <tbody>__BANDS__</tbody></table></div>
  </div>
  <div class="card">
    <h2>Choosing an entry threshold</h2>
    <p class="note">If crews enter only where the 30-day forecast is below the cut-off, this is what happened on test data. The acceptable level is a decision for emergency managers, not the model.</p>
    <div class="tw"><table><thead><tr><th class="num">Cut-off</th><th class="num">Cells cleared</th><th class="num">Event rate there</th><th class="num">Events missed</th></tr></thead>
    <tbody>__THRESH__</tbody></table></div>
  </div>
</div>

<footer>6-month risk = 1 − Π(1 − p) over windows 1–6. Treating windows as independent overstates risk where quakes cluster, so the six-month figure errs on the cautious side at moderate and high risk
(test: forecast __P6__ vs observed __O6__ on average; 0.379 bits of information gain). In the lowest decile the six-month forecast was 0.06% while 0.45% was observed, so treat values under 1% as "under 1%", not as zero.
Grid, windows and model are provisional development outputs, not an official hazard product.</footer>
</main>

<script>
const D = __DATA__;
const BANDS = [[0, .01, "Low", "#cde2fb"], [.01, .05, "Guarded", "#86b6ef"], [.05, .2, "Elevated", "#3987e5"],
               [.2, .5, "High", "#1c5cab"], [.5, 1.01, "Very high", "#0d366b"]];
const CITIES = [["New Madrid, MO", 36.59, -89.53], ["Memphis, TN", 35.15, -89.95], ["Knoxville, TN", 35.96, -83.92],
  ["Charleston, SC", 32.78, -79.93], ["Atlanta, GA", 33.75, -84.39], ["Charlotte, NC", 35.23, -80.84],
  ["Washington, DC", 38.9, -77.04], ["New York, NY", 40.71, -74.0], ["Boston, MA", 42.36, -71.06],
  ["Montréal, QC", 45.5, -73.57], ["Québec City, QC", 46.81, -71.21], ["Chicago, IL", 41.88, -87.63],
  ["Cleveland, OH", 41.5, -81.69], ["Miami, FL", 25.76, -80.19]];
const LAT0 = 24.6, LAT1 = 50.4, LON0 = -90.4, LON1 = -64.9, K = Math.cos(37.5 * Math.PI / 180);
const cv = document.getElementById("map"), ctx = cv.getContext("2d");
const W = cv.width, Hc = cv.height, PAD = 36;
const sx = (W - 2 * PAD) / ((LON1 - LON0) * K), sy = (Hc - 2 * PAD) / (LAT1 - LAT0), S = Math.min(sx, sy);
const ox = (W - (LON1 - LON0) * K * S) / 2, oy = (Hc - (LAT1 - LAT0) * S) / 2;
const X = lon => ox + (lon - LON0) * K * S, Y = lat => oy + (LAT1 - lat) * S;
const band = p => BANDS.find(b => p >= b[0] && p < b[1]);
const fmt = p => p < 0.001 ? "<0.1%" : p >= 0.999 ? ">99.9%" : (p < 0.1 ? (p * 100).toFixed(1) : (p * 100).toFixed(0)) + "%";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const view = document.getElementById("view"), citySel = document.getElementById("city");
let sel = null, clicked = null;

view.add(new Option("6-month risk (windows 1–6)", "6"));
D.windows.forEach((w, i) => view.add(new Option(`30-day risk: window ${i + 1} (${w.slice(0, 10)})`, String(i))));
view.value = "0";
CITIES.forEach((c, i) => citySel.add(new Option(c[0], String(i))));
document.getElementById("legend").innerHTML = BANDS.map(b =>
  `<span><i class="sw" style="background:${b[3]}"></i>${b[2]} ${b[0] * 100}–${Math.min(b[1], 1) * 100}%</span>`).join("");
document.querySelectorAll(".sw[data-band]").forEach(e => { const b = BANDS.find(x => x[2] === e.dataset.band); e.style.background = b[3]; });

function value(c) { return view.value === "6" ? c[4] : c[3][+view.value]; }

function draw() {
  ctx.clearRect(0, 0, W, Hc);
  ctx.fillStyle = "#fcfcfb"; ctx.fillRect(0, 0, W, Hc);
  ctx.strokeStyle = "#e1e0d9"; ctx.lineWidth = 1; ctx.font = "12px 'IBM Plex Mono', monospace"; ctx.fillStyle = "#6b7176";
  for (let lat = 25; lat <= 50; lat += 5) { const y = Y(lat); ctx.beginPath(); ctx.moveTo(X(LON0), y); ctx.lineTo(X(LON1), y); ctx.stroke(); ctx.fillText(lat + "°N", 2, y + 4); }
  for (let lon = -90; lon <= -65; lon += 5) { const x = X(lon); ctx.beginPath(); ctx.moveTo(x, Y(LAT1)); ctx.lineTo(x, Y(LAT0)); ctx.stroke(); ctx.fillText(-lon + "°W", x - 14, Hc - 8); }
  const cw = 0.5 * K * S - 1, ch = 0.5 * S - 1;
  for (const c of D.cells) { ctx.fillStyle = band(value(c))[3]; ctx.fillRect(X(c[1]) - cw / 2, Y(c[0]) - ch / 2, cw, ch); }
  ctx.fillStyle = "#15181b"; ctx.strokeStyle = "#fcfcfb"; ctx.lineWidth = 3; ctx.font = "600 12px 'IBM Plex Sans', sans-serif";
  for (const [n, la, lo] of CITIES) {
    const x = X(lo), y = Y(la); ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 7); ctx.fill();
    const label = n.split(",")[0]; ctx.strokeText(label, x + 7, y + 4); ctx.fillText(label, x + 7, y + 4);
  }
  if (sel) {
    ctx.strokeStyle = "#eb6834"; ctx.lineWidth = 2.5;
    ctx.strokeRect(X(sel[1]) - cw / 2 - 2, Y(sel[0]) - ch / 2 - 2, cw + 4, ch + 4);
    const r = 100 / 111 * S;  // 100 km in latitude pixels
    ctx.beginPath(); ctx.ellipse(X(sel[1]), Y(sel[0]), r / Math.cos(sel[0] * Math.PI / 180) * K, r, 0, 0, 7); ctx.stroke();
  }
  if (clicked) { ctx.fillStyle = "#eb6834"; ctx.beginPath(); ctx.arc(X(clicked[1]), Y(clicked[0]), 5, 0, 7); ctx.fill(); }
}

function nearest(lat, lon) {
  let best = null, bd = 1e9;
  for (const c of D.cells) { const d = Math.hypot((c[0] - lat), (c[1] - lon) * Math.cos(lat * Math.PI / 180)); if (d < bd) { bd = d; best = c; } }
  return bd <= 0.4 ? best : null;
}

function spark(c) {
  const w = 320, h = 120, pl = 34, pb = 20, n = c[3].length, bw = (w - pl - 4) / n;
  const bars = c[3].map((p, i) => {
    const bh = Math.max(1.5, p * (h - pb - 8)); const x = pl + i * bw + 1;
    return `<rect x="${x.toFixed(1)}" y="${(h - pb - bh).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${bh.toFixed(1)}" rx="2" fill="${band(p)[3]}"><title>Window ${i + 1} (${D.windows[i].slice(0, 10)}): ${fmt(p)}</title></rect>`;
  }).join("");
  const grid = [0, .5, 1].map(v => { const y = h - pb - v * (h - pb - 8);
    return `<line x1="${pl}" x2="${w}" y1="${y}" y2="${y}" stroke="var(--rule)"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="var(--muted)" font-family="IBM Plex Mono, monospace">${v * 100}%</text>`; }).join("");
  const xl = `<text x="${pl}" y="${h - 5}" font-size="10" fill="var(--muted)">Jan 2025</text><text x="${w}" y="${h - 5}" text-anchor="end" font-size="10" fill="var(--muted)">Feb 2026</text>`;
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="30-day probability for each of the 15 windows">${grid}${bars}${xl}</svg>`;
}

function show(lat, lon, label) {
  clicked = [lat, lon]; sel = nearest(lat, lon);
  const panel = document.getElementById("panel");
  if (!sel) {
    panel.innerHTML = `<span class="eyebrow">Selected point</span><div class="place">${label || "Outside the grid"}</div>
      <div class="coords">${lat.toFixed(2)}°N ${(-lon).toFixed(2)}°W</div>
      <p class="note">This point is outside the forecast grid (25–50°N, 65–90°W). Click inside the colored area.</p>`;
    draw(); return;
  }
  const w1 = sel[3][0], b30 = band(w1), b6 = band(sel[4]);
  const t30 = D.bands30.find(b => b.band === b30[2]), t6 = D.bands6.find(b => b.band === b6[2]);
  panel.innerHTML = `<span class="eyebrow">Nearest grid cell ${sel[2]} · 100 km radius</span>
    <div class="place">${label || "Selected point"}</div>
    <div class="coords">clicked ${lat.toFixed(2)}°N ${(-lon).toFixed(2)}°W · cell centre ${sel[0].toFixed(1)}°N ${(-sel[1]).toFixed(1)}°W</div>
    <div class="big">
      <div><span class="v">${fmt(w1)}</span><span class="k">Next 30 days (window 1)</span></div>
      <div><span class="v">${fmt(sel[4])}</span><span class="k">Next 6 months (windows 1–6)</span></div>
    </div>
    <span class="pill"><i class="sw" style="background:${b30[3]};margin:0"></i>30-day band: ${b30[2]}</span>
    <p class="note">In 2021–2023 testing, 30-day forecasts in the ${b30[2]} band averaged ${fmt(t30.mean_forecast)} and an event followed ${fmt(t30.observed_rate)} of the time.
    ${t6 ? `Six-month forecasts in the ${b6[2]} band averaged ${fmt(t6.mean_forecast)}; observed ${fmt(t6.observed_rate)}.` : ""}</p>
    <div><h3 style="font-size:.95rem;margin-bottom:4px">30-day probability, window by window</h3>${spark(sel)}</div>`;
  draw();
}

cv.addEventListener("click", e => {
  const r = cv.getBoundingClientRect(), x = (e.clientX - r.left) * W / r.width, y = (e.clientY - r.top) * Hc / r.height;
  show(LAT1 - (y - oy) / S, LON0 + (x - ox) / (K * S));
});
view.addEventListener("change", draw);
citySel.addEventListener("change", () => { const c = CITIES[+citySel.value]; if (c) show(c[1], c[2], c[0]); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
show(CITIES[0][1], CITIES[0][2], CITIES[0][0]);
</script>
"""
page = (page.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__BANDS__", band_rows).replace("__THRESH__", threshold_rows)
        .replace("__P6__", f"{six['mean_forecast']:.1%}").replace("__O6__", f"{six['observed_rate']:.1%}"))
(HERE / "stakeholder" / "risk_map.html").write_text(page)
print("wrote", HERE / "stakeholder" / "risk_map.html", f"{len(page) / 1e3:.0f} kB")

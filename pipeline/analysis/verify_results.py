#!/usr/bin/env python3
"""Check the rerun artifacts against every recorded number and the forecast file.

Run after ``rerun_all.py``.  Writes ``verification.json`` and prints a
PASS/FAIL line per check.  Never reads the judge catalog.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RERUN = HERE / "rerun"
sys.path.insert(0, str(ROOT))

from run_contract_interface import contract_rows, load_mask, load_windows  # noqa: E402
from run_full_grid_development_comparison import ACCEL, catalog, snapshot  # noqa: E402

H = 15
N_CELLS = 1551
TOL = 1e-9          # reruns are deterministic; anything above this is a real change
SEED = 42
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail) -> None:
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def load_json(path: Path):
    return json.loads(path.read_text())


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
# 1. Reproducibility of recorded artifacts
# --------------------------------------------------------------------------

def check_historical_examples() -> None:
    old = np.load(ROOT / "full_grid_development_comparison" / "historical_examples.npz")
    new = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    diffs = {k: (not np.array_equal(old[k], new[k])) for k in old.files}
    check("historical examples regenerate bit-identically from raw catalogs",
          not any(diffs.values()), {k: ("differs" if d else "identical") for k, d in diffs.items()})


def by_model(rows):
    return {r["model"]: r for r in rows}


def check_full_grid() -> None:
    old = by_model(load_json(ROOT / "full_grid_development_comparison" / "comparison_metrics.json")["overall"])
    new = by_model(load_json(RERUN / "full_grid_development_comparison" / "comparison_metrics.json")["overall"])
    detail, ok = {}, True
    for name in old:
        for key in ("ig_bits", "log_loss", "brier", "auc"):
            same = close(old[name][key], new[name][key])
            ok &= same
            if not same:
                detail[f"{name}.{key}"] = [old[name][key], new[name][key]]
    check("full-grid development comparison metrics reproduce", ok, detail or "all models identical")
    cand = new["pooled_numeric_horizon_index"]
    check("headline candidate matches FINAL_RESULTS.md exact values",
          close(cand["ig_bits"], 0.14582367854116265)
          and close(cand["log_loss"], 0.1687691602520702)
          and close(cand["brier"], 0.04962805239809624),
          {"ig_bits": cand["ig_bits"], "log_loss": cand["log_loss"], "brier": cand["brier"]})


def check_reporting_gaps() -> None:
    new = load_json(RERUN / "full_grid_development_comparison" / "reporting_gap_diagnostics.json")
    c = "candidate_pooled_numeric_horizon_index"
    got = {
        "overall_ece": new["overall"][c]["ece_10_equal_width"],
        "early_ece": new["horizon_groups"][c]["early"]["ece_10_equal_width"],
        "middle_ece": new["horizon_groups"][c]["middle"]["ece_10_equal_width"],
        "late_ece": new["horizon_groups"][c]["late"]["ece_10_equal_width"],
        "spatial_prior_ig_bits": new["overall"]["spatial_prior_train_cell_horizon"]["ig_bits"],
    }
    recorded = load_json(ROOT / "frozen_pipeline_manifest.json")["comparison_status"]["reporting_diagnostics"]
    # The manifest stores these rounded to 6 decimals.
    ok = all(close(got[k], recorded[k], 5e-7) for k in recorded)
    check("ECE and spatial-prior diagnostics match manifest", ok,
          {k: [recorded[k], round(got[k], 6)] for k in recorded})
    check("all ECE values under the Deck's 0.02 reference",
          all(v < 0.02 for k, v in got.items() if k.endswith("ece")),
          {k: round(v, 6) for k, v in got.items() if k.endswith("ece")})


def check_external() -> None:
    old = load_json(ROOT / "final_matched_external_driver_comparison" / "metrics.json")
    new = load_json(RERUN / "final_matched_external_driver_comparison" / "metrics.json")
    o, n = by_model(old["overall"]), by_model(new["overall"])
    ok = all(close(o[m][k], n[m][k]) for m in o for k in ("ig_bits", "log_loss", "brier"))
    po = old["paired_contiguous_origin_block_bootstrap"]
    pn = new["paired_contiguous_origin_block_bootstrap"]
    for m in po:
        if m == "seismic_only":
            continue
        for k in ("ci95_ig_improvement_bits_low", "ci95_ig_improvement_bits_high"):
            ok &= close(po[m][k], pn[m][k])
    check("external-driver comparison (metrics + block intervals) reproduces", ok,
          {m: round(n[m]["ig_bits"], 6) for m in n})
    check("external seismic-only baseline equals headline candidate",
          close(n["seismic_only"]["ig_bits"], 0.14582367854116265), n["seismic_only"]["ig_bits"])
    no_gain = all(pn[m]["ci95_ig_improvement_bits_low"] < 0 or pn[m]["mean_ig_improvement_bits"] < 0
                  for m in pn if m != "seismic_only")
    check("no external driver shows a reliable gain (decision holds)", no_gain,
          {m: [round(pn[m]["ci95_ig_improvement_bits_low"], 5), round(pn[m]["ci95_ig_improvement_bits_high"], 5)]
           for m in pn if m != "seismic_only"})


def check_etas() -> None:
    old = load_json(ROOT / "matched_etas_frozen_comparison" / "metrics.json")
    new = load_json(RERUN / "matched_etas_frozen_comparison" / "metrics.json")
    ok = all(close(old["metrics"][m][k], new["metrics"][m][k])
             for m in old["metrics"] for k in ("ig_bits", "log_loss", "brier"))
    ob, nb = old["etas_vs_frozen_paired_origin_bootstrap"], new["etas_vs_frozen_paired_origin_bootstrap"]
    ok &= all(close(ob[k], nb[k]) for k in ob)
    manifest = load_json(ROOT / "frozen_pipeline_manifest.json")["comparison_status"]["matched_full_grid_etas"]
    ok &= close(new["metrics"]["etas"]["ig_bits"], manifest["ig_bits"])
    ok &= close(nb["ci95_ig_bits_low"], manifest["paired_ci95_bits"][0])
    check("matched ETAS comparison reproduces and matches manifest", ok,
          {"etas_ig": new["metrics"]["etas"]["ig_bits"],
           "delta_bits": nb["mean_log_loss_improvement"] / np.log(2),
           "ci95": [nb["ci95_ig_bits_low"], nb["ci95_ig_bits_high"]]})


def check_qml() -> None:
    old = load_json(ROOT / "day7_qml_ablation" / "metrics.json")
    new = load_json(RERUN / "day7_qml_ablation" / "metrics.json")
    keys = ("qml_quantum_features", "classical_two_input_control")
    ok = all(close(old[m][k], new[m][k]) for m in keys for k in ("ig_bits", "log_loss", "brier", "auc"))
    check("Day-7 QML ablation reproduces", ok,
          {m: round(new[m]["ig_bits"], 6) for m in keys})


# --------------------------------------------------------------------------
# 2. Forecast file
# --------------------------------------------------------------------------

def check_forecast() -> dict:
    mask = load_mask(ROOT / "provisional_contract" / "provisional_mask.csv")
    windows = load_windows(ROOT / "provisional_contract" / "provisional_windows.csv")
    rank = windows["window_start"].rank(method="first").astype(int).to_numpy() - 1
    rows = contract_rows(mask, windows)
    old = pd.read_csv(ROOT / "frozen_provisional_forecast" / "predictions.csv")
    new = pd.read_csv(RERUN / "frozen_forecast_fixed" / "predictions.csv")
    matrix = np.load(RERUN / "frozen_forecast_fixed" / "probability_matrix.npy")

    # Structure
    keys_match = (
        new[["lat", "lon"]].to_numpy().tolist() == rows[["lat", "lon"]].to_numpy().tolist()
        and (pd.to_datetime(new.window_start).to_numpy() == pd.to_datetime(rows.window_start).to_numpy()).all()
        and (new.mask_id.to_numpy() == rows.mask_id.to_numpy()).all()
    )
    p = new.probability.to_numpy()
    check("fixed forecast: 23,265 rows in exact contract order",
          len(new) == 23265 and keys_match, {"rows": len(new), "order_matches_contract": keys_match})
    check("fixed forecast: probabilities finite, clipped to [1e-4, 1-1e-4], no duplicate keys",
          np.isfinite(p).all() and p.min() >= 1e-4 and p.max() <= 1 - 1e-4
          and not new[["mask_id", "window_start"]].duplicated().any(),
          {"min": float(p.min()), "max": float(p.max()), "mean": float(p.mean())})
    expected = matrix[np.repeat(np.arange(N_CELLS), H), np.tile(rank, N_CELLS)]
    check("fixed forecast: every row equals matrix[cell, window rank]",
          np.allclose(p, expected, rtol=0, atol=1e-12),
          {"max_abs_diff": float(np.abs(p - expected).max()), "note": "CSV float round-trip only"})

    # Diagnose the recorded file
    same_order = np.isclose(old.probability.to_numpy(), p)
    as_horizon_major = matrix[:, :].T.reshape(-1)  # horizon-major, cell-minor
    recorded_is_transposed = np.allclose(old.probability.to_numpy(), as_horizon_major)
    check("RECORDED forecast rows are correctly aligned (known bug; expected FAIL)", bool(same_order.all()),
          {"rows_wrong": int((~same_order).sum()),
           "recorded_equals_horizon_major_vector": bool(recorded_is_transposed)})

    # Physical sanity: prob should track each cell's recent activity.
    events = catalog()
    cells = mask[["lat", "lon"]].to_numpy(float)
    snap = snapshot(events, pd.Timestamp("2025-01-01"), cells)
    count365 = np.repeat(snap[:, 5], H)  # 100-km 365-day count, per contract row
    rho_new = spearmanr(count365, p).statistic
    rho_old = spearmanr(count365, old.probability.to_numpy()).statistic
    check("fixed forecast tracks each cell's 2024 activity better than recorded file",
          rho_new > rho_old and rho_new > 0.5,
          {"spearman_fixed": round(float(rho_new), 4), "spearman_recorded": round(float(rho_old), 4)})
    return {"snapshot": snap, "mask": mask, "windows": windows, "new": new}


def check_forecast_independent(ctx: dict) -> None:
    """Refit the frozen model here and recompute 200 random rows by key."""
    z = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED)
    model = HistGradientBoostingClassifier(**args).fit(
        np.column_stack([z["x_train"][:, ACCEL], z["h_train"]]), z["y_train"])
    raw_cal = model.predict_proba(np.column_stack([z["x_cal"][:, ACCEL], z["h_cal"]]))[:, 1]
    iso = {h: IsotonicRegression(out_of_bounds="clip").fit(
        raw_cal[z["h_cal"] == np.float32(h / 14)], z["y_cal"][z["h_cal"] == np.float32(h / 14)])
        for h in range(H)}
    mask, windows, new = ctx["mask"], ctx["windows"], ctx["new"]
    starts = sorted(pd.to_datetime(windows.window_start))
    picks = new.sample(200, random_state=7)
    id_to_pos = {m: i for i, m in enumerate(mask.mask_id)}
    worst = 0.0
    for _, row in picks.iterrows():
        i = id_to_pos[row.mask_id]
        h = starts.index(pd.Timestamp(row.window_start))
        x = np.append(ctx["snapshot"][i, ACCEL], np.float32(h / 14))[None, :]
        want = np.clip(iso[h].predict(model.predict_proba(x)[:, 1]), 1e-4, 1 - 1e-4)[0]
        worst = max(worst, abs(want - row.probability))
    check("fixed forecast: 200 random rows recomputed independently by (mask_id, window)",
          worst < 1e-9, {"max_abs_diff": worst})


# --------------------------------------------------------------------------
# 3. Protocol checks
# --------------------------------------------------------------------------

HEADLINE_CHAIN = [
    "run_full_grid_development_comparison.py", "diagnose_full_grid_reporting_gaps.py",
    "run_final_matched_external_driver_comparison.py", "run_external_driver_ablation.py",
    "run_matched_etas_frozen_comparison.py", "run_day7_qml_ablation.py", "run_multiscale_experiment.py",
    "run_contract_interface.py", "official_grid_contract.py", "run_frozen_provisional_forecast.py",
]
DOCUMENTED_JUDGE_USE = {"audit_dataset.py": "data audit (hashes/counts)",
                        "prepare_derived_catalogs.py": "derived-catalog preparation",
                        "run_judge_comparison.py": "documented earlier judge self-check on older models"}


def check_judge_not_read() -> None:
    hits = []
    for script in [ROOT / n for n in HEADLINE_CHAIN] + [HERE / "run_frozen_forecast_fixed.py"]:
        text = script.read_text()
        if re.search(r"earthquakeq_judge|judge\.csv|[\"']judge[\"']\s*[,)]", text):
            hits.append(script.name)
    others = sorted(n.name for n in ROOT.glob("*.py")
                    if n.name not in HEADLINE_CHAIN and re.search(r"judge", n.read_text())
                    and re.search(r"\(\"train\", \"test\", \"judge\"\)|read_split\(\"judge\"\)", n.read_text()))
    check("headline pipeline scripts never read the judge catalog", not hits,
          {"headline_hits": hits or "none",
           "other_scripts_that_read_judge": {n: DOCUMENTED_JUDGE_USE.get(n, "UNDOCUMENTED") for n in others}})


def check_label_overlap_sensitivity() -> None:
    """Fit/calibration labels reach up to 15 months past their origins, so late
    fit rows label events inside the calibration period and late calibration
    rows label events inside the test period.  Drop those rows and refit."""
    z = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    days = 30 * (np.arange(H) + 1)

    def keep(origins, boundary, n_rows):
        o = pd.DatetimeIndex(origins).to_numpy()
        end = o[:, None] + days[None, :].astype("timedelta64[D]")
        ok = np.repeat((end <= np.datetime64(boundary)).reshape(-1), N_CELLS)
        assert ok.size == n_rows
        return ok

    fit_ok = keep(pd.date_range("2001-01-01", "2018-12-01", freq="MS"), "2019-01-01", len(z["y_train"]))
    cal_ok = keep(pd.date_range("2019-01-01", "2020-12-01", freq="MS"), "2021-01-01", len(z["y_cal"]))
    # Calibration origins after 2019-12 have no fully purged horizon>0 rows;
    # per-horizon isotonic needs rows for every horizon, so report counts.
    cal_h = np.rint(z["h_cal"] * 14).astype(int)
    per_h = np.bincount(cal_h[cal_ok], minlength=H)
    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED)
    xtr = np.column_stack([z["x_train"][:, ACCEL], z["h_train"]])
    xca = np.column_stack([z["x_cal"][:, ACCEL], z["h_cal"]])
    xte = np.column_stack([z["x_test"][:, ACCEL], z["h_test"]])
    yte = z["y_test"]
    base = float(z["y_train"].mean())
    ref_ll = log_loss(yte, np.full(len(yte), base))
    model = HistGradientBoostingClassifier(**args).fit(xtr[fit_ok], z["y_train"][fit_ok])
    raw_ca, raw_te = model.predict_proba(xca)[:, 1], model.predict_proba(xte)[:, 1]
    p = np.empty(len(yte))
    te_h = np.rint(z["h_test"] * 14).astype(int)
    for h in range(H):
        a = (cal_h == h) & cal_ok
        if np.unique(z["y_cal"][a]).size < 2:
            a = cal_h == h  # no purged rows left for this horizon; fall back
        p[te_h == h] = IsotonicRegression(out_of_bounds="clip").fit(raw_ca[a], z["y_cal"][a]).predict(raw_te[te_h == h])
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ig = (ref_ll - log_loss(yte, p)) / np.log(2)
    test_rate_ig = (log_loss(yte, np.full(len(yte), yte.mean())) - log_loss(yte, p)) / np.log(2)
    check("purged-split sensitivity (no label windows crossing split boundaries)",
          ig > 0.13,
          {"purged_ig_bits": round(float(ig), 6), "recorded_ig_bits": 0.145824,
           "delta_bits": round(float(ig - 0.14582367854116265), 6),
           "fit_rows_kept": int(fit_ok.sum()), "cal_rows_kept_per_horizon": per_h.tolist(),
           "ig_vs_test_period_climatology_bits": round(float(test_rate_ig), 6)})


def check_reference_rate() -> None:
    pred = np.load(RERUN / "full_grid_development_comparison" / "test_predictions.npz")
    y, p = pred["y"], pred["pooled_numeric_horizon_index"]
    ex = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    train_rate, test_rate = float(ex["y_train"].mean()), float(y.mean())
    ig_train = (log_loss(y, np.full(len(y), train_rate)) - log_loss(y, p)) / np.log(2)
    ig_test = (log_loss(y, np.full(len(y), test_rate)) - log_loss(y, p)) / np.log(2)
    check("candidate beats climatology under either reference rate",
          ig_train > 0 and ig_test > 0,
          {"train_rate": round(train_rate, 5), "test_rate": round(test_rate, 5),
           "ig_vs_train_rate": round(float(ig_train), 6), "ig_vs_test_rate": round(float(ig_test), 6)})


def main() -> None:
    steps = [check_historical_examples, check_full_grid, check_reporting_gaps, check_external,
             check_etas, check_qml, check_judge_not_read, check_reference_rate]
    for step in steps:
        try:
            step()
        except Exception as exc:
            check(step.__name__, False, f"error: {type(exc).__name__}: {exc}")
    try:
        ctx = check_forecast()
        check_forecast_independent(ctx)
    except Exception as exc:
        check("forecast checks", False, f"error: {type(exc).__name__}: {exc}")
    try:
        check_label_overlap_sensitivity()
    except Exception as exc:
        check("purged-split sensitivity", False, f"error: {type(exc).__name__}: {exc}")
    (HERE / "verification.json").write_text(json.dumps(CHECKS, indent=2, default=str) + "\n")
    failed = [c["check"] for c in CHECKS if not c["pass"]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()

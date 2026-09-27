#!/usr/bin/env python3
"""Quantum feature-map test on the full grid, with a paired block bootstrap.

Following the Study Deck (Module 6, "Approach 1"): encode the time-varying
("dynamic") seismic features by angle encoding, entangle, read Pauli Z and
ZZ expectation values, and append them to the classical features.  Compare
classical-only (the frozen candidate) with classical+quantum on identical
rows, splits and per-horizon isotonic calibration.  Uncertainty: paired
contiguous 3-origin block bootstrap (2,000 reps) of the IG and AUC
differences.  Test data only; the judge set is not used again.

Circuit (4 qubits, exactly simulated; every gate is real so the state is a
real 16-vector):  RY(pi*x_i) on each qubit -> CNOT ring 0>1>2>3>0 ->
RY(pi*x_i) re-upload -> CNOT ring.  Read-out: <Z_i> (4) and <Z_i Z_i+1> (4).
Inputs x_i in [0,1], scaled with fit-period statistics only:
365-day count, log days-since-last, 7/30 d and 30/365 d acceleration ratios.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
RERUN = HERE / "rerun"
OUT = HERE / "quantum_test"
sys.path.insert(0, str(HERE.parent))
from run_full_grid_development_comparison import ACCEL  # noqa: E402

H, SEED, REPS, BLOCK = 15, 42, 2000, 3
DYNAMIC = [5, 9, 10, 11]          # raw columns: count_365d, log1p recency, ratio 7/30, ratio 30/365
NQ = 4


def ry(state, q, theta):
    """Apply RY(theta) (per-row angles) to qubit q of a (n, 2,2,2,2) real state."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    c = c.reshape(-1, *([1] * (NQ - 1))); s = s.reshape(-1, *([1] * (NQ - 1)))
    a0 = np.take(state, 0, axis=q + 1); a1 = np.take(state, 1, axis=q + 1)
    return np.stack([c * a0 - s * a1, s * a0 + c * a1], axis=q + 1)


def cnot(state, ctl, tgt):
    out = state.copy()
    idx1 = [slice(None)] * (NQ + 1); idx1[ctl + 1] = 1
    sub = out[tuple(idx1)]
    t = tgt if tgt < ctl else tgt - 1        # target axis after removing control axis
    out[tuple(idx1)] = np.flip(sub, axis=t + 1)
    return out


def qfeatures(x: np.ndarray) -> np.ndarray:
    n = len(x)
    state = np.zeros((n,) + (2,) * NQ, dtype=np.float64); state[(slice(None),) + (0,) * NQ] = 1.0
    for layer in range(2):
        for q in range(NQ):
            state = ry(state, q, np.pi * x[:, q])
        for q in range(NQ):
            state = cnot(state, q, (q + 1) % NQ)
    prob = (state ** 2).reshape(n, -1)
    bits = ((np.arange(2 ** NQ)[:, None] >> (NQ - 1 - np.arange(NQ))[None, :]) & 1)  # qubit 0 = first axis
    z = 1 - 2 * bits                                    # (16, NQ)
    zi = prob @ z
    zz = prob @ np.stack([z[:, i] * z[:, (i + 1) % NQ] for i in range(NQ)], axis=1)
    return np.hstack([zi, zz]).astype(np.float32)


def check_against_qiskit(x):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    ours = qfeatures(x)
    for r, row in enumerate(x):
        qc = QuantumCircuit(NQ)
        for _ in range(2):
            for q in range(NQ):
                qc.ry(np.pi * row[q], q)
            for q in range(NQ):
                qc.cx(q, (q + 1) % NQ)
        sv = Statevector(qc)
        ops = []
        for i in range(NQ):
            s = ["I"] * NQ; s[NQ - 1 - i] = "Z"; ops.append("".join(s))       # qiskit little-endian
        for i in range(NQ):
            s = ["I"] * NQ; s[NQ - 1 - i] = "Z"; s[NQ - 1 - (i + 1) % NQ] = "Z"; ops.append("".join(s))
        ref = [float(np.real(sv.expectation_value(SparsePauliOp(o)))) for o in ops]
        if not np.allclose(ref, ours[r], atol=1e-5):
            raise AssertionError(f"simulator mismatch on row {r}: {ref} vs {ours[r]}")
    return len(x)


def calibrate(raw_cal, y_cal, h_cal, raw_te, h_te):
    p = np.empty(len(raw_te))
    for h in range(H):
        a, b = h_cal == h, h_te == h
        p[b] = IsotonicRegression(out_of_bounds="clip").fit(raw_cal[a], y_cal[a]).predict(raw_te[b])
    return np.clip(p, 1e-4, 1 - 1e-4)


def block_bootstrap(y, pa, pb, origins):
    """Paired 3-origin block bootstrap of (b - a) for IG bits and AUC."""
    uo = np.unique(origins)
    rows = [np.flatnonzero(origins == o) for o in uo]
    ll = lambda p, i: -(y[i] * np.log(p[i]) + (1 - y[i]) * np.log1p(-p[i]))
    d_ll = [(ll(pa, r) - ll(pb, r)).mean() for r in rows]      # >0 means b better
    blocks = [list(range(i, min(i + BLOCK, len(uo)))) for i in range(0, len(uo), BLOCK)]
    rng = np.random.default_rng(SEED)
    ig_draws, auc_draws = [], []
    for k in range(REPS):
        pick = [o for bi in rng.integers(0, len(blocks), len(blocks)) for o in blocks[bi]]
        ig_draws.append(np.mean([d_ll[o] for o in pick]) / np.log(2))
        if k < 300:     # AUC is costly; 300 paired resamples
            idx = np.concatenate([rows[o] for o in pick])
            auc_draws.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    return {"delta_ig_bits": float(np.mean(d_ll) / np.log(2)),
            "ig_ci95": [float(v) for v in np.quantile(ig_draws, [0.025, 0.975])],
            "delta_auc": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            "auc_ci95": [float(v) for v in np.quantile(auc_draws, [0.025, 0.975])],
            "ig_reps": REPS, "auc_reps": len(auc_draws), "block_origins": BLOCK, "origins": int(len(uo))}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    z = np.load(RERUN / "full_grid_development_comparison" / "historical_examples.npz")
    fit = z["x_train"][:, DYNAMIC]
    tr = [np.log1p(np.maximum(fit[:, 0], 0)), fit[:, 1], np.log1p(fit[:, 2]), np.log1p(fit[:, 3])]
    hi = [float(np.percentile(c, 99.5)) or 1.0 for c in tr]

    def encode(x):
        cols = [np.log1p(np.maximum(x[:, 5], 0)), x[:, 9], np.log1p(x[:, 10]), np.log1p(x[:, 11])]
        return np.column_stack([np.clip(c / h, 0, 1) for c, h in zip(cols, hi)]).astype(np.float64)

    checked = check_against_qiskit(encode(z["x_test"][:: 150_000]))

    def q(x):
        return np.vstack([qfeatures(encode(x[i:i + 500_000])) for i in range(0, len(x), 500_000)])

    q_tr, q_ca, q_te = q(z["x_train"]), q(z["x_cal"]), q(z["x_test"])
    args = dict(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
    feats = lambda x, h, qf: np.column_stack([x[:, ACCEL], h, qf])
    model = HistGradientBoostingClassifier(**args).fit(feats(z["x_train"], z["h_train"], q_tr), z["y_train"])
    hc, ht = np.rint(z["h_cal"] * 14).astype(int), np.rint(z["h_test"] * 14).astype(int)
    p_q = calibrate(model.predict_proba(feats(z["x_cal"], z["h_cal"], q_ca))[:, 1], z["y_cal"], hc,
                    model.predict_proba(feats(z["x_test"], z["h_test"], q_te))[:, 1], ht)
    pred = np.load(RERUN / "full_grid_development_comparison" / "test_predictions.npz")
    p_c = pred["pooled_numeric_horizon_index"]
    y = z["y_test"]
    base = float(z["y_train"].mean())
    ig = lambda p: float((log_loss(y, np.full(len(y), base)) - log_loss(y, p)) / np.log(2))

    # Does the model actually use the quantum columns? Split-gain proxy: how often trees split on them.
    used = np.zeros(len(ACCEL) + 1 + 2 * NQ)
    for pr in model._predictors:
        nodes = pr[0].nodes
        for f in nodes["feature_idx"][~nodes["is_leaf"].astype(bool)]:
            used[f] += 1
    q_split_share = float(used[len(ACCEL) + 1:].sum() / used.sum())

    res = {"design": __doc__.strip().splitlines()[2:13],
           "simulator_checked_rows_vs_qiskit": checked,
           "classical_only": {"ig_bits": ig(p_c), "auc": float(roc_auc_score(y, p_c))},
           "classical_plus_quantum": {"ig_bits": ig(p_q), "auc": float(roc_auc_score(y, p_q))},
           "quantum_feature_split_share": q_split_share,
           "paired_block_bootstrap_quantum_minus_classical": block_bootstrap(y, p_c, p_q, z["origin_test"]),
           "rows": {"fit": int(len(z["y_train"])), "cal": int(len(z["y_cal"])), "test": int(len(y))},
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "quantum_paired_test.json").write_text(json.dumps(res, indent=2) + "\n")
    np.savez_compressed(OUT / "predictions.npz", y=y, origin=z["origin_test"], classical=p_c, quantum=p_q)
    print(json.dumps({k: v for k, v in res.items() if k != "design"}, indent=2))


if __name__ == "__main__":
    main()

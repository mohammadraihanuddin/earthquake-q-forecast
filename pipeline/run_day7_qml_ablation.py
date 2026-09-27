#!/usr/bin/env python3
"""Bounded Day-7 QML ablation on the frozen historical protocol.

The teaching notebook's two-qubit map is evaluated as a small quantum feature
generator. This intentionally uses a fixed stratified sample because the
notebook's full kernel SVM is O(n^2) and cannot consume the full grid rows.
Judge data is never loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "full_grid_development_comparison" / "historical_examples.npz"
OUT = ROOT / "day7_qml_ablation"
SEED = 42
HORIZONS = 15
TRAIN_N, CAL_N, TEST_N = 12000, 6000, 24000


def entangling_map(x: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.ry(np.pi * x[0], 0)
    qc.ry(np.pi * x[1], 1)
    qc.cx(0, 1)
    qc.ry(np.pi * x[0] * x[1], 1)
    return qc


OBSERVABLES = [
    SparsePauliOp.from_list([("ZI", 1.0)]),
    SparsePauliOp.from_list([("IZ", 1.0)]),
    SparsePauliOp.from_list([("ZZ", 1.0)]),
]


def qfeatures(x: np.ndarray) -> np.ndarray:
    states = [Statevector(entangling_map(row)) for row in x]
    return np.asarray([
        [float(np.real(state.expectation_value(obs))) for obs in OBSERVABLES]
        for state in states
    ], dtype=np.float32)


def stratified_indices(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    n_pos = min(len(positives), max(1, round(n * float(y.mean()))))
    n_neg = n - n_pos
    chosen = np.concatenate([
        rng.choice(positives, n_pos, replace=False),
        rng.choice(negatives, n_neg, replace=False),
    ])
    rng.shuffle(chosen)
    return chosen


def calibrate(raw_cal, y_cal, h_cal, raw_test, h_test):
    out = np.empty(len(raw_test), dtype=float)
    for h in range(HORIZONS):
        cal = h_cal == h / (HORIZONS - 1)
        test = h_test == h / (HORIZONS - 1)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal[cal], y_cal[cal])
        out[test] = iso.predict(raw_test[test])
    return np.clip(out, 1e-4, 1 - 1e-4)


def metrics(y, p, base):
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "log_loss": float(log_loss(y, p)),
        "ig_bits": float((log_loss(y, np.full(len(y), base)) - log_loss(y, p)) / np.log(2)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def main():
    z = np.load(EXAMPLES)
    rng = np.random.default_rng(SEED)
    # Two inputs: a recent seismic count and normalized horizon. The count
    # uses only the historical ACCEL feature matrix; horizon is explicit.
    def sample(x, y, h, n):
        idx = stratified_indices(y, n, rng)
        return x[idx], y[idx], h[idx]

    xtr, ytr, htr = sample(z["x_train"], z["y_train"], z["h_train"], TRAIN_N)
    xca, yca, hca = sample(z["x_cal"], z["y_cal"], z["h_cal"], CAL_N)
    xte, yte, hte = sample(z["x_test"], z["y_test"], z["h_test"], TEST_N)
    # Column 2 is the 7-day 100-km count; scaling uses fit-only bounds.
    count_scale = max(float(xtr[:, 2].max()), 1.0)
    def encode(x, h):
        return np.column_stack([
            np.clip(x[:, 2] / count_scale, 0.0, 1.0),
            h,
        ]).astype(np.float64)

    qtr, qca, qte = map(qfeatures, (encode(xtr, htr), encode(xca, hca), encode(xte, hte)))
    model_args = dict(
        max_iter=120, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=SEED,
    )
    q_model = HistGradientBoostingClassifier(**model_args).fit(qtr, ytr)
    q_p = calibrate(
        q_model.predict_proba(qca)[:, 1], yca, hca,
        q_model.predict_proba(qte)[:, 1], hte,
    )
    classical_model = HistGradientBoostingClassifier(**model_args).fit(
        encode(xtr, htr), ytr
    )
    classical_p = calibrate(
        classical_model.predict_proba(encode(xca, hca))[:, 1], yca, hca,
        classical_model.predict_proba(encode(xte, hte))[:, 1], hte,
    )
    base = float(ytr.mean())
    result = {
        "protocol": {
            "source": "Day-7 entangling_map",
            "judge_loaded": False,
            "train_sample": TRAIN_N,
            "calibration_sample": CAL_N,
            "test_sample": TEST_N,
            "sampling": "fixed seed 42, stratified by binary label",
            "inputs": "7-day seismic count normalized fit-only plus h/14",
            "quantum_features": ["Z0", "Z1", "ZZ"],
            "calibration": "independent isotonic per horizon",
            "warning": "Exploratory sample ablation; not comparable to full-row candidate headline",
        },
        "qml_quantum_features": metrics(yte, q_p, base),
        "classical_two_input_control": metrics(yte, classical_p, base),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        OUT / "predictions.npz", y=yte, horizon=hte,
        qml_probability=q_p, classical_probability=classical_p,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rerun every pipeline behind the recorded results into ``rerun/``.

Each original script is imported unchanged and its hard-coded output paths
are redirected, so the recorded artifacts in ``pipeline/`` are never
overwritten.  Downstream stages read the *rerun* historical examples, so the
whole chain is regenerated from the raw catalogs.

Usage: python rerun_all.py [stage ...]   (default: all stages, in order)
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RERUN = HERE / "rerun"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

EXAMPLES = RERUN / "full_grid_development_comparison" / "historical_examples.npz"


def full_grid():
    import run_full_grid_development_comparison as m
    m.OUT = RERUN / "full_grid_development_comparison"
    m.main()


def reporting_gaps():
    import diagnose_full_grid_reporting_gaps as m
    art = RERUN / "full_grid_development_comparison"
    m.ART, m.PREDICTIONS, m.EXAMPLES = art, art / "test_predictions.npz", EXAMPLES
    m.OUT_JSON = art / "reporting_gap_diagnostics.json"
    m.OUT_MD = art / "reporting_gap_diagnostics.md"
    m.main()


def external_drivers():
    import run_final_matched_external_driver_comparison as m
    m.EXAMPLES, m.OUT = EXAMPLES, RERUN / "final_matched_external_driver_comparison"
    m.main()


def etas():
    import run_matched_etas_frozen_comparison as m
    m.EXAMPLES, m.OUT = EXAMPLES, RERUN / "matched_etas_frozen_comparison"
    m.main()


def qml():
    import run_day7_qml_ablation as m
    m.EXAMPLES, m.OUT = EXAMPLES, RERUN / "day7_qml_ablation"
    m.main()


def forecast():
    import run_frozen_forecast_fixed as m
    m.EXAMPLES = EXAMPLES
    m.main()


def contract_synthetic():
    import run_contract_interface as m
    m.run_synthetic_test()


STAGES = {
    "full_grid": full_grid,
    "reporting_gaps": reporting_gaps,
    "external_drivers": external_drivers,
    "etas": etas,
    "qml": qml,
    "forecast": forecast,
    "contract_synthetic": contract_synthetic,
}


def main() -> None:
    RERUN.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or list(STAGES)
    status_path = RERUN / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for name in wanted:
        print(f"=== {name}", flush=True)
        log = RERUN / f"{name}.log"
        start = time.time()
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                STAGES[name]()
            outcome = "ok"
        except Exception as exc:  # record and continue with later stages
            outcome = f"failed: {type(exc).__name__}: {exc}"
        log.write_text(buffer.getvalue())
        status[name] = {"outcome": outcome, "seconds": round(time.time() - start, 1)}
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"    {outcome} ({status[name]['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()

"""Forecastability ceiling and variance decomposition on MU-Glioma-Post.

Uses the same trajectories as the switching tests (duplicates removed, split
at intervals of 28 days or less, segments with at least three scans) and runs
``gbm_pinn.ceiling.analyze_cohort``: maximum-likelihood fit, nested tests,
patient-bootstrap intervals, and the power to detect between-patient
growth-rate variation.

Requires ``outputs/surveillance/mu_scan_volumes.csv`` from
``scripts/test_switching_mu.py``.

Usage:
    uv run --extra analysis --extra dev python scripts/analyze_ceiling_mu.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from test_switching_mu import EXCLUSION_DAYS

from gbm_pinn.ceiling import analyze_cohort
from gbm_pinn.switching import Cohort, radius_mm, split_segments


def segments_with_patients(
    scans: pd.DataFrame, measure: str, *, drop_zero_segments: bool = False
) -> tuple[Cohort, list[str]]:
    segments, groups = [], []
    for patient, rows in scans.sort_values("day").groupby("patient"):
        radii = radius_mm(rows[f"{measure}_ml"].to_numpy())
        for t, r in split_segments(rows.day.to_numpy(), radii, exclusion_days=EXCLUSION_DAYS):
            if len(t) < 3 or (drop_zero_segments and np.any(r == 0)):
                continue
            segments.append((t, r))
            groups.append(patient)
    return Cohort.from_segments(segments, min_scans=3), groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surveillance"))
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--power-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    scans = pd.read_csv(args.output_dir / "mu_scan_volumes.csv")
    rng = np.random.default_rng(args.seed)
    settings = {"resamples": args.resamples, "power_simulations": args.power_simulations}
    report = {"cohort": "MU-Glioma-Post", "role": "exploratory (discovery cohort)"}
    for label, measure, drop_zero in (
        ("enhancing", "enhancing", False),
        ("enhancing_no_zero_segments", "enhancing", True),
        ("core", "core", False),
    ):
        cohort, groups = segments_with_patients(scans, measure, drop_zero_segments=drop_zero)
        report[label] = analyze_cohort(cohort, groups, rng=rng, **settings)
        print(f"{label} done", flush=True)
    (args.output_dir / "ceiling_mu.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

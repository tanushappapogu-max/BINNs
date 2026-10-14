"""``gbm-ceiling``: how forecastable is a longitudinal tumor-volume dataset?

Give it a CSV with one row per scan and it reports, before anyone builds a
forecaster, how much of persistence's error any size-history forecaster could
remove, how scan-to-scan change splits into reverting, interval-independent
and time-accumulating parts, and how large between-patient growth-rate
differences would have to be for the data to detect them.

Input columns: ``patient``, ``day``, ``volume_ml`` and optionally ``segment``
(scans in different segments are never paired, e.g. across a surgery).
Trajectories are split wherever two scans are ``--exclusion-days`` apart or
less, and only pieces with at least three scans are used.

Example::

    gbm-ceiling scans.csv --json report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gbm_pinn.ceiling import analyze_cohort
from gbm_pinn.switching import Cohort, radius_mm, split_segments

REQUIRED = ("patient", "day", "volume_ml")


def build_cohort(frame: pd.DataFrame, exclusion_days: float) -> tuple[Cohort, list[str]]:
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if (frame.volume_ml < 0).any() or frame[list(REQUIRED)].isna().any().any():
        raise ValueError("volumes must be present and nonnegative")
    keys = ["patient", "segment"] if "segment" in frame.columns else ["patient"]
    segments, groups = [], []
    for key, rows in frame.sort_values("day").groupby(keys, sort=True):
        patient = key[0] if isinstance(key, tuple) else key
        radii = radius_mm(rows.volume_ml.to_numpy(dtype=float))
        for t, r in split_segments(
            rows.day.to_numpy(dtype=float), radii, exclusion_days=exclusion_days
        ):
            if len(t) >= 3:
                segments.append((t, r))
                groups.append(str(patient))
    if not segments:
        raise ValueError("no trajectory has three or more scans after splitting")
    return Cohort.from_segments(segments, min_scans=3), groups


def render(report: dict) -> str:
    fit, ceiling, ci = report["fit"], report["ceiling"], report["bootstrap_95ci"]

    def interval(name: str) -> str:
        return f"[{ci[name]['ci_low']:.3f}, {ci[name]['ci_high']:.3f}]"

    tests = report["nested_tests"]
    lines = [
        f"Segments: {report['segments']}   patients: {report['patients']}   "
        f"transitions: {ceiling['transitions']}",
        "",
        "Forecastability (share of persistence's squared radius error removable)",
        f"  from size history : {ceiling['history_ceiling']:.3f} {interval('history_ceiling')}",
        f"  by an oracle      : {ceiling['oracle_ceiling']:.3f} {interval('oracle_ceiling')}",
        "",
        "Change decomposition (equivalent radius)",
        f"  reverting SD            : {fit['reverting_sd_mm']:.2f} mm  (p drop = "
        f"{tests['no_reverting_change']['p_value']:.3g})",
        f"  interval-independent SD : {fit['interval_independent_sd_mm']:.2f} mm  (p drop = "
        f"{tests['no_interval_independent_change']['p_value']:.3g})",
        f"  time-accumulating SD    : {fit['clock_sd_mm_per_90_days']:.2f} mm per sqrt(90 d)  "
        f"(p drop = {tests['no_clock']['p_value']:.3g}, power {report['clock_test_power']:.2f})",
        f"  common drift            : {fit['drift_mean_mm_per_year']:.2f} mm/yr",
        f"  between-patient drift SD: {fit['drift_sd_mm_per_year']:.2f} mm/yr  (p drop = "
        f"{tests['no_patient_rates']['p_value']:.3g})",
        "",
        "Smallest between-patient growth-rate SD detectable with 80% power: "
        + (
            f"{report['minimum_detectable_drift_sd_mm_per_year']:g} mm/yr"
            if report["minimum_detectable_drift_sd_mm_per_year"] is not None
            else "beyond the tested range"
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gbm-ceiling", description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--exclusion-days", type=float, default=28.0)
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--power-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args(argv)

    cohort, groups = build_cohort(pd.read_csv(args.csv), args.exclusion_days)
    report = analyze_cohort(
        cohort,
        groups,
        rng=np.random.default_rng(args.seed),
        resamples=args.resamples,
        power_simulations=args.power_simulations,
    )
    print(render(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Confirmatory forecastability analysis on LUMIERE.

Follows ``docs/preregistration_ceiling_lumiere.md``. Trajectories use the
explicit surgery markers in LUMIERE: every post-operative session starts a new
segment, and sessions before the first post-operative scan (the untreated
tumor) are dropped. Segments are further split
at intervals of 28 days or less, exactly as for MU, and only segments with at
least three scans are kept. Scan day is week x 7.

Requires ``outputs/surveillance/lumiere_scan_volumes.csv`` from
``scripts/prepare_lumiere_volumes.py``.

Usage:
    uv run --extra analysis --extra dev python scripts/analyze_ceiling_lumiere.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gbm_pinn.ceiling import analyze_cohort
from gbm_pinn.switching import Cohort, radius_mm, split_segments

PROTOCOL = Path("docs/preregistration_ceiling_lumiere.md")
EXCLUSION_DAYS = 28.0


def starts_segment(row: pd.Series) -> bool:
    """A post-operative session: any later part of a surgery day, or rated Post-Op."""
    return row.surgery_part >= 2 or "Post-Op" in str(row.rano)


def surgery_segments(rows: pd.DataFrame) -> list[pd.DataFrame]:
    """Split one patient's sessions at every post-operative scan.

    Sessions before the first post-operative scan show the untreated tumor and
    are dropped. Pre-operative scans before a later surgery are kept: they end
    the preceding segment, just as the last scan before re-operation does in MU.
    """
    ordered = rows.sort_values(["week", "surgery_part"])
    starts = ordered.apply(starts_segment, axis=1).to_numpy()
    labels = np.cumsum(starts)
    return [part for label, part in ordered.groupby(labels, sort=True) if label > 0]


def build(
    scans: pd.DataFrame, measure: str, *, surgery_only: bool = False
) -> tuple[Cohort, list[str]]:
    segments, groups = [], []
    for patient, rows in scans.groupby("patient"):
        for part in surgery_segments(rows):
            days = part.week.to_numpy(dtype=float) * 7.0
            radii = radius_mm(part[f"{measure}_ml"].to_numpy())
            pieces = (
                [(days, radii)]
                if surgery_only
                else split_segments(days, radii, exclusion_days=EXCLUSION_DAYS)
            )
            for t, r in pieces:
                if len(t) >= 3 and np.all(np.diff(t) > 0):
                    segments.append((t, r))
                    groups.append(patient)
    return Cohort.from_segments(segments, min_scans=3), groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surveillance"))
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--power-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    scans = pd.read_csv(args.output_dir / "lumiere_scan_volumes.csv")
    rng = np.random.default_rng(args.seed)
    settings = {"resamples": args.resamples, "power_simulations": args.power_simulations}
    report: dict[str, object] = {
        "cohort": "LUMIERE",
        "role": "confirmatory",
        "protocol": str(PROTOCOL),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    }
    for label, measure, surgery_only in (
        ("enhancing", "enhancing", False),
        ("enhancing_surgery_splits_only", "enhancing", True),
        ("whole", "whole", False),
    ):
        cohort, groups = build(scans, measure, surgery_only=surgery_only)
        report[label] = analyze_cohort(cohort, groups, rng=rng, **settings)
        print(f"{label} done", flush=True)
    (args.output_dir / "ceiling_lumiere.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

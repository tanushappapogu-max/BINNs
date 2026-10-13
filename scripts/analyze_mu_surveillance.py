"""Interval-versus-change analysis on MU-Glioma-Post.

Builds one row per consecutive scan pair across all three shared splits,
joins first-progression dates from the clinical spreadsheet, and writes the
statistics panel from ``gbm_pinn.surveillance_stats``.

Usage:
    uv run --extra analysis --extra dev python scripts/analyze_mu_surveillance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from gbm_pinn.surveillance_stats import add_change_columns, summarize

DERIVED = Path("data/derived/mu_glioma_post")
CLINICAL = Path("data/raw/mu_glioma_post/MU-Glioma-Post_ClinicalData-July2025.xlsx")
SPLITS = ("training", "model_selection", "final_test")
WHOLE_LABELS = (1, 2, 3)


def whole_volume_ml(path: str) -> float:
    image = nib.load(path)
    labels = np.asarray(image.dataobj)
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return float(np.isin(labels, WHOLE_LABELS).sum() * voxel_ml)


def load_transitions() -> pd.DataFrame:
    transitions = []
    for split in SPLITS:
        payload = json.loads((DERIVED / f"shared_{split}_transitions.json").read_text())
        transitions += [dict(t, split=split) for t in payload["transitions"]]

    # Source volumes are cached in the manifests; a patient's last target is not.
    volumes = {
        (t["patient_id"], t["source_timepoint"]): t["source_morphology"][
            "whole_abnormality_volume_ml"
        ]
        for t in transitions
    }
    for t in transitions:
        key = (t["patient_id"], t["target_timepoint"])
        if key not in volumes:
            volumes[key] = whole_volume_ml(t["target_segmentation"])

    clinical = pd.read_excel(CLINICAL, sheet_name="MU Glioma Post").set_index("Patient_ID")
    progression = pd.to_numeric(
        clinical["Number of days from Diagnosis to date of First Progression"], errors="coerce"
    )
    rows = []
    for t in transitions:
        first_progression = progression.get(t["patient_id"], np.nan)
        source_day, target_day = float(t["source_day"]), float(t["target_day"])
        if np.isnan(first_progression):
            phase = "no_progression_recorded"
        elif target_day <= first_progression:
            phase = "before_progression"
        elif source_day < first_progression:
            phase = "spans_progression"
        else:
            phase = "after_progression"
        rows.append(
            {
                "patient": t["patient_id"],
                "split": t["split"],
                "source_day": source_day,
                "target_day": target_day,
                "horizon_days": float(t["horizon_days"]),
                "source_volume": volumes[(t["patient_id"], t["source_timepoint"])],
                "target_volume": volumes[(t["patient_id"], t["target_timepoint"])],
                "phase": phase,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surveillance"))
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = args.output_dir / "mu_transitions.csv"
    if table.is_file():
        frame = pd.read_csv(table)
    else:
        frame = load_transitions()
        frame.to_csv(table, index=False)

    panel = summarize(frame, resamples=args.resamples)
    enriched = add_change_columns(frame)
    panel["by_phase"] = (
        enriched.groupby("phase")
        .agg(
            transitions=("horizon_days", "size"),
            median_horizon_days=("horizon_days", "median"),
            median_abs_change=("abs_change", "median"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    (args.output_dir / "mu_panel.json").write_text(json.dumps(panel, indent=2))
    print(json.dumps(panel, indent=2))


if __name__ == "__main__":
    main()

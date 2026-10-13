"""Pre-registered bunching test on MU-Glioma-Post.

Implements ``docs/preregistration_switching.md``: per-scan label volumes,
segments split at intervals of 28 days or less, a steady-growth null fitted
per segment, and a parametric-bootstrap test of bunching, with power and
type-I checks that decide whether the result is interpretable.

Usage:
    uv run --extra analysis --extra dev python scripts/test_switching_mu.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from gbm_pinn.switching import (
    Cohort,
    bunching_test,
    largest_change_is_growth,
    radius_mm,
    rejection_rate,
    split_segments,
)

MANIFEST = Path("data/derived/mu_glioma_post/shared_split_manifest.json")
CLINICAL = Path("data/raw/mu_glioma_post/MU-Glioma-Post_ClinicalData-July2025.xlsx")
NIFTI = Path("data/raw/mu_glioma_post/nifti")
DUPLICATE_PATIENTS = {"PatientID_0062", "PatientID_0132"}
DUPLICATE_SCANS = {("PatientID_0018", 5), ("PatientID_0032", 5)}
MEASURES = {"enhancing": (3,), "core": (1, 3), "whole": (1, 2, 3)}
EXCLUSION_DAYS = 28.0


def scan_day_columns(clinical: pd.DataFrame) -> dict[int, str]:
    ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}
    columns = {}
    for k, word in ordinals.items():
        matches = [c for c in clinical.columns if f"to {word} MRI" in c]
        if len(matches) != 1:
            raise ValueError(f"expected one scan-day column for timepoint {k}")
        columns[k] = matches[0]
    return columns


def load_scan_volumes() -> pd.DataFrame:
    manifest = json.loads(MANIFEST.read_text())
    patients = [
        p["patient_id"] for p in manifest["patients"] if p["patient_id"] not in DUPLICATE_PATIENTS
    ]
    clinical = pd.read_excel(CLINICAL, sheet_name="MU Glioma Post").set_index("Patient_ID")
    day_column = scan_day_columns(clinical)
    rows = []
    for patient in patients:
        for k, column in day_column.items():
            if (patient, k) in DUPLICATE_SCANS:
                continue
            name = f"{patient}_Timepoint_{k}_tumorMask.nii.gz"
            mask_path = NIFTI / patient / f"Timepoint_{k}" / name
            day = pd.to_numeric(clinical.at[patient, column], errors="coerce")
            if not mask_path.is_file() or np.isnan(day):
                continue
            image = nib.load(mask_path)
            labels = np.asarray(image.dataobj)
            voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
            row = {"patient": patient, "timepoint": k, "day": float(day)}
            for name, codes in MEASURES.items():
                row[f"{name}_ml"] = float(np.isin(labels, codes).sum() * voxel_ml)
            row["multiple_surgeries"] = int(clinical.at[patient, "Multiple surgeries"] == 1)
            rows.append(row)
    return pd.DataFrame(rows)


def build_cohort(
    scans: pd.DataFrame,
    measure: str,
    *,
    drop_first_interval: bool = False,
    drop_zero_segments: bool = False,
) -> Cohort:
    segments = []
    for _, patient in scans.sort_values("day").groupby("patient"):
        days = patient.day.to_numpy()
        radii = radius_mm(patient[f"{measure}_ml"].to_numpy())
        if drop_first_interval:
            days, radii = days[1:], radii[1:]
        for t, r in split_segments(days, radii, exclusion_days=EXCLUSION_DAYS):
            if drop_zero_segments and np.any(r == 0):
                continue
            segments.append((t, r))
    return Cohort.from_segments(segments, min_scans=3)


def run(cohort: Cohort, rng: np.random.Generator, **kwargs: float) -> dict[str, object]:
    result = bunching_test(cohort, rng=rng, **kwargs).as_dict()
    growth = largest_change_is_growth(cohort.radius)
    result["largest_interval_is_growth_fraction"] = float(np.mean(growth))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surveillance"))
    parser.add_argument("--power-cohorts", type=int, default=200)
    parser.add_argument("--power-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = args.output_dir / "mu_scan_volumes.csv"
    if table.is_file():
        scans = pd.read_csv(table)
    else:
        scans = load_scan_volumes()
        scans.to_csv(table, index=False)

    rng = np.random.default_rng(args.seed)
    primary = build_cohort(scans, "enhancing")
    report: dict[str, object] = {
        "protocol": "docs/preregistration_switching.md",
        "scans": len(scans),
        "patients": int(scans.patient.nunique()),
        "primary_enhancing": run(primary, rng),
        "power_switching": rejection_rate(
            primary,
            alternative="switching",
            cohorts=args.power_cohorts,
            draws=args.power_draws,
            rng=rng,
        ),
        "type_one_rate": rejection_rate(
            primary,
            alternative="steady",
            cohorts=args.power_cohorts,
            draws=args.power_draws,
            rng=rng,
        ),
    }

    primary_p = report["primary_enhancing"]["p_value"]
    power = report["power_switching"]
    type_one = report["type_one_rate"]
    if power < 0.80:
        decision = "inconclusive: defer to LUMIERE"
    elif primary_p < 0.05 and type_one <= 0.10:
        decision = "supported: passes kill criterion 1"
    elif primary_p >= 0.05:
        decision = "killed for MU: steady growth plus noise explains bunching"
    else:
        decision = "uninterpretable: type-I rate above 0.10"
    report["decision"] = decision

    no_multi = scans[scans.multiple_surgeries == 0]
    report["secondary"] = {
        "core": run(build_cohort(scans, "core"), rng),
        "whole": run(build_cohort(scans, "whole"), rng),
        "enhancing_no_multiple_surgeries": run(build_cohort(no_multi, "enhancing"), rng),
        "enhancing_no_zero_segments": run(
            build_cohort(scans, "enhancing", drop_zero_segments=True), rng
        ),
        "enhancing_drop_first_interval": run(
            build_cohort(scans, "enhancing", drop_first_interval=True), rng
        ),
        "enhancing_half_sigma": run(primary, rng, sigma_scale=0.5),
    }
    (args.output_dir / "switching_test_mu.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

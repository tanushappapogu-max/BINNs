"""Patient-level split construction for shared MU-Glioma-Post training."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

MU_ROLES = ("training", "model_selection", "final_test")


@dataclass(frozen=True, slots=True)
class MuTreatmentEvent:
    """One normalized treatment record relative to diagnosis."""

    modality: str
    raw_name: str
    start_day: float | None
    end_day: float | None
    timing_known: bool
    right_censored: bool = False
    dose_gy: float | None = None
    fractions: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "raw_name": self.raw_name,
            "start_day": self.start_day,
            "end_day": self.end_day,
            "timing_known": self.timing_known,
            "right_censored": self.right_censored,
            "dose_gy": self.dose_gy,
            "fractions": self.fractions,
        }


@dataclass(frozen=True, slots=True)
class MuPatient:
    """One eligible MU patient and their ordered relative scan days."""

    patient_id: str
    scan_days: tuple[float, ...]
    diagnosis: str
    grade: str
    treatment_events: tuple[MuTreatmentEvent, ...] = ()

    @property
    def transition_count(self) -> int:
        return len(self.scan_days) - 1


def eligible_mu_patients(rows: Iterable[Mapping[str, Any]]) -> tuple[MuPatient, ...]:
    """Select diagnosed GBM patients with at least two valid, ordered scan dates."""
    patients: list[MuPatient] = []
    seen: set[str] = set()
    for row in rows:
        raw_patient_id = row.get("Patient_ID")
        patient_id = "" if raw_patient_id is None else str(raw_patient_id).strip()
        if not patient_id and all(value in (None, "") for value in row.values()):
            continue
        if not patient_id or patient_id in seen:
            raise ValueError("MU patient IDs must be nonempty and unique")
        seen.add(patient_id)
        diagnosis = str(row.get("Primary Diagnosis", "")).strip()
        if diagnosis.upper() != "GBM":
            continue
        scan_days = tuple(
            float(value)
            for key, value in row.items()
            if "MRI (Timepoint_" in str(key) and _finite(value)
        )
        if len(scan_days) < 2 or any(
            later <= earlier for earlier, later in pairwise(scan_days)
        ):
            continue
        patients.append(
            MuPatient(
                patient_id=patient_id,
                scan_days=scan_days,
                diagnosis=diagnosis,
                grade=str(row.get("Grade of Primary Brain Tumor", "")).strip(),
                treatment_events=extract_mu_treatment_events(row, last_scan_day=scan_days[-1]),
            )
        )
    return tuple(sorted(patients, key=lambda patient: patient.patient_id))


def assign_mu_roles(
    patients: Sequence[MuPatient],
    *,
    preserved_roles: Mapping[str, str] | None = None,
    seed: int = 162,
) -> dict[str, str]:
    """Create a reproducible 70/15/15 split while preserving prior assignments."""
    if not patients:
        raise ValueError("at least one eligible MU patient is required")
    ids = {patient.patient_id for patient in patients}
    preserved = dict(preserved_roles or {})
    unknown = preserved.keys() - ids
    if unknown:
        raise ValueError(f"preserved split contains ineligible patients: {sorted(unknown)}")
    invalid = set(preserved.values()) - set(MU_ROLES)
    if invalid:
        raise ValueError(f"invalid preserved roles: {sorted(invalid)}")
    targets = _split_targets(len(patients))
    assigned_counts = Counter(preserved.values())
    for role in MU_ROLES:
        if assigned_counts[role] > targets[role]:
            raise ValueError(f"preserved {role} patients exceed the target split size")

    remaining_by_scans: dict[int, list[MuPatient]] = defaultdict(list)
    for patient in patients:
        if patient.patient_id not in preserved:
            remaining_by_scans[len(patient.scan_days)].append(patient)
    rng = random.Random(seed)
    for group in remaining_by_scans.values():
        rng.shuffle(group)

    assignments = dict(preserved)
    ordered: list[MuPatient] = []
    buckets = [remaining_by_scans[key] for key in sorted(remaining_by_scans, reverse=True)]
    while any(buckets):
        for bucket in buckets:
            if bucket:
                ordered.append(bucket.pop())
    for patient in ordered:
        deficits = {role: targets[role] - assigned_counts[role] for role in MU_ROLES}
        role = max(MU_ROLES, key=lambda value: (deficits[value], -MU_ROLES.index(value)))
        if deficits[role] <= 0:
            raise RuntimeError("MU split assignment exhausted all role targets")
        assignments[patient.patient_id] = role
        assigned_counts[role] += 1
    if any(assigned_counts[role] != targets[role] for role in MU_ROLES):
        raise RuntimeError("MU split assignment did not meet its target counts")
    return assignments


def build_mu_shared_manifest(
    patients: Sequence[MuPatient],
    assignments: Mapping[str, str],
    *,
    metadata_path: Path,
    local_nifti_root: Path | None = None,
    seed: int = 162,
) -> dict[str, Any]:
    """Build the locked source manifest without generating scan-level splits."""
    patient_ids = {patient.patient_id for patient in patients}
    if set(assignments) != patient_ids:
        raise ValueError("assignments must contain every eligible patient exactly once")
    records = []
    for patient in patients:
        local_timepoints = 0
        if local_nifti_root is not None:
            directory = local_nifti_root / patient.patient_id
            local_timepoints = sum(path.is_dir() for path in directory.glob("Timepoint_*"))
        records.append(
            {
                "patient_id": patient.patient_id,
                "role": assignments[patient.patient_id],
                "source": "MU-Glioma-Post",
                "diagnosis": patient.diagnosis,
                "grade": patient.grade,
                "scan_days": list(patient.scan_days),
                "transition_count": patient.transition_count,
                "local_timepoint_count": local_timepoints,
                "local_images_complete": local_timepoints == len(patient.scan_days),
                "treatment_events": [event.to_record() for event in patient.treatment_events],
            }
        )
    return {
        "dataset": "MU-Glioma-Post",
        "source_url": "https://doi.org/10.7937/7K9K-3C83",
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "split_protocol": {
            "unit": "patient",
            "seed": seed,
            "fractions": {
                "training": 0.70,
                "model_selection": 0.15,
                "final_test": 0.15,
            },
            "eligibility": "Primary Diagnosis equals GBM and at least two ordered dated MRIs",
            "preserves_previous_assignments": True,
        },
        "dataset_contract": {
            "label_convention": {
                "0": "background",
                "1": "nonenhancing_or_necrotic_core",
                "2": "flair_abnormality",
                "3": "enhancing_tissue",
                "4": "resection_cavity",
            },
            "spatial_alignment": "SRI24_atlas_space",
            "treatment_timing": "relative_to_diagnosis",
        },
        "patients": records,
    }


def _split_targets(patient_count: int) -> dict[str, int]:
    fractions = {"training": 0.70, "model_selection": 0.15, "final_test": 0.15}
    exact = {role: fractions[role] * patient_count for role in MU_ROLES}
    targets = {role: int(exact[role]) for role in MU_ROLES}
    remaining = patient_count - sum(targets.values())
    priority = sorted(
        MU_ROLES,
        key=lambda role: (exact[role] - targets[role], -MU_ROLES.index(role)),
        reverse=True,
    )
    for role in priority[:remaining]:
        targets[role] += 1
    return targets


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def extract_mu_treatment_events(
    row: Mapping[str, Any], *, last_scan_day: float
) -> tuple[MuTreatmentEvent, ...]:
    """Normalize MU treatment metadata without inferring missing start dates."""
    events: list[MuTreatmentEvent] = []
    _append_duration_events(
        events,
        raw_name=row.get("Name of Initial Chemo Therapy"),
        start=row.get(" Number of days from Diagnosis to Initial Chemo Therapy Start date"),
        end=row.get(" Number of days from Diagnosis to Initial Chemo Therapy end date"),
        last_scan_day=last_scan_day,
    )
    if _truthy_treatment(row.get("Radiation Therapy")):
        events.append(
            _duration_event(
                "radiation",
                str(row.get("Radiation Therapy", "Radiation")),
                row.get("Number of days from Diagnosis to Radiation Therapy Start date"),
                row.get("Number of days from Diagnosis to Radiation Therapy end date"),
                last_scan_day,
                dose_gy=_number_from_text(row.get("Dose")),
                fractions=_integer(row.get("Number of Fractions")),
            )
        )
    for name_key, start_key, end_key in (
        (
            "Additional Therapy",
            "Number of Days from Diagnosis to Starting Additional Therapy ",
            "Number of Days from Diagnosis to Complete Additional Therapy ",
        ),
        (
            "2nd_Additional Therapy",
            "Number of Days from Diagnosis to Starting 2nd_Additional Therapy ",
            "Number of Days from Dagnosis to Complete 2nd_Additional Therapy ",
        ),
        (
            "Immuno therapy",
            "Number of Days from Diagnosis to Start Immunotherapy ",
            "Number of Days from Diagnosis to Complete Immunotherapy ",
        ),
        (
            "Other Types of Therapy (LITT, more chemo, proton therapy)",
            "Number of Days from Diagnosis to Start Other Additional Therapy ",
            "Number of Days from Diagnosis to Complete Other Additional Therapy ",
        ),
    ):
        _append_duration_events(
            events,
            raw_name=row.get(name_key),
            start=row.get(start_key),
            end=row.get(end_key),
            last_scan_day=last_scan_day,
        )
    brachy_name = _clean_text(row.get("Brachy therapy"))
    if _truthy_treatment(brachy_name):
        day = _number(
            row.get(
                "Number of Days from Diagnosis to the day of Insertion of Brachytherapy "
            )
        )
        events.append(
            MuTreatmentEvent(
                modality=_normalize_modalities(brachy_name)[0],
                raw_name=brachy_name,
                start_day=day,
                end_day=day,
                timing_known=day is not None,
            )
        )
    return tuple(events)


def _append_duration_events(
    events: list[MuTreatmentEvent],
    *,
    raw_name: Any,
    start: Any,
    end: Any,
    last_scan_day: float,
) -> None:
    name = _clean_text(raw_name)
    if not _truthy_treatment(name):
        return
    for modality in _normalize_modalities(name):
        events.append(_duration_event(modality, name, start, end, last_scan_day))


def _duration_event(
    modality: str,
    raw_name: str,
    raw_start: Any,
    raw_end: Any,
    last_scan_day: float,
    *,
    dose_gy: float | None = None,
    fractions: int | None = None,
) -> MuTreatmentEvent:
    start = _number(raw_start)
    end = _number(raw_end)
    timing_known = start is not None
    right_censored = timing_known and end is None
    if right_censored:
        end = max(start, float(last_scan_day))
    if timing_known and end is not None and end < start:
        timing_known = False
        start = None
        end = None
        right_censored = False
    if not timing_known:
        start = None
        end = None
    return MuTreatmentEvent(
        modality=modality,
        raw_name=raw_name,
        start_day=start,
        end_day=end,
        timing_known=timing_known,
        right_censored=right_censored,
        dose_gy=dose_gy,
        fractions=fractions,
    )


def _normalize_modalities(name: str) -> tuple[str, ...]:
    normalized = name.casefold()
    modalities: list[str] = []
    rules = (
        (("temozolomide", "temodar", "tmz"), "temozolomide"),
        (("avastin", "bevacizumab"), "bevacizumab"),
        (("lomustine", "ccnu"), "lomustine"),
        (("carboplatin",), "carboplatin"),
        (("irinotecan",), "irinotecan"),
        (("keytruda", "pembrolizumab"), "pembrolizumab"),
        (("retifanlimab",), "retifanlimab"),
        (("optune", "ttf"), "optune_ttf"),
        (("litt",), "litt"),
        (("cyberknife",), "cyberknife"),
        (("gliadel",), "brachytherapy_gliadel"),
        (("gamma tile",), "brachytherapy_gamma_tiles"),
    )
    for aliases, modality in rules:
        if any(alias in normalized for alias in aliases):
            modalities.append(modality)
    return tuple(dict.fromkeys(modalities)) or ("other",)


def _truthy_treatment(value: Any) -> bool:
    text = _clean_text(value).casefold()
    return bool(text) and text not in {"no", "none", "n/a", "na", "0", "false"}


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    if _finite(value):
        return float(value)
    return None


def _number_from_text(value: Any) -> float | None:
    number = _number(value)
    if number is not None:
        return number
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text.casefold().replace("gy", "").strip())
    except ValueError:
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None

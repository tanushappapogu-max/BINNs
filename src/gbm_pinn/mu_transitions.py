"""Leakage-safe transition indexing for shared MU-Glioma-Post training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gbm_pinn.clinical import load_longitudinal_segmentations
from gbm_pinn.multicompartment_cohort import COHORT_ROLES

EXPOSURE_GROUPS = {
    "radiation_exposure": {"radiation", "cyberknife"},
    "systemic_cytotoxic_exposure": {
        "temozolomide",
        "lomustine",
        "carboplatin",
        "irinotecan",
    },
    "antiangiogenic_exposure": {"bevacizumab"},
    "device_exposure": {"optune_ttf"},
    "local_intervention_exposure": {
        "litt",
        "brachytherapy_gliadel",
        "brachytherapy_gamma_tiles",
    },
}


def build_mu_transition_index(
    manifest_path: Path,
    nifti_root: Path,
    *,
    included_roles: set[str] | None = None,
    allow_final_test: bool = False,
) -> dict[str, Any]:
    """Index consecutive scan pairs while keeping patient roles isolated."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patients = manifest.get("patients")
    if manifest.get("dataset") != "MU-Glioma-Post" or not isinstance(patients, list):
        raise ValueError("a locked MU-Glioma-Post patient manifest is required")
    roles = {"training"} if included_roles is None else set(included_roles)
    if unknown := roles - COHORT_ROLES:
        raise ValueError(f"unknown cohort roles: {', '.join(sorted(unknown))}")
    if "final_test" in roles and not allow_final_test:
        raise ValueError("final_test transitions require explicit allow_final_test=True")

    transitions: list[dict[str, Any]] = []
    missing_patients: list[str] = []
    selected_patients = [patient for patient in patients if patient.get("role") in roles]
    seen: set[str] = set()
    for patient in selected_patients:
        patient_id = patient.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id or patient_id in seen:
            raise ValueError("patient IDs must be nonempty and unique")
        seen.add(patient_id)
        scan_days = tuple(float(day) for day in patient.get("scan_days", ()))
        if len(scan_days) < 2 or np.any(np.diff(scan_days) <= 0):
            raise ValueError(f"{patient_id} requires at least two increasing scan days")
        patient_directory = nifti_root / patient_id
        if not patient_directory.is_dir():
            missing_patients.append(patient_id)
            continue
        try:
            scans = load_longitudinal_segmentations(patient_directory, scan_days)
        except (FileNotFoundError, ValueError):
            missing_patients.append(patient_id)
            continue
        for source_index in range(len(scans.paths) - 1):
            target_index = source_index + 1
            exposure = treatment_exposure_features(
                patient.get("treatment_events", []),
                source_day=scans.days[source_index],
                target_day=scans.days[target_index],
            )
            source_morphology = source_morphology_features(
                scans.labels[source_index], scans.affine
            )
            history_available = source_index > 0
            previous_index = source_index - 1 if history_available else source_index
            transitions.append(
                {
                    "transition_id": f"{patient_id}_T{source_index + 1}_to_T{target_index + 1}",
                    "patient_id": patient_id,
                    "role": patient["role"],
                    "source": patient.get("source", "MU-Glioma-Post"),
                    "source_timepoint": source_index + 1,
                    "target_timepoint": target_index + 1,
                    "source_day": scans.days[source_index],
                    "target_day": scans.days[target_index],
                    "horizon_days": scans.days[target_index] - scans.days[source_index],
                    "source_segmentation": str(scans.paths[source_index]),
                    "target_segmentation": str(scans.paths[target_index]),
                    "history_available": history_available,
                    "previous_day": scans.days[previous_index] if history_available else None,
                    "previous_segmentation": (
                        str(scans.paths[previous_index]) if history_available else None
                    ),
                    "treatment_exposure": exposure,
                    "source_morphology": source_morphology,
                }
            )
    return {
        "dataset": "MU-Glioma-Post",
        "patient_manifest": str(manifest_path),
        "roles": sorted(roles),
        "patient_count": len(selected_patients),
        "locally_complete_patient_count": len(selected_patients) - len(missing_patients),
        "missing_patient_ids": sorted(missing_patients),
        "transition_count": len(transitions),
        "transitions": transitions,
    }


def treatment_exposure_features(
    treatment_events: list[dict[str, Any]], *, source_day: float, target_day: float
) -> dict[str, float | int]:
    """Summarize known treatment overlap using only interval metadata."""
    horizon = float(target_day) - float(source_day)
    if horizon <= 0:
        raise ValueError("treatment exposure requires an increasing scan interval")
    features: dict[str, float | int] = {name: 0.0 for name in EXPOSURE_GROUPS}
    features["unknown_treatment_timing_count"] = 0
    for event in treatment_events:
        if not event.get("timing_known"):
            features["unknown_treatment_timing_count"] += 1
            continue
        start = event.get("start_day")
        end = event.get("end_day")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            features["unknown_treatment_timing_count"] += 1
            continue
        if start == end:
            exposure = float(source_day < start <= target_day)
        else:
            overlap = max(0.0, min(float(target_day), end) - max(float(source_day), start))
            exposure = min(1.0, overlap / horizon)
        modality = event.get("modality")
        for feature, modalities in EXPOSURE_GROUPS.items():
            if modality in modalities:
                features[feature] = max(float(features[feature]), exposure)
    return features


def source_morphology_features(
    labels: np.ndarray, affine: np.ndarray
) -> dict[str, float | int | list[float] | None]:
    """Compute target-independent tumor geometry from one source segmentation."""
    if labels.ndim != 3 or not np.all(np.isfinite(labels)):
        raise ValueError("source labels must be a finite three-dimensional volume")
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("source affine must be a finite 4x4 matrix")
    rounded = np.rint(labels)
    if not np.array_equal(labels, rounded) or not set(np.unique(rounded)).issubset(
        {0, 1, 2, 3, 4}
    ):
        raise ValueError("source labels must use the MU label convention 0 through 4")
    voxel_volume_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1_000.0
    counts = {
        f"label_{label}_voxel_count": int(np.count_nonzero(rounded == label))
        for label in range(1, 5)
    }
    features: dict[str, float | int | list[float] | None] = {
        **counts,
        "voxel_volume_ml": voxel_volume_ml,
    }
    for label, name in ((1, "core"), (2, "flair"), (3, "enhancing"), (4, "cavity")):
        features[f"{name}_volume_ml"] = counts[f"label_{label}_voxel_count"] * voxel_volume_ml
    abnormality = np.isin(rounded, (1, 2, 3))
    features["whole_abnormality_volume_ml"] = int(np.count_nonzero(abnormality)) * voxel_volume_ml
    if np.any(abnormality):
        coordinates = np.argwhere(abnormality)
        centroid_voxel = coordinates.mean(axis=0)
        centroid_world = affine @ np.append(centroid_voxel, 1.0)
        features["whole_abnormality_centroid_mm"] = centroid_world[:3].tolist()
        spans = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        features["whole_abnormality_extent_mm"] = (spans * spacing).tolist()
    else:
        features["whole_abnormality_centroid_mm"] = None
        features["whole_abnormality_extent_mm"] = None
    return features

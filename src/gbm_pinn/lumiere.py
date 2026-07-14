"""Selective preparation helpers for the longitudinal LUMIERE dataset."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LUMIERE_ARCHIVE_URL = "https://ndownloader.figshare.com/files/38249697"
_SEGMENTATION_PATTERN = re.compile(
    r"^Imaging/(?P<patient>Patient-\d+)/week-(?P<week>\d+)/"
    r"DeepBraTumIA-segmentation/atlas/segmentation/seg_mask\.nii\.gz$"
)


@dataclass(frozen=True, slots=True)
class LumiereSession:
    """One atlas-aligned LUMIERE scan and its remote archive members."""

    patient_id: str
    week: int
    segmentation_member: str
    brain_mask_member: str

    @property
    def day(self) -> float:
        return float(self.week * 7)


def index_lumiere_sessions(member_names: Iterable[str]) -> dict[str, tuple[LumiereSession, ...]]:
    """Index atlas-space segmentations from a LUMIERE archive member listing."""
    sessions: dict[str, list[LumiereSession]] = {}
    available = set(member_names)
    for name in available:
        match = _SEGMENTATION_PATTERN.match(name)
        if match is None:
            continue
        patient_id = match.group("patient")
        week = int(match.group("week"))
        prefix = name.removesuffix("segmentation/seg_mask.nii.gz")
        brain_mask = f"{prefix}skull_strip/brain_mask.nii.gz"
        if brain_mask not in available:
            continue
        sessions.setdefault(patient_id, []).append(
            LumiereSession(patient_id, week, name, brain_mask)
        )
    indexed: dict[str, tuple[LumiereSession, ...]] = {}
    for patient_id, patient_sessions in sessions.items():
        ordered = tuple(sorted(patient_sessions, key=lambda session: session.week))
        if any(right.week <= left.week for left, right in pairwise(ordered)):
            raise ValueError(f"duplicate or unordered weeks for {patient_id}")
        indexed[patient_id] = ordered
    return dict(sorted(indexed.items()))


def remap_lumiere_segmentation(labels: NDArray[np.integer]) -> NDArray[np.int16]:
    """Map DeepBraTumIA labels into the MU-compatible compartment convention.

    LUMIERE label 1 is enhancing tumor, 2 is necrotic/nonenhancing tumor,
    and 3 is edema. The project convention is 1 necrotic, 2 edema,
    3 enhancing, and 4 cavity. LUMIERE does not provide a cavity label.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3 or np.any(~np.isfinite(labels)) or not np.allclose(
        labels, np.rint(labels), atol=1e-6
    ):
        raise ValueError("LUMIERE segmentation must be a finite integer 3D volume")
    integer = np.rint(labels).astype(np.int16)
    unexpected = set(np.unique(integer).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise ValueError(f"unexpected LUMIERE labels: {sorted(unexpected)}")
    remapped = np.zeros(integer.shape, dtype=np.int16)
    remapped[integer == 1] = 3
    remapped[integer == 2] = 1
    remapped[integer == 3] = 2
    return remapped


def validate_prepared_lumiere(
    prepared_root: Path,
    selected: dict[str, tuple[LumiereSession, ...]],
) -> None:
    """Require one completed segmentation and brain mask for every selected session."""
    missing: list[str] = []
    for patient_id, sessions in selected.items():
        for session in sessions:
            directory = prepared_root / patient_id / f"week-{session.week:03d}"
            segmentation = directory / (
                f"{patient_id}_week-{session.week:03d}_tumorMask.nii.gz"
            )
            brain_mask = directory / (
                f"{patient_id}_week-{session.week:03d}_brain_mask.nii.gz"
            )
            if not segmentation.is_file() or not brain_mask.is_file():
                missing.append(f"{patient_id}/week-{session.week:03d}")
    if missing:
        preview = ", ".join(missing[:5])
        remainder = len(missing) - min(len(missing), 5)
        suffix = f" and {remainder} more" if remainder else ""
        raise RuntimeError(f"incomplete LUMIERE preparation: {preview}{suffix}")


def build_lumiere_manifest(
    prepared_root: Path,
    protocol: dict[str, Any],
    *,
    minimum_sessions: int = 4,
) -> dict[str, Any]:
    """Build a cohort manifest from selectively prepared patient directories."""
    if minimum_sessions < 4:
        raise ValueError("minimum_sessions must be at least four for held-out forecasting")
    patients: list[dict[str, Any]] = []
    for patient_directory in sorted(prepared_root.glob("Patient-*")):
        sessions: list[tuple[int, Path]] = []
        for path in patient_directory.glob("week-*/**/*tumorMask.nii.gz"):
            match = re.fullmatch(r"week-(\d+)", path.relative_to(patient_directory).parts[0])
            if match is not None:
                sessions.append((int(match.group(1)), path))
        sessions.sort()
        weeks = [week for week, _ in sessions]
        if len(weeks) < minimum_sessions:
            continue
        if any(right <= left for left, right in pairwise(weeks)):
            raise ValueError(f"duplicate or unordered prepared weeks for {patient_directory.name}")
        patients.append(
            {
                "patient_id": patient_directory.name,
                "role": "training",
                "scan_days": [float(week * 7) for week in weeks],
                "cell_kill_windows": [],
                "edema_treatment_windows": [],
                "source": "LUMIERE",
            }
        )
    if not patients:
        raise ValueError("no prepared LUMIERE patient has enough sessions")
    return {
        "dataset": "LUMIERE",
        "source_url": "https://springernature.figshare.com/articles/dataset/21249516",
        "license": "CC0",
        "dataset_contract": {
            "spatial_alignment": "within_patient_atlas_space",
            "label_convention": {
                "0": "background",
                "1": "necrotic_or_nonenhancing_tumor",
                "2": "edema",
                "3": "enhancing_tumor",
            },
            "unavailable_labels": ["resection_cavity"],
            "treatment_timing": "unavailable",
        },
        "protocol": protocol,
        "patients": patients,
    }

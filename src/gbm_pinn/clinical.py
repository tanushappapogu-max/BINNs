"""Leakage-safe preprocessing for longitudinal postoperative segmentations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class LongitudinalSegmentations:
    """Aligned label volumes and acquisition times for one patient."""

    labels: tuple[IntegerArray, ...]
    days: tuple[float, ...]
    affine: FloatArray
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if len(self.labels) < 2 or len(self.labels) != len(self.days):
            raise ValueError("at least two label volumes with matching days are required")
        if len(self.paths) != len(self.labels):
            raise ValueError("paths must match label volumes")
        if any(right <= left for left, right in pairwise(self.days)):
            raise ValueError("days must be strictly increasing")
        shape = self.labels[0].shape
        if len(shape) != 3 or any(volume.shape != shape for volume in self.labels):
            raise ValueError("label volumes must share one three-dimensional shape")


def segmentation_to_density(
    labels: IntegerArray,
    *,
    infiltrative_density: float = 0.3,
) -> FloatArray:
    """Map MU-Glioma-Post labels to a scalar tumor-density observation."""
    if not 0.0 <= infiltrative_density <= 1.0:
        raise ValueError("infiltrative_density must lie in [0, 1]")
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3):
        raise ValueError("labels must be a two- or three-dimensional array")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"unexpected segmentation labels: {sorted(unique - {0, 1, 2, 3, 4})}")
    density = np.zeros(labels.shape, dtype=np.float64)
    density[labels == 2] = infiltrative_density
    density[np.isin(labels, (1, 3))] = 1.0
    return density


def select_observation_slice(labels: tuple[IntegerArray, ...], observation_count: int) -> int:
    """Choose an axial slice using observation scans only, preventing target leakage."""
    if not 1 <= observation_count < len(labels):
        raise ValueError("observation_count must leave at least one held-out scan")
    observed = labels[:observation_count]
    if any(volume.ndim != 3 or volume.shape != observed[0].shape for volume in observed):
        raise ValueError("observed label volumes must share one three-dimensional shape")
    tumor_counts = np.zeros(observed[0].shape[2], dtype=np.int64)
    for volume in observed:
        tumor_counts += np.count_nonzero(np.isin(volume, (1, 2, 3)), axis=(0, 1))
    if int(tumor_counts.max()) == 0:
        raise ValueError("observation scans contain no tumor labels")
    return int(np.argmax(tumor_counts))


def normalized_elapsed_times(days: tuple[float, ...], forecast_index: int) -> FloatArray:
    """Return elapsed time from the first scan, normalized by the forecast horizon."""
    if not 1 <= forecast_index < len(days):
        raise ValueError("forecast_index must identify a later scan")
    selected = np.asarray(days[: forecast_index + 1], dtype=np.float64)
    elapsed = selected - selected[0]
    if np.any(np.diff(elapsed) <= 0) or elapsed[-1] <= 0:
        raise ValueError("selected days must be strictly increasing")
    return elapsed / elapsed[-1]


def discover_segmentation_paths(patient_directory: str | Path) -> tuple[Path, ...]:
    """Find and naturally sort segmentation NIfTI files below a patient directory."""
    patient_directory = Path(patient_directory)
    candidates = tuple(
        path for path in patient_directory.rglob("*.nii*") if is_segmentation_path(path)
    )
    if not candidates:
        raise FileNotFoundError(f"no segmentation NIfTI files found under {patient_directory}")
    return tuple(sorted(candidates, key=_natural_path_key))


def is_segmentation_path(path: str | Path) -> bool:
    """Recognize common segmentation names, including MU-Glioma-Post tumor masks."""
    name = Path(path).name.lower()
    return "seg" in name or "tumormask" in name or "tumor_mask" in name


def load_longitudinal_segmentations(
    patient_directory: str | Path,
    days: tuple[float, ...],
) -> LongitudinalSegmentations:
    """Load aligned integer segmentation volumes from one patient directory."""
    try:
        import nibabel as nib
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise ImportError("install the 'imaging' extra to load NIfTI files") from error

    paths = discover_segmentation_paths(patient_directory)
    if len(paths) != len(days):
        raise ValueError(f"found {len(paths)} segmentations but received {len(days)} scan days")
    labels: list[IntegerArray] = []
    reference_affine: FloatArray | None = None
    reference_shape: tuple[int, ...] | None = None
    for path in paths:
        image = nib.as_closest_canonical(nib.load(path))
        data = np.asanyarray(image.dataobj)
        if not np.all(np.isfinite(data)) or not np.allclose(data, np.rint(data), atol=1e-6):
            raise ValueError(f"segmentation must contain finite integer labels: {path}")
        integer_data = np.rint(data).astype(np.int16)
        segmentation_to_density(integer_data)
        affine = np.asarray(image.affine, dtype=np.float64)
        if reference_shape is None:
            reference_shape = integer_data.shape
            reference_affine = affine
        elif integer_data.shape != reference_shape or not np.allclose(
            affine, reference_affine, atol=1e-4
        ):
            raise ValueError("segmentations are not aligned to the same voxel grid")
        labels.append(integer_data)
    assert reference_affine is not None
    return LongitudinalSegmentations(tuple(labels), days, reference_affine, paths)


def _natural_path_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))
    )

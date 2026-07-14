"""CPU-safe shared 3D residual forecaster for the MU training gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn

TREATMENT_FEATURES = (
    "radiation_exposure",
    "systemic_cytotoxic_exposure",
    "antiangiogenic_exposure",
    "device_exposure",
    "local_intervention_exposure",
    "unknown_treatment_timing_count",
)
MORPHOLOGY_FEATURES = (
    "core_volume_ml",
    "flair_volume_ml",
    "enhancing_volume_ml",
    "cavity_volume_ml",
    "whole_abnormality_volume_ml",
)


@dataclass(frozen=True, slots=True)
class PreparedTransition:
    """Downsampled source-conditioned features and future whole-abnormality target."""

    transition_id: str
    patient_id: str
    features: np.ndarray
    persistence: np.ndarray
    target: np.ndarray
    shape: tuple[int, int, int]


class SharedResidualForecaster(nn.Module):
    """Predict a correction to persistence from geometry, time, and treatment."""

    def __init__(self, feature_count: int, hidden_width: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features: torch.Tensor, persistence: torch.Tensor) -> torch.Tensor:
        prior_logit = torch.where(
            persistence.reshape(-1, 1) > 0.5,
            torch.full_like(persistence.reshape(-1, 1), 3.0),
            torch.full_like(persistence.reshape(-1, 1), -3.0),
        )
        return prior_logit + self.network(features)


def load_transition_manifest(path: Path, *, required_role: str) -> list[dict[str, Any]]:
    """Load one role-specific transition index and reject mixed-role input."""
    value = json.loads(path.read_text(encoding="utf-8"))
    transitions = value.get("transitions")
    if value.get("dataset") != "MU-Glioma-Post" or not isinstance(transitions, list):
        raise ValueError("MU-Glioma-Post transition index required")
    if set(value.get("roles", [])) != {required_role}:
        raise ValueError(f"transition index must contain only the {required_role} role")
    if any(item.get("role") != required_role for item in transitions):
        raise ValueError("transition role contamination detected")
    return transitions


def prepare_transition(
    transition: dict[str, Any], *, downsample: int = 4
) -> PreparedTransition:
    """Construct source-only voxel features and a separate future target."""
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    try:
        import nibabel as nib
    except ImportError as error:  # pragma: no cover
        raise ImportError("install the imaging extra to train the shared model") from error
    source_image = nib.as_closest_canonical(nib.load(transition["source_segmentation"]))
    target_image = nib.as_closest_canonical(nib.load(transition["target_segmentation"]))
    source = np.rint(np.asanyarray(source_image.dataobj)[::downsample, ::downsample, ::downsample])
    target = np.rint(np.asanyarray(target_image.dataobj)[::downsample, ::downsample, ::downsample])
    if source.shape != target.shape or not np.allclose(source_image.affine, target_image.affine):
        raise ValueError("source and target segmentations must be aligned")
    if not set(np.unique(source)).issubset({0, 1, 2, 3, 4}) or not set(
        np.unique(target)
    ).issubset({0, 1, 2, 3, 4}):
        raise ValueError("MU labels must lie between 0 and 4")
    source_abnormality = np.isin(source, (1, 2, 3))
    target_abnormality = np.isin(target, (1, 2, 3))
    spacing = np.linalg.norm(source_image.affine[:3, :3], axis=0) * downsample
    outside = ndimage.distance_transform_edt(~source_abnormality, sampling=spacing)
    inside = ndimage.distance_transform_edt(source_abnormality, sampling=spacing)
    signed_distance = np.clip(outside - inside, -40.0, 40.0) / 40.0
    history_available = bool(transition.get("history_available", False))
    if history_available:
        previous_image = nib.as_closest_canonical(nib.load(transition["previous_segmentation"]))
        previous = np.rint(
            np.asanyarray(previous_image.dataobj)[::downsample, ::downsample, ::downsample]
        )
        if previous.shape != source.shape or not np.allclose(
            previous_image.affine, source_image.affine
        ):
            raise ValueError("previous and source segmentations must be aligned")
        history_days = float(transition["source_day"]) - float(transition["previous_day"])
        if history_days <= 0:
            raise ValueError("history interval must be positive")
    else:
        previous = source
        history_days = 1.0
    previous_abnormality = np.isin(previous, (1, 2, 3))
    previous_outside = ndimage.distance_transform_edt(~previous_abnormality, sampling=spacing)
    previous_inside = ndimage.distance_transform_edt(previous_abnormality, sampling=spacing)
    previous_distance = np.clip(previous_outside - previous_inside, -40.0, 40.0) / 40.0
    distance_velocity = (signed_distance - previous_distance) / history_days
    grid = np.indices(source.shape, dtype=np.float32)
    coordinates = [
        2.0 * grid[axis].reshape(-1) / max(source.shape[axis] - 1, 1) - 1.0
        for axis in range(3)
    ]
    one_hot = [(source.reshape(-1) == label).astype(np.float32) for label in range(1, 5)]
    previous_one_hot = [
        (previous.reshape(-1) == label).astype(np.float32) for label in range(1, 5)
    ]
    horizon = np.full(source.size, np.log1p(float(transition["horizon_days"])), np.float32)
    exposure = transition["treatment_exposure"]
    treatments = [
        np.full(source.size, float(exposure[name]), np.float32) for name in TREATMENT_FEATURES
    ]
    morphology = transition["source_morphology"]
    morphologies = [
        np.full(source.size, np.log1p(float(morphology[name])), np.float32)
        for name in MORPHOLOGY_FEATURES
    ]
    features = np.column_stack(
        [
            *coordinates,
            signed_distance.reshape(-1),
            previous_distance.reshape(-1),
            distance_velocity.reshape(-1),
            *one_hot,
            *previous_one_hot,
            np.full(source.size, float(history_available), np.float32),
            horizon,
            *treatments,
            *morphologies,
        ]
    ).astype(np.float32)
    return PreparedTransition(
        transition_id=str(transition["transition_id"]),
        patient_id=str(transition["patient_id"]),
        features=features,
        persistence=source_abnormality.reshape(-1).astype(np.float32),
        target=target_abnormality.reshape(-1).astype(np.float32),
        shape=source.shape,
    )


def stratified_sample_indices(
    transition: PreparedTransition, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Balance changed, persistent, and background voxels for training."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    changed = np.flatnonzero(transition.persistence != transition.target)
    positive = np.flatnonzero(transition.target > 0.5)
    persistent = np.flatnonzero(transition.persistence > 0.5)
    background = np.flatnonzero((transition.persistence + transition.target) == 0)
    strata = (changed, positive, persistent, background)
    per_stratum = count // len(strata)
    samples = [
        rng.choice(
            values if values.size else np.arange(transition.target.size),
            per_stratum,
            replace=True,
        )
        for values in strata
    ]
    remainder = count - per_stratum * len(strata)
    if remainder:
        samples.append(rng.integers(0, transition.target.size, size=remainder))
    result = np.concatenate(samples)
    rng.shuffle(result)
    return result


def uniform_sample_indices(
    transition: PreparedTransition, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample the true voxel prevalence so probability calibration remains valid."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    return rng.choice(
        transition.target.size,
        size=count,
        replace=count > transition.target.size,
    )


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(prediction & target)) / denominator

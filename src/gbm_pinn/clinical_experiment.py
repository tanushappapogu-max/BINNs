"""Two-dimensional real-patient postoperative forecasting pilot."""

from __future__ import annotations

import time as time_module
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

from gbm_pinn.clinical import (
    LongitudinalSegmentations,
    is_segmentation_path,
    load_longitudinal_segmentations,
    segmentation_to_density,
    select_observation_slice,
)
from gbm_pinn.pinn import PINNConfig, TrainingConfig, TumorPINN, fit_pinn, resolve_torch_device

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ClinicalPilotConfig:
    """Configuration for an early-scan to held-out-scan patient forecast."""

    patient_directory: Path
    scan_days: tuple[float, ...]
    observation_count: int = 2
    forecast_index: int = 2
    infiltrative_density: float = 0.3
    data_points_per_time: int = 2_048
    collocation_points: int = 4_096
    boundary_points: int = 1_024
    threshold: float = 0.1
    seed: int = 162
    device: str = "auto"
    epochs: int = 2_000
    hidden_width: int = 32
    hidden_layers: int = 3
    diffusivity_bounds: tuple[float, float] = (0.01, 2.0)
    proliferation_bounds: tuple[float, float] = (0.001, 0.05)
    initial_diffusivity: float = 0.13
    initial_proliferation_rate: float = 0.012
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 2e-3
    data_weight: float = 10.0
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    data_batch_size: int | None = 1_024
    collocation_batch_size: int | None = 1_024
    boundary_batch_size: int | None = 512
    causal_time_chunks: int = 4
    checkpoint_interval: int = 100
    checkpoint_path: Path | None = None
    resume_from_checkpoint: bool = False
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.observation_count <= self.forecast_index < len(self.scan_days):
            raise ValueError("observations must precede a valid held-out forecast index")
        if (
            self.data_points_per_time <= 0
            or min(self.collocation_points, self.boundary_points) <= 0
        ):
            raise ValueError("sample counts must be positive")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class ClinicalPilotResult:
    """Metrics and provenance for one held-out real-patient forecast."""

    config: dict[str, object]
    resolved_device: str
    axial_slice: int
    voxel_spacing_mm: tuple[float, float]
    observation_days: tuple[float, ...]
    forecast_day: float
    training_seconds: float
    estimated_diffusivity_mm2_per_day: float
    estimated_proliferation_per_day: float
    forecast_rmse_full_brain: float
    forecast_dice_full_brain: float
    forecast_dice_fixed_cavity_domain: float
    forecast_volume_relative_error_full_brain: float
    persistence_dice_full_brain: float
    dice_skill_over_persistence: float
    beats_persistence: bool
    initial_total_loss: float
    final_total_loss: float
    final_data_loss: float
    final_physics_loss: float
    final_boundary_loss: float


@dataclass(frozen=True, slots=True)
class PreparedClinicalPilot:
    """Leakage-safe arrays and samples used by the patient PINN."""

    axial_slice: int
    spacing: tuple[float, float]
    elapsed_days: FloatArray
    observation_density: FloatArray
    target_density: FloatArray
    brain_mask: BoolArray
    baseline_cavity_mask: BoolArray
    data_coordinates: torch.Tensor
    data_density: torch.Tensor
    collocation_coordinates: torch.Tensor
    boundary_coordinates: torch.Tensor
    boundary_normals: torch.Tensor


def run_clinical_pilot(config: ClinicalPilotConfig) -> ClinicalPilotResult:
    """Train on early scans and evaluate once against the locked later scan."""
    device = resolve_torch_device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    segmentations = load_longitudinal_segmentations(config.patient_directory, config.scan_days)
    prepared = prepare_clinical_pilot(segmentations, config, rng)
    spatial_upper = (
        (prepared.brain_mask.shape[0] - 1) * prepared.spacing[0],
        (prepared.brain_mask.shape[1] - 1) * prepared.spacing[1],
    )
    model = TumorPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([spatial_upper[0], spatial_upper[1], prepared.elapsed_days[-1]]),
        config=PINNConfig(
            hidden_width=config.hidden_width,
            hidden_layers=config.hidden_layers,
            diffusivity_bounds=config.diffusivity_bounds,
            proliferation_bounds=config.proliferation_bounds,
            initial_diffusivity=config.initial_diffusivity,
            initial_proliferation_rate=config.initial_proliferation_rate,
        ),
    ).to(device)
    tensors = (
        prepared.data_coordinates,
        prepared.data_density,
        prepared.collocation_coordinates,
        prepared.boundary_coordinates,
        prepared.boundary_normals,
    )
    data_coordinates, data_density, collocation, boundary, normals = (
        tensor.to(device) for tensor in tensors
    )
    _synchronize_device(device)
    start = time_module.perf_counter()
    training = fit_pinn(
        model,
        data_coordinates,
        data_density,
        collocation,
        boundary_coordinates=boundary,
        boundary_normals=normals,
        config=TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.network_learning_rate,
            parameter_learning_rate=config.parameter_learning_rate,
            data_weight=config.data_weight,
            physics_weight=config.physics_weight,
            boundary_weight=config.boundary_weight,
            data_batch_size=config.data_batch_size,
            collocation_batch_size=config.collocation_batch_size,
            boundary_batch_size=config.boundary_batch_size,
            causal_time_chunks=config.causal_time_chunks,
            checkpoint_interval=config.checkpoint_interval,
        ),
        checkpoint_path=config.checkpoint_path,
        resume_from_checkpoint=config.resume_from_checkpoint,
    )
    _synchronize_device(device)
    training_seconds = time_module.perf_counter() - start

    coordinates = _grid_coordinates(
        prepared.brain_mask.shape,
        prepared.spacing,
        prepared.elapsed_days[-1],
    ).to(device)
    with torch.no_grad():
        prediction = model(coordinates).reshape(prepared.brain_mask.shape).cpu().numpy()
    prediction[~prepared.brain_mask] = 0.0
    prediction[prepared.baseline_cavity_mask] = 0.0
    target = prepared.target_density
    persistence = prepared.observation_density[-1]
    fixed_domain = prepared.brain_mask & ~prepared.baseline_cavity_mask
    difference = prediction[prepared.brain_mask] - target[prepared.brain_mask]

    if config.artifact_path is not None:
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.artifact_path,
            prediction=prediction,
            target=target,
            observations=prepared.observation_density,
            brain_mask=prepared.brain_mask,
            baseline_cavity_mask=prepared.baseline_cavity_mask,
            axial_slice=prepared.axial_slice,
            elapsed_days=prepared.elapsed_days,
        )

    serialized_config = asdict(config)
    for key, value in tuple(serialized_config.items()):
        if isinstance(value, Path):
            serialized_config[key] = str(value)
    return ClinicalPilotResult(
        config=serialized_config,
        resolved_device=str(device),
        axial_slice=prepared.axial_slice,
        voxel_spacing_mm=prepared.spacing,
        observation_days=tuple(config.scan_days[: config.observation_count]),
        forecast_day=config.scan_days[config.forecast_index],
        training_seconds=training_seconds,
        estimated_diffusivity_mm2_per_day=float(model.diffusivity.detach().cpu()),
        estimated_proliferation_per_day=float(model.proliferation_rate.detach().cpu()),
        forecast_rmse_full_brain=float(np.sqrt(np.mean(difference**2))),
        forecast_dice_full_brain=_masked_dice(
            prediction, target, prepared.brain_mask, config.threshold
        ),
        forecast_dice_fixed_cavity_domain=_masked_dice(
            prediction, target, fixed_domain, config.threshold
        ),
        forecast_volume_relative_error_full_brain=_masked_volume_error(
            prediction, target, prepared.brain_mask, config.threshold
        ),
        persistence_dice_full_brain=(
            persistence_dice := _masked_dice(
                persistence, target, prepared.brain_mask, config.threshold
            )
        ),
        dice_skill_over_persistence=(
            forecast_dice := _masked_dice(prediction, target, prepared.brain_mask, config.threshold)
        )
        - persistence_dice,
        beats_persistence=forecast_dice > persistence_dice,
        initial_total_loss=training.total_loss[0],
        final_total_loss=training.total_loss[-1],
        final_data_loss=training.data_loss[-1],
        final_physics_loss=training.physics_loss[-1],
        final_boundary_loss=training.boundary_loss[-1],
    )


def prepare_clinical_pilot(
    segmentations: LongitudinalSegmentations,
    config: ClinicalPilotConfig,
    rng: np.random.Generator,
) -> PreparedClinicalPilot:
    """Build training samples without consulting the held-out scan for slice selection."""
    selected_labels = segmentations.labels[: config.forecast_index + 1]
    axial_slice = select_observation_slice(selected_labels, config.observation_count)
    spacing3d = _voxel_spacing(segmentations.affine)
    spacing = (float(spacing3d[0]), float(spacing3d[1]))
    brain_volume = _load_observation_brain_mask(
        config.patient_directory,
        segmentations.paths[0],
        segmentations.affine,
        segmentations.labels[0].shape,
    )
    brain_mask = brain_volume[:, :, axial_slice]
    sliced_labels = tuple(volume[:, :, axial_slice] for volume in selected_labels)
    baseline_cavity = sliced_labels[0] == 4
    observation_density = np.stack(
        [
            segmentation_to_density(labels, infiltrative_density=config.infiltrative_density)
            for labels in sliced_labels[: config.observation_count]
        ]
    )
    target_density = segmentation_to_density(
        sliced_labels[config.forecast_index],
        infiltrative_density=config.infiltrative_density,
    )
    elapsed_days = np.asarray(segmentations.days[: config.forecast_index + 1], dtype=np.float64)
    elapsed_days -= elapsed_days[0]
    data_coordinates, data_density = _sample_data(
        observation_density,
        sliced_labels[: config.observation_count],
        brain_mask,
        elapsed_days[: config.observation_count],
        spacing,
        config.data_points_per_time,
        rng,
    )
    active_mask = brain_mask & ~baseline_cavity
    collocation = _sample_interior(
        active_mask,
        spacing,
        elapsed_days[-1],
        config.collocation_points,
        rng,
    )
    boundary_coordinates, boundary_normals = _sample_mask_boundary(
        active_mask,
        spacing,
        elapsed_days[-1],
        config.boundary_points,
        rng,
    )
    return PreparedClinicalPilot(
        axial_slice=axial_slice,
        spacing=spacing,
        elapsed_days=elapsed_days,
        observation_density=observation_density,
        target_density=target_density,
        brain_mask=brain_mask,
        baseline_cavity_mask=baseline_cavity,
        data_coordinates=data_coordinates,
        data_density=data_density,
        collocation_coordinates=collocation,
        boundary_coordinates=boundary_coordinates,
        boundary_normals=boundary_normals,
    )


def _load_observation_brain_mask(
    patient_directory: Path,
    first_segmentation_path: Path,
    reference_affine: FloatArray,
    reference_shape: tuple[int, ...],
) -> BoolArray:
    try:
        import nibabel as nib
    except ImportError as error:  # pragma: no cover
        raise ImportError("install the 'imaging' extra to load NIfTI files") from error
    relative = first_segmentation_path.relative_to(patient_directory)
    timepoint_directory = patient_directory / relative.parts[0]
    candidates = tuple(
        path for path in timepoint_directory.rglob("*.nii*") if not is_segmentation_path(path)
    )
    if not candidates:
        raise FileNotFoundError("no skull-stripped MRI was found for the first observation")
    mask = np.zeros(reference_shape, dtype=bool)
    for path in candidates:
        image = nib.as_closest_canonical(nib.load(path))
        if image.shape != reference_shape or not np.allclose(
            image.affine, reference_affine, atol=1e-4
        ):
            raise ValueError("first-observation MRI is not aligned with the segmentations")
        data = np.asanyarray(image.dataobj)
        mask |= np.isfinite(data) & (np.abs(data) > 1e-8)
    if not np.any(mask):
        raise ValueError("first-observation MRI contains no nonzero brain voxels")
    return mask


def _sample_data(
    densities: FloatArray,
    labels: tuple[NDArray[np.integer], ...],
    brain_mask: BoolArray,
    times: FloatArray,
    spacing: tuple[float, float],
    points_per_time: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate_blocks: list[FloatArray] = []
    density_blocks: list[FloatArray] = []
    for density, label, time in zip(densities, labels, times, strict=True):
        active = brain_mask & (label != 4)
        tumor_indices = np.flatnonzero(active & np.isin(label, (1, 2, 3)))
        background_indices = np.flatnonzero(active & ~np.isin(label, (1, 2, 3)))
        tumor_count = min(tumor_indices.size, points_per_time // 2)
        background_count = min(background_indices.size, points_per_time - tumor_count)
        selected = np.concatenate(
            (
                rng.choice(tumor_indices, tumor_count, replace=False),
                rng.choice(background_indices, background_count, replace=False),
            )
        )
        rng.shuffle(selected)
        ij = np.column_stack(np.unravel_index(selected, density.shape))
        coordinates = np.column_stack(
            (ij[:, 0] * spacing[0], ij[:, 1] * spacing[1], np.full(selected.size, time))
        )
        coordinate_blocks.append(coordinates)
        density_blocks.append(density.ravel()[selected, None])
    return (
        torch.as_tensor(np.concatenate(coordinate_blocks), dtype=torch.float32),
        torch.as_tensor(np.concatenate(density_blocks), dtype=torch.float32),
    )


def _sample_interior(
    active_mask: BoolArray,
    spacing: tuple[float, float],
    maximum_time: float,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    active = np.column_stack(np.nonzero(active_mask))
    selected = active[rng.integers(0, active.shape[0], count)]
    coordinates = np.column_stack(
        (
            selected[:, 0] * spacing[0],
            selected[:, 1] * spacing[1],
            np.linspace(0.0, maximum_time, count),
        )
    )
    rng.shuffle(coordinates[:, 2])
    return torch.as_tensor(coordinates, dtype=torch.float32)


def _sample_mask_boundary(
    active_mask: BoolArray,
    spacing: tuple[float, float],
    maximum_time: float,
    count: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    boundary = active_mask & ~ndimage.binary_erosion(active_mask)
    points = np.column_stack(np.nonzero(boundary))
    selected = points[rng.integers(0, points.shape[0], count)]
    signed_distance = ndimage.distance_transform_edt(active_mask, sampling=spacing) - (
        ndimage.distance_transform_edt(~active_mask, sampling=spacing)
    )
    gradient = np.stack(np.gradient(signed_distance, *spacing), axis=-1)
    normals = gradient[selected[:, 0], selected[:, 1]]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, np.finfo(np.float64).eps)
    coordinates = np.column_stack(
        (
            selected[:, 0] * spacing[0],
            selected[:, 1] * spacing[1],
            np.linspace(0.0, maximum_time, count),
        )
    )
    rng.shuffle(coordinates[:, 2])
    return (
        torch.as_tensor(coordinates, dtype=torch.float32),
        torch.as_tensor(normals, dtype=torch.float32),
    )


def _grid_coordinates(
    shape: tuple[int, int], spacing: tuple[float, float], time: float
) -> torch.Tensor:
    first, second = np.meshgrid(
        np.arange(shape[0]) * spacing[0],
        np.arange(shape[1]) * spacing[1],
        indexing="ij",
    )
    return torch.as_tensor(
        np.column_stack((first.ravel(), second.ravel(), np.full(first.size, time))),
        dtype=torch.float32,
    )


def _voxel_spacing(affine: FloatArray) -> FloatArray:
    return np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=np.float64) ** 2, axis=0))


def _masked_dice(
    prediction: FloatArray, target: FloatArray, mask: BoolArray, threshold: float
) -> float:
    predicted = (prediction >= threshold) & mask
    observed = (target >= threshold) & mask
    denominator = int(predicted.sum() + observed.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.sum(predicted & observed)) / denominator


def _masked_volume_error(
    prediction: FloatArray, target: FloatArray, mask: BoolArray, threshold: float
) -> float:
    predicted = int(np.sum((prediction >= threshold) & mask))
    observed = int(np.sum((target >= threshold) & mask))
    if observed == 0:
        return 0.0 if predicted == 0 else float("inf")
    return abs(predicted - observed) / observed


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

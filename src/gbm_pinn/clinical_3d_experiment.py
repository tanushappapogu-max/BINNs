"""Memory-bounded three-dimensional real-patient postoperative forecasting pilot."""

from __future__ import annotations

import time as time_module
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

from gbm_pinn.clinical import load_longitudinal_segmentations, segmentation_to_density
from gbm_pinn.clinical_experiment import (
    _load_observation_brain_mask,
    _masked_dice,
    _masked_volume_error,
    _synchronize_device,
    _voxel_spacing,
)
from gbm_pinn.pinn import PINNConfig, TrainingConfig, TumorPINN, fit_pinn, resolve_torch_device
from gbm_pinn.treatment import TreatmentAwareTumorPINN, TreatmentWindow

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Clinical3DPilotConfig:
    """Configuration for a full-volume early-scan to held-out-scan forecast."""

    patient_directory: Path
    scan_days: tuple[float, ...]
    observation_count: int = 2
    forecast_index: int = 2
    infiltrative_density: float = 0.3
    tumor_sample_fraction: float = 0.1
    data_points_per_time: int = 8_192
    collocation_points: int = 16_384
    boundary_points: int = 4_096
    evaluation_batch_size: int = 65_536
    threshold: float = 0.1
    seed: int = 162
    device: str = "auto"
    epochs: int = 2_000
    hidden_width: int = 48
    hidden_layers: int = 4
    fourier_frequencies: tuple[float, ...] = ()
    diffusivity_bounds: tuple[float, float] = (0.01, 2.0)
    proliferation_bounds: tuple[float, float] = (0.001, 0.05)
    initial_diffusivity: float = 0.13
    initial_proliferation_rate: float = 0.012
    treatment_windows: tuple[TreatmentWindow, ...] = ()
    treatment_response_bounds: tuple[float, float] = (0.0, 0.2)
    initial_treatment_response: float = 0.02
    learn_diffusivity: bool = True
    learn_proliferation_rate: bool = True
    learn_treatment_response: bool = True
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 2e-3
    data_weight: float = 10.0
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    data_batch_size: int | None = 2_048
    collocation_batch_size: int | None = 2_048
    boundary_batch_size: int | None = 1_024
    causal_time_chunks: int = 4
    checkpoint_interval: int = 100
    checkpoint_path: Path | None = None
    resume_from_checkpoint: bool = False
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.observation_count <= self.forecast_index < len(self.scan_days):
            raise ValueError("observations must precede a valid held-out forecast index")
        counts = (
            self.data_points_per_time,
            self.collocation_points,
            self.boundary_points,
            self.evaluation_batch_size,
        )
        if min(counts) <= 0:
            raise ValueError("sample and batch counts must be positive")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if not 0.0 < self.tumor_sample_fraction < 1.0:
            raise ValueError("tumor_sample_fraction must lie in (0, 1)")
        if not self.treatment_windows and self.learn_treatment_response:
            object.__setattr__(self, "learn_treatment_response", False)


@dataclass(frozen=True, slots=True)
class Clinical3DPilotResult:
    """Full-volume held-out metrics and provenance for one patient forecast."""

    config: dict[str, object]
    resolved_device: str
    volume_shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    observation_days: tuple[float, ...]
    forecast_day: float
    training_seconds: float
    evaluation_seconds: float
    estimated_diffusivity_mm2_per_day: float
    estimated_proliferation_per_day: float
    estimated_treatment_response_per_day: float | None
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


def run_clinical_3d_pilot(config: Clinical3DPilotConfig) -> Clinical3DPilotResult:
    """Fit a 3D PINN on early masks and evaluate the locked later volume."""
    device = resolve_torch_device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    segmentations = load_longitudinal_segmentations(config.patient_directory, config.scan_days)
    selected_labels = segmentations.labels[: config.forecast_index + 1]
    spacing_array = _voxel_spacing(segmentations.affine)
    spacing = tuple(float(value) for value in spacing_array)
    brain_mask = _load_observation_brain_mask(
        config.patient_directory,
        segmentations.paths[0],
        segmentations.affine,
        segmentations.labels[0].shape,
    )
    baseline_cavity = selected_labels[0] == 4
    active_mask = brain_mask & ~baseline_cavity
    elapsed_days = np.asarray(segmentations.days[: config.forecast_index + 1], dtype=np.float64)
    elapsed_days -= elapsed_days[0]
    data_coordinates, data_density = _sample_volume_data(
        selected_labels[: config.observation_count],
        brain_mask,
        elapsed_days[: config.observation_count],
        spacing,
        config.infiltrative_density,
        config.data_points_per_time,
        config.tumor_sample_fraction,
        rng,
    )
    collocation = _sample_volume_interior(
        active_mask,
        spacing,
        elapsed_days[-1],
        config.collocation_points,
        rng,
    )
    boundary, normals = _sample_volume_boundary(
        active_mask,
        spacing,
        elapsed_days[-1],
        config.boundary_points,
        rng,
    )
    lower = torch.zeros(4)
    upper = torch.tensor(
        [
            (active_mask.shape[0] - 1) * spacing[0],
            (active_mask.shape[1] - 1) * spacing[1],
            (active_mask.shape[2] - 1) * spacing[2],
            elapsed_days[-1],
        ]
    )
    pinn_config = PINNConfig(
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        diffusivity_bounds=config.diffusivity_bounds,
        proliferation_bounds=config.proliferation_bounds,
        initial_diffusivity=config.initial_diffusivity,
        initial_proliferation_rate=config.initial_proliferation_rate,
        fourier_frequencies=config.fourier_frequencies,
    )
    model: TumorPINN
    if config.treatment_windows:
        model = TreatmentAwareTumorPINN(
            lower,
            upper,
            config.treatment_windows,
            pinn_config,
            treatment_response_bounds=config.treatment_response_bounds,
            initial_treatment_response=config.initial_treatment_response,
        )
    else:
        model = TumorPINN(lower, upper, pinn_config)
    model = model.to(device)
    tensors = (data_coordinates, data_density, collocation, boundary, normals)
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
        learn_diffusivity=config.learn_diffusivity,
        learn_proliferation_rate=config.learn_proliferation_rate,
        learn_treatment_response=config.learn_treatment_response,
        checkpoint_path=config.checkpoint_path,
        resume_from_checkpoint=config.resume_from_checkpoint,
    )
    _synchronize_device(device)
    training_seconds = time_module.perf_counter() - start
    evaluation_start = time_module.perf_counter()
    prediction = _predict_volume(
        model,
        active_mask.shape,
        spacing,
        elapsed_days[-1],
        device,
        config.evaluation_batch_size,
    )
    prediction[~brain_mask] = 0.0
    prediction[baseline_cavity] = 0.0
    target = segmentation_to_density(
        selected_labels[config.forecast_index],
        infiltrative_density=config.infiltrative_density,
    )
    persistence = segmentation_to_density(
        selected_labels[config.observation_count - 1],
        infiltrative_density=config.infiltrative_density,
    )
    evaluation_seconds = time_module.perf_counter() - evaluation_start
    fixed_domain = brain_mask & ~baseline_cavity
    difference = prediction[brain_mask] - target[brain_mask]
    forecast_dice = _masked_dice(prediction, target, brain_mask, config.threshold)
    persistence_dice = _masked_dice(persistence, target, brain_mask, config.threshold)
    if config.artifact_path is not None:
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.artifact_path,
            prediction=prediction,
            target=target,
            brain_mask=brain_mask,
            baseline_cavity_mask=baseline_cavity,
            elapsed_days=elapsed_days,
        )
    serialized_config = asdict(config)
    for key, value in tuple(serialized_config.items()):
        if isinstance(value, Path):
            serialized_config[key] = str(value)
    response = getattr(model, "treatment_response_rate", None)
    return Clinical3DPilotResult(
        config=serialized_config,
        resolved_device=str(device),
        volume_shape=active_mask.shape,
        voxel_spacing_mm=spacing,
        observation_days=tuple(config.scan_days[: config.observation_count]),
        forecast_day=config.scan_days[config.forecast_index],
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        estimated_diffusivity_mm2_per_day=float(model.diffusivity.detach().cpu()),
        estimated_proliferation_per_day=float(model.proliferation_rate.detach().cpu()),
        estimated_treatment_response_per_day=(
            None if response is None else float(response.detach().cpu())
        ),
        forecast_rmse_full_brain=float(np.sqrt(np.mean(difference**2))),
        forecast_dice_full_brain=forecast_dice,
        forecast_dice_fixed_cavity_domain=_masked_dice(
            prediction, target, fixed_domain, config.threshold
        ),
        forecast_volume_relative_error_full_brain=_masked_volume_error(
            prediction, target, brain_mask, config.threshold
        ),
        persistence_dice_full_brain=persistence_dice,
        dice_skill_over_persistence=forecast_dice - persistence_dice,
        beats_persistence=forecast_dice > persistence_dice,
        initial_total_loss=training.total_loss[0],
        final_total_loss=training.total_loss[-1],
        final_data_loss=training.data_loss[-1],
        final_physics_loss=training.physics_loss[-1],
        final_boundary_loss=training.boundary_loss[-1],
    )


def _sample_volume_data(
    labels: tuple[NDArray[np.integer], ...],
    brain_mask: BoolArray,
    times: FloatArray,
    spacing: tuple[float, float, float],
    infiltrative_density: float,
    points_per_time: int,
    tumor_sample_fraction: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates: list[FloatArray] = []
    densities: list[FloatArray] = []
    for label, time in zip(labels, times, strict=True):
        active = brain_mask & (label != 4)
        tumor = np.flatnonzero(active & np.isin(label, (1, 2, 3)))
        background = np.flatnonzero(active & ~np.isin(label, (1, 2, 3)))
        tumor_count = min(tumor.size, round(points_per_time * tumor_sample_fraction))
        background_count = min(background.size, points_per_time - tumor_count)
        selected = np.concatenate(
            (
                rng.choice(tumor, tumor_count, replace=False),
                rng.choice(background, background_count, replace=False),
            )
        )
        rng.shuffle(selected)
        ijk = np.column_stack(np.unravel_index(selected, label.shape))
        coordinates.append(
            np.column_stack(
                (
                    ijk[:, 0] * spacing[0],
                    ijk[:, 1] * spacing[1],
                    ijk[:, 2] * spacing[2],
                    np.full(selected.size, time),
                )
            )
        )
        density = segmentation_to_density(label, infiltrative_density=infiltrative_density)
        densities.append(density.ravel()[selected, None])
    return (
        torch.as_tensor(np.concatenate(coordinates), dtype=torch.float32),
        torch.as_tensor(np.concatenate(densities), dtype=torch.float32),
    )


def _sample_volume_interior(
    active_mask: BoolArray,
    spacing: tuple[float, float, float],
    maximum_time: float,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    active = np.column_stack(np.nonzero(active_mask))
    selected = active[rng.integers(0, active.shape[0], count)]
    times = np.linspace(0.0, maximum_time, count)
    rng.shuffle(times)
    return torch.as_tensor(
        np.column_stack((selected * np.asarray(spacing), times)), dtype=torch.float32
    )


def _sample_volume_boundary(
    active_mask: BoolArray,
    spacing: tuple[float, float, float],
    maximum_time: float,
    count: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    boundary = active_mask & ~ndimage.binary_erosion(active_mask)
    points = np.column_stack(np.nonzero(boundary))
    selected = points[rng.integers(0, points.shape[0], count)]
    distance = ndimage.distance_transform_edt(active_mask, sampling=spacing)
    normals = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        lower = selected.copy()
        upper = selected.copy()
        lower[:, axis] = np.maximum(lower[:, axis] - 1, 0)
        upper[:, axis] = np.minimum(upper[:, axis] + 1, active_mask.shape[axis] - 1)
        normals[:, axis] = (distance[tuple(upper.T)] - distance[tuple(lower.T)]) / (
            2.0 * spacing[axis]
        )
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    zero = norm[:, 0] <= np.finfo(np.float64).eps
    if np.any(zero):
        center = np.asarray(active_mask.shape, dtype=np.float64) / 2.0
        normals[zero] = selected[zero] - center
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norm, np.finfo(np.float64).eps)
    times = np.linspace(0.0, maximum_time, count)
    rng.shuffle(times)
    coordinates = np.column_stack((selected * np.asarray(spacing), times))
    return (
        torch.as_tensor(coordinates, dtype=torch.float32),
        torch.as_tensor(normals, dtype=torch.float32),
    )


def _predict_volume(
    model: TumorPINN,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    time: float,
    device: torch.device,
    batch_size: int,
) -> NDArray[np.float32]:
    prediction = np.empty(int(np.prod(shape)), dtype=np.float32)
    total = prediction.size
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            flat = np.arange(start, stop)
            ijk = np.column_stack(np.unravel_index(flat, shape))
            coordinates = torch.as_tensor(
                np.column_stack(
                    (
                        ijk * np.asarray(spacing),
                        np.full(stop - start, time),
                    )
                ),
                dtype=torch.float32,
                device=device,
            )
            prediction[start:stop] = model(coordinates).squeeze(1).cpu().numpy()
    return prediction.reshape(shape)

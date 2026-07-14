"""Leakage-safe real-patient fitting and rollout for the compartment model."""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import ndimage

from gbm_pinn.clinical import load_longitudinal_segmentations
from gbm_pinn.clinical_3d_experiment import (
    _sample_volume_boundary,
    _sample_volume_interior,
)
from gbm_pinn.clinical_experiment import (
    _load_observation_brain_mask,
    _masked_dice,
    _masked_volume_error,
    _synchronize_device,
    _voxel_spacing,
)
from gbm_pinn.multicompartment import MultiCompartmentParameters, mri_surrogate_channels
from gbm_pinn.multicompartment_pinn import (
    MultiCompartmentPINN,
    MultiCompartmentPINNConfig,
    MultiCompartmentTrainingConfig,
    fit_multicompartment_pinn,
    segmentation_to_observation_channels,
)
from gbm_pinn.multicompartment_solver import MultiCompartmentSolver
from gbm_pinn.pinn import resolve_torch_device
from gbm_pinn.treatment import TreatmentWindow

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MultiCompartmentClinicalConfig:
    """Settings for fitting early scans and forecasting one held-out scan."""

    patient_directory: Path
    scan_days: tuple[float, ...]
    observation_count: int = 3
    forecast_index: int = 3
    data_points_per_time: int = 8_192
    collocation_points: int = 16_384
    boundary_points: int = 4_096
    samples_per_stratum: tuple[float, float, float, float] = (0.2, 0.3, 0.2, 0.3)
    infiltrative_viable_density: float = 0.3
    latent_seed_dilation_mm: float = 10.0
    evaluation_batch_size: int = 65_536
    threshold: float = 0.5
    seed: int = 162
    device: str = "auto"
    treatment_windows: tuple[TreatmentWindow, ...] = ()
    edema_treatment_windows: tuple[TreatmentWindow, ...] = ()
    pinn: MultiCompartmentPINNConfig = field(default_factory=MultiCompartmentPINNConfig)
    training: MultiCompartmentTrainingConfig = field(
        default_factory=MultiCompartmentTrainingConfig
    )
    checkpoint_path: Path | None = None
    resume_from_checkpoint: bool = False
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.observation_count <= self.forecast_index < len(self.scan_days):
            raise ValueError("observations must precede a valid held-out forecast index")
        if min(self.data_points_per_time, self.collocation_points, self.boundary_points) <= 0:
            raise ValueError("sample counts must be positive")
        if len(self.samples_per_stratum) != 4 or any(
            value < 0 for value in self.samples_per_stratum
        ):
            raise ValueError("four nonnegative sampling fractions are required")
        if not np.isclose(sum(self.samples_per_stratum), 1.0):
            raise ValueError("sampling fractions must sum to one")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if not 0.0 <= self.infiltrative_viable_density <= 1.0:
            raise ValueError("infiltrative_viable_density must lie in [0, 1]")
        if self.latent_seed_dilation_mm < 0 or self.evaluation_batch_size <= 0:
            raise ValueError("latent seed dilation must be nonnegative and batch size positive")
        if self.resume_from_checkpoint and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required to resume training")


@dataclass(frozen=True, slots=True)
class MultiCompartmentClinicalResult:
    """Learned coefficients, losses, and held-out channel metrics."""

    resolved_device: str
    training_seconds: float
    simulation_seconds: float
    forecast_horizon_days: float
    parameters: dict[str, float | None]
    losses: dict[str, float]
    channel_metrics: dict[str, dict[str, float | bool | int | None]]
    whole_abnormality_metrics: dict[str, float | bool | int | None]


def run_multicompartment_clinical(
    config: MultiCompartmentClinicalConfig,
) -> MultiCompartmentClinicalResult:
    """Estimate patient coefficients from observed scans, then solve the PDE forward."""
    device = resolve_torch_device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    scans = load_longitudinal_segmentations(config.patient_directory, config.scan_days)
    labels = scans.labels[: config.forecast_index + 1]
    spacing = tuple(float(value) for value in _voxel_spacing(scans.affine))
    brain = _load_observation_brain_mask(
        config.patient_directory, scans.paths[0], scans.affine, labels[0].shape
    )
    baseline_cavity = labels[0] == 4
    inverse_domain = brain & ~baseline_cavity
    elapsed = np.asarray(scans.days[: config.forecast_index + 1], dtype=np.float64)
    elapsed -= elapsed[0]
    observation_end = float(elapsed[config.observation_count - 1])

    data_coordinates, targets = _sample_multicompartment_data(
        labels[: config.observation_count],
        brain,
        elapsed[: config.observation_count],
        spacing,
        config.data_points_per_time,
        config.samples_per_stratum,
        config.infiltrative_viable_density,
        rng,
    )
    collocation = _sample_volume_interior(
        inverse_domain, spacing, observation_end, config.collocation_points, rng
    )
    boundary, normals = _sample_volume_boundary(
        inverse_domain, spacing, observation_end, config.boundary_points, rng
    )
    shifted_windows = tuple(
        TreatmentWindow(
            window.start_day - scans.days[0],
            window.end_day - scans.days[0],
            window.intensity,
            window.decay_days,
        )
        for window in config.treatment_windows
    )
    shifted_edema_windows = tuple(
        TreatmentWindow(
            window.start_day - scans.days[0],
            window.end_day - scans.days[0],
            window.intensity,
            window.decay_days,
        )
        for window in config.edema_treatment_windows
    )
    upper = torch.tensor(
        [
            (brain.shape[0] - 1) * spacing[0],
            (brain.shape[1] - 1) * spacing[1],
            (brain.shape[2] - 1) * spacing[2],
            observation_end,
        ],
        dtype=torch.float32,
    )
    model = MultiCompartmentPINN(
        torch.zeros(4),
        upper,
        shifted_windows,
        config.pinn,
        edema_treatment_windows=shifted_edema_windows,
    ).to(device)
    data_coordinates, targets, collocation, boundary, normals = (
        value.to(device)
        for value in (data_coordinates, targets, collocation, boundary, normals)
    )
    _synchronize_device(device)
    start = time_module.perf_counter()
    training = fit_multicompartment_pinn(
        model,
        data_coordinates,
        targets,
        collocation,
        boundary_coordinates=boundary,
        boundary_normals=normals,
        config=config.training,
        checkpoint_path=config.checkpoint_path,
        resume_from_checkpoint=config.resume_from_checkpoint,
    )
    _synchronize_device(device)
    training_seconds = time_module.perf_counter() - start

    parameters = _model_parameters(
        model,
        cell_kill_identifiable=bool(config.treatment_windows),
        edema_treatment_identifiable=bool(config.edema_treatment_windows),
    )
    latest = labels[config.observation_count - 1]
    forward_cavity = (latest == 4) & brain
    forward_domain = brain & ~forward_cavity
    solver = MultiCompartmentSolver(
        np.full(brain.shape, parameters["viable_diffusivity_mm2_per_day"]),
        np.full(brain.shape, parameters["edema_diffusivity_mm2_per_day"]),
        brain,
        MultiCompartmentParameters(
            proliferation_rate=float(parameters["proliferation_per_day"]),
            edema_generation_rate=float(parameters["edema_generation_per_day"]),
            edema_clearance_rate=float(parameters["edema_clearance_per_day"]),
            necrosis_clearance_rate=float(parameters["necrosis_clearance_per_day"]),
            edema_half_saturation=model.multicompartment_config.edema_half_saturation,
            spontaneous_necrosis_rate=(
                model.multicompartment_config.spontaneous_necrosis_rate
            ),
            necrosis_threshold=model.multicompartment_config.necrosis_threshold,
            necrosis_transition_width=(
                model.multicompartment_config.necrosis_transition_width
            ),
        ),
        spacing=spacing,
        cavity_mask=forward_cavity,
    )
    horizon = float(scans.days[config.forecast_index] - scans.days[config.observation_count - 1])
    forecast_start_day = float(scans.days[config.observation_count - 1])
    treatment_exposure = _numpy_treatment_exposure(config.treatment_windows, forecast_start_day)
    edema_treatment_exposure = _numpy_treatment_exposure(
        config.edema_treatment_windows, forecast_start_day
    )
    latent_at_forecast_start = _predict_latent_volume(
        model,
        brain.shape,
        spacing,
        observation_end,
        device,
        config.evaluation_batch_size,
    )
    initial_viable, initial_edema, initial_necrotic = _latent_seeded_initial_state(
        latent_at_forecast_start,
        labels[: config.observation_count],
        brain,
        spacing,
        config.infiltrative_viable_density,
        config.latent_seed_dilation_mm,
        config.training.detection_limits,
    )
    simulation_start = time_module.perf_counter()
    simulation = solver.simulate(
        initial_viable,
        initial_edema,
        initial_necrotic,
        np.asarray([horizon]),
        treatment_cell_kill=lambda time: (parameters["treatment_cell_kill_per_day"] or 0.0)
        * treatment_exposure(time),
        treatment_edema_leakage_suppression=lambda time: (
            parameters["antiangiogenic_leakage_suppression"] or 0.0
        )
        * edema_treatment_exposure(time),
    )
    simulation_seconds = time_module.perf_counter() - simulation_start
    predicted = mri_surrogate_channels(
        simulation.viable[-1], simulation.edema[-1], simulation.necrotic[-1]
    )
    target = segmentation_to_observation_channels(
        labels[config.forecast_index],
        infiltrative_viable_density=config.infiltrative_viable_density,
    )
    persistence = segmentation_to_observation_channels(
        latest, infiltrative_viable_density=config.infiltrative_viable_density
    )
    channel_names = (
        "enhancing_tissue",
        "flair_abnormality",
        "nonenhancing_or_necrotic_core",
    )
    channel_metrics: dict[str, dict[str, float | bool]] = {}
    for index, name in enumerate(channel_names):
        channel_metrics[name] = _metrics(
            predicted[name], target[..., index], persistence[..., index], brain, config.threshold
        )
    predicted_whole = np.maximum.reduce(tuple(predicted.values()))
    target_whole = np.isin(labels[config.forecast_index], (1, 2, 3)).astype(np.float32)
    persistence_whole = np.isin(latest, (1, 2, 3)).astype(np.float32)
    whole_metrics = _metrics(
        predicted_whole, target_whole, persistence_whole, brain, config.threshold
    )
    if config.artifact_path is not None:
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.artifact_path,
            predicted_enhancing=predicted["enhancing_tissue"],
            predicted_flair=predicted["flair_abnormality"],
            predicted_core=predicted["nonenhancing_or_necrotic_core"],
            target_channels=target,
            persistence_channels=persistence,
            brain_mask=brain,
            forward_domain=forward_domain,
            initial_viable=initial_viable,
            initial_edema=initial_edema,
            initial_necrotic=initial_necrotic,
        )
    return MultiCompartmentClinicalResult(
        resolved_device=str(device),
        training_seconds=training_seconds,
        simulation_seconds=simulation_seconds,
        forecast_horizon_days=horizon,
        parameters=parameters,
        losses={
            "initial_total": training.total_loss[0],
            "final_total": training.total_loss[-1],
            "final_data": training.data_loss[-1],
            "final_physics": training.physics_loss[-1],
            "final_boundary": training.boundary_loss[-1],
            "final_radiation_jump": training.radiation_jump_loss[-1],
            **{
                f"scale_{name}": value for name, value in training.loss_scales.items()
            },
        },
        channel_metrics=channel_metrics,
        whole_abnormality_metrics=whole_metrics,
    )


def _sample_multicompartment_data(
    labels: tuple[NDArray[np.integer], ...],
    brain_mask: BoolArray,
    times: FloatArray,
    spacing: tuple[float, float, float],
    points_per_time: int,
    fractions: tuple[float, float, float, float],
    infiltrative_viable_density: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each MRI class separately so rare compartments retain influence."""
    all_coordinates: list[FloatArray] = []
    all_targets: list[NDArray[np.float32]] = []
    strata = (1, 2, 3, 0)
    requested = np.floor(np.asarray(fractions) * points_per_time).astype(int)
    requested[-1] += points_per_time - int(requested.sum())
    for volume, time in zip(labels, times, strict=True):
        active = brain_mask & (volume != 4)
        selected_parts: list[NDArray[np.int64]] = []
        for label, count in zip(strata, requested, strict=True):
            mask = active & ((volume == label) if label else ~np.isin(volume, (1, 2, 3, 4)))
            candidates = np.flatnonzero(mask)
            if candidates.size and count:
                selected_parts.append(
                    rng.choice(candidates, count, replace=candidates.size < count)
                )
        if not selected_parts:
            raise ValueError("an observation scan contains no sampleable brain voxels")
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)
        ijk = np.column_stack(np.unravel_index(selected, volume.shape))
        all_coordinates.append(
            np.column_stack((ijk * np.asarray(spacing), np.full(selected.size, time)))
        )
        targets = segmentation_to_observation_channels(
            volume, infiltrative_viable_density=infiltrative_viable_density
        )
        all_targets.append(targets.reshape(-1, 3)[selected])
    return (
        torch.as_tensor(np.concatenate(all_coordinates), dtype=torch.float32),
        torch.as_tensor(np.concatenate(all_targets), dtype=torch.float32),
    )


def _model_parameters(
    model: MultiCompartmentPINN,
    *,
    cell_kill_identifiable: bool,
    edema_treatment_identifiable: bool,
) -> dict[str, float | None]:
    return {
        "viable_diffusivity_mm2_per_day": float(model.diffusivity.detach().cpu()),
        "proliferation_per_day": float(model.proliferation_rate.detach().cpu()),
        "edema_diffusivity_mm2_per_day": float(model.edema_diffusivity.detach().cpu()),
        "edema_generation_per_day": float(model.edema_generation_rate.detach().cpu()),
        "edema_clearance_per_day": float(model.edema_clearance_rate.detach().cpu()),
        "necrosis_clearance_per_day": float(model.necrosis_clearance_rate.detach().cpu()),
        "treatment_cell_kill_per_day": (
            float(model.treatment_cell_kill_rate.detach().cpu())
            if cell_kill_identifiable
            else None
        ),
        "antiangiogenic_leakage_suppression": (
            float(model.antiangiogenic_leakage_suppression.detach().cpu())
            if edema_treatment_identifiable
            else None
        ),
    }


def _numpy_treatment_exposure(
    windows: tuple[TreatmentWindow, ...], offset_day: float
):
    def exposure(elapsed_day: float) -> float:
        day = offset_day + elapsed_day
        total = 0.0
        for window in windows:
            if window.start_day <= day <= window.end_day:
                total = max(total, window.intensity)
            elif day > window.end_day and window.decay_days > 0:
                total = max(
                    total,
                    window.intensity * np.exp(-(day - window.end_day) / window.decay_days),
                )
        return float(total)

    return exposure


def _predict_latent_volume(
    model: MultiCompartmentPINN,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    time: float,
    device: torch.device,
    batch_size: int,
) -> NDArray[np.float32]:
    prediction = np.empty((int(np.prod(shape)), 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, prediction.shape[0], batch_size):
            stop = min(start + batch_size, prediction.shape[0])
            flat = np.arange(start, stop)
            ijk = np.column_stack(np.unravel_index(flat, shape))
            coordinates = torch.as_tensor(
                np.column_stack((ijk * np.asarray(spacing), np.full(stop - start, time))),
                dtype=torch.float32,
                device=device,
            )
            prediction[start:stop] = model(coordinates).cpu().numpy()
    return prediction.reshape(*shape, 3)


def _latent_seeded_initial_state(
    latent: NDArray[np.floating],
    observed_labels: tuple[NDArray[np.integer], ...],
    brain_mask: BoolArray,
    spacing: tuple[float, float, float],
    infiltrative_viable_density: float,
    dilation_mm: float,
    latent_caps: tuple[float, float, float],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Retain sub-threshold history-supported seeds and hard-anchor visible labels."""
    latest = observed_labels[-1]
    if latent.shape != (*latest.shape, 3):
        raise ValueError("latent fields must match the observed volume shape")
    historical = np.logical_or.reduce(
        tuple(np.isin(volume, (1, 2, 3)) for volume in observed_labels)
    )
    seed_support = ndimage.distance_transform_edt(~historical, sampling=spacing) <= dilation_mm
    hidden = brain_mask & (latest == 0) & seed_support
    viable = np.where(hidden, np.minimum(latent[..., 0], latent_caps[0]), 0.0)
    edema = np.where(hidden, np.minimum(latent[..., 1], latent_caps[1]), 0.0)
    necrotic = np.where(hidden, np.minimum(latent[..., 2], latent_caps[2]), 0.0)
    viable[latest == 2] = infiltrative_viable_density
    viable[latest == 3] = 1.0
    edema[latest == 2] = 1.0
    necrotic[latest == 1] = 1.0
    cavity = latest == 4
    for state in (viable, edema, necrotic):
        state[~brain_mask | cavity] = 0.0
    return viable.astype(np.float64), edema.astype(np.float64), necrotic.astype(np.float64)


def _metrics(
    prediction: FloatArray,
    target: FloatArray,
    persistence: FloatArray,
    mask: BoolArray,
    threshold: float,
) -> dict[str, float | bool | int | None]:
    target_voxels = int(np.sum((target >= threshold) & mask))
    persistence_voxels = int(np.sum((persistence >= threshold) & mask))
    predicted_voxels = int(np.sum((prediction >= threshold) & mask))
    if target_voxels == 0:
        return {
            "evaluable": False,
            "target_voxels": target_voxels,
            "predicted_voxels": predicted_voxels,
            "persistence_voxels": persistence_voxels,
            "forecast_dice": None,
            "persistence_dice": None,
            "dice_skill_over_persistence": None,
            "beats_persistence": False,
            "volume_relative_error": None,
        }
    forecast_dice = _masked_dice(prediction, target, mask, threshold)
    persistence_dice = _masked_dice(persistence, target, mask, threshold)
    skill = forecast_dice - persistence_dice
    return {
        "evaluable": True,
        "target_voxels": target_voxels,
        "predicted_voxels": predicted_voxels,
        "persistence_voxels": persistence_voxels,
        "forecast_dice": forecast_dice,
        "persistence_dice": persistence_dice,
        "dice_skill_over_persistence": skill,
        "beats_persistence": skill > 0,
        "volume_relative_error": _masked_volume_error(
            prediction, target, mask, threshold
        ),
    }

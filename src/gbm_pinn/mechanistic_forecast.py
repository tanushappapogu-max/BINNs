"""Three-dimensional forward forecasts initialized from the last observed mask."""

from __future__ import annotations

import time as time_module
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gbm_pinn.clinical import load_longitudinal_segmentations, segmentation_to_density
from gbm_pinn.clinical_experiment import (
    _load_observation_brain_mask,
    _masked_dice,
    _masked_volume_error,
    _voxel_spacing,
)
from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.solver import FiniteVolumeSolver
from gbm_pinn.treatment import TreatmentWindow


@dataclass(frozen=True, slots=True)
class MechanisticForecastConfig:
    """Inputs for a held-out rollout from the last observed clinical volume."""

    patient_directory: Path
    scan_days: tuple[float, ...]
    observation_count: int
    forecast_index: int
    diffusivity_mm2_per_day: float
    proliferation_per_day: float
    treatment_response_per_day: float = 0.0
    treatment_windows: tuple[TreatmentWindow, ...] = ()
    treatment_time_offset_days: float = 0.0
    infiltrative_density: float = 0.3
    threshold: float = 0.1
    maximum_time_step: float | None = None
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.observation_count <= self.forecast_index < len(self.scan_days):
            raise ValueError("observations must precede a valid held-out forecast index")
        rates = (
            self.diffusivity_mm2_per_day,
            self.proliferation_per_day,
            self.treatment_response_per_day,
        )
        if any(not np.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("physical rates must be finite and nonnegative")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class MechanisticForecastResult:
    """Metrics from a forward equation solve against a locked future mask."""

    config: dict[str, object]
    volume_shape: tuple[int, int, int]
    forecast_horizon_days: float
    simulation_seconds: float
    stable_time_step_days: float
    forecast_rmse_full_brain: float
    forecast_dice_full_brain: float
    forecast_dice_last_observation_cavity_domain: float
    forecast_volume_relative_error_full_brain: float
    persistence_dice_full_brain: float
    dice_skill_over_persistence: float
    beats_persistence: bool


def run_mechanistic_forecast(config: MechanisticForecastConfig) -> MechanisticForecastResult:
    """Advance the learned equation from the last observed 3D density."""
    segmentations = load_longitudinal_segmentations(config.patient_directory, config.scan_days)
    observation_index = config.observation_count - 1
    initial_labels = segmentations.labels[observation_index]
    target_labels = segmentations.labels[config.forecast_index]
    brain_mask = _load_observation_brain_mask(
        config.patient_directory,
        segmentations.paths[observation_index],
        segmentations.affine,
        initial_labels.shape,
    )
    cavity = initial_labels == 4
    initial_density = segmentation_to_density(
        initial_labels, infiltrative_density=config.infiltrative_density
    )
    target = segmentation_to_density(
        target_labels, infiltrative_density=config.infiltrative_density
    )
    diffusivity = np.full(initial_labels.shape, config.diffusivity_mm2_per_day)
    solver = FiniteVolumeSolver(
        diffusivity,
        brain_mask,
        ReactionDiffusionParameters(proliferation_rate=config.proliferation_per_day),
        spacing=tuple(float(value) for value in _voxel_spacing(segmentations.affine)),
        cavity_mask=cavity,
    )
    horizon = config.scan_days[config.forecast_index] - config.scan_days[observation_index]

    def treatment(time: float) -> float:
        schedule_time = time + config.treatment_time_offset_days
        exposure = sum(
            _window_exposure(window, schedule_time) for window in config.treatment_windows
        )
        return config.treatment_response_per_day * exposure

    maximum_treatment_rate = config.treatment_response_per_day * sum(
        window.intensity for window in config.treatment_windows
    )
    stable_step = solver.stable_time_step(maximum_treatment_rate)
    start = time_module.perf_counter()
    result = solver.simulate(
        initial_density,
        np.asarray([horizon]),
        maximum_time_step=config.maximum_time_step,
        treatment=treatment if config.treatment_windows else None,
    )
    simulation_seconds = time_module.perf_counter() - start
    prediction = result.density[-1]
    fixed_domain = brain_mask & ~cavity
    difference = prediction[brain_mask] - target[brain_mask]
    forecast_dice = _masked_dice(prediction, target, brain_mask, config.threshold)
    persistence_dice = _masked_dice(initial_density, target, brain_mask, config.threshold)
    if config.artifact_path is not None:
        config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.artifact_path,
            prediction=prediction,
            target=target,
            initial_density=initial_density,
            brain_mask=brain_mask,
            initial_cavity_mask=cavity,
        )
    serialized_config = asdict(config)
    for key, value in tuple(serialized_config.items()):
        if isinstance(value, Path):
            serialized_config[key] = str(value)
    return MechanisticForecastResult(
        config=serialized_config,
        volume_shape=initial_labels.shape,
        forecast_horizon_days=horizon,
        simulation_seconds=simulation_seconds,
        stable_time_step_days=stable_step,
        forecast_rmse_full_brain=float(np.sqrt(np.mean(difference**2))),
        forecast_dice_full_brain=forecast_dice,
        forecast_dice_last_observation_cavity_domain=_masked_dice(
            prediction, target, fixed_domain, config.threshold
        ),
        forecast_volume_relative_error_full_brain=_masked_volume_error(
            prediction, target, brain_mask, config.threshold
        ),
        persistence_dice_full_brain=persistence_dice,
        dice_skill_over_persistence=forecast_dice - persistence_dice,
        beats_persistence=forecast_dice > persistence_dice,
    )


def _window_exposure(window: TreatmentWindow, time: float) -> float:
    if window.start_day <= time <= window.end_day:
        return window.intensity
    if time > window.end_day and window.decay_days > 0:
        return window.intensity * float(np.exp(-(time - window.end_day) / window.decay_days))
    return 0.0

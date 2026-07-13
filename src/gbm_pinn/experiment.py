"""End-to-end synthetic forecasting experiments."""

from __future__ import annotations

import time as time_module
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from gbm_pinn.cavity import CavityAwareTumorPINN, PiecewiseCavityTumorPINN
from gbm_pinn.equation import GrowthLaw, ReactionDiffusionParameters
from gbm_pinn.pinn import (
    PINNConfig,
    TrainingConfig,
    TumorPINN,
    fit_pinn,
    resolve_torch_device,
)
from gbm_pinn.solver import FiniteVolumeSolver
from gbm_pinn.synthetic import gaussian_density
from gbm_pinn.tissue import TissueAwareTumorPINN

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SyntheticExperimentConfig:
    """Configuration for a homogeneous synthetic forecast."""

    grid_size: int = 21
    domain_length: float = 1.0
    observation_times: tuple[float, ...] = (0.0, 0.4)
    forecast_time: float = 1.0
    true_diffusivity: float = 0.005
    true_proliferation_rate: float = 0.4
    growth_law: GrowthLaw = "logistic"
    allee_parameter: float = 0.0
    fitted_growth_law: GrowthLaw | None = None
    fitted_allee_parameter: float = 0.0
    white_to_gray_diffusivity_ratio: float = 1.0
    tissue_transition_center: float = 0.5
    tissue_transition_width: float = 0.08
    cavity_radius: float = 0.0
    cavity_center: tuple[float, float] = (0.5, 0.5)
    cavity_interface_radius: float = 0.0
    interface_points: int = 128
    diffusivity_bounds: tuple[float, float] = (0.001, 0.02)
    proliferation_bounds: tuple[float, float] = (0.05, 0.8)
    initial_diffusivity: float = 0.01
    initial_proliferation_rate: float = 0.2
    learn_diffusivity: bool = True
    learn_proliferation_rate: bool = True
    initial_peak: float = 0.6
    initial_standard_deviation: float = 0.125
    data_points_per_time: int = 256
    observation_noise_standard_deviation: float = 0.0
    collocation_points: int = 512
    boundary_points: int = 128
    threshold: float = 0.1
    seed: int = 17
    device: str = "cpu"
    epochs: int = 1_000
    hidden_width: int = 32
    hidden_layers: int = 3
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 5e-3
    data_weight: float = 10.0
    physics_weight: float = 5.0
    boundary_weight: float = 1.0
    interface_weight: float = 1.0
    lbfgs_max_iterations: int = 0
    data_batch_size: int | None = None
    collocation_batch_size: int | None = None
    boundary_batch_size: int | None = None
    interface_batch_size: int | None = None
    causal_time_chunks: int = 1

    def __post_init__(self) -> None:
        if self.grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if self.domain_length <= 0:
            raise ValueError("domain_length must be positive")
        times = np.asarray(self.observation_times, dtype=np.float64)
        if (
            times.ndim != 1
            or times.size == 0
            or times[0] != 0.0
            or np.any(np.diff(times) <= 0)
            or times[-1] >= self.forecast_time
        ):
            raise ValueError(
                "observation_times must start at zero, increase, and precede forecast_time"
            )
        if self.true_diffusivity <= 0 or self.true_proliferation_rate <= 0:
            raise ValueError("true physical parameters must be positive")
        if self.growth_law not in ("logistic", "weak_allee"):
            raise ValueError("growth_law must be 'logistic' or 'weak_allee'")
        if self.allee_parameter < 0:
            raise ValueError("allee_parameter must be nonnegative")
        if self.growth_law == "logistic" and self.allee_parameter != 0:
            raise ValueError("allee_parameter must be zero for logistic growth")
        if self.fitted_growth_law not in (None, "logistic", "weak_allee"):
            raise ValueError("fitted_growth_law must be 'logistic', 'weak_allee', or None")
        if self.fitted_allee_parameter < 0:
            raise ValueError("fitted_allee_parameter must be nonnegative")
        if self.fitted_growth_law in (None, "logistic") and self.fitted_allee_parameter != 0:
            raise ValueError("fitted_allee_parameter requires fitted_growth_law='weak_allee'")
        if self.white_to_gray_diffusivity_ratio < 1:
            raise ValueError("white_to_gray_diffusivity_ratio must be at least one")
        if not 0 < self.tissue_transition_center < self.domain_length:
            raise ValueError("tissue_transition_center must lie inside the spatial domain")
        if self.tissue_transition_width <= 0:
            raise ValueError("tissue_transition_width must be positive")
        if self.cavity_radius < 0:
            raise ValueError("cavity_radius must be nonnegative")
        if len(self.cavity_center) != 2 or any(
            not self.cavity_radius < coordinate < self.domain_length - self.cavity_radius
            for coordinate in self.cavity_center
        ):
            raise ValueError("cavity must lie strictly inside the spatial domain")
        if self.cavity_interface_radius < 0:
            raise ValueError("cavity_interface_radius must be nonnegative")
        if self.cavity_interface_radius > 0:
            if self.cavity_radius == 0:
                raise ValueError("cavity_interface_radius requires a positive cavity_radius")
            maximum_interface_radius = min(
                self.cavity_center[0],
                self.cavity_center[1],
                self.domain_length - self.cavity_center[0],
                self.domain_length - self.cavity_center[1],
            )
            if not self.cavity_radius < self.cavity_interface_radius < maximum_interface_radius:
                raise ValueError(
                    "cavity_interface_radius must exceed the cavity and lie inside the domain"
                )
        if (
            len(self.diffusivity_bounds) != 2
            or self.diffusivity_bounds[0] < 0
            or self.diffusivity_bounds[0] >= self.diffusivity_bounds[1]
        ):
            raise ValueError("diffusivity_bounds must be an increasing nonnegative pair")
        if (
            len(self.proliferation_bounds) != 2
            or self.proliferation_bounds[0] < 0
            or self.proliferation_bounds[0] >= self.proliferation_bounds[1]
        ):
            raise ValueError("proliferation_bounds must be an increasing nonnegative pair")
        if not self.diffusivity_bounds[0] < self.initial_diffusivity < self.diffusivity_bounds[1]:
            raise ValueError("initial_diffusivity must lie strictly inside diffusivity_bounds")
        if not (
            self.proliferation_bounds[0]
            < self.initial_proliferation_rate
            < self.proliferation_bounds[1]
        ):
            raise ValueError(
                "initial_proliferation_rate must lie strictly inside proliferation_bounds"
            )
        if not 0 < self.initial_peak <= 1:
            raise ValueError("initial_peak must lie in (0, 1]")
        if not 0 < self.initial_standard_deviation < self.domain_length:
            raise ValueError("initial_standard_deviation must lie inside the spatial domain")
        if (
            min(
                self.data_points_per_time,
                self.collocation_points,
                self.boundary_points,
                self.interface_points,
            )
            <= 0
        ):
            raise ValueError("sample counts must be positive")
        if self.observation_noise_standard_deviation < 0:
            raise ValueError("observation_noise_standard_deviation must be nonnegative")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie in (0, 1)")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
        if self.hidden_width <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.network_learning_rate <= 0 or self.parameter_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if (
            min(
                self.data_weight,
                self.physics_weight,
                self.boundary_weight,
                self.interface_weight,
            )
            < 0
        ):
            raise ValueError("loss weights must be nonnegative")
        if self.lbfgs_max_iterations < 0:
            raise ValueError("lbfgs_max_iterations must be nonnegative")
        for name, value in (
            ("data_batch_size", self.data_batch_size),
            ("collocation_batch_size", self.collocation_batch_size),
            ("boundary_batch_size", self.boundary_batch_size),
            ("interface_batch_size", self.interface_batch_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.causal_time_chunks <= 0:
            raise ValueError("causal_time_chunks must be positive")


@dataclass(frozen=True, slots=True)
class SyntheticExperimentResult:
    """Forecast, parameter, and optimization metrics."""

    config: dict[str, int | float]
    resolved_device: str
    training_seconds: float
    peak_accelerator_memory_mb: float | None
    forecast_rmse: float
    forecast_mae: float
    forecast_dice: float
    forecast_volume_relative_error: float
    estimated_diffusivity: float
    estimated_proliferation_rate: float
    diffusivity_relative_error: float
    proliferation_relative_error: float
    initial_total_loss: float
    final_total_loss: float
    final_data_loss: float
    final_physics_loss: float
    final_boundary_loss: float
    final_interface_loss: float


def run_synthetic_forecast(
    config: SyntheticExperimentConfig | None = None,
) -> SyntheticExperimentResult:
    """Fit a PINN to early snapshots and forecast a hidden future field."""
    config = config or SyntheticExperimentConfig()
    device = resolve_torch_device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    grid = np.linspace(0.0, config.domain_length, config.grid_size)
    spacing = config.domain_length / (config.grid_size - 1)
    shape = (config.grid_size, config.grid_size)
    cavity_mask = _cavity_mask(grid, config)
    active_mask = ~cavity_mask
    white_matter_probability = _white_matter_probability(grid, config)
    reference_diffusivity = config.true_diffusivity * (
        1.0 + (config.white_to_gray_diffusivity_ratio - 1.0) * white_matter_probability
    )
    initial_density = gaussian_density(
        shape,
        standard_deviation=config.initial_standard_deviation / spacing,
        peak=config.initial_peak,
    )
    initial_density[cavity_mask] = 0.0
    reference_solver = FiniteVolumeSolver(
        diffusivity=reference_diffusivity,
        brain_mask=np.ones(shape, dtype=bool),
        cavity_mask=cavity_mask,
        spacing=(spacing, spacing),
        parameters=ReactionDiffusionParameters(
            proliferation_rate=config.true_proliferation_rate,
            growth_law=config.growth_law,
            allee_parameter=config.allee_parameter,
        ),
    )
    reference_times = np.array((*config.observation_times, config.forecast_time))
    reference = reference_solver.simulate(initial_density, reference_times).density

    data_coordinates, data_density = _sample_observations(
        grid,
        np.asarray(config.observation_times),
        reference[:-1],
        config.data_points_per_time,
        config.observation_noise_standard_deviation,
        active_mask,
        rng,
    )
    collocation_coordinates = _sample_collocation(config, rng)
    boundary_coordinates, boundary_normals = _sample_boundaries(config, rng)
    interface_coordinates, interface_normals = _sample_interface(config, rng)

    coordinate_lower_bounds = torch.tensor([0.0, 0.0, 0.0])
    coordinate_upper_bounds = torch.tensor(
        [config.domain_length, config.domain_length, config.forecast_time]
    )
    fitted_growth_law = config.fitted_growth_law or config.growth_law
    fitted_allee_parameter = (
        config.allee_parameter
        if config.fitted_growth_law is None
        else config.fitted_allee_parameter
    )
    pinn_config = PINNConfig(
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        diffusivity_bounds=config.diffusivity_bounds,
        proliferation_bounds=config.proliferation_bounds,
        initial_diffusivity=config.initial_diffusivity,
        initial_proliferation_rate=config.initial_proliferation_rate,
        growth_law=fitted_growth_law,
        allee_parameter=fitted_allee_parameter,
    )
    if config.cavity_radius > 0 and config.white_to_gray_diffusivity_ratio > 1.0:
        raise ValueError("combined tissue and cavity features are not implemented")
    if config.cavity_interface_radius > 0:
        model = PiecewiseCavityTumorPINN(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            config.cavity_center,
            config.cavity_radius,
            config.cavity_interface_radius,
            config=pinn_config,
        )
    elif config.cavity_radius > 0:
        model = CavityAwareTumorPINN(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            config.cavity_center,
            config.cavity_radius,
            config=pinn_config,
        )
    elif config.white_to_gray_diffusivity_ratio == 1.0:
        model = TumorPINN(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            config=pinn_config,
        )
    else:
        model = TissueAwareTumorPINN(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            white_matter_probability,
            white_to_gray_diffusivity_ratio=config.white_to_gray_diffusivity_ratio,
            config=pinn_config,
        )
    model = model.to(device)
    data_coordinates = data_coordinates.to(device)
    data_density = data_density.to(device)
    collocation_coordinates = collocation_coordinates.to(device)
    boundary_coordinates = boundary_coordinates.to(device)
    boundary_normals = boundary_normals.to(device)
    if interface_coordinates is not None:
        interface_coordinates = interface_coordinates.to(device)
        interface_normals = interface_normals.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize_device(device)
    training_start = time_module.perf_counter()
    training = fit_pinn(
        model,
        data_coordinates,
        data_density,
        collocation_coordinates,
        boundary_coordinates=boundary_coordinates,
        boundary_normals=boundary_normals,
        interface_coordinates=interface_coordinates,
        interface_normals=interface_normals,
        config=TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.network_learning_rate,
            parameter_learning_rate=config.parameter_learning_rate,
            data_weight=config.data_weight,
            physics_weight=config.physics_weight,
            boundary_weight=config.boundary_weight,
            interface_weight=config.interface_weight,
            lbfgs_max_iterations=config.lbfgs_max_iterations,
            data_batch_size=config.data_batch_size,
            collocation_batch_size=config.collocation_batch_size,
            boundary_batch_size=config.boundary_batch_size,
            interface_batch_size=config.interface_batch_size,
            causal_time_chunks=config.causal_time_chunks,
        ),
        learn_diffusivity=config.learn_diffusivity,
        learn_proliferation_rate=config.learn_proliferation_rate,
    )
    _synchronize_device(device)
    training_seconds = time_module.perf_counter() - training_start
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2) if device.type == "cuda" else None
    )

    forecast_coordinates = _grid_coordinates(grid, config.forecast_time).to(device)
    with torch.no_grad():
        forecast = (
            model(forecast_coordinates).reshape(shape).detach().cpu().numpy().astype(np.float64)
        )
    forecast[cavity_mask] = 0.0
    target = reference[-1]
    difference = forecast - target
    estimated_diffusivity = float(model.diffusivity.detach())
    estimated_proliferation = float(model.proliferation_rate.detach())

    return SyntheticExperimentResult(
        config=asdict(config),
        resolved_device=str(device),
        training_seconds=training_seconds,
        peak_accelerator_memory_mb=peak_memory_mb,
        forecast_rmse=float(np.sqrt(np.mean(difference**2))),
        forecast_mae=float(np.mean(np.abs(difference))),
        forecast_dice=_dice_at_threshold(forecast, target, config.threshold),
        forecast_volume_relative_error=_relative_volume_error(forecast, target, config.threshold),
        estimated_diffusivity=estimated_diffusivity,
        estimated_proliferation_rate=estimated_proliferation,
        diffusivity_relative_error=abs(estimated_diffusivity - config.true_diffusivity)
        / config.true_diffusivity,
        proliferation_relative_error=abs(estimated_proliferation - config.true_proliferation_rate)
        / config.true_proliferation_rate,
        initial_total_loss=training.total_loss[0],
        final_total_loss=training.total_loss[-1],
        final_data_loss=training.data_loss[-1],
        final_physics_loss=training.physics_loss[-1],
        final_boundary_loss=training.boundary_loss[-1],
        final_interface_loss=training.interface_loss[-1],
    )


def _sample_observations(
    grid: FloatArray,
    times: FloatArray,
    density: FloatArray,
    points_per_time: int,
    noise_standard_deviation: float,
    active_mask: NDArray[np.bool_],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_coordinates = _grid_coordinates(grid, 0.0).numpy()
    active_indices = np.flatnonzero(active_mask.ravel())
    sample_count = min(points_per_time, active_indices.size)
    coordinate_blocks: list[FloatArray] = []
    density_blocks: list[FloatArray] = []
    for index, time in enumerate(times):
        selected = rng.choice(active_indices, sample_count, replace=False)
        coordinates = spatial_coordinates[selected].copy()
        coordinates[:, 2] = time
        coordinate_blocks.append(coordinates)
        selected_density = density[index].reshape(-1)[selected, None]
        if noise_standard_deviation > 0:
            selected_density = selected_density + rng.normal(
                0.0, noise_standard_deviation, size=selected_density.shape
            )
            selected_density = np.clip(selected_density, 0.0, 1.0)
        density_blocks.append(selected_density)
    return (
        torch.as_tensor(np.concatenate(coordinate_blocks), dtype=torch.float32),
        torch.as_tensor(np.concatenate(density_blocks), dtype=torch.float32),
    )


def _white_matter_probability(grid: FloatArray, config: SyntheticExperimentConfig) -> FloatArray:
    probability_x = 1.0 / (
        1.0 + np.exp(-(grid - config.tissue_transition_center) / config.tissue_transition_width)
    )
    return np.tile(probability_x, (config.grid_size, 1))


def _cavity_mask(grid: FloatArray, config: SyntheticExperimentConfig) -> NDArray[np.bool_]:
    if config.cavity_radius == 0:
        return np.zeros((config.grid_size, config.grid_size), dtype=bool)
    y_coordinates, x_coordinates = np.meshgrid(grid, grid, indexing="ij")
    squared_distance = (x_coordinates - config.cavity_center[0]) ** 2 + (
        y_coordinates - config.cavity_center[1]
    ) ** 2
    return squared_distance <= config.cavity_radius**2


def _sample_collocation(
    config: SyntheticExperimentConfig, rng: np.random.Generator
) -> torch.Tensor:
    accepted: list[FloatArray] = []
    accepted_count = 0
    while accepted_count < config.collocation_points:
        candidates = rng.random((config.collocation_points, 3))
        candidates[:, :2] *= config.domain_length
        candidates[:, 2] *= config.forecast_time
        if config.cavity_radius > 0:
            squared_distance = (candidates[:, 0] - config.cavity_center[0]) ** 2 + (
                candidates[:, 1] - config.cavity_center[1]
            ) ** 2
            candidates = candidates[squared_distance > config.cavity_radius**2]
        accepted.append(candidates)
        accepted_count += candidates.shape[0]
    coordinates = np.concatenate(accepted)[: config.collocation_points]
    coordinates[:, 2] = _stratified_times(config.collocation_points, config.forecast_time, rng)
    return torch.as_tensor(coordinates, dtype=torch.float32)


def _sample_boundaries(
    config: SyntheticExperimentConfig, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    cavity_count = config.boundary_points // 2 if config.cavity_radius > 0 else 0
    outer_count = config.boundary_points - cavity_count
    coordinates = rng.random((outer_count, 3))
    coordinates[:, :2] *= config.domain_length
    coordinates[:, 2] *= config.forecast_time
    normals = np.zeros((outer_count, 2), dtype=np.float64)
    sides = rng.integers(0, 4, size=outer_count)
    side_definitions = (
        (0, 0.0, 0, -1.0),
        (0, config.domain_length, 0, 1.0),
        (1, 0.0, 1, -1.0),
        (1, config.domain_length, 1, 1.0),
    )
    for side, (axis, position, normal_axis, normal_value) in enumerate(side_definitions):
        selected = sides == side
        coordinates[selected, axis] = position
        normals[selected, normal_axis] = normal_value
    if cavity_count > 0:
        angles = rng.uniform(0.0, 2.0 * np.pi, cavity_count)
        radial = np.column_stack((np.cos(angles), np.sin(angles)))
        cavity_coordinates = np.empty((cavity_count, 3), dtype=np.float64)
        cavity_coordinates[:, :2] = np.asarray(config.cavity_center) + config.cavity_radius * radial
        cavity_coordinates[:, 2] = rng.uniform(0.0, config.forecast_time, cavity_count)
        coordinates = np.concatenate((coordinates, cavity_coordinates))
        normals = np.concatenate((normals, -radial))
    coordinates[:, 2] = _stratified_times(config.boundary_points, config.forecast_time, rng)
    return (
        torch.as_tensor(coordinates, dtype=torch.float32),
        torch.as_tensor(normals, dtype=torch.float32),
    )


def _sample_interface(
    config: SyntheticExperimentConfig, rng: np.random.Generator
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if config.cavity_interface_radius == 0:
        return None, None
    angles = rng.uniform(0.0, 2.0 * np.pi, config.interface_points)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    coordinates = np.empty((config.interface_points, 3), dtype=np.float64)
    coordinates[:, :2] = np.asarray(config.cavity_center) + config.cavity_interface_radius * normals
    coordinates[:, 2] = _stratified_times(config.interface_points, config.forecast_time, rng)
    return (
        torch.as_tensor(coordinates, dtype=torch.float32),
        torch.as_tensor(normals, dtype=torch.float32),
    )


def _grid_coordinates(grid: FloatArray, time: float) -> torch.Tensor:
    y_coordinates, x_coordinates = np.meshgrid(grid, grid, indexing="ij")
    coordinates = np.column_stack(
        (
            x_coordinates.ravel(),
            y_coordinates.ravel(),
            np.full(x_coordinates.size, time),
        )
    )
    return torch.as_tensor(coordinates, dtype=torch.float32)


def _stratified_times(count: int, maximum_time: float, rng: np.random.Generator) -> FloatArray:
    times = np.linspace(0.0, maximum_time, count, dtype=np.float64)
    rng.shuffle(times)
    return times


def _dice_at_threshold(prediction: FloatArray, target: FloatArray, threshold: float) -> float:
    prediction_mask = prediction >= threshold
    target_mask = target >= threshold
    denominator = int(prediction_mask.sum() + target_mask.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction_mask, target_mask).sum()) / denominator


def _relative_volume_error(prediction: FloatArray, target: FloatArray, threshold: float) -> float:
    prediction_volume = int(np.count_nonzero(prediction >= threshold))
    target_volume = int(np.count_nonzero(target >= threshold))
    if target_volume == 0:
        return 0.0 if prediction_volume == 0 else float("inf")
    return abs(prediction_volume - target_volume) / target_volume


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

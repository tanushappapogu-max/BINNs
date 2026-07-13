"""End-to-end synthetic forecasting experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.pinn import PINNConfig, TrainingConfig, TumorPINN, fit_pinn
from gbm_pinn.solver import FiniteVolumeSolver
from gbm_pinn.synthetic import gaussian_density

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
    initial_diffusivity: float = 0.01
    initial_proliferation_rate: float = 0.2
    learn_diffusivity: bool = True
    learn_proliferation_rate: bool = True
    initial_peak: float = 0.6
    initial_standard_deviation: float = 2.5
    data_points_per_time: int = 256
    collocation_points: int = 512
    boundary_points: int = 128
    threshold: float = 0.1
    seed: int = 17
    epochs: int = 1_000
    hidden_width: int = 32
    hidden_layers: int = 3
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 5e-3
    data_weight: float = 10.0
    physics_weight: float = 5.0
    boundary_weight: float = 1.0
    lbfgs_max_iterations: int = 0

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
        if not 0.001 < self.initial_diffusivity < 0.02:
            raise ValueError("initial_diffusivity must lie inside (0.001, 0.02)")
        if not 0.05 < self.initial_proliferation_rate < 0.8:
            raise ValueError("initial_proliferation_rate must lie inside (0.05, 0.8)")
        if not 0 < self.initial_peak <= 1:
            raise ValueError("initial_peak must lie in (0, 1]")
        if self.initial_standard_deviation <= 0:
            raise ValueError("initial_standard_deviation must be positive")
        if min(self.data_points_per_time, self.collocation_points, self.boundary_points) <= 0:
            raise ValueError("sample counts must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must lie in (0, 1)")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.hidden_width <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.network_learning_rate <= 0 or self.parameter_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if min(self.data_weight, self.physics_weight, self.boundary_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.lbfgs_max_iterations < 0:
            raise ValueError("lbfgs_max_iterations must be nonnegative")


@dataclass(frozen=True, slots=True)
class SyntheticExperimentResult:
    """Forecast, parameter, and optimization metrics."""

    config: dict[str, int | float]
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


def run_synthetic_forecast(
    config: SyntheticExperimentConfig | None = None,
) -> SyntheticExperimentResult:
    """Fit a PINN to early snapshots and forecast a hidden future field."""
    config = config or SyntheticExperimentConfig()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    grid = np.linspace(0.0, config.domain_length, config.grid_size)
    spacing = config.domain_length / (config.grid_size - 1)
    shape = (config.grid_size, config.grid_size)
    initial_density = gaussian_density(
        shape,
        standard_deviation=config.initial_standard_deviation,
        peak=config.initial_peak,
    )
    reference_solver = FiniteVolumeSolver(
        diffusivity=np.full(shape, config.true_diffusivity),
        brain_mask=np.ones(shape, dtype=bool),
        spacing=(spacing, spacing),
        parameters=ReactionDiffusionParameters(proliferation_rate=config.true_proliferation_rate),
    )
    reference_times = np.array((*config.observation_times, config.forecast_time))
    reference = reference_solver.simulate(initial_density, reference_times).density

    data_coordinates, data_density = _sample_observations(
        grid,
        np.asarray(config.observation_times),
        reference[:-1],
        config.data_points_per_time,
        rng,
    )
    collocation_coordinates = _sample_collocation(config, rng)
    boundary_coordinates, boundary_normals = _sample_boundaries(config, rng)

    model = TumorPINN(
        coordinate_lower_bounds=torch.tensor([0.0, 0.0, 0.0]),
        coordinate_upper_bounds=torch.tensor(
            [config.domain_length, config.domain_length, config.forecast_time]
        ),
        config=PINNConfig(
            hidden_width=config.hidden_width,
            hidden_layers=config.hidden_layers,
            diffusivity_bounds=(0.001, 0.02),
            proliferation_bounds=(0.05, 0.8),
            initial_diffusivity=config.initial_diffusivity,
            initial_proliferation_rate=config.initial_proliferation_rate,
        ),
    )
    training = fit_pinn(
        model,
        data_coordinates,
        data_density,
        collocation_coordinates,
        boundary_coordinates=boundary_coordinates,
        boundary_normals=boundary_normals,
        config=TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.network_learning_rate,
            parameter_learning_rate=config.parameter_learning_rate,
            data_weight=config.data_weight,
            physics_weight=config.physics_weight,
            boundary_weight=config.boundary_weight,
            lbfgs_max_iterations=config.lbfgs_max_iterations,
        ),
        learn_diffusivity=config.learn_diffusivity,
        learn_proliferation_rate=config.learn_proliferation_rate,
    )

    forecast_coordinates = _grid_coordinates(grid, config.forecast_time)
    with torch.no_grad():
        forecast = (
            model(forecast_coordinates).reshape(shape).detach().cpu().numpy().astype(np.float64)
        )
    target = reference[-1]
    difference = forecast - target
    estimated_diffusivity = float(model.diffusivity.detach())
    estimated_proliferation = float(model.proliferation_rate.detach())

    return SyntheticExperimentResult(
        config=asdict(config),
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
    )


def _sample_observations(
    grid: FloatArray,
    times: FloatArray,
    density: FloatArray,
    points_per_time: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_coordinates = _grid_coordinates(grid, 0.0).numpy()
    sample_count = min(points_per_time, spatial_coordinates.shape[0])
    coordinate_blocks: list[FloatArray] = []
    density_blocks: list[FloatArray] = []
    for index, time in enumerate(times):
        selected = rng.choice(spatial_coordinates.shape[0], sample_count, replace=False)
        coordinates = spatial_coordinates[selected].copy()
        coordinates[:, 2] = time
        coordinate_blocks.append(coordinates)
        density_blocks.append(density[index].reshape(-1)[selected, None])
    return (
        torch.as_tensor(np.concatenate(coordinate_blocks), dtype=torch.float32),
        torch.as_tensor(np.concatenate(density_blocks), dtype=torch.float32),
    )


def _sample_collocation(
    config: SyntheticExperimentConfig, rng: np.random.Generator
) -> torch.Tensor:
    coordinates = rng.random((config.collocation_points, 3))
    coordinates[:, :2] *= config.domain_length
    coordinates[:, 2] *= config.forecast_time
    return torch.as_tensor(coordinates, dtype=torch.float32)


def _sample_boundaries(
    config: SyntheticExperimentConfig, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = rng.random((config.boundary_points, 3))
    coordinates[:, :2] *= config.domain_length
    coordinates[:, 2] *= config.forecast_time
    normals = np.zeros((config.boundary_points, 2), dtype=np.float64)
    sides = rng.integers(0, 4, size=config.boundary_points)
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

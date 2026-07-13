"""Physics-informed neural network for the homogeneous Fisher-KPP equation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PINNConfig:
    """Network structure and bounded physical-parameter configuration."""

    hidden_width: int = 64
    hidden_layers: int = 4
    diffusivity_bounds: tuple[float, float] = (1e-4, 1.0)
    proliferation_bounds: tuple[float, float] = (1e-4, 1.0)
    initial_diffusivity: float = 0.1
    initial_proliferation_rate: float = 0.1
    carrying_capacity: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_width <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")
        _validate_bounded_initial_value(
            self.initial_diffusivity, self.diffusivity_bounds, "initial_diffusivity"
        )
        _validate_bounded_initial_value(
            self.initial_proliferation_rate,
            self.proliferation_bounds,
            "initial_proliferation_rate",
        )
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization settings and loss weights."""

    epochs: int = 1_000
    learning_rate: float = 1e-3
    parameter_learning_rate: float | None = None
    data_weight: float = 1.0
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    lbfgs_max_iterations: int = 0
    lbfgs_learning_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.parameter_learning_rate is not None and self.parameter_learning_rate <= 0:
            raise ValueError("parameter_learning_rate must be positive")
        if min(self.data_weight, self.physics_weight, self.boundary_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.lbfgs_max_iterations < 0:
            raise ValueError("lbfgs_max_iterations must be nonnegative")
        if self.lbfgs_learning_rate <= 0:
            raise ValueError("lbfgs_learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Per-epoch optimization losses."""

    total_loss: tuple[float, ...]
    data_loss: tuple[float, ...]
    physics_loss: tuple[float, ...]
    boundary_loss: tuple[float, ...]


class TumorPINN(nn.Module):
    """Map physical space-time coordinates to normalized tumor density."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        config: PINNConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or PINNConfig()
        lower = torch.as_tensor(coordinate_lower_bounds, dtype=torch.float32)
        upper = torch.as_tensor(coordinate_upper_bounds, dtype=torch.float32)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("coordinate bounds must contain x, y, and t")
        if not torch.all(torch.isfinite(lower)) or not torch.all(torch.isfinite(upper)):
            raise ValueError("coordinate bounds must be finite")
        if torch.any(lower >= upper):
            raise ValueError("lower coordinate bounds must be below upper bounds")

        self.config = config
        self.register_buffer("coordinate_lower_bounds", lower)
        self.register_buffer("coordinate_upper_bounds", upper)

        layers: list[nn.Module] = []
        input_width = 3
        for _ in range(config.hidden_layers):
            layers.extend((nn.Linear(input_width, config.hidden_width), nn.Tanh()))
            input_width = config.hidden_width
        layers.append(nn.Linear(input_width, 1))
        self.network = nn.Sequential(*layers)

        self.raw_diffusivity = nn.Parameter(
            torch.tensor(
                _inverse_bounded_transform(config.initial_diffusivity, config.diffusivity_bounds),
                dtype=torch.float32,
            )
        )
        self.raw_proliferation_rate = nn.Parameter(
            torch.tensor(
                _inverse_bounded_transform(
                    config.initial_proliferation_rate, config.proliferation_bounds
                ),
                dtype=torch.float32,
            )
        )
        self.reset_parameters()

    @property
    def diffusivity(self) -> Tensor:
        """Return diffusivity constrained to its configured interval."""
        return _bounded_transform(self.raw_diffusivity, self.config.diffusivity_bounds)

    @property
    def proliferation_rate(self) -> Tensor:
        """Return proliferation constrained to its configured interval."""
        return _bounded_transform(self.raw_proliferation_rate, self.config.proliferation_bounds)

    def reset_parameters(self) -> None:
        """Initialize linear layers with tanh-compatible Xavier weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Predict density for physical coordinates with columns x, y, and t."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (sample, 3)")
        normalized = (
            2.0
            * (
                (coordinates - self.coordinate_lower_bounds)
                / (self.coordinate_upper_bounds - self.coordinate_lower_bounds)
            )
            - 1.0
        )
        return self.config.carrying_capacity * torch.sigmoid(self.network(normalized))


def pde_residual(model: nn.Module, coordinates: Tensor) -> Tensor:
    """Return the Fisher-KPP residual at interior collocation coordinates."""
    coordinates = coordinates.detach().clone().requires_grad_(True)
    density = model(coordinates)
    gradient = torch.autograd.grad(
        density,
        coordinates,
        grad_outputs=torch.ones_like(density),
        create_graph=True,
    )[0]
    density_x = gradient[:, 0:1]
    density_y = gradient[:, 1:2]
    density_t = gradient[:, 2:3]
    density_xx = torch.autograd.grad(
        density_x,
        coordinates,
        grad_outputs=torch.ones_like(density_x),
        create_graph=True,
    )[0][:, 0:1]
    density_yy = torch.autograd.grad(
        density_y,
        coordinates,
        grad_outputs=torch.ones_like(density_y),
        create_graph=True,
    )[0][:, 1:2]
    return (
        density_t
        - model.diffusivity * (density_xx + density_yy)
        - model.proliferation_rate * density * (1.0 - density / model.config.carrying_capacity)
    )


def normal_flux(model: nn.Module, coordinates: Tensor, normals: Tensor) -> Tensor:
    """Return outward diffusive flux at boundary coordinates."""
    if normals.ndim != 2 or normals.shape[1] != 2:
        raise ValueError("normals must have shape (sample, 2)")
    if normals.shape[0] != coordinates.shape[0]:
        raise ValueError("normals and coordinates must contain the same number of samples")
    coordinates = coordinates.detach().clone().requires_grad_(True)
    density = model(coordinates)
    gradient = torch.autograd.grad(
        density,
        coordinates,
        grad_outputs=torch.ones_like(density),
        create_graph=True,
    )[0][:, :2]
    return model.diffusivity * torch.sum(gradient * normals, dim=1, keepdim=True)


def fit_pinn(
    model: TumorPINN,
    data_coordinates: Tensor,
    data_density: Tensor,
    collocation_coordinates: Tensor,
    *,
    boundary_coordinates: Tensor | None = None,
    boundary_normals: Tensor | None = None,
    config: TrainingConfig | None = None,
    learn_diffusivity: bool = True,
    learn_proliferation_rate: bool = True,
) -> TrainingResult:
    """Fit network and physical parameters with Adam."""
    config = config or TrainingConfig()
    if data_density.ndim != 2 or data_density.shape[1] != 1:
        raise ValueError("data_density must have shape (sample, 1)")
    if data_coordinates.shape[0] != data_density.shape[0]:
        raise ValueError("data coordinates and density must contain equal samples")
    if (boundary_coordinates is None) != (boundary_normals is None):
        raise ValueError("boundary coordinates and normals must be provided together")

    model.raw_diffusivity.requires_grad_(learn_diffusivity)
    model.raw_proliferation_rate.requires_grad_(learn_proliferation_rate)
    parameter_learning_rate = config.parameter_learning_rate or config.learning_rate
    parameter_groups: list[dict[str, object]] = [
        {"params": model.network.parameters(), "lr": config.learning_rate}
    ]
    physical_parameters = [
        parameter
        for parameter, enabled in (
            (model.raw_diffusivity, learn_diffusivity),
            (model.raw_proliferation_rate, learn_proliferation_rate),
        )
        if enabled
    ]
    if physical_parameters:
        parameter_groups.append({"params": physical_parameters, "lr": parameter_learning_rate})
    optimizer = torch.optim.Adam(parameter_groups)
    total_history: list[float] = []
    data_history: list[float] = []
    physics_history: list[float] = []
    boundary_history: list[float] = []

    def loss_terms() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        data_loss = torch.mean((model(data_coordinates) - data_density) ** 2)
        physics_loss = torch.mean(pde_residual(model, collocation_coordinates) ** 2)
        if boundary_coordinates is None:
            boundary_loss = torch.zeros((), device=data_coordinates.device)
        else:
            boundary_loss = torch.mean(
                normal_flux(model, boundary_coordinates, boundary_normals) ** 2
            )
        total_loss = (
            config.data_weight * data_loss
            + config.physics_weight * physics_loss
            + config.boundary_weight * boundary_loss
        )
        return total_loss, data_loss, physics_loss, boundary_loss

    def record_losses(
        total_loss: Tensor, data_loss: Tensor, physics_loss: Tensor, boundary_loss: Tensor
    ) -> None:
        total_history.append(float(total_loss.detach()))
        data_history.append(float(data_loss.detach()))
        physics_history.append(float(physics_loss.detach()))
        boundary_history.append(float(boundary_loss.detach()))

    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss, data_loss, physics_loss, boundary_loss = loss_terms()
        total_loss.backward()
        optimizer.step()
        record_losses(total_loss, data_loss, physics_loss, boundary_loss)

    if config.lbfgs_max_iterations > 0:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        lbfgs = torch.optim.LBFGS(
            trainable_parameters,
            lr=config.lbfgs_learning_rate,
            max_iter=config.lbfgs_max_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            total_loss, data_loss, physics_loss, boundary_loss = loss_terms()
            total_loss.backward()
            record_losses(total_loss, data_loss, physics_loss, boundary_loss)
            return total_loss

        lbfgs.step(closure)

    return TrainingResult(
        total_loss=tuple(total_history),
        data_loss=tuple(data_history),
        physics_loss=tuple(physics_history),
        boundary_loss=tuple(boundary_history),
    )


def _validate_bounded_initial_value(value: float, bounds: tuple[float, float], name: str) -> None:
    lower, upper = bounds
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or lower >= upper:
        raise ValueError(f"{name} bounds must be nonnegative, finite, and strictly ordered")
    if not lower < value < upper:
        raise ValueError(f"{name} must lie strictly inside its bounds")


def _bounded_transform(raw_value: Tensor, bounds: tuple[float, float]) -> Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * torch.sigmoid(raw_value)


def _inverse_bounded_transform(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    fraction = (value - lower) / (upper - lower)
    return float(np.log(fraction / (1.0 - fraction)))

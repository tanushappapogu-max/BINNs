"""Physics-informed neural network for the homogeneous Fisher-KPP equation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from gbm_pinn.equation import GrowthLaw


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve an execution device and reject unavailable accelerators."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if requested != "cpu" and requested != "cuda" and requested != "mps":
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    return torch.device(requested)


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
    growth_law: GrowthLaw = "logistic"
    allee_parameter: float = 0.0
    fourier_frequencies: tuple[float, ...] = ()

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
        if self.growth_law not in ("logistic", "weak_allee"):
            raise ValueError("growth_law must be 'logistic' or 'weak_allee'")
        if self.allee_parameter < 0:
            raise ValueError("allee_parameter must be nonnegative")
        if self.growth_law == "logistic" and self.allee_parameter != 0:
            raise ValueError("allee_parameter must be zero for logistic growth")
        if any(not np.isfinite(value) or value <= 0 for value in self.fourier_frequencies):
            raise ValueError("fourier frequencies must be finite and positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization settings and loss weights."""

    epochs: int = 1_000
    learning_rate: float = 1e-3
    parameter_learning_rate: float | None = None
    data_weight: float = 1.0
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    interface_weight: float = 1.0
    lbfgs_max_iterations: int = 0
    lbfgs_learning_rate: float = 1.0
    data_batch_size: int | None = None
    collocation_batch_size: int | None = None
    boundary_batch_size: int | None = None
    interface_batch_size: int | None = None
    causal_time_chunks: int = 1
    checkpoint_interval: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.parameter_learning_rate is not None and self.parameter_learning_rate <= 0:
            raise ValueError("parameter_learning_rate must be positive")
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
        if self.lbfgs_learning_rate <= 0:
            raise ValueError("lbfgs_learning_rate must be positive")
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
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when provided")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Per-epoch optimization losses."""

    total_loss: tuple[float, ...]
    data_loss: tuple[float, ...]
    physics_loss: tuple[float, ...]
    boundary_loss: tuple[float, ...]
    interface_loss: tuple[float, ...]


class TumorPINN(nn.Module):
    """Map physical space-time coordinates to normalized tumor density."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        config: PINNConfig | None = None,
        *,
        additional_input_width: int = 0,
    ) -> None:
        super().__init__()
        config = config or PINNConfig()
        lower = torch.as_tensor(coordinate_lower_bounds, dtype=torch.float32)
        upper = torch.as_tensor(coordinate_upper_bounds, dtype=torch.float32)
        if lower.ndim != 1 or upper.shape != lower.shape or lower.numel() < 2:
            raise ValueError("coordinate bounds must contain spatial dimensions followed by time")
        if not torch.all(torch.isfinite(lower)) or not torch.all(torch.isfinite(upper)):
            raise ValueError("coordinate bounds must be finite")
        if torch.any(lower >= upper):
            raise ValueError("lower coordinate bounds must be below upper bounds")
        if additional_input_width < 0:
            raise ValueError("additional_input_width must be nonnegative")

        self.config = config
        self.spatial_dimensions = lower.numel() - 1
        self.register_buffer("coordinate_lower_bounds", lower)
        self.register_buffer("coordinate_upper_bounds", upper)

        feature_width = (
            lower.numel()
            + 2 * self.spatial_dimensions * len(config.fourier_frequencies)
            + additional_input_width
        )
        self.network = self._make_network(feature_width)

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

    def _make_network(self, input_width: int) -> nn.Sequential:
        """Build a density network for the requested feature width."""
        layers: list[nn.Module] = []
        for _ in range(self.config.hidden_layers):
            layers.extend((nn.Linear(input_width, self.config.hidden_width), nn.Tanh()))
            input_width = self.config.hidden_width
        layers.append(nn.Linear(input_width, 1))
        return nn.Sequential(*layers)

    def field_parameters(self) -> Iterator[nn.Parameter]:
        """Return trainable neural-field parameters without physical coefficients."""
        return self.network.parameters()

    @property
    def diffusivity(self) -> Tensor:
        """Return diffusivity constrained to its configured interval."""
        return _bounded_transform(self.raw_diffusivity, self.config.diffusivity_bounds)

    @property
    def proliferation_rate(self) -> Tensor:
        """Return proliferation constrained to its configured interval."""
        return _bounded_transform(self.raw_proliferation_rate, self.config.proliferation_bounds)

    def diffusivity_at(self, coordinates: Tensor) -> Tensor:
        """Return diffusivity at each space-time coordinate."""
        return self.diffusivity.expand(coordinates.shape[0], 1)

    def diffusivity_gradient_at(self, coordinates: Tensor) -> Tensor:
        """Return one zero diffusivity-gradient column per spatial dimension."""
        return torch.zeros(
            (coordinates.shape[0], self.spatial_dimensions), device=coordinates.device
        )

    def treatment_rate_at(self, coordinates: Tensor) -> Tensor:
        """Return treatment-mediated loss rate; untreated models use zero."""
        return torch.zeros((coordinates.shape[0], 1), device=coordinates.device)

    def reset_parameters(self) -> None:
        """Initialize linear layers with tanh-compatible Xavier weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Predict density from spatial coordinates followed by time."""
        expected_width = self.coordinate_lower_bounds.numel()
        if coordinates.ndim != 2 or coordinates.shape[1] != expected_width:
            raise ValueError(f"coordinates must have shape (sample, {expected_width})")
        return self.config.carrying_capacity * torch.sigmoid(
            self.network(self.coordinate_features(coordinates))
        )

    def coordinate_features(self, coordinates: Tensor) -> Tensor:
        """Return normalized network features for physical coordinates."""
        normalized = (
            2.0
            * (
                (coordinates - self.coordinate_lower_bounds)
                / (self.coordinate_upper_bounds - self.coordinate_lower_bounds)
            )
            - 1.0
        )
        features = [normalized]
        spatial = normalized[:, : self.spatial_dimensions]
        for frequency in self.config.fourier_frequencies:
            angle = torch.pi * frequency * spatial
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.cat(features, dim=1)


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
    spatial_dimensions = getattr(model, "spatial_dimensions", coordinates.shape[1] - 1)
    spatial_gradient = gradient[:, :spatial_dimensions]
    density_t = gradient[:, -1:]
    laplacian = torch.zeros_like(density)
    for dimension in range(spatial_dimensions):
        first_derivative = spatial_gradient[:, dimension : dimension + 1]
        laplacian = (
            laplacian
            + torch.autograd.grad(
                first_derivative,
                coordinates,
                grad_outputs=torch.ones_like(first_derivative),
                create_graph=True,
            )[0][:, dimension : dimension + 1]
        )
    diffusivity = model.diffusivity_at(coordinates)
    diffusivity_gradient = model.diffusivity_gradient_at(coordinates)
    if diffusivity_gradient.shape != spatial_gradient.shape:
        raise ValueError("diffusivity gradient must match the spatial coordinate dimensions")
    diffusion_divergence = diffusivity * laplacian + torch.sum(
        diffusivity_gradient * spatial_gradient, dim=1, keepdim=True
    )
    normalized_density = density / model.config.carrying_capacity
    if model.config.growth_law == "logistic":
        proliferation = model.proliferation_rate * density * (1.0 - normalized_density)
    else:
        proliferation = (
            model.proliferation_rate
            * density
            * (normalized_density + model.config.allee_parameter)
            * (1.0 - normalized_density)
        )
    treatment_rate_function = getattr(model, "treatment_rate_at", None)
    treatment_rate = (
        treatment_rate_function(coordinates)
        if treatment_rate_function is not None
        else torch.zeros_like(density)
    )
    treatment_loss = treatment_rate * density
    return density_t - diffusion_divergence - proliferation + treatment_loss


def normal_flux(model: nn.Module, coordinates: Tensor, normals: Tensor) -> Tensor:
    """Return outward diffusive flux at boundary coordinates."""
    spatial_dimensions = getattr(model, "spatial_dimensions", coordinates.shape[1] - 1)
    if normals.ndim != 2 or normals.shape[1] != spatial_dimensions:
        raise ValueError(f"normals must have shape (sample, {spatial_dimensions})")
    if normals.shape[0] != coordinates.shape[0]:
        raise ValueError("normals and coordinates must contain the same number of samples")
    coordinates = coordinates.detach().clone().requires_grad_(True)
    density = model(coordinates)
    gradient = torch.autograd.grad(
        density,
        coordinates,
        grad_outputs=torch.ones_like(density),
        create_graph=True,
    )[0][:, :spatial_dimensions]
    return model.diffusivity_at(coordinates) * torch.sum(gradient * normals, dim=1, keepdim=True)


def fit_pinn(
    model: TumorPINN,
    data_coordinates: Tensor,
    data_density: Tensor,
    collocation_coordinates: Tensor,
    *,
    boundary_coordinates: Tensor | None = None,
    boundary_normals: Tensor | None = None,
    interface_coordinates: Tensor | None = None,
    interface_normals: Tensor | None = None,
    config: TrainingConfig | None = None,
    learn_diffusivity: bool = True,
    learn_proliferation_rate: bool = True,
    learn_treatment_response: bool = True,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
) -> TrainingResult:
    """Fit network and physical parameters with Adam."""
    config = config or TrainingConfig()
    if data_density.ndim != 2 or data_density.shape[1] != 1:
        raise ValueError("data_density must have shape (sample, 1)")
    if data_coordinates.shape[0] != data_density.shape[0]:
        raise ValueError("data coordinates and density must contain equal samples")
    if (boundary_coordinates is None) != (boundary_normals is None):
        raise ValueError("boundary coordinates and normals must be provided together")
    if (interface_coordinates is None) != (interface_normals is None):
        raise ValueError("interface coordinates and normals must be provided together")
    if interface_coordinates is not None and not hasattr(model, "interface_residual"):
        raise ValueError("model must define interface_residual for interface training")

    model.raw_diffusivity.requires_grad_(learn_diffusivity)
    model.raw_proliferation_rate.requires_grad_(learn_proliferation_rate)
    treatment_parameter = getattr(model, "raw_treatment_response", None)
    if treatment_parameter is not None:
        treatment_parameter.requires_grad_(learn_treatment_response)
    parameter_learning_rate = config.parameter_learning_rate or config.learning_rate
    parameter_groups: list[dict[str, object]] = [
        {"params": model.field_parameters(), "lr": config.learning_rate}
    ]
    physical_parameters = [
        parameter
        for parameter, enabled in (
            (model.raw_diffusivity, learn_diffusivity),
            (model.raw_proliferation_rate, learn_proliferation_rate),
        )
        if enabled
    ]
    if treatment_parameter is not None and learn_treatment_response:
        physical_parameters.append(treatment_parameter)
    if physical_parameters:
        parameter_groups.append({"params": physical_parameters, "lr": parameter_learning_rate})
    optimizer = torch.optim.Adam(parameter_groups)
    total_history: list[float] = []
    data_history: list[float] = []
    physics_history: list[float] = []
    boundary_history: list[float] = []
    interface_history: list[float] = []
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume_from_checkpoint and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when resume_from_checkpoint is true")
    start_epoch = 0
    if resume_from_checkpoint:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=data_coordinates.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["completed_epochs"])
        if start_epoch > config.epochs:
            raise ValueError("checkpoint has more completed epochs than requested training")
        histories = checkpoint["histories"]
        total_history.extend(histories["total_loss"])
        data_history.extend(histories["data_loss"])
        physics_history.extend(histories["physics_loss"])
        boundary_history.extend(histories["boundary_loss"])
        interface_history.extend(histories["interface_loss"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    def loss_terms(
        *, use_batches: bool, time_upper_bound: float | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        available_data_coordinates, available_data_density = _paired_time_prefix(
            data_coordinates,
            data_density,
            time_upper_bound,
        )
        selected_data_coordinates, selected_data_density = _paired_batch(
            available_data_coordinates,
            available_data_density,
            config.data_batch_size if use_batches else None,
        )
        available_collocation = _single_time_prefix(
            collocation_coordinates,
            time_upper_bound,
        )
        selected_collocation = _single_batch(
            available_collocation,
            config.collocation_batch_size if use_batches else None,
        )
        data_loss = torch.mean((model(selected_data_coordinates) - selected_data_density) ** 2)
        physics_loss = torch.mean(pde_residual(model, selected_collocation) ** 2)
        if boundary_coordinates is None:
            boundary_loss = torch.zeros((), device=data_coordinates.device)
        else:
            available_boundary_coordinates, available_boundary_normals = _paired_time_prefix(
                boundary_coordinates,
                boundary_normals,
                time_upper_bound,
            )
            selected_boundary_coordinates, selected_boundary_normals = _paired_batch(
                available_boundary_coordinates,
                available_boundary_normals,
                config.boundary_batch_size if use_batches else None,
            )
            boundary_loss = torch.mean(
                normal_flux(model, selected_boundary_coordinates, selected_boundary_normals) ** 2
            )
        if interface_coordinates is None:
            interface_loss = torch.zeros((), device=data_coordinates.device)
        else:
            available_interface_coordinates, available_interface_normals = _paired_time_prefix(
                interface_coordinates,
                interface_normals,
                time_upper_bound,
            )
            selected_interface_coordinates, selected_interface_normals = _paired_batch(
                available_interface_coordinates,
                available_interface_normals,
                config.interface_batch_size if use_batches else None,
            )
            density_jump, flux_jump = model.interface_residual(
                selected_interface_coordinates, selected_interface_normals
            )
            interface_loss = torch.mean(density_jump**2) + torch.mean(flux_jump**2)
        total_loss = (
            config.data_weight * data_loss
            + config.physics_weight * physics_loss
            + config.boundary_weight * boundary_loss
            + config.interface_weight * interface_loss
        )
        return total_loss, data_loss, physics_loss, boundary_loss, interface_loss

    def record_losses(
        total_loss: Tensor,
        data_loss: Tensor,
        physics_loss: Tensor,
        boundary_loss: Tensor,
        interface_loss: Tensor,
    ) -> None:
        total_history.append(float(total_loss.detach()))
        data_history.append(float(data_loss.detach()))
        physics_history.append(float(physics_loss.detach()))
        boundary_history.append(float(boundary_loss.detach()))
        interface_history.append(float(interface_loss.detach()))

    time_lower = float(model.coordinate_lower_bounds[-1].detach().cpu())
    time_upper = float(model.coordinate_upper_bounds[-1].detach().cpu())
    for epoch in range(start_epoch, config.epochs):
        current_time_upper: float | None = None
        if config.causal_time_chunks > 1:
            chunk = min(
                config.causal_time_chunks - 1,
                epoch * config.causal_time_chunks // config.epochs,
            )
            current_time_upper = time_lower + (time_upper - time_lower) * (
                (chunk + 1) / config.causal_time_chunks
            )
        optimizer.zero_grad(set_to_none=True)
        total_loss, data_loss, physics_loss, boundary_loss, interface_loss = loss_terms(
            use_batches=True,
            time_upper_bound=current_time_upper,
        )
        total_loss.backward()
        optimizer.step()
        record_losses(total_loss, data_loss, physics_loss, boundary_loss, interface_loss)
        completed_epochs = epoch + 1
        if checkpoint_path is not None and (
            completed_epochs == config.epochs
            or (
                config.checkpoint_interval is not None
                and completed_epochs % config.checkpoint_interval == 0
            )
        ):
            _save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                completed_epochs,
                total_history,
                data_history,
                physics_history,
                boundary_history,
                interface_history,
            )

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
            total_loss, data_loss, physics_loss, boundary_loss, interface_loss = loss_terms(
                use_batches=False,
            )
            total_loss.backward()
            record_losses(total_loss, data_loss, physics_loss, boundary_loss, interface_loss)
            return total_loss

        lbfgs.step(closure)

    return TrainingResult(
        total_loss=tuple(total_history),
        data_loss=tuple(data_history),
        physics_loss=tuple(physics_history),
        boundary_loss=tuple(boundary_history),
        interface_loss=tuple(interface_history),
    )


def _save_training_checkpoint(
    path: Path,
    model: TumorPINN,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    total_history: list[float],
    data_history: list[float],
    physics_history: list[float],
    boundary_history: list[float],
    interface_history: list[float],
) -> None:
    """Atomically persist enough state to resume Adam training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "completed_epochs": completed_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "histories": {
                "total_loss": total_history,
                "data_loss": data_history,
                "physics_loss": physics_history,
                "boundary_loss": boundary_history,
                "interface_loss": interface_history,
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _single_batch(values: Tensor, batch_size: int | None) -> Tensor:
    if batch_size is None or batch_size >= values.shape[0]:
        return values
    indices = torch.randperm(values.shape[0], device=values.device)[:batch_size]
    return values[indices]


def _paired_batch(left: Tensor, right: Tensor, batch_size: int | None) -> tuple[Tensor, Tensor]:
    if left.shape[0] != right.shape[0]:
        raise ValueError("paired batch tensors must contain equal samples")
    if batch_size is None or batch_size >= left.shape[0]:
        return left, right
    indices = torch.randperm(left.shape[0], device=left.device)[:batch_size]
    return left[indices], right[indices]


def _single_time_prefix(values: Tensor, time_upper_bound: float | None) -> Tensor:
    if time_upper_bound is None:
        return values
    selected = values[:, -1] <= time_upper_bound + 1e-7
    if not torch.any(selected):
        raise ValueError("causal time window contains no samples")
    return values[selected]


def _paired_time_prefix(
    coordinates: Tensor,
    values: Tensor,
    time_upper_bound: float | None,
) -> tuple[Tensor, Tensor]:
    if coordinates.shape[0] != values.shape[0]:
        raise ValueError("paired time-prefix tensors must contain equal samples")
    if time_upper_bound is None:
        return coordinates, values
    selected = coordinates[:, -1] <= time_upper_bound + 1e-7
    if not torch.any(selected):
        raise ValueError("causal time window contains no samples")
    return coordinates[selected], values[selected]


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

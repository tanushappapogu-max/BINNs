"""Three-output PINN for viable tumor, edema, and necrotic tissue."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from gbm_pinn.pinn import PINNConfig, TumorPINN
from gbm_pinn.treatment import TreatmentWindow


@dataclass(frozen=True, slots=True)
class MultiCompartmentPINNConfig:
    """Network and bounded kinetic parameters for the coupled PINN."""

    network: PINNConfig = field(
        default_factory=lambda: PINNConfig(
            diffusivity_bounds=(0.01, 2.0),
            proliferation_bounds=(0.001, 0.05),
            initial_diffusivity=0.13,
            initial_proliferation_rate=0.012,
        )
    )
    edema_diffusivity_bounds: tuple[float, float] = (1e-4, 2.0)
    edema_generation_bounds: tuple[float, float] = (1e-5, 0.2)
    edema_clearance_bounds: tuple[float, float] = (1e-5, 0.2)
    necrosis_clearance_bounds: tuple[float, float] = (1e-5, 0.2)
    treatment_cell_kill_bounds: tuple[float, float] = (0.0, 0.2)
    antiangiogenic_suppression_bounds: tuple[float, float] = (0.0, 1.0)
    initial_edema_diffusivity: float = 0.1
    initial_edema_generation: float = 0.01
    initial_edema_clearance: float = 0.01
    initial_necrosis_clearance: float = 0.01
    initial_treatment_cell_kill: float = 0.01
    initial_antiangiogenic_suppression: float = 0.1
    edema_half_saturation: float = 0.1
    spontaneous_necrosis_rate: float = 0.01
    necrosis_threshold: float = 0.8
    necrosis_transition_width: float = 0.05

    def __post_init__(self) -> None:
        values = (
            (
                self.initial_edema_diffusivity,
                self.edema_diffusivity_bounds,
                "initial_edema_diffusivity",
            ),
            (
                self.initial_edema_generation,
                self.edema_generation_bounds,
                "initial_edema_generation",
            ),
            (
                self.initial_edema_clearance,
                self.edema_clearance_bounds,
                "initial_edema_clearance",
            ),
            (
                self.initial_necrosis_clearance,
                self.necrosis_clearance_bounds,
                "initial_necrosis_clearance",
            ),
            (
                self.initial_treatment_cell_kill,
                self.treatment_cell_kill_bounds,
                "initial_treatment_cell_kill",
            ),
            (
                self.initial_antiangiogenic_suppression,
                self.antiangiogenic_suppression_bounds,
                "initial_antiangiogenic_suppression",
            ),
        )
        for value, bounds, name in values:
            lower, upper = bounds
            if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or lower >= upper:
                raise ValueError(f"{name} bounds must be finite, nonnegative, and ordered")
            if not lower < value < upper:
                raise ValueError(f"{name} must lie strictly inside its bounds")
        if not np.isfinite(self.edema_half_saturation) or self.edema_half_saturation <= 0:
            raise ValueError("edema_half_saturation must be finite and positive")
        if not np.isfinite(self.spontaneous_necrosis_rate) or self.spontaneous_necrosis_rate < 0:
            raise ValueError("spontaneous_necrosis_rate must be finite and nonnegative")
        if not 0 < self.necrosis_threshold <= self.network.carrying_capacity:
            raise ValueError("necrosis_threshold must lie in (0, carrying_capacity]")
        if not np.isfinite(self.necrosis_transition_width) or self.necrosis_transition_width <= 0:
            raise ValueError("necrosis_transition_width must be finite and positive")


@dataclass(frozen=True, slots=True)
class MultiCompartmentTrainingConfig:
    """Optimization settings for label-aware coupled PINN fitting."""

    epochs: int = 1_000
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 2e-3
    data_weight: float = 10.0
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    data_batch_size: int | None = 2_048
    collocation_batch_size: int | None = 2_048
    boundary_batch_size: int | None = 1_024
    detection_limits: tuple[float, float, float] = (0.1, 0.1, 0.1)
    checkpoint_interval: int | None = 100
    radiation_jump_weight: float = 1.0
    normalize_loss_terms: bool = False
    loss_scale_floor: float = 1e-8
    field_warmup_epochs: int = 0
    parameter_calibration_epochs: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.field_warmup_epochs < 0 or self.parameter_calibration_epochs < 0:
            raise ValueError("staged-training epoch counts must be nonnegative")
        if self.field_warmup_epochs + self.parameter_calibration_epochs >= self.epochs:
            raise ValueError("staged phases must leave at least one joint-training epoch")
        if self.network_learning_rate <= 0 or self.parameter_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if min(
            self.data_weight,
            self.physics_weight,
            self.boundary_weight,
            self.radiation_jump_weight,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")
        for value in (
            self.data_batch_size,
            self.collocation_batch_size,
            self.boundary_batch_size,
        ):
            if value is not None and value <= 0:
                raise ValueError("batch sizes must be positive when provided")
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive when provided")
        if not np.isfinite(self.loss_scale_floor) or self.loss_scale_floor <= 0:
            raise ValueError("loss_scale_floor must be finite and positive")
        if len(self.detection_limits) != 3 or any(
            not 0.0 <= value < 1.0 for value in self.detection_limits
        ):
            raise ValueError("three detection limits in [0, 1) are required")


@dataclass(frozen=True, slots=True)
class MultiCompartmentTrainingResult:
    """Per-epoch coupled PINN loss histories."""

    total_loss: tuple[float, ...]
    data_loss: tuple[float, ...]
    physics_loss: tuple[float, ...]
    boundary_loss: tuple[float, ...]
    radiation_jump_loss: tuple[float, ...]
    loss_scales: dict[str, float]


class MultiCompartmentPINN(TumorPINN):
    """Predict three latent biological fields from physical space-time coordinates."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        treatment_windows: tuple[TreatmentWindow, ...] = (),
        config: MultiCompartmentPINNConfig | None = None,
        *,
        edema_treatment_windows: tuple[TreatmentWindow, ...] = (),
    ) -> None:
        self.multicompartment_config = config or MultiCompartmentPINNConfig()
        super().__init__(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            self.multicompartment_config.network,
        )
        self.treatment_windows = treatment_windows
        self.edema_treatment_windows = edema_treatment_windows
        parameter_specs = (
            (
                "raw_edema_diffusivity",
                self.multicompartment_config.initial_edema_diffusivity,
                self.multicompartment_config.edema_diffusivity_bounds,
            ),
            (
                "raw_edema_generation",
                self.multicompartment_config.initial_edema_generation,
                self.multicompartment_config.edema_generation_bounds,
            ),
            (
                "raw_edema_clearance",
                self.multicompartment_config.initial_edema_clearance,
                self.multicompartment_config.edema_clearance_bounds,
            ),
            (
                "raw_necrosis_clearance",
                self.multicompartment_config.initial_necrosis_clearance,
                self.multicompartment_config.necrosis_clearance_bounds,
            ),
            (
                "raw_treatment_cell_kill",
                self.multicompartment_config.initial_treatment_cell_kill,
                self.multicompartment_config.treatment_cell_kill_bounds,
            ),
            (
                "raw_antiangiogenic_suppression",
                self.multicompartment_config.initial_antiangiogenic_suppression,
                self.multicompartment_config.antiangiogenic_suppression_bounds,
            ),
        )
        for name, initial, bounds in parameter_specs:
            self.register_parameter(
                name,
                nn.Parameter(torch.tensor(_inverse_bounded(initial, bounds), dtype=torch.float32)),
            )

    def _make_network(self, input_width: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(self.config.hidden_layers):
            layers.extend((nn.Linear(input_width, self.config.hidden_width), nn.Tanh()))
            input_width = self.config.hidden_width
        layers.append(nn.Linear(input_width, 3))
        return nn.Sequential(*layers)

    def forward(self, coordinates: Tensor) -> Tensor:
        expected_width = self.coordinate_lower_bounds.numel()
        if coordinates.ndim != 2 or coordinates.shape[1] != expected_width:
            raise ValueError(f"coordinates must have shape (sample, {expected_width})")
        unit_fields = torch.sigmoid(self.network(self.coordinate_features(coordinates)))
        scale = torch.tensor(
            [self.config.carrying_capacity, 1.0, self.config.carrying_capacity],
            device=coordinates.device,
        )
        return unit_fields * scale

    @property
    def edema_diffusivity(self) -> Tensor:
        return _bounded(
            self.raw_edema_diffusivity,
            self.multicompartment_config.edema_diffusivity_bounds,
        )

    @property
    def edema_generation_rate(self) -> Tensor:
        return _bounded(
            self.raw_edema_generation,
            self.multicompartment_config.edema_generation_bounds,
        )

    @property
    def edema_clearance_rate(self) -> Tensor:
        return _bounded(
            self.raw_edema_clearance,
            self.multicompartment_config.edema_clearance_bounds,
        )

    @property
    def necrosis_clearance_rate(self) -> Tensor:
        return _bounded(
            self.raw_necrosis_clearance,
            self.multicompartment_config.necrosis_clearance_bounds,
        )

    @property
    def treatment_cell_kill_rate(self) -> Tensor:
        return _bounded(
            self.raw_treatment_cell_kill,
            self.multicompartment_config.treatment_cell_kill_bounds,
        )

    @property
    def antiangiogenic_leakage_suppression(self) -> Tensor:
        return _bounded(
            self.raw_antiangiogenic_suppression,
            self.multicompartment_config.antiangiogenic_suppression_bounds,
        )

    def cell_kill_exposure_at(self, coordinates: Tensor) -> Tensor:
        times = coordinates[:, -1:]
        exposure = torch.zeros_like(times)
        for window in self.treatment_windows:
            exposure = torch.maximum(exposure, window.exposure_at(times))
        return exposure

    def edema_treatment_exposure_at(self, coordinates: Tensor) -> Tensor:
        times = coordinates[:, -1:]
        exposure = torch.zeros_like(times)
        for window in self.edema_treatment_windows:
            exposure = torch.maximum(exposure, window.exposure_at(times))
        return exposure


class PiecewiseTimeMultiCompartmentPINN(MultiCompartmentPINN):
    """Use one smooth neural field per interval between instantaneous events."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        event_times: tuple[float, ...],
        treatment_windows: tuple[TreatmentWindow, ...] = (),
        config: MultiCompartmentPINNConfig | None = None,
        *,
        edema_treatment_windows: tuple[TreatmentWindow, ...] = (),
    ) -> None:
        super().__init__(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            treatment_windows,
            config,
            edema_treatment_windows=edema_treatment_windows,
        )
        lower_time = float(self.coordinate_lower_bounds[-1])
        upper_time = float(self.coordinate_upper_bounds[-1])
        if any(not np.isfinite(time) for time in event_times):
            raise ValueError("event times must be finite")
        if any(right <= left for left, right in pairwise(event_times)):
            raise ValueError("event times must be strictly increasing")
        if any(not lower_time < time < upper_time for time in event_times):
            raise ValueError("event times must lie strictly inside the modeled time range")
        self.event_times = event_times
        input_width = next(
            layer.in_features for layer in self.network if isinstance(layer, nn.Linear)
        )
        self.post_event_networks = nn.ModuleList(
            self._make_network(input_width) for _ in event_times
        )
        for network in self.post_event_networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                    nn.init.zeros_(layer.bias)

    def field_parameters(self) -> Iterator[nn.Parameter]:
        """Return parameters from every smooth time-interval field."""
        yield from self.network.parameters()
        yield from self.post_event_networks.parameters()

    def forward(self, coordinates: Tensor) -> Tensor:
        expected_width = self.coordinate_lower_bounds.numel()
        if coordinates.ndim != 2 or coordinates.shape[1] != expected_width:
            raise ValueError(f"coordinates must have shape (sample, {expected_width})")
        fields = self._interval_fields(0, coordinates)
        times = coordinates[:, -1:]
        for interval_index, event_time in enumerate(self.event_times, start=1):
            post_fields = self._interval_fields(interval_index, coordinates)
            fields = torch.where(times >= event_time, post_fields, fields)
        return fields

    def radiation_jump_residual(
        self,
        event_index: int,
        spatial_coordinates: Tensor,
        survival: Tensor | float,
    ) -> Tensor:
        """Return viable, edema, and necrotic LQ jump-condition residuals."""
        if not 0 <= event_index < len(self.event_times):
            raise IndexError("event_index is outside the configured event times")
        if spatial_coordinates.ndim != 2 or spatial_coordinates.shape[1] != (
            self.spatial_dimensions
        ):
            raise ValueError("spatial coordinates must contain one column per dimension")
        event_column = torch.full(
            (spatial_coordinates.shape[0], 1),
            self.event_times[event_index],
            dtype=spatial_coordinates.dtype,
            device=spatial_coordinates.device,
        )
        coordinates = torch.cat((spatial_coordinates, event_column), dim=1)
        before = self._interval_fields(event_index, coordinates)
        after = self._interval_fields(event_index + 1, coordinates)
        survival_tensor = torch.as_tensor(
            survival,
            dtype=spatial_coordinates.dtype,
            device=spatial_coordinates.device,
        )
        try:
            survival_tensor = torch.broadcast_to(
                survival_tensor,
                (spatial_coordinates.shape[0], 1),
            )
        except RuntimeError as error:
            raise ValueError("survival must be scalar or have shape (sample, 1)") from error
        if torch.any(~torch.isfinite(survival_tensor)) or torch.any(
            (survival_tensor < 0) | (survival_tensor > 1)
        ):
            raise ValueError("survival must be finite and lie in [0, 1]")
        viable_before, edema_before, necrotic_before = before.split(1, dim=1)
        expected_after = torch.cat(
            (
                survival_tensor * viable_before,
                edema_before,
                necrotic_before + (1.0 - survival_tensor) * viable_before,
            ),
            dim=1,
        )
        return after - expected_after

    def _interval_fields(self, interval_index: int, coordinates: Tensor) -> Tensor:
        network = (
            self.network
            if interval_index == 0
            else self.post_event_networks[interval_index - 1]
        )
        unit_fields = torch.sigmoid(network(self.coordinate_features(coordinates)))
        scale = torch.tensor(
            [self.config.carrying_capacity, 1.0, self.config.carrying_capacity],
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        return unit_fields * scale


def multicompartment_pde_residual(model: MultiCompartmentPINN, coordinates: Tensor) -> Tensor:
    """Return viable, edema, and necrosis residual columns."""
    coordinates = coordinates.detach().clone().requires_grad_(True)
    fields = model(coordinates)
    gradients: list[Tensor] = []
    laplacians: list[Tensor] = []
    for channel in range(3):
        field_value = fields[:, channel : channel + 1]
        gradient = torch.autograd.grad(
            field_value,
            coordinates,
            grad_outputs=torch.ones_like(field_value),
            create_graph=True,
        )[0]
        gradients.append(gradient)
        laplacian = torch.zeros_like(field_value)
        for dimension in range(model.spatial_dimensions):
            first = gradient[:, dimension : dimension + 1]
            laplacian += torch.autograd.grad(
                first,
                coordinates,
                grad_outputs=torch.ones_like(first),
                create_graph=True,
            )[0][:, dimension : dimension + 1]
        laplacians.append(laplacian)

    viable, edema, necrotic = fields.split(1, dim=1)
    cell_kill = model.treatment_cell_kill_rate * model.cell_kill_exposure_at(coordinates)
    edema_suppression = (
        model.antiangiogenic_leakage_suppression
        * model.edema_treatment_exposure_at(coordinates)
    )
    occupied = (viable + necrotic) / model.config.carrying_capacity
    proliferation = model.proliferation_rate * viable * (1.0 - occupied)
    treatment_death = cell_kill * viable
    crowding_switch = 0.5 * (
        1.0
        + torch.tanh(
            (viable + necrotic - model.multicompartment_config.necrosis_threshold)
            / model.multicompartment_config.necrosis_transition_width
        )
    )
    spontaneous_death = (
        model.multicompartment_config.spontaneous_necrosis_rate * crowding_switch * viable
    )
    viable_reaction = proliferation - spontaneous_death - treatment_death
    edema_source = (
        model.edema_generation_rate
        * viable
        / (viable + model.multicompartment_config.edema_half_saturation)
        * (1.0 - edema)
    )
    edema_reaction = (
        (1.0 - edema_suppression) * edema_source - model.edema_clearance_rate * edema
    )
    necrotic_reaction = (
        spontaneous_death + treatment_death - model.necrosis_clearance_rate * necrotic
    )
    return torch.cat(
        (
            gradients[0][:, -1:] - model.diffusivity * laplacians[0] - viable_reaction,
            gradients[1][:, -1:] - model.edema_diffusivity * laplacians[1] - edema_reaction,
            gradients[2][:, -1:] - necrotic_reaction,
        ),
        dim=1,
    )


def multicompartment_observation_channels(latent_fields: Tensor) -> Tensor:
    """Map latent PINN outputs to enhancing, FLAIR, and necrotic channels."""
    if latent_fields.ndim != 2 or latent_fields.shape[1] != 3:
        raise ValueError("latent_fields must have shape (sample, 3)")
    viable, edema, necrotic = latent_fields.split(1, dim=1)
    flair = viable + edema - viable * edema
    return torch.cat((viable, flair, necrotic), dim=1)


def multicompartment_normal_flux(
    model: MultiCompartmentPINN,
    coordinates: Tensor,
    normals: Tensor,
) -> Tensor:
    """Return viable and edema diffusive fluxes at no-flux boundaries."""
    if normals.ndim != 2 or normals.shape != (
        coordinates.shape[0],
        model.spatial_dimensions,
    ):
        raise ValueError("normals must match samples and spatial dimensions")
    coordinates = coordinates.detach().clone().requires_grad_(True)
    fields = model(coordinates)
    fluxes: list[Tensor] = []
    for channel, diffusivity in ((0, model.diffusivity), (1, model.edema_diffusivity)):
        field_value = fields[:, channel : channel + 1]
        gradient = torch.autograd.grad(
            field_value,
            coordinates,
            grad_outputs=torch.ones_like(field_value),
            create_graph=True,
        )[0][:, : model.spatial_dimensions]
        fluxes.append(diffusivity * torch.sum(gradient * normals, dim=1, keepdim=True))
    return torch.cat(fluxes, dim=1)


def fit_multicompartment_pinn(
    model: MultiCompartmentPINN,
    data_coordinates: Tensor,
    observation_targets: Tensor,
    collocation_coordinates: Tensor,
    *,
    boundary_coordinates: Tensor | None = None,
    boundary_normals: Tensor | None = None,
    radiation_jump_spatial_coordinates: tuple[Tensor, ...] = (),
    radiation_survival: tuple[Tensor | float, ...] = (),
    config: MultiCompartmentTrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = False,
) -> MultiCompartmentTrainingResult:
    """Fit latent fields and coupled physical coefficients with Adam."""
    config = config or MultiCompartmentTrainingConfig()
    if observation_targets.ndim != 2 or observation_targets.shape[1] != 3:
        raise ValueError("observation_targets must have shape (sample, 3)")
    if data_coordinates.shape[0] != observation_targets.shape[0]:
        raise ValueError("data coordinates and targets must contain equal samples")
    if (boundary_coordinates is None) != (boundary_normals is None):
        raise ValueError("boundary coordinates and normals must be provided together")
    if len(radiation_jump_spatial_coordinates) != len(radiation_survival):
        raise ValueError("radiation jump coordinates and survival values must align")
    if radiation_jump_spatial_coordinates and not isinstance(
        model,
        PiecewiseTimeMultiCompartmentPINN,
    ):
        raise TypeError("radiation jump training requires a piecewise-time PINN")
    if isinstance(model, PiecewiseTimeMultiCompartmentPINN) and len(
        radiation_jump_spatial_coordinates
    ) not in (0, len(model.event_times)):
        raise ValueError("provide one radiation jump sample set per event")
    physical_parameters = [
        model.raw_diffusivity,
        model.raw_proliferation_rate,
        model.raw_edema_diffusivity,
        model.raw_edema_generation,
        model.raw_edema_clearance,
        model.raw_necrosis_clearance,
        model.raw_treatment_cell_kill,
        model.raw_antiangiogenic_suppression,
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": model.field_parameters(), "lr": config.network_learning_rate},
            {"params": physical_parameters, "lr": config.parameter_learning_rate},
        ]
    )
    histories: dict[str, list[float]] = {
        "total": [],
        "data": [],
        "physics": [],
        "boundary": [],
        "radiation_jump": [],
    }
    loss_scales: dict[str, float] = {}
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
        saved_histories = checkpoint["histories"]
        for name in histories:
            histories[name].extend(saved_histories.get(name, []))
        loss_scales.update(checkpoint.get("loss_scales", {}))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    for epoch in range(start_epoch, config.epochs):
        in_field_warmup = epoch < config.field_warmup_epochs
        in_parameter_calibration = (
            config.field_warmup_epochs
            <= epoch
            < config.field_warmup_epochs + config.parameter_calibration_epochs
        )
        for parameter in model.field_parameters():
            parameter.requires_grad_(not in_parameter_calibration)
        for parameter in physical_parameters:
            parameter.requires_grad_(not in_field_warmup)
        data_indices = _batch_indices(
            data_coordinates.shape[0], config.data_batch_size, data_coordinates.device
        )
        collocation_indices = _batch_indices(
            collocation_coordinates.shape[0],
            config.collocation_batch_size,
            collocation_coordinates.device,
        )
        predicted = multicompartment_observation_channels(model(data_coordinates[data_indices]))
        data_loss = censored_observation_loss(
            predicted,
            observation_targets[data_indices],
            config.detection_limits,
        )
        residual = multicompartment_pde_residual(
            model, collocation_coordinates[collocation_indices]
        )
        physics_loss = torch.mean(residual**2)
        if boundary_coordinates is None or boundary_normals is None:
            boundary_loss = torch.zeros((), device=data_coordinates.device)
        else:
            boundary_indices = _batch_indices(
                boundary_coordinates.shape[0],
                config.boundary_batch_size,
                boundary_coordinates.device,
            )
            flux = multicompartment_normal_flux(
                model,
                boundary_coordinates[boundary_indices],
                boundary_normals[boundary_indices],
            )
            boundary_loss = torch.mean(flux**2)
        radiation_jump_loss = torch.zeros((), device=data_coordinates.device)
        if radiation_jump_spatial_coordinates:
            jump_losses = [
                torch.mean(model.radiation_jump_residual(index, coordinates, survival) ** 2)
                for index, (coordinates, survival) in enumerate(
                    zip(
                        radiation_jump_spatial_coordinates,
                        radiation_survival,
                        strict=True,
                    )
                )
            ]
            radiation_jump_loss = torch.stack(jump_losses).mean()
        raw_loss_terms = {
            "data": data_loss,
            "physics": physics_loss,
            "boundary": boundary_loss,
            "radiation_jump": radiation_jump_loss,
        }
        if config.normalize_loss_terms and not loss_scales:
            loss_scales.update(
                {
                    name: max(float(value.detach().cpu()), config.loss_scale_floor)
                    for name, value in raw_loss_terms.items()
                }
            )

        scaled_loss_terms = (
            {
                name: value / loss_scales[name]
                for name, value in raw_loss_terms.items()
            }
            if config.normalize_loss_terms
            else raw_loss_terms
        )
        total = (
            config.data_weight * scaled_loss_terms["data"]
            + config.physics_weight * scaled_loss_terms["physics"]
            + config.boundary_weight * scaled_loss_terms["boundary"]
            + config.radiation_jump_weight * scaled_loss_terms["radiation_jump"]
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        for name, value in (
            ("total", total),
            ("data", data_loss),
            ("physics", physics_loss),
            ("boundary", boundary_loss),
            ("radiation_jump", radiation_jump_loss),
        ):
            histories[name].append(float(value.detach().cpu()))
        completed_epochs = epoch + 1
        if checkpoint_path is not None and (
            completed_epochs == config.epochs
            or (
                config.checkpoint_interval is not None
                and completed_epochs % config.checkpoint_interval == 0
            )
        ):
            _save_multicompartment_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                completed_epochs,
                histories,
                loss_scales,
            )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return MultiCompartmentTrainingResult(
        total_loss=tuple(histories["total"]),
        data_loss=tuple(histories["data"]),
        physics_loss=tuple(histories["physics"]),
        boundary_loss=tuple(histories["boundary"]),
        radiation_jump_loss=tuple(histories["radiation_jump"]),
        loss_scales=loss_scales
        or {
            name: 1.0
            for name in ("data", "physics", "boundary", "radiation_jump")
        },
    )


def segmentation_to_observation_channels(
    labels: NDArray[np.integer], *, infiltrative_viable_density: float = 0.3
) -> NDArray[np.float32]:
    """Convert exclusive labels to viable-proxy, FLAIR, and core targets.

    The FLAIR target is deliberately non-exclusive: enhancing tissue lies inside
    the broader MRI abnormality represented by the viable-plus-edema observation
    channel. Label 2 is a mixture of edema and infiltrative tumor, so it receives
    a lower viable-density proxy instead of being forced to contain no viable
    tumor. This density is a modeling assumption, not a direct MRI measurement.
    """
    labels = np.asarray(labels)
    if not 0.0 <= infiltrative_viable_density <= 1.0:
        raise ValueError("infiltrative_viable_density must lie in [0, 1]")
    if np.any(~np.isin(labels, (0, 1, 2, 3, 4))):
        raise ValueError("segmentation labels must lie in {0, 1, 2, 3, 4}")
    viable_proxy = np.zeros(labels.shape, dtype=np.float32)
    viable_proxy[labels == 2] = infiltrative_viable_density
    viable_proxy[labels == 3] = 1.0
    return np.stack(
        (viable_proxy, np.isin(labels, (2, 3)), labels == 1), axis=-1
    ).astype(np.float32)


def censored_observation_loss(
    prediction: Tensor,
    target: Tensor,
    detection_limits: tuple[float, float, float],
) -> Tensor:
    """Use exact positive targets and one-sided below-detection negative targets."""
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction and target must have matching shape (sample, 3)")
    limits = torch.as_tensor(detection_limits, dtype=prediction.dtype, device=prediction.device)
    positive = target > 0
    exact_error = (prediction - target) ** 2
    censored_error = torch.relu(prediction - limits) ** 2
    return torch.mean(torch.where(positive, exact_error, censored_error))


def _bounded(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
    lower, upper = bounds
    return lower + (upper - lower) * torch.sigmoid(raw)


def _inverse_bounded(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    fraction = (value - lower) / (upper - lower)
    return float(np.log(fraction / (1.0 - fraction)))


def _batch_indices(count: int, batch_size: int | None, device: torch.device) -> Tensor:
    if count <= 0:
        raise ValueError("training tensors must contain at least one sample")
    if batch_size is None or batch_size >= count:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device)[:batch_size]


def _save_multicompartment_checkpoint(
    path: Path,
    model: MultiCompartmentPINN,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    histories: dict[str, list[float]],
    loss_scales: dict[str, float],
) -> None:
    """Atomically save the coupled model so long Mac runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "completed_epochs": completed_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "histories": histories,
            "loss_scales": loss_scales,
        },
        temporary,
    )
    temporary.replace(path)

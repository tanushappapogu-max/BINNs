"""Geometry-aware PINN for a circular resection cavity."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from gbm_pinn.pinn import PINNConfig, TumorPINN


class CavityAwareTumorPINN(TumorPINN):
    """Encode cavity geometry with a hard zero-normal-gradient feature map."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        cavity_center: tuple[float, float],
        cavity_radius: float,
        config: PINNConfig | None = None,
    ) -> None:
        if cavity_radius <= 0:
            raise ValueError("cavity_radius must be positive")
        if len(cavity_center) != 2:
            raise ValueError("cavity_center must contain x and y")
        super().__init__(
            coordinate_lower_bounds,
            coordinate_upper_bounds,
            config,
            additional_input_width=1,
        )
        center = torch.as_tensor(cavity_center, dtype=torch.float32)
        if not torch.all(torch.isfinite(center)):
            raise ValueError("cavity_center must be finite")
        self.register_buffer("cavity_center", center)
        self.cavity_radius = float(cavity_radius)

    def coordinate_features(self, coordinates: Tensor) -> Tensor:
        """Return angular, squared-radial, and normalized-time features."""
        normalized_time = super().coordinate_features(coordinates)[:, 2:3]
        displacement = coordinates[:, :2] - self.cavity_center
        radial_distance = torch.linalg.vector_norm(
            displacement,
            dim=1,
            keepdim=True,
        )
        direction = displacement / radial_distance.clamp_min(torch.finfo(coordinates.dtype).eps)
        spatial_extent = self.coordinate_upper_bounds[:2] - self.coordinate_lower_bounds[:2]
        scale = torch.linalg.vector_norm(spatial_extent)
        squared_radial_distance = ((radial_distance - self.cavity_radius) / scale) ** 2
        radial_feature = 2.0 * squared_radial_distance - 1.0
        return torch.cat((direction, radial_feature, normalized_time), dim=1)


class PiecewiseCavityTumorPINN(TumorPINN):
    """Use separate near-cavity and far-field networks with a circular interface."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        cavity_center: tuple[float, float],
        cavity_radius: float,
        interface_radius: float,
        config: PINNConfig | None = None,
    ) -> None:
        if cavity_radius <= 0:
            raise ValueError("cavity_radius must be positive")
        if interface_radius <= cavity_radius:
            raise ValueError("interface_radius must exceed cavity_radius")
        if len(cavity_center) != 2:
            raise ValueError("cavity_center must contain x and y")
        super().__init__(coordinate_lower_bounds, coordinate_upper_bounds, config)
        center = torch.as_tensor(cavity_center, dtype=torch.float32)
        if not torch.all(torch.isfinite(center)):
            raise ValueError("cavity_center must be finite")
        available_radius = torch.minimum(
            center - self.coordinate_lower_bounds[:2],
            self.coordinate_upper_bounds[:2] - center,
        ).min()
        if interface_radius >= float(available_radius):
            raise ValueError("interface circle must lie strictly inside the spatial domain")

        self.register_buffer("cavity_center", center)
        self.cavity_radius = float(cavity_radius)
        self.interface_radius = float(interface_radius)
        self.near_network = self._make_network(4)
        for layer in self.near_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)

    def field_parameters(self) -> Iterator[nn.Parameter]:
        """Return parameters from both neural fields."""
        yield from self.network.parameters()
        yield from self.near_network.parameters()

    def forward(self, coordinates: Tensor) -> Tensor:
        """Route coordinates to the near-cavity or far-field neural representation."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (sample, 3)")
        radial_distance = self._radial_distance(coordinates)
        near_density = self._near_density(coordinates)
        far_density = self._far_density(coordinates)
        return torch.where(radial_distance <= self.interface_radius, near_density, far_density)

    def interface_residual(self, coordinates: Tensor, normals: Tensor) -> tuple[Tensor, Tensor]:
        """Return density and normal-flux jumps across the artificial interface."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (sample, 3)")
        if normals.shape != (coordinates.shape[0], 2):
            raise ValueError("normals must have shape (sample, 2)")
        coordinates = coordinates.detach().clone().requires_grad_(True)
        near_density = self._near_density(coordinates)
        far_density = self._far_density(coordinates)
        near_gradient = torch.autograd.grad(
            near_density,
            coordinates,
            grad_outputs=torch.ones_like(near_density),
            create_graph=True,
        )[0][:, :2]
        far_gradient = torch.autograd.grad(
            far_density,
            coordinates,
            grad_outputs=torch.ones_like(far_density),
            create_graph=True,
        )[0][:, :2]
        diffusivity = self.diffusivity_at(coordinates)
        near_flux = diffusivity * torch.sum(near_gradient * normals, dim=1, keepdim=True)
        far_flux = diffusivity * torch.sum(far_gradient * normals, dim=1, keepdim=True)
        return near_density - far_density, near_flux - far_flux

    def _near_density(self, coordinates: Tensor) -> Tensor:
        return self.config.carrying_capacity * torch.sigmoid(
            self.near_network(self._near_features(coordinates))
        )

    def _far_density(self, coordinates: Tensor) -> Tensor:
        return self.config.carrying_capacity * torch.sigmoid(
            self.network(super().coordinate_features(coordinates))
        )

    def _near_features(self, coordinates: Tensor) -> Tensor:
        normalized_time = super().coordinate_features(coordinates)[:, 2:3]
        displacement = coordinates[:, :2] - self.cavity_center
        radial_distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
        direction = displacement / radial_distance.clamp_min(torch.finfo(coordinates.dtype).eps)
        annulus_width = self.interface_radius - self.cavity_radius
        squared_distance = ((radial_distance - self.cavity_radius) / annulus_width) ** 2
        radial_feature = 2.0 * squared_distance - 1.0
        return torch.cat((direction, radial_feature, normalized_time), dim=1)

    def _radial_distance(self, coordinates: Tensor) -> Tensor:
        return torch.linalg.vector_norm(
            coordinates[:, :2] - self.cavity_center, dim=1, keepdim=True
        )

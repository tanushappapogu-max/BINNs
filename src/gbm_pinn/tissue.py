"""Tissue-dependent diffusivity for the tumor PINN."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional

from gbm_pinn.pinn import PINNConfig, TumorPINN

FloatArray = NDArray[np.float64]


class TissueAwareTumorPINN(TumorPINN):
    """Use a tissue probability map to vary scalar diffusivity in space."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        white_matter_probability: FloatArray,
        *,
        white_to_gray_diffusivity_ratio: float,
        config: PINNConfig | None = None,
    ) -> None:
        super().__init__(coordinate_lower_bounds, coordinate_upper_bounds, config)
        probability = np.asarray(white_matter_probability, dtype=np.float32)
        if probability.ndim != 2 or min(probability.shape) < 2:
            raise ValueError("white_matter_probability must be a two-dimensional grid")
        if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
            raise ValueError("white_matter_probability must contain values in [0, 1]")
        if white_to_gray_diffusivity_ratio < 1:
            raise ValueError("white_to_gray_diffusivity_ratio must be at least one")

        lower = self.coordinate_lower_bounds.detach().cpu().numpy()
        upper = self.coordinate_upper_bounds.detach().cpu().numpy()
        dy = float(upper[1] - lower[1]) / (probability.shape[0] - 1)
        dx = float(upper[0] - lower[0]) / (probability.shape[1] - 1)
        probability_y, probability_x = np.gradient(probability, dy, dx)

        self.white_to_gray_diffusivity_ratio = float(white_to_gray_diffusivity_ratio)
        self.register_buffer("white_matter_probability", torch.from_numpy(probability)[None, None])
        self.register_buffer(
            "white_matter_probability_x", torch.from_numpy(probability_x)[None, None]
        )
        self.register_buffer(
            "white_matter_probability_y", torch.from_numpy(probability_y)[None, None]
        )

    @property
    def gray_matter_diffusivity(self) -> Tensor:
        """Return the learned gray-matter diffusivity."""
        return self.diffusivity

    @property
    def white_matter_diffusivity(self) -> Tensor:
        """Return white-matter diffusivity from the configured ratio."""
        return self.gray_matter_diffusivity * self.white_to_gray_diffusivity_ratio

    def diffusivity_at(self, coordinates: Tensor) -> Tensor:
        """Interpolate tissue probability and return local diffusivity."""
        probability = self._sample(self.white_matter_probability, coordinates)
        scale = 1.0 + (self.white_to_gray_diffusivity_ratio - 1.0) * probability
        return self.gray_matter_diffusivity * scale

    def diffusivity_gradient_at(self, coordinates: Tensor) -> Tensor:
        """Interpolate the precomputed spatial diffusivity gradient."""
        probability_x = self._sample(self.white_matter_probability_x, coordinates)
        probability_y = self._sample(self.white_matter_probability_y, coordinates)
        scale = self.gray_matter_diffusivity * (self.white_to_gray_diffusivity_ratio - 1.0)
        return scale * torch.cat((probability_x, probability_y), dim=1)

    def _sample(self, field: Tensor, coordinates: Tensor) -> Tensor:
        lower = self.coordinate_lower_bounds
        upper = self.coordinate_upper_bounds
        normalized_x = 2.0 * (coordinates[:, 0] - lower[0]) / (upper[0] - lower[0]) - 1.0
        normalized_y = 2.0 * (coordinates[:, 1] - lower[1]) / (upper[1] - lower[1]) - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=1).reshape(1, -1, 1, 2)
        sampled = functional.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(-1, 1).detach()

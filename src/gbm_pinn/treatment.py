"""Schedule-driven treatment exposure for treatment-aware tumor PINNs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from gbm_pinn.pinn import PINNConfig, TumorPINN


@dataclass(frozen=True, slots=True)
class TreatmentWindow:
    """Known treatment interval and optional exponentially decaying post-effect."""

    start_day: float
    end_day: float
    intensity: float = 1.0
    decay_days: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_day) or not np.isfinite(self.end_day):
            raise ValueError("treatment days must be finite")
        if self.end_day < self.start_day:
            raise ValueError("treatment end_day must not precede start_day")
        if not np.isfinite(self.intensity) or self.intensity < 0:
            raise ValueError("treatment intensity must be finite and nonnegative")
        if not np.isfinite(self.decay_days) or self.decay_days < 0:
            raise ValueError("treatment decay_days must be finite and nonnegative")

    def exposure_at(self, times: Tensor) -> Tensor:
        """Return known exposure during treatment and its optional post-effect."""
        active = (times >= self.start_day) & (times <= self.end_day)
        exposure = torch.where(active, self.intensity, 0.0)
        if self.decay_days > 0:
            after = times > self.end_day
            post_effect = self.intensity * torch.exp(-(times - self.end_day) / self.decay_days)
            exposure = torch.where(after, post_effect, exposure)
        return exposure


class TreatmentAwareTumorPINN(TumorPINN):
    """Tumor PINN with a known schedule and a bounded treatment response rate."""

    def __init__(
        self,
        coordinate_lower_bounds: Tensor,
        coordinate_upper_bounds: Tensor,
        treatment_windows: tuple[TreatmentWindow, ...],
        config: PINNConfig | None = None,
        *,
        treatment_response_bounds: tuple[float, float] = (0.0, 0.2),
        initial_treatment_response: float = 0.02,
    ) -> None:
        if not treatment_windows:
            raise ValueError("at least one treatment window is required")
        lower, upper = treatment_response_bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or lower >= upper:
            raise ValueError("treatment response bounds must be finite and strictly ordered")
        if not lower < initial_treatment_response < upper:
            raise ValueError("initial treatment response must lie strictly inside its bounds")
        super().__init__(coordinate_lower_bounds, coordinate_upper_bounds, config)
        self.treatment_windows = treatment_windows
        self.treatment_response_bounds = treatment_response_bounds
        fraction = (initial_treatment_response - lower) / (upper - lower)
        self.raw_treatment_response = nn.Parameter(
            torch.tensor(float(np.log(fraction / (1.0 - fraction))), dtype=torch.float32)
        )

    @property
    def treatment_response_rate(self) -> Tensor:
        """Return the bounded response coefficient in inverse-day units."""
        lower, upper = self.treatment_response_bounds
        return lower + (upper - lower) * torch.sigmoid(self.raw_treatment_response)

    def treatment_rate_at(self, coordinates: Tensor) -> Tensor:
        """Return response coefficient times summed known treatment exposure."""
        times = coordinates[:, -1:]
        exposure = torch.zeros_like(times)
        for window in self.treatment_windows:
            exposure = exposure + window.exposure_at(times)
        return self.treatment_response_rate * exposure

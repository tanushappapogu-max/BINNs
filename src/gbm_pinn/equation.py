"""Parameter definitions for the tumor reaction-diffusion equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike

GrowthLaw = Literal["logistic", "weak_allee"]


@dataclass(frozen=True, slots=True)
class ReactionDiffusionParameters:
    """Scalar kinetic parameters expressed in consistent spatial and temporal units."""

    proliferation_rate: float
    carrying_capacity: float = 1.0
    growth_law: GrowthLaw = "logistic"
    allee_parameter: float = 0.0

    def __post_init__(self) -> None:
        if self.proliferation_rate < 0:
            raise ValueError("proliferation_rate must be nonnegative")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")
        if self.growth_law not in ("logistic", "weak_allee"):
            raise ValueError("growth_law must be 'logistic' or 'weak_allee'")
        if self.allee_parameter < 0:
            raise ValueError("allee_parameter must be nonnegative")
        if self.growth_law == "logistic" and self.allee_parameter != 0:
            raise ValueError("allee_parameter must be zero for logistic growth")

    @property
    def maximum_reaction_slope(self) -> float:
        """Return a conservative reaction-rate scale for explicit stepping."""
        if self.growth_law == "logistic":
            return self.proliferation_rate
        return self.proliferation_rate * (1.0 + self.allee_parameter)

    def reaction(self, density: ArrayLike, treatment_rate: ArrayLike) -> ArrayLike:
        """Return density-dependent proliferation minus treatment-mediated loss."""
        normalized_density = density / self.carrying_capacity
        if self.growth_law == "logistic":
            growth = self.proliferation_rate * density * (1.0 - normalized_density)
        else:
            growth = (
                self.proliferation_rate
                * density
                * (normalized_density + self.allee_parameter)
                * (1.0 - normalized_density)
            )
        return growth - treatment_rate * density

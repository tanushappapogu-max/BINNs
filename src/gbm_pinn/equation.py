"""Parameter definitions for the tumor reaction-diffusion equation."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ReactionDiffusionParameters:
    """Scalar kinetic parameters expressed in consistent spatial and temporal units."""

    proliferation_rate: float
    carrying_capacity: float = 1.0

    def __post_init__(self) -> None:
        if self.proliferation_rate < 0:
            raise ValueError("proliferation_rate must be nonnegative")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")

    def reaction(self, density: ArrayLike, treatment_rate: ArrayLike) -> ArrayLike:
        """Return logistic proliferation minus treatment-mediated loss."""
        return (
            self.proliferation_rate * density * (1.0 - density / self.carrying_capacity)
            - treatment_rate * density
        )

"""Finite-volume reference solver for a masked reaction-diffusion domain."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gbm_pinn.equation import ReactionDiffusionParameters

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
TreatmentFunction = Callable[[float], FloatArray | float]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Density snapshots and their corresponding simulation times."""

    times: FloatArray
    density: FloatArray


class FiniteVolumeSolver:
    """Solve a two-dimensional variable-coefficient Fisher-KPP equation."""

    def __init__(
        self,
        diffusivity: FloatArray,
        brain_mask: BoolArray,
        parameters: ReactionDiffusionParameters,
        *,
        spacing: tuple[float, float] = (1.0, 1.0),
        cavity_mask: BoolArray | None = None,
    ) -> None:
        diffusivity = np.asarray(diffusivity, dtype=np.float64)
        brain_mask = np.asarray(brain_mask, dtype=bool)
        if diffusivity.ndim != 2:
            raise ValueError("diffusivity must be a two-dimensional array")
        if brain_mask.shape != diffusivity.shape:
            raise ValueError("brain_mask must match diffusivity shape")
        if np.any(~np.isfinite(diffusivity)) or np.any(diffusivity < 0):
            raise ValueError("diffusivity must contain finite nonnegative values")
        if len(spacing) != 2 or any(value <= 0 for value in spacing):
            raise ValueError("spacing values must be positive")

        if cavity_mask is None:
            cavity_mask = np.zeros_like(brain_mask)
        else:
            cavity_mask = np.asarray(cavity_mask, dtype=bool)
            if cavity_mask.shape != diffusivity.shape:
                raise ValueError("cavity_mask must match diffusivity shape")
            if np.any(cavity_mask & ~brain_mask):
                raise ValueError("cavity_mask must be contained within brain_mask")

        self.parameters = parameters
        self.spacing = (float(spacing[0]), float(spacing[1]))
        self.cavity_mask = cavity_mask.copy()
        self.active_mask = brain_mask & ~cavity_mask
        self.diffusivity = np.where(self.active_mask, diffusivity, 0.0)

    def stable_time_step(self, maximum_treatment_rate: float = 0.0) -> float:
        """Return a conservative explicit-Euler stability limit."""
        if maximum_treatment_rate < 0:
            raise ValueError("maximum_treatment_rate must be nonnegative")
        maximum_diffusivity = float(np.max(self.diffusivity, initial=0.0))
        dy, dx = self.spacing
        diffusion_rate = 2.0 * maximum_diffusivity * (1.0 / dx**2 + 1.0 / dy**2)
        reaction_rate = self.parameters.maximum_reaction_slope + maximum_treatment_rate
        total_rate = diffusion_rate + reaction_rate
        return np.inf if total_rate == 0 else 0.9 / total_rate

    def spatial_derivative(self, density: FloatArray) -> FloatArray:
        """Return the conservative divergence of diffusive face fluxes."""
        density = self._validated_density(density)
        dy, dx = self.spacing

        x_flux = np.zeros((density.shape[0], density.shape[1] + 1), dtype=np.float64)
        y_flux = np.zeros((density.shape[0] + 1, density.shape[1]), dtype=np.float64)

        x_face_diffusivity = self._harmonic_mean(self.diffusivity[:, :-1], self.diffusivity[:, 1:])
        y_face_diffusivity = self._harmonic_mean(self.diffusivity[:-1, :], self.diffusivity[1:, :])
        x_flux[:, 1:-1] = x_face_diffusivity * np.diff(density, axis=1) / dx
        y_flux[1:-1, :] = y_face_diffusivity * np.diff(density, axis=0) / dy

        divergence = np.diff(x_flux, axis=1) / dx + np.diff(y_flux, axis=0) / dy
        return np.where(self.active_mask, divergence, 0.0)

    def simulate(
        self,
        initial_density: FloatArray,
        output_times: FloatArray,
        *,
        maximum_time_step: float | None = None,
        treatment: TreatmentFunction | None = None,
    ) -> SimulationResult:
        """Advance density and return states at each requested time."""
        times = np.asarray(output_times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("output_times must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("output_times must be finite, nonnegative, and strictly increasing")

        density = self._validated_density(initial_density).copy()
        density[~self.active_mask] = 0.0
        snapshots = np.empty((times.size, *density.shape), dtype=np.float64)

        current_time = 0.0
        for index, target_time in enumerate(times):
            while current_time < target_time:
                treatment_rate = self._treatment_at(treatment, current_time)
                stable_step = self.stable_time_step(float(np.max(treatment_rate, initial=0.0)))
                step = target_time - current_time
                if maximum_time_step is not None:
                    if maximum_time_step <= 0:
                        raise ValueError("maximum_time_step must be positive")
                    step = min(step, maximum_time_step)
                step = min(step, stable_step)
                if not np.isfinite(step) or step <= 0:
                    current_time = target_time
                    break

                derivative = self.spatial_derivative(density)
                derivative += self.parameters.reaction(density, treatment_rate)
                density = density + step * derivative
                density[~self.active_mask] = 0.0
                density = np.clip(density, 0.0, self.parameters.carrying_capacity)
                current_time += step

            snapshots[index] = density

        return SimulationResult(times=times.copy(), density=snapshots)

    def _treatment_at(self, treatment: TreatmentFunction | None, time: float) -> FloatArray:
        if treatment is None:
            return np.zeros_like(self.diffusivity)
        rate = np.asarray(treatment(time), dtype=np.float64)
        try:
            rate = np.broadcast_to(rate, self.diffusivity.shape)
        except ValueError as error:
            raise ValueError("treatment output must be scalar or match the domain shape") from error
        if np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("treatment rates must be finite and nonnegative")
        return np.where(self.active_mask, rate, 0.0)

    def _validated_density(self, density: FloatArray) -> FloatArray:
        density = np.asarray(density, dtype=np.float64)
        if density.shape != self.diffusivity.shape:
            raise ValueError("density must match diffusivity shape")
        if np.any(~np.isfinite(density)) or np.any(density < 0):
            raise ValueError("density must contain finite nonnegative values")
        return density

    @staticmethod
    def _harmonic_mean(left: FloatArray, right: FloatArray) -> FloatArray:
        denominator = left + right
        return np.divide(
            2.0 * left * right,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

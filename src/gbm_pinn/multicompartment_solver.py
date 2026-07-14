"""Finite-volume reference solver for the MRI-aware compartment system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.multicompartment import MultiCompartmentParameters
from gbm_pinn.solver import FiniteVolumeSolver

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
RateFunction = Callable[[float], FloatArray | float]


@dataclass(frozen=True, slots=True)
class RadiationFraction:
    """One instantaneous linear-quadratic radiation event."""

    time: float
    dose: ArrayLike
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        dose = np.asarray(self.dose, dtype=np.float64)
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("radiation time must be finite and nonnegative")
        if np.any(~np.isfinite(dose)) or np.any(dose < 0):
            raise ValueError("radiation dose must be finite and nonnegative")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("radiation alpha must be finite and nonnegative")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("radiation beta must be finite and nonnegative")
        object.__setattr__(self, "dose", dose.copy())


@dataclass(frozen=True, slots=True)
class MultiCompartmentSimulationResult:
    """Latent compartment snapshots at requested simulation times."""

    times: FloatArray
    viable: FloatArray
    edema: FloatArray
    necrotic: FloatArray


class MultiCompartmentSolver:
    """Advance viable tumor, edema, and necrosis on a fixed brain domain."""

    def __init__(
        self,
        viable_diffusivity: FloatArray,
        edema_diffusivity: FloatArray,
        brain_mask: BoolArray,
        parameters: MultiCompartmentParameters,
        *,
        spacing: tuple[float, ...] = (1.0, 1.0),
        cavity_mask: BoolArray | None = None,
    ) -> None:
        viable_diffusivity = np.asarray(viable_diffusivity, dtype=np.float64)
        edema_diffusivity = np.asarray(edema_diffusivity, dtype=np.float64)
        if viable_diffusivity.shape != edema_diffusivity.shape:
            raise ValueError("compartment diffusivity fields must have matching shapes")
        zero_reaction = ReactionDiffusionParameters(proliferation_rate=0.0)
        self._viable_spatial = FiniteVolumeSolver(
            viable_diffusivity,
            brain_mask,
            zero_reaction,
            spacing=spacing,
            cavity_mask=cavity_mask,
        )
        self._edema_spatial = FiniteVolumeSolver(
            edema_diffusivity,
            brain_mask,
            zero_reaction,
            spacing=spacing,
            cavity_mask=cavity_mask,
        )
        self.parameters = parameters
        self.active_mask = self._viable_spatial.active_mask
        self.shape = viable_diffusivity.shape

    def stable_time_step(
        self,
        maximum_cell_kill_rate: float = 0.0,
        maximum_edema_leakage_suppression: float = 0.0,
    ) -> float:
        """Return a conservative explicit-Euler step for the coupled system."""
        treatment_rates = (maximum_cell_kill_rate, maximum_edema_leakage_suppression)
        if any(not np.isfinite(rate) or rate < 0 for rate in treatment_rates):
            raise ValueError("maximum treatment rates must be finite and nonnegative")
        viable_spatial_rate = 0.9 / self._viable_spatial.stable_time_step()
        edema_spatial_rate = 0.9 / self._edema_spatial.stable_time_step()
        viable_rate = viable_spatial_rate + self.parameters.proliferation_rate
        viable_rate += maximum_cell_kill_rate
        edema_rate = edema_spatial_rate + self.parameters.edema_generation_rate
        edema_rate += self.parameters.edema_clearance_rate
        rate = max(
            viable_rate + self.parameters.spontaneous_necrosis_rate,
            edema_rate,
            self.parameters.necrosis_clearance_rate,
        )
        return np.inf if rate == 0 else 0.9 / rate

    def simulate(
        self,
        initial_viable: FloatArray,
        initial_edema: FloatArray,
        initial_necrotic: FloatArray,
        output_times: FloatArray,
        *,
        maximum_time_step: float | None = None,
        treatment_cell_kill: RateFunction | None = None,
        treatment_edema_leakage_suppression: RateFunction | None = None,
        radiation_fractions: tuple[RadiationFraction, ...] = (),
    ) -> MultiCompartmentSimulationResult:
        """Advance all compartments and return requested snapshots."""
        times = np.asarray(output_times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("output_times must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("output_times must be finite, nonnegative, and strictly increasing")
        if maximum_time_step is not None and (
            not np.isfinite(maximum_time_step) or maximum_time_step <= 0
        ):
            raise ValueError("maximum_time_step must be finite and positive")
        if any(
            later.time < earlier.time
            for earlier, later in pairwise(radiation_fractions)
        ):
            raise ValueError("radiation fractions must be ordered by time")

        viable = self._validated_state(initial_viable, "initial_viable")
        edema = self._validated_state(initial_edema, "initial_edema")
        necrotic = self._validated_state(initial_necrotic, "initial_necrotic")
        viable = np.where(self.active_mask, viable, 0.0)
        edema = np.where(self.active_mask, edema, 0.0)
        necrotic = np.where(self.active_mask, necrotic, 0.0)
        output_shape = (times.size, *self.shape)
        viable_output = np.empty(output_shape, dtype=np.float64)
        edema_output = np.empty(output_shape, dtype=np.float64)
        necrotic_output = np.empty(output_shape, dtype=np.float64)

        current_time = 0.0
        next_fraction = 0
        for index, target_time in enumerate(times):
            while (
                next_fraction < len(radiation_fractions)
                and radiation_fractions[next_fraction].time <= current_time
            ):
                viable, necrotic = self._apply_radiation_fraction(
                    viable, necrotic, radiation_fractions[next_fraction]
                )
                next_fraction += 1
            while current_time < target_time:
                cell_kill = self._rate_at(treatment_cell_kill, current_time)
                edema_suppression = self._rate_at(
                    treatment_edema_leakage_suppression, current_time
                )
                stable_step = self.stable_time_step(
                    float(np.max(cell_kill, initial=0.0)),
                    float(np.max(edema_suppression, initial=0.0)),
                )
                next_event_time = (
                    radiation_fractions[next_fraction].time
                    if next_fraction < len(radiation_fractions)
                    else np.inf
                )
                step = min(target_time - current_time, next_event_time - current_time, stable_step)
                if maximum_time_step is not None:
                    step = min(step, maximum_time_step)
                if not np.isfinite(step) or step <= 0:
                    current_time = target_time
                    break

                viable_reaction, edema_reaction, necrotic_reaction = self.parameters.reaction(
                    viable,
                    edema,
                    necrotic,
                    cell_kill,
                    edema_suppression,
                )
                viable += step * (self._viable_spatial.spatial_derivative(viable) + viable_reaction)
                edema += step * (self._edema_spatial.spatial_derivative(edema) + edema_reaction)
                necrotic += step * necrotic_reaction
                viable = np.clip(viable, 0.0, self.parameters.carrying_capacity)
                edema = np.clip(edema, 0.0, 1.0)
                necrotic = np.clip(necrotic, 0.0, self.parameters.carrying_capacity)
                for field in (viable, edema, necrotic):
                    field[~self.active_mask] = 0.0
                current_time += step

                while (
                    next_fraction < len(radiation_fractions)
                    and radiation_fractions[next_fraction].time <= current_time
                    and radiation_fractions[next_fraction].time <= target_time
                ):
                    viable, necrotic = self._apply_radiation_fraction(
                        viable, necrotic, radiation_fractions[next_fraction]
                    )
                    next_fraction += 1

            viable_output[index] = viable
            edema_output[index] = edema
            necrotic_output[index] = necrotic

        return MultiCompartmentSimulationResult(
            times=times.copy(),
            viable=viable_output,
            edema=edema_output,
            necrotic=necrotic_output,
        )

    def _validated_state(self, value: FloatArray, name: str) -> FloatArray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != self.shape:
            raise ValueError(f"{name} must match the solver shape")
        if np.any(~np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"{name} must contain finite nonnegative values")
        return value.copy()

    def _apply_radiation_fraction(
        self,
        viable: FloatArray,
        necrotic: FloatArray,
        fraction: RadiationFraction,
    ) -> tuple[FloatArray, FloatArray]:
        dose = np.asarray(fraction.dose, dtype=np.float64)
        try:
            dose = np.broadcast_to(dose, self.shape)
        except ValueError as error:
            raise ValueError("radiation dose must be scalar or match the domain") from error
        survival = np.exp(-(fraction.alpha * dose + fraction.beta * dose**2))
        killed = (1.0 - survival) * viable
        viable = viable - killed
        necrotic = np.clip(
            necrotic + killed,
            0.0,
            self.parameters.carrying_capacity,
        )
        for field in (viable, necrotic):
            field[~self.active_mask] = 0.0
        return viable, necrotic

    def _rate_at(self, function: RateFunction | None, time: float) -> FloatArray:
        if function is None:
            return np.zeros(self.shape, dtype=np.float64)
        rate = np.asarray(function(time), dtype=np.float64)
        try:
            rate = np.broadcast_to(rate, self.shape)
        except ValueError as error:
            raise ValueError("treatment output must be scalar or match the domain") from error
        if np.any(~np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("treatment rates must be finite and nonnegative")
        return np.where(self.active_mask, rate, 0.0)

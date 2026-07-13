"""Parameter estimation against finite-volume density observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.solver import FiniteVolumeSolver

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """Estimated homogeneous coefficients and optimizer diagnostics."""

    diffusivity: float
    proliferation_rate: float
    residual_sum_squares: float
    function_evaluations: int
    converged: bool


def fit_homogeneous_parameters(
    initial_density: FloatArray,
    observation_times: FloatArray,
    observed_density: FloatArray,
    brain_mask: BoolArray,
    *,
    diffusivity_bounds: tuple[float, float],
    proliferation_bounds: tuple[float, float],
    spacing: tuple[float, float] = (1.0, 1.0),
    cavity_mask: BoolArray | None = None,
    observation_mask: BoolArray | None = None,
    initial_guess: tuple[float, float] | None = None,
    maximum_time_step: float | None = None,
    maximum_function_evaluations: int = 100,
) -> ParameterEstimate:
    """Estimate scalar diffusivity and proliferation from density snapshots."""
    initial_density = np.asarray(initial_density, dtype=np.float64)
    observation_times = np.asarray(observation_times, dtype=np.float64)
    observed_density = np.asarray(observed_density, dtype=np.float64)
    brain_mask = np.asarray(brain_mask, dtype=bool)

    if initial_density.ndim != 2:
        raise ValueError("initial_density must be a two-dimensional array")
    if brain_mask.shape != initial_density.shape:
        raise ValueError("brain_mask must match initial_density shape")
    if observed_density.shape != (observation_times.size, *initial_density.shape):
        raise ValueError("observed_density must have shape (time, row, column)")
    if np.any(~np.isfinite(observed_density)) or np.any(observed_density < 0):
        raise ValueError("observed_density must contain finite nonnegative values")
    if maximum_function_evaluations <= 0:
        raise ValueError("maximum_function_evaluations must be positive")

    lower_bounds = np.array([diffusivity_bounds[0], proliferation_bounds[0]])
    upper_bounds = np.array([diffusivity_bounds[1], proliferation_bounds[1]])
    if np.any(lower_bounds < 0) or np.any(lower_bounds >= upper_bounds):
        raise ValueError("parameter bounds must be nonnegative and strictly ordered")

    if observation_mask is None:
        observation_mask = brain_mask
    else:
        observation_mask = np.asarray(observation_mask, dtype=bool)
        if observation_mask.shape != initial_density.shape:
            raise ValueError("observation_mask must match initial_density shape")
        if np.any(observation_mask & ~brain_mask):
            raise ValueError("observation_mask must be contained within brain_mask")
    if not np.any(observation_mask):
        raise ValueError("observation_mask must select at least one location")

    if initial_guess is None:
        guess = (lower_bounds + upper_bounds) / 2.0
    else:
        guess = np.asarray(initial_guess, dtype=np.float64)
        if guess.shape != (2,) or np.any(guess < lower_bounds) or np.any(guess > upper_bounds):
            raise ValueError("initial_guess must lie within the parameter bounds")

    target = observed_density[:, observation_mask]

    def residual(parameters: FloatArray) -> FloatArray:
        diffusivity, proliferation_rate = parameters
        solver = FiniteVolumeSolver(
            diffusivity=np.full(initial_density.shape, diffusivity),
            brain_mask=brain_mask,
            cavity_mask=cavity_mask,
            spacing=spacing,
            parameters=ReactionDiffusionParameters(proliferation_rate=proliferation_rate),
        )
        prediction = solver.simulate(
            initial_density,
            observation_times,
            maximum_time_step=maximum_time_step,
        ).density
        return (prediction[:, observation_mask] - target).ravel()

    result = least_squares(
        residual,
        guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=maximum_function_evaluations,
        method="trf",
    )
    return ParameterEstimate(
        diffusivity=float(result.x[0]),
        proliferation_rate=float(result.x[1]),
        residual_sum_squares=float(np.dot(result.fun, result.fun)),
        function_evaluations=int(result.nfev),
        converged=bool(result.success),
    )

"""Monte Carlo predictive distributions for the next scan.

The state-space model in ``gbm_pinn.ceiling`` cannot say which way a tumor
will move, but it does say how uncertain the next scan is and how that
uncertainty grows with the interval. This module turns that into forecasts:

* parameter uncertainty comes from refitting on patient-bootstrap resamples;
* for each parameter draw the Kalman filter gives a Gaussian one-step
  predictive distribution, from which radii are sampled (clipped at zero);
* pooling the samples over draws gives a mixture predictive distribution.

From the samples come central prediction intervals and event probabilities,
such as the chance that the next radius exceeds a RANO-style threshold above
the segment's smallest radius so far (the nadir).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gbm_pinn.ceiling import StateSpace, fit_state_space, kalman
from gbm_pinn.switching import Cohort

FloatArray = NDArray[np.float64]

# RANO progression is a 25% increase in the product of perpendicular
# diameters. For a sphere that product scales with radius squared, so the
# equivalent radius threshold is sqrt(1.25) times the nadir radius.
RANO_RADIUS_FACTOR = float(np.sqrt(1.25))


@dataclass(frozen=True, slots=True)
class Predictive:
    """Monte Carlo samples of the next radius for every transition."""

    samples: FloatArray  # (transitions, draws)
    target: FloatArray
    previous: FloatArray
    nadir: FloatArray
    horizon: FloatArray
    segment: NDArray[np.int64]
    position: NDArray[np.int64]


def _nadir_and_position(
    cohort: Cohort, segment: NDArray[np.int64], previous_count: int
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Running minimum radius up to each source scan, and the target's index."""
    nadir, position = [], []
    seen: dict[int, int] = {}
    for s in segment:
        k = seen.get(int(s), 1)
        seen[int(s)] = k + 1
        position.append(k)
        nadir.append(np.nanmin(cohort.radius[s, :k]))
    assert len(position) == previous_count
    return np.asarray(nadir), np.asarray(position)


def bootstrap_draws(
    cohort: Cohort,
    groups: Sequence[str],
    *,
    draws: int,
    rng: np.random.Generator,
    starts: int = 2,
) -> list[StateSpace]:
    """Parameter draws from refitting on patient-bootstrap resamples."""
    groups = np.asarray(groups)
    patients = np.unique(groups)
    rows = {p: np.flatnonzero(groups == p) for p in patients}
    fits = []
    for _ in range(draws):
        picked = rng.choice(patients, size=patients.size, replace=True)
        index = np.concatenate([rows[p] for p in picked])
        sample = Cohort(cohort.days[index], cohort.radius[index])
        fits.append(fit_state_space(sample, starts=starts).params)
    return fits


def monte_carlo(
    cohort: Cohort,
    params: Sequence[StateSpace],
    *,
    samples_per_draw: int,
    rng: np.random.Generator,
    fixed_variance: float | None = None,
) -> Predictive:
    """Pooled predictive samples over parameter draws.

    With ``fixed_variance`` every transition uses that variance instead of
    the filter's, which ignores the interval and the history: the
    time-blind comparator with the same centre.
    """
    pieces = []
    reference = None
    for p in params:
        _, t = kalman(cohort, p)
        if reference is None:
            reference = t
        variance = (
            t.kalman_var if fixed_variance is None else np.full_like(t.kalman_var, fixed_variance)
        )
        z = rng.standard_normal((t.target.size, samples_per_draw))
        pieces.append(np.clip(t.predicted_mean[:, None] + np.sqrt(variance)[:, None] * z, 0, None))
    assert reference is not None
    nadir, position = _nadir_and_position(cohort, reference.segment, reference.target.size)
    return Predictive(
        np.concatenate(pieces, axis=1),
        reference.target,
        reference.previous,
        nadir,
        reference.horizon,
        reference.segment,
        position,
    )


def coverage(pred: Predictive, level: float) -> float:
    """Share of observed next radii inside the central ``level`` interval."""
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(pred.samples, [tail, 1.0 - tail], axis=1)
    return float(np.mean((pred.target >= low) & (pred.target <= high)))


def crps(pred: Predictive) -> FloatArray:
    """Continuous ranked probability score per transition (lower is better).

    Uses the sorted-sample form of E|X - y| - E|X - X'| / 2.
    """
    x = np.sort(pred.samples, axis=1)
    m = x.shape[1]
    first = np.mean(np.abs(x - pred.target[:, None]), axis=1)
    weights = (2 * np.arange(1, m + 1) - m - 1) / m**2
    second = np.sum(x * weights[None, :], axis=1)
    return first - second


def progression_probability(pred: Predictive, factor: float = RANO_RADIUS_FACTOR) -> FloatArray:
    """Probability the next radius exceeds ``factor`` times the nadir so far."""
    return np.mean(pred.samples > factor * pred.nadir[:, None], axis=1)

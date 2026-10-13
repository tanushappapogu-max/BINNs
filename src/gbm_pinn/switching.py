"""Test whether tumor change is bunched beyond steady growth plus noise.

Trajectories are equivalent-sphere radii at irregular scan days. Under the
steady-growth null each segment's radius is linear in time with independent
Gaussian measurement noise; a pooled noise level is fitted from the residuals.
The bunching statistic is the share of a segment's total absolute radius
change that falls in its single largest interval, averaged over segments, and
its null distribution comes from a parametric bootstrap at the observed days.

Segments of different lengths are stored as NaN-padded rows so every step is
vectorized over segments and bootstrap draws. The protocol is fixed in
``docs/preregistration_switching.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def radius_mm(volume_ml: FloatArray | float) -> FloatArray:
    """Equivalent-sphere radius in millimetres for a volume in millilitres."""
    volume_mm3 = np.asarray(volume_ml, dtype=np.float64) * 1000.0
    if np.any(volume_mm3 < 0):
        raise ValueError("volumes must be nonnegative")
    return np.cbrt(3.0 * volume_mm3 / (4.0 * np.pi))


def split_segments(
    days: Sequence[float], values: Sequence[float], *, exclusion_days: float = 28.0
) -> list[tuple[FloatArray, FloatArray]]:
    """Cut one patient's sorted trajectory at every interval <= ``exclusion_days``."""
    t = np.asarray(days, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    cuts = np.flatnonzero(np.diff(t) <= exclusion_days) + 1
    return list(zip(np.split(t, cuts), np.split(y, cuts), strict=True))


@dataclass(frozen=True, slots=True)
class Cohort:
    """NaN-padded segments: ``days`` and ``radius`` have shape (segments, max_scans)."""

    days: FloatArray
    radius: FloatArray

    @classmethod
    def from_segments(
        cls, segments: Sequence[tuple[FloatArray, FloatArray]], *, min_scans: int = 3
    ) -> Cohort:
        kept = [(t, r) for t, r in segments if len(t) >= min_scans]
        if not kept:
            raise ValueError("no segment has enough scans")
        width = max(len(t) for t, _ in kept)
        days = np.full((len(kept), width), np.nan)
        radius = np.full((len(kept), width), np.nan)
        for i, (t, r) in enumerate(kept):
            days[i, : len(t)] = t
            radius[i, : len(r)] = r
        return cls(days, radius)

    @property
    def mask(self) -> NDArray[np.bool_]:
        return ~np.isnan(self.days)

    def with_radius(self, radius: FloatArray) -> Cohort:
        return Cohort(self.days, np.where(self.mask, radius, np.nan))


@dataclass(frozen=True, slots=True)
class LinearFit:
    intercept: FloatArray
    slope: FloatArray
    sigma: float


def fit_linear(cohort: Cohort) -> LinearFit:
    """Per-segment OLS of radius on time with a noise level pooled over segments.

    Time is centred per segment, so the intercept is the fitted radius at the
    segment's mean scan day.
    """
    mask = cohort.mask
    n = mask.sum(axis=1)
    t_mean = np.nanmean(cohort.days, axis=1, keepdims=True)
    tc = np.where(mask, cohort.days - t_mean, 0.0)
    y = np.where(mask, cohort.radius, 0.0)
    y_mean = y.sum(axis=1, keepdims=True) / n[:, None]
    slope = (tc * (y - y_mean)).sum(axis=1) / (tc**2).sum(axis=1)
    residual = np.where(mask, y - y_mean - slope[:, None] * tc, 0.0)
    dof = int((n - 2).sum())
    sigma = float(np.sqrt((residual**2).sum() / dof)) if dof > 0 else 0.0
    return LinearFit(y_mean[:, 0], slope, sigma)


def _centred_days(cohort: Cohort) -> FloatArray:
    return cohort.days - np.nanmean(cohort.days, axis=1, keepdims=True)


def bunching(radius: FloatArray) -> FloatArray:
    """Mean bunching over segments. Works on (..., segments, scans) arrays.

    Segments with no total change are left out of the mean, since every
    interval is equally (un)informative there.
    """
    change = np.abs(np.diff(radius, axis=-1))
    total = np.nansum(change, axis=-1)
    largest = np.nanmax(np.where(np.isnan(change), -np.inf, change), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, largest / total, np.nan)
    return np.nanmean(share, axis=-1)


def largest_change_is_growth(radius: FloatArray) -> NDArray[np.bool_]:
    """Whether each segment's largest interval is an increase."""
    change = np.diff(radius, axis=-1)
    index = np.nanargmax(np.where(np.isnan(change), -np.inf, np.abs(change)), axis=-1)
    return np.take_along_axis(change, index[..., None], axis=-1)[..., 0] > 0


def simulate_steady(
    cohort: Cohort, fit: LinearFit, draws: int, rng: np.random.Generator, *, sigma: float | None
) -> FloatArray:
    """H0 trajectories at the observed days, clipped at zero radius."""
    noise = fit.sigma if sigma is None else sigma
    tc = _centred_days(cohort)
    mean = fit.intercept[:, None] + fit.slope[:, None] * tc
    sample = mean[None] + noise * rng.standard_normal((draws, *mean.shape))
    return np.where(cohort.mask, np.clip(sample, 0.0, None), np.nan)


def simulate_switching(
    cohort: Cohort, fit: LinearFit, draws: int, rng: np.random.Generator
) -> FloatArray:
    """Flat-then-linear trajectories with the same net change as the H0 fit."""
    tc = _centred_days(cohort)
    first = np.nanmin(tc, axis=1)
    last = np.nanmax(tc, axis=1)
    span = last - first
    start = fit.intercept + fit.slope * first
    net = fit.slope * span
    shape = (draws, *tc.shape)
    switch = first[None, :] + rng.uniform(size=shape[:2]) * span[None, :]
    elapsed = np.clip(tc[None] - switch[..., None], 0.0, None)
    remaining = np.maximum(last[None, :] - switch, 1e-9)
    mean = start[None, :, None] + (net[None, :] / remaining)[..., None] * elapsed
    sample = mean + fit.sigma * rng.standard_normal(shape)
    return np.where(cohort.mask, np.clip(sample, 0.0, None), np.nan)


@dataclass(frozen=True, slots=True)
class TestResult:
    observed: float
    null_mean: float
    null_q95: float
    p_value: float
    sigma: float
    segments: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "observed_bunching": self.observed,
            "null_mean_bunching": self.null_mean,
            "null_95th_percentile": self.null_q95,
            "p_value": self.p_value,
            "pooled_sigma_mm": self.sigma,
            "segments": self.segments,
        }


def bunching_test(
    cohort: Cohort,
    *,
    draws: int = 5000,
    rng: np.random.Generator,
    sigma_scale: float = 1.0,
) -> TestResult:
    """One-sided parametric-bootstrap test of bunching against steady growth."""
    fit = fit_linear(cohort)
    observed = float(bunching(cohort.radius))
    null = bunching(simulate_steady(cohort, fit, draws, rng, sigma=fit.sigma * sigma_scale))
    p_value = float((1 + np.sum(null >= observed)) / (draws + 1))
    return TestResult(
        observed,
        float(null.mean()),
        float(np.quantile(null, 0.95)),
        p_value,
        fit.sigma,
        int(cohort.days.shape[0]),
    )


def rejection_rate(
    cohort: Cohort,
    *,
    alternative: str,
    cohorts: int,
    draws: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> float:
    """Share of synthetic cohorts the test rejects, under H0 or switching.

    Each synthetic cohort reuses the real scan days and the H0 fit to the real
    radii, then goes through the full test (refit and bootstrap).
    """
    fit = fit_linear(cohort)
    if alternative == "steady":
        synthetic = simulate_steady(cohort, fit, cohorts, rng, sigma=None)
    elif alternative == "switching":
        synthetic = simulate_switching(cohort, fit, cohorts, rng)
    else:
        raise ValueError("alternative must be 'steady' or 'switching'")
    rejected = 0
    for radius in synthetic:
        result = bunching_test(cohort.with_radius(radius), draws=draws, rng=rng)
        rejected += result.p_value < alpha
    return rejected / cohorts

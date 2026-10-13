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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

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


# ---------------------------------------------------------------------------
# Persistence test: regime change versus measurement outliers
# ---------------------------------------------------------------------------


def pooled_sigma(cohort: Cohort, radius: FloatArray) -> FloatArray:
    """Pooled OLS residual sigma for (..., segments, scans) radii at the cohort's days."""
    mask = cohort.mask
    n = mask.sum(axis=1)
    tc = np.where(mask, _centred_days(cohort), 0.0)
    y = np.where(mask, radius, 0.0)
    y_mean = y.sum(axis=-1, keepdims=True) / n[:, None]
    slope = (tc * (y - y_mean)).sum(axis=-1) / (tc**2).sum(axis=1)
    residual = np.where(mask, y - y_mean - slope[..., None] * tc, 0.0)
    return np.sqrt((residual**2).sum(axis=(-2, -1)) / (n - 2).sum())


def _consecutive_changes(radius: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    change = np.diff(radius, axis=-1)
    first, second = change[..., :-1], change[..., 1:]
    valid = ~(np.isnan(first) | np.isnan(second))
    lead = radius.shape[:-2]
    return (
        np.where(valid, first, 0.0).reshape(*lead, -1),
        np.where(valid, second, 0.0).reshape(*lead, -1),
        valid.reshape(*lead, -1),
    )


def persistence(radius: FloatArray) -> FloatArray:
    """Pearson correlation of consecutive radius changes, pooled over segments."""
    first, second, valid = _consecutive_changes(radius)
    count = valid.sum(axis=-1)
    df = np.where(valid, first - (first.sum(axis=-1) / count)[..., None], 0.0)
    ds = np.where(valid, second - (second.sum(axis=-1) / count)[..., None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (df * ds).sum(axis=-1) / np.sqrt((df**2).sum(axis=-1) * (ds**2).sum(axis=-1))


def sign_concordance(radius: FloatArray) -> FloatArray:
    """Mean of sign(Δr_k)·sign(Δr_k+1) over consecutive interval pairs."""
    first, second, valid = _consecutive_changes(radius)
    agreement = np.where(valid, np.sign(first) * np.sign(second), 0.0)
    return agreement.sum(axis=-1) / valid.sum(axis=-1)


@dataclass(frozen=True, slots=True)
class Draws:
    """Common random numbers so rival simulations vary smoothly in their parameters."""

    normal: FloatArray
    contamination: FloatArray
    switch: FloatArray

    @classmethod
    def sample(cls, cohort: Cohort, draws: int, rng: np.random.Generator) -> Draws:
        shape = (draws, *cohort.days.shape)
        return cls(
            rng.standard_normal(shape),
            rng.uniform(size=shape),
            rng.uniform(size=shape[:2]),
        )


def _finish(cohort: Cohort, sample: FloatArray) -> FloatArray:
    return np.where(cohort.mask, np.clip(sample, 0.0, None), np.nan)


def simulate_outliers(
    cohort: Cohort, fit: LinearFit, draws: Draws, *, rate: float, noise: float, scale: float = 4.0
) -> FloatArray:
    """Steady growth with contaminated noise: a share ``rate`` of scans err ``scale``x more."""
    tc = _centred_days(cohort)
    mean = fit.intercept[:, None] + fit.slope[:, None] * tc
    factor = np.where(draws.contamination < rate, scale, 1.0)
    return _finish(cohort, mean[None] + noise * factor * draws.normal)


def simulate_regime_change(
    cohort: Cohort, fit: LinearFit, draws: Draws, *, amplitude: float, noise: float
) -> FloatArray:
    """Flat until a uniform switch time, then linear to the H0 net change plus ``amplitude``."""
    tc = _centred_days(cohort)
    first = np.nanmin(tc, axis=1)
    last = np.nanmax(tc, axis=1)
    start = fit.intercept + fit.slope * first
    net = fit.slope * (last - first)
    total = net + np.where(net >= 0, 1.0, -1.0) * amplitude
    switch = first[None, :] + draws.switch * (last - first)[None, :]
    elapsed = np.clip(tc[None] - switch[..., None], 0.0, None)
    remaining = np.maximum(last[None, :] - switch, 1e-9)
    mean = start[None, :, None] + (total[None, :] / remaining)[..., None] * elapsed
    return _finish(cohort, mean + noise * draws.normal)


@dataclass(frozen=True, slots=True)
class Calibration:
    parameter: float
    noise: float
    bunching: float
    reproduces: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "parameter": self.parameter,
            "noise_mm": self.noise,
            "mean_bunching": self.bunching,
            "reproduces_observed_bunching": self.reproduces,
        }


def calibrate(
    cohort: Cohort,
    simulate: Callable[[float, float], FloatArray],
    grid: Sequence[float],
    *,
    target_bunching: float,
    target_sigma: float,
    tolerance: float = 0.005,
) -> Calibration:
    """Pick the grid value whose noise-matched simulations best reproduce the bunching.

    For each grid value the noise level is bisected so the mean refitted
    pooled sigma equals ``target_sigma``; the grid value with mean bunching
    closest to ``target_bunching`` is returned.
    """
    best: Calibration | None = None
    for parameter in grid:
        low, high = 1e-6, 2.0 * target_sigma
        for _ in range(30):
            mid = 0.5 * (low + high)
            if pooled_sigma(cohort, simulate(parameter, mid)).mean() < target_sigma:
                low = mid
            else:
                high = mid
        noise = 0.5 * (low + high)
        mean_bunching = float(bunching(simulate(parameter, noise)).mean())
        if best is None or abs(mean_bunching - target_bunching) < abs(
            best.bunching - target_bunching
        ):
            best = Calibration(float(parameter), noise, mean_bunching, False)
    assert best is not None
    reproduces = abs(best.bunching - target_bunching) <= tolerance
    return Calibration(best.parameter, best.noise, best.bunching, reproduces)


def separation_auc(higher: FloatArray, lower: FloatArray) -> float:
    """Probability a draw from ``higher`` exceeds one from ``lower`` (ties count half)."""
    ranks = rankdata(np.concatenate([higher, lower]))
    u = ranks[: higher.size].sum() - higher.size * (higher.size + 1) / 2.0
    return float(u / (higher.size * lower.size))


def central_interval(sample: FloatArray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(sample, [tail, 1.0 - tail])
    return float(low), float(high)


def persistence_verdict(
    observed: float,
    regime: FloatArray,
    outliers: FloatArray,
    *,
    min_auc: float = 0.80,
) -> dict[str, object]:
    """Pre-registered decision for the persistence test."""
    auc = separation_auc(regime, outliers)
    regime_low, regime_high = central_interval(regime)
    outlier_low, outlier_high = central_interval(outliers)
    regime_ok = regime_low <= observed <= regime_high
    outlier_ok = outlier_low <= observed <= outlier_high
    if auc < min_auc:
        verdict = "inconclusive: rivals not separable"
    elif regime_ok and not outlier_ok:
        verdict = "switching favored"
    elif outlier_ok and not regime_ok:
        verdict = "outliers favored"
    elif regime_ok and outlier_ok:
        verdict = "undetermined: both rivals compatible"
    else:
        verdict = "neither rival explains the data"
    return {
        "auc_regime_above_outliers": auc,
        "regime_interval": [regime_low, regime_high],
        "outlier_interval": [outlier_low, outlier_high],
        "regime_compatible": regime_ok,
        "outliers_compatible": outlier_ok,
        "verdict": verdict,
    }

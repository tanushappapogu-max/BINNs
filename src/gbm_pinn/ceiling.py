"""How much of scan-to-scan tumor change is forecastable at all?

Each trajectory segment (equivalent radius y at irregular days) is modelled as

    x_{k+1} = x_k + v * h_k + w_k,   w_k ~ N(0, q0 + q1 * h_k)   lasting change
    y_k     = x_k + e_k,             e_k ~ N(0, r)               reverting change
    v ~ N(mu, tau2), one drift per segment.

``q1`` is change that accumulates with elapsed time (the "clock" that growth
models assume), ``q0`` is lasting change of the same size whatever the
interval, ``r`` is change that reverts at the next scan (measurement error or
transient imaging effects), and ``tau2`` is between-patient variation in a
steady growth rate. The model is fitted by maximum likelihood with a Kalman
filter that conditions the latent state on each segment's first scan.

Three forecast errors are compared on the same transitions:

* persistence, which predicts no change;
* the Kalman one-step forecast, the best predictor from a segment's own size
  history under the fitted model;
* the oracle floor ``q0 + q1 * h + r``, the error that remains even when the
  true current state and drift are known exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import chi2

from gbm_pinn.switching import Cohort

FloatArray = NDArray[np.float64]
PARAMETERS = ("reverting_var", "jump_var", "diffusion_rate", "drift_mean", "drift_var")
_VARIANCES = {"reverting_var", "jump_var", "diffusion_rate", "drift_var"}
_LOG_FLOOR = np.log(1e-10)


@dataclass(frozen=True, slots=True)
class StateSpace:
    """Model parameters in mm^2 (variances), mm^2/day (diffusion) and mm/day (drift)."""

    reverting_var: float
    jump_var: float
    diffusion_rate: float
    drift_mean: float
    drift_var: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Transitions:
    """Per-transition forecast errors under a fitted model."""

    horizon: FloatArray
    persistence_sq: FloatArray
    kalman_sq: FloatArray
    kalman_var: FloatArray
    oracle_var: FloatArray
    segment: NDArray[np.int64]


def kalman(cohort: Cohort, params: StateSpace) -> tuple[float, Transitions]:
    """Log-likelihood of every segment after its first scan, and forecast errors."""
    days, y = cohort.days, cohort.radius
    n_seg, width = y.shape
    r, q0, q1 = params.reverting_var, params.jump_var, params.diffusion_rate
    mean = np.stack([y[:, 0], np.full(n_seg, params.drift_mean)], axis=1)
    cov = np.zeros((n_seg, 2, 2))
    cov[:, 0, 0], cov[:, 1, 1] = r, params.drift_var

    loglik = 0.0
    columns: dict[str, list[FloatArray]] = {k: [] for k in Transitions.__dataclass_fields__}
    for k in range(1, width):
        observed = ~np.isnan(y[:, k])
        h = np.where(observed, days[:, k] - days[:, k - 1], 0.0)
        transition = np.zeros((n_seg, 2, 2))
        transition[:, 0, 0] = transition[:, 1, 1] = 1.0
        transition[:, 0, 1] = h
        mean_pred = np.einsum("nij,nj->ni", transition, mean)
        cov_pred = transition @ cov @ transition.transpose(0, 2, 1)
        cov_pred[:, 0, 0] += q0 + q1 * h
        variance = cov_pred[:, 0, 0] + r
        innovation = np.where(observed, y[:, k] - mean_pred[:, 0], 0.0)
        loglik += float(
            np.sum(
                np.where(
                    observed,
                    -0.5 * (np.log(2.0 * np.pi * variance) + innovation**2 / variance),
                    0.0,
                )
            )
        )
        gain = cov_pred[:, :, 0] / variance[:, None]
        mean = np.where(observed[:, None], mean_pred + gain * innovation[:, None], mean)
        updated = cov_pred - np.einsum("ni,nj->nij", gain, cov_pred[:, 0, :])
        cov = np.where(observed[:, None, None], updated, cov)

        index = np.flatnonzero(observed)
        columns["horizon"].append(h[index])
        columns["persistence_sq"].append((y[index, k] - y[index, k - 1]) ** 2)
        columns["kalman_sq"].append(innovation[index] ** 2)
        columns["kalman_var"].append(variance[index])
        columns["oracle_var"].append(q0 + q1 * h[index] + r)
        columns["segment"].append(index)
    return loglik, Transitions(**{k: np.concatenate(v) for k, v in columns.items()})


def _unpack(vector: FloatArray, free: Sequence[str], fixed: dict[str, float]) -> StateSpace:
    values = dict(fixed)
    for name, value in zip(free, vector, strict=True):
        values[name] = float(np.exp(value)) if name in _VARIANCES else float(value)
    return StateSpace(**{name: values[name] for name in PARAMETERS})


@dataclass(frozen=True, slots=True)
class Fit:
    params: StateSpace
    loglik: float
    free: tuple[str, ...]


def fit_state_space(
    cohort: Cohort,
    *,
    fixed: dict[str, float] | None = None,
    starts: int = 4,
    seed: int = 0,
) -> Fit:
    """Maximum-likelihood fit; names in ``fixed`` are held at the given values.

    Variances are optimized on a log scale with a tiny floor, so a variance
    the data do not support converges toward zero rather than failing.
    Several Nelder-Mead starts guard against local optima.
    """
    fixed = dict(fixed or {})
    free = tuple(name for name in PARAMETERS if name not in fixed)
    rng = np.random.default_rng(seed)
    scale = float(np.nanvar(np.diff(cohort.radius, axis=1))) or 1.0
    span = float(np.nanmax(cohort.days) - np.nanmin(cohort.days)) or 1.0
    base = {
        "reverting_var": np.log(scale / 4.0),
        "jump_var": np.log(scale / 4.0),
        "diffusion_rate": np.log(scale / (4.0 * span)),
        "drift_mean": 0.0,
        "drift_var": np.log(scale / span**2),
    }

    def floor(vector: FloatArray) -> FloatArray:
        pairs = zip(free, vector, strict=True)
        return np.array([max(v, _LOG_FLOOR) if n in _VARIANCES else v for n, v in pairs])

    def objective(vector: FloatArray) -> float:
        return -kalman(cohort, _unpack(floor(vector), free, fixed))[0]

    best = None
    for attempt in range(starts):
        start = np.array([base[n] for n in free])
        if attempt:
            start = start + rng.normal(0.0, 1.0, size=start.size) * np.array(
                [1.0 if n in _VARIANCES else 1e-3 for n in free]
            )
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 40000, "maxfev": 40000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    return Fit(_unpack(floor(best.x), free, fixed), -float(best.fun), free)


def boundary_lr_test(full: Fit, reduced: Fit) -> dict[str, float]:
    """Likelihood-ratio test of one variance set to zero (boundary of the space).

    Under the null the statistic follows a 50:50 mixture of chi-square with
    0 and 1 degrees of freedom, so the p-value is half the chi-square(1) tail.
    """
    statistic = max(0.0, 2.0 * (full.loglik - reduced.loglik))
    return {"lr_statistic": statistic, "p_value": float(0.5 * chi2.sf(statistic, 1))}


def ceiling(cohort: Cohort, params: StateSpace) -> dict[str, float]:
    """Share of persistence's squared error removable from history and by an oracle."""
    _, t = kalman(cohort, params)
    persistence = float(t.persistence_sq.mean())
    return {
        "transitions": int(t.horizon.size),
        "persistence_mse": persistence,
        "history_mse": float(t.kalman_sq.mean()),
        "oracle_mse": float(t.oracle_var.mean()),
        "history_ceiling": 1.0 - float(t.kalman_sq.mean()) / persistence,
        "oracle_ceiling": 1.0 - float(t.oracle_var.mean()) / persistence,
        "reverting_share": params.reverting_var / persistence,
    }


def simulate(
    cohort: Cohort, params: StateSpace, rng: np.random.Generator, *, clip: bool = False
) -> FloatArray:
    """One synthetic radius array at the cohort's scan days under ``params``."""
    days = cohort.days
    n_seg, width = days.shape
    drift = params.drift_mean + np.sqrt(params.drift_var) * rng.standard_normal(n_seg)
    state = np.empty((n_seg, width))
    state[:, 0] = np.nanmean(cohort.radius[:, :1], axis=1)
    for k in range(1, width):
        h = np.nan_to_num(days[:, k] - days[:, k - 1])
        step = np.sqrt(params.jump_var + params.diffusion_rate * h)
        state[:, k] = state[:, k - 1] + drift * h + step * rng.standard_normal(n_seg)
    sample = state + np.sqrt(params.reverting_var) * rng.standard_normal((n_seg, width))
    if clip:
        sample = np.clip(sample, 0.0, None)
    return np.where(cohort.mask, sample, np.nan)


def with_params(params: StateSpace, **changes: float) -> StateSpace:
    return replace(params, **changes)


# ---------------------------------------------------------------------------
# Cohort-level analysis: nested tests, patient bootstrap, power
# ---------------------------------------------------------------------------

NESTED = {
    "no_clock": {"diffusion_rate": 0.0},
    "no_patient_rates": {"drift_var": 0.0},
    "no_interval_independent_change": {"jump_var": 0.0},
    "no_reverting_change": {"reverting_var": 0.0},
}
_PER_90_DAYS = 90.0


def summarize_params(params: StateSpace) -> dict[str, float]:
    """Parameters on interpretable scales (mm standard deviations, mm per year)."""
    return {
        "reverting_sd_mm": float(np.sqrt(params.reverting_var)),
        "interval_independent_sd_mm": float(np.sqrt(params.jump_var)),
        "clock_sd_mm_per_90_days": float(np.sqrt(params.diffusion_rate * _PER_90_DAYS)),
        "drift_mean_mm_per_year": float(params.drift_mean * 365.25),
        "drift_sd_mm_per_year": float(np.sqrt(params.drift_var) * 365.25),
    }


def nested_tests(cohort: Cohort, full: Fit, *, starts: int = 4) -> dict[str, dict[str, float]]:
    """Boundary likelihood-ratio test for dropping each variance component."""
    results = {}
    for name, fixed in NESTED.items():
        reduced = fit_state_space(cohort, fixed=fixed, starts=starts)
        results[name] = boundary_lr_test(full, reduced)
    return results


def _subset(cohort: Cohort, rows: NDArray[np.int64]) -> Cohort:
    return Cohort(cohort.days[rows], cohort.radius[rows])


def patient_bootstrap(
    cohort: Cohort,
    groups: Sequence[str],
    *,
    resamples: int,
    rng: np.random.Generator,
    starts: int = 2,
) -> dict[str, dict[str, float]]:
    """Percentile 95% intervals from resampling patients with all their segments."""
    groups = np.asarray(groups)
    patients = np.unique(groups)
    rows_by_patient = {p: np.flatnonzero(groups == p) for p in patients}
    draws: dict[str, list[float]] = {}
    for _ in range(resamples):
        picked = rng.choice(patients, size=patients.size, replace=True)
        sample = _subset(cohort, np.concatenate([rows_by_patient[p] for p in picked]))
        fit = fit_state_space(sample, starts=starts)
        values = {**summarize_params(fit.params), **ceiling(sample, fit.params)}
        for key, value in values.items():
            draws.setdefault(key, []).append(float(value))
    return {
        key: {"ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))}
        for key, v in draws.items()
        if key != "transitions"
    }


def drift_power_curve(
    cohort: Cohort,
    fitted: StateSpace,
    drift_sd_mm_per_year: Sequence[float],
    *,
    simulations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Power to detect between-patient growth-rate variation of a given size.

    Synthetic cohorts keep the observed scan days and the fitted model, with
    the drift standard deviation replaced by each value; the boundary
    likelihood-ratio test for ``drift_var`` is applied to each.
    """
    curve = {}
    for sd_year in drift_sd_mm_per_year:
        params = with_params(fitted, drift_var=float((sd_year / 365.25) ** 2))
        rejected = 0
        for _ in range(simulations):
            sample = cohort.with_radius(simulate(cohort, params, rng))
            full = fit_state_space(sample, starts=1)
            reduced = fit_state_space(sample, fixed={"drift_var": 0.0}, starts=1)
            rejected += boundary_lr_test(full, reduced)["p_value"] < alpha
        curve[f"{sd_year:g}"] = rejected / simulations
    return curve


def clock_power(
    cohort: Cohort,
    fitted: StateSpace,
    *,
    simulations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> float:
    """Power of the no-clock test if all lasting change accumulated with time.

    The fitted interval-independent variance is converted into clock-like
    variance with the same total at the median interval, so the alternative
    has as much lasting change as the data but obeys the growth-model
    assumption. The share of synthetic cohorts in which dropping the clock
    term is rejected is the test's power against that alternative.
    """
    gaps = np.diff(cohort.days, axis=1)
    median_gap = float(np.nanmedian(gaps))
    clock = with_params(
        fitted,
        jump_var=0.0,
        diffusion_rate=fitted.diffusion_rate + fitted.jump_var / median_gap,
    )
    rejected = 0
    for _ in range(simulations):
        sample = cohort.with_radius(simulate(cohort, clock, rng))
        full = fit_state_space(sample, starts=1)
        reduced = fit_state_space(sample, fixed={"diffusion_rate": 0.0}, starts=1)
        rejected += boundary_lr_test(full, reduced)["p_value"] < alpha
    return rejected / simulations


def minimum_detectable(curve: dict[str, float], power: float = 0.8) -> float | None:
    """Smallest drift SD (mm/year) on the curve reaching the requested power."""
    reached = [float(k) for k, v in curve.items() if v >= power]
    return min(reached) if reached else None


def analyze_cohort(
    cohort: Cohort,
    groups: Sequence[str],
    *,
    rng: np.random.Generator,
    resamples: int = 300,
    power_simulations: int = 100,
    power_grid: Sequence[float] = (2.0, 5.0, 10.0, 20.0, 40.0),
) -> dict[str, object]:
    """Full pre-specified analysis for one cohort."""
    full = fit_state_space(cohort, starts=6)
    curve = drift_power_curve(
        cohort, full.params, power_grid, simulations=power_simulations, rng=rng
    )
    return {
        "segments": int(cohort.days.shape[0]),
        "patients": int(np.unique(np.asarray(groups)).size),
        "fit": {**summarize_params(full.params), "loglik": full.loglik},
        "ceiling": ceiling(cohort, full.params),
        "nested_tests": nested_tests(cohort, full),
        "bootstrap_95ci": patient_bootstrap(cohort, groups, resamples=resamples, rng=rng),
        "drift_power_curve": curve,
        "clock_test_power": clock_power(
            cohort, full.params, simulations=power_simulations, rng=rng
        ),
        "minimum_detectable_drift_sd_mm_per_year": minimum_detectable(curve),
    }

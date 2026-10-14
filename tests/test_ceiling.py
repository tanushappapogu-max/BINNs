import numpy as np
import pytest

from gbm_pinn.ceiling import (
    StateSpace,
    boundary_lr_test,
    ceiling,
    fit_state_space,
    kalman,
    simulate,
)
from gbm_pinn.switching import Cohort


def _cohort(rng, segments=150, scans=6):
    gaps = rng.uniform(40.0, 160.0, size=(segments, scans - 1))
    days = np.concatenate([np.zeros((segments, 1)), np.cumsum(gaps, axis=1)], axis=1)
    return Cohort(days, np.full_like(days, 15.0))


TRUTH = StateSpace(
    reverting_var=4.0, jump_var=6.0, diffusion_rate=0.03, drift_mean=0.01, drift_var=1e-4
)


def test_kalman_matches_brute_force_gaussian_likelihood():
    rng = np.random.default_rng(0)
    base = _cohort(rng, segments=1, scans=5)
    y = simulate(base, TRUTH, rng)[0]
    t = base.days[0]
    r, q0, q1 = TRUTH.reverting_var, TRUTH.jump_var, TRUTH.diffusion_rate
    # Given y_0 with a flat prior, x_0 ~ N(y_0, r). Scans 1..n-1 are then jointly
    # Gaussian: shared x_0 error, drift, accumulated lasting change, own noise.
    later = np.arange(1, len(t))
    elapsed = t[later] - t[0]
    gaps = np.diff(t)
    walk = np.array(
        [[sum(q0 + q1 * gaps[s] for s in range(min(i, j))) for j in later] for i in later]
    )
    cov = r + TRUTH.drift_var * np.outer(elapsed, elapsed) + walk + r * np.eye(later.size)
    diff = y[later] - (y[0] + TRUTH.drift_mean * elapsed)
    expected = -0.5 * (np.linalg.slogdet(2 * np.pi * cov)[1] + diff @ np.linalg.solve(cov, diff))
    assert kalman(base.with_radius(y[None]), TRUTH)[0] == pytest.approx(expected, rel=1e-9)


def test_fit_recovers_robust_quantities():
    # Monte Carlo (20 cohorts) showed the variance split between reverting and
    # lasting change is unbiased but noisy, while the total one-step variance
    # and the forecasting ceiling are estimated tightly. Test the latter here.
    rng = np.random.default_rng(1)
    base = _cohort(rng, segments=400)
    cohort = base.with_radius(simulate(base, TRUTH, rng))
    fit = fit_state_space(cohort, starts=3)

    def one_step(p):
        return 2 * p.reverting_var + p.jump_var + 100 * p.diffusion_rate

    assert one_step(fit.params) == pytest.approx(one_step(TRUTH), rel=0.15)
    assert fit.params.drift_mean == pytest.approx(TRUTH.drift_mean, abs=0.005)
    assert ceiling(cohort, fit.params)["history_ceiling"] == pytest.approx(
        ceiling(cohort, TRUTH)["history_ceiling"], abs=0.02
    )
    assert fit.loglik >= kalman(cohort, TRUTH)[0]


def test_boundary_test_detects_present_and_absent_drift_variation():
    rng = np.random.default_rng(2)
    base = _cohort(rng, segments=300)
    present = StateSpace(4.0, 2.0, 0.0, 0.0, 4e-4)
    absent = StateSpace(4.0, 2.0, 0.0, 0.0, 0.0)
    for params, expect_signal in ((present, True), (absent, False)):
        cohort = base.with_radius(simulate(base, params, rng))
        full = fit_state_space(cohort, starts=2)
        reduced = fit_state_space(cohort, fixed={"drift_var": 0.0}, starts=2)
        p = boundary_lr_test(full, reduced)["p_value"]
        assert (p < 0.05) == expect_signal


def test_ceiling_is_near_zero_for_pure_random_walk_and_high_for_pure_noise():
    rng = np.random.default_rng(3)
    base = _cohort(rng, segments=300)
    walk = StateSpace(1e-8, 9.0, 0.0, 0.0, 0.0)
    noise = StateSpace(9.0, 1e-8, 0.0, 0.0, 0.0)
    walk_ceiling = ceiling(base.with_radius(simulate(base, walk, rng)), walk)
    noise_ceiling = ceiling(base.with_radius(simulate(base, noise, rng)), noise)
    assert walk_ceiling["history_ceiling"] == pytest.approx(0.0, abs=0.05)
    # With pure noise the best forecast after k scans is their mean, error
    # r(1 + 1/k), against 2r for persistence; averaged over k = 1..5 that
    # removes 1 - (1 + mean(1, 1/2, 1/3, 1/4, 1/5)) / 2 = 0.27 of the error.
    assert noise_ceiling["history_ceiling"] == pytest.approx(0.27, abs=0.05)
    assert noise_ceiling["oracle_ceiling"] == pytest.approx(0.5, abs=0.05)


def test_nested_tests_power_and_bootstrap_shapes():
    from gbm_pinn.ceiling import (
        drift_power_curve,
        minimum_detectable,
        nested_tests,
        patient_bootstrap,
    )

    rng = np.random.default_rng(7)
    # Intervals must vary widely for interval-independent and clock-like
    # change to be told apart, as they do in clinical follow-up.
    gaps = rng.uniform(14.0, 400.0, size=(150, 4))
    days = np.concatenate([np.zeros((150, 1)), np.cumsum(gaps, axis=1)], axis=1)
    base = Cohort(days, np.full_like(days, 15.0))
    params = StateSpace(4.0, 6.0, 0.0, 0.0, 0.0)
    cohort = base.with_radius(simulate(base, params, rng))
    full = fit_state_space(cohort, starts=2)
    tests = nested_tests(cohort, full, starts=2)
    assert set(tests) == {
        "no_clock",
        "no_patient_rates",
        "no_interval_independent_change",
        "no_reverting_change",
    }
    # Both components that are present are needed; the absent ones are not.
    assert tests["no_reverting_change"]["p_value"] < 0.05
    assert tests["no_interval_independent_change"]["p_value"] < 0.05
    assert tests["no_patient_rates"]["p_value"] > 0.01

    groups = [f"p{i // 2}" for i in range(150)]
    intervals = patient_bootstrap(cohort, groups, resamples=15, rng=rng)
    assert intervals["history_ceiling"]["ci_low"] <= intervals["history_ceiling"]["ci_high"]

    curve = drift_power_curve(cohort, full.params, (0.5, 200.0), simulations=10, rng=rng)
    assert curve["200"] > curve["0.5"]
    assert minimum_detectable({"5": 0.3, "10": 0.85, "20": 1.0}) == 10.0
    assert minimum_detectable({"5": 0.3}) is None


def test_clock_power_is_high_when_intervals_vary_widely():
    from gbm_pinn.ceiling import clock_power

    rng = np.random.default_rng(8)
    gaps = rng.uniform(14.0, 400.0, size=(150, 4))
    days = np.concatenate([np.zeros((150, 1)), np.cumsum(gaps, axis=1)], axis=1)
    base = Cohort(days, np.full_like(days, 15.0))
    params = StateSpace(4.0, 6.0, 0.0, 0.0, 0.0)
    assert clock_power(base, params, simulations=20, rng=rng) >= 0.8

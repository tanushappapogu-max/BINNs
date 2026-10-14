import numpy as np
import pytest

from gbm_pinn.ceiling import StateSpace, simulate
from gbm_pinn.predictive import (
    RANO_RADIUS_FACTOR,
    Predictive,
    coverage,
    crps,
    monte_carlo,
    progression_probability,
)
from gbm_pinn.switching import Cohort

PARAMS = StateSpace(
    reverting_var=4.0, jump_var=2.0, diffusion_rate=0.05, drift_mean=0.005, drift_var=0.0
)


def _cohort(rng, segments=300, scans=6):
    gaps = rng.uniform(14.0, 300.0, size=(segments, scans - 1))
    days = np.concatenate([np.zeros((segments, 1)), np.cumsum(gaps, axis=1)], axis=1)
    return Cohort(days, np.full_like(days, 20.0))


def test_intervals_have_nominal_coverage_when_the_model_is_true():
    rng = np.random.default_rng(0)
    base = _cohort(rng)
    cohort = base.with_radius(simulate(base, PARAMS, rng))
    pred = monte_carlo(cohort, [PARAMS], samples_per_draw=2000, rng=rng)
    for level in (0.5, 0.8, 0.95):
        assert coverage(pred, level) == pytest.approx(level, abs=0.03)


def test_time_blind_intervals_misfit_short_and_long_gaps():
    rng = np.random.default_rng(1)
    base = _cohort(rng)
    cohort = base.with_radius(simulate(base, PARAMS, rng))
    aware = monte_carlo(cohort, [PARAMS], samples_per_draw=1000, rng=rng)
    average = float(np.var(aware.samples, axis=1).mean())
    blind = monte_carlo(cohort, [PARAMS], samples_per_draw=1000, rng=rng, fixed_variance=average)
    assert crps(aware).mean() < crps(blind).mean()


def test_crps_matches_closed_form_for_a_point_forecast():
    samples = np.full((1, 50), 3.0)
    pred = Predictive(
        samples,
        np.array([5.0]),
        np.array([3.0]),
        np.array([3.0]),
        np.array([90.0]),
        np.array([0]),
        np.array([1]),
    )
    # A point forecast's CRPS is its absolute error.
    assert crps(pred)[0] == pytest.approx(2.0)


def test_nadir_tracks_running_minimum_and_probability_is_bounded():
    days = np.array([[0.0, 100.0, 200.0, 300.0]])
    radius = np.array([[10.0, 8.0, 9.0, 12.0]])
    cohort = Cohort(days, radius)
    rng = np.random.default_rng(2)
    pred = monte_carlo(cohort, [PARAMS], samples_per_draw=500, rng=rng)
    assert pred.nadir.tolist() == [10.0, 8.0, 8.0]
    assert pred.position.tolist() == [1, 2, 3]
    p = progression_probability(pred)
    assert np.all((p >= 0) & (p <= 1))
    assert pytest.approx(1.1180, abs=1e-4) == RANO_RADIUS_FACTOR

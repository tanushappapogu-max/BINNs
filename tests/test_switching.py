import numpy as np
import pytest

from gbm_pinn.switching import (
    Cohort,
    bunching,
    bunching_test,
    fit_linear,
    largest_change_is_growth,
    radius_mm,
    rejection_rate,
    simulate_switching,
    split_segments,
)


def test_radius_of_sphere_volume():
    volume_ml = 4.0 / 3.0 * np.pi * 10.0**3 / 1000.0
    assert radius_mm(volume_ml) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        radius_mm(-1.0)


def test_split_segments_cuts_short_intervals_and_sorts():
    segments = split_segments([200, 0, 90, 100, 300], [5, 1, 2, 3, 4], exclusion_days=28)
    days = [list(t) for t, _ in segments]
    assert days == [[0, 90], [100, 200, 300]]
    assert list(segments[1][1]) == [3, 5, 4]


def test_cohort_padding_and_minimum_scans():
    segments = [(np.array([0.0, 1.0]), np.array([1.0, 2.0])), (np.arange(4.0), np.ones(4))]
    cohort = Cohort.from_segments(segments, min_scans=3)
    assert cohort.days.shape == (1, 4)
    with pytest.raises(ValueError):
        Cohort.from_segments(segments[:1], min_scans=3)


def test_fit_linear_recovers_slope_and_zero_noise():
    days = np.array([[0.0, 50.0, 100.0, 200.0]])
    cohort = Cohort(days, 3.0 + 0.02 * days)
    fit = fit_linear(cohort)
    assert fit.slope[0] == pytest.approx(0.02)
    assert fit.sigma == pytest.approx(0.0, abs=1e-12)


def test_bunching_is_interval_share_for_steady_growth_and_one_for_a_jump():
    steady = np.array([[0.0, 1.0, 2.0, 3.0]])
    jump = np.array([[0.0, 0.0, 5.0, 5.0]])
    padded = np.array([[0.0, 1.0, np.nan]])
    assert bunching(steady) == pytest.approx(1.0 / 3.0)
    assert bunching(jump) == pytest.approx(1.0)
    assert bunching(padded) == pytest.approx(1.0)
    assert bool(largest_change_is_growth(jump)[0])


def _synthetic_cohort(rng, segments=60):
    days = np.sort(rng.uniform(0, 600, size=(segments, 5)), axis=1)
    days[:, 0] = 0.0
    return Cohort(days, np.full_like(days, 10.0))


def test_steady_data_is_not_called_bunched():
    rng = np.random.default_rng(1)
    base = _synthetic_cohort(rng)
    radius = 10.0 + 0.01 * base.days + 0.5 * rng.standard_normal(base.days.shape)
    result = bunching_test(base.with_radius(radius), draws=2000, rng=rng)
    assert result.p_value > 0.05


def test_clear_switching_is_detected():
    rng = np.random.default_rng(2)
    base = _synthetic_cohort(rng)
    fit = fit_linear(base.with_radius(10.0 + 0.02 * base.days))
    radius = simulate_switching(base, fit, 1, rng)[0]
    # Strong switch, small noise: refit sees bunching well beyond steady growth.
    result = bunching_test(base.with_radius(radius), draws=2000, rng=rng)
    assert result.p_value < 0.05


def test_type_one_rate_is_controlled():
    rng = np.random.default_rng(3)
    base = _synthetic_cohort(rng, segments=40)
    radius = 10.0 + 0.01 * base.days + 0.5 * rng.standard_normal(base.days.shape)
    rate = rejection_rate(
        base.with_radius(radius), alternative="steady", cohorts=60, draws=300, rng=rng
    )
    assert rate <= 0.15

import numpy as np
import pytest

from gbm_pinn.switching import (
    Cohort,
    Draws,
    bunching,
    bunching_test,
    calibrate,
    fit_linear,
    largest_change_is_growth,
    persistence,
    persistence_verdict,
    pooled_sigma,
    radius_mm,
    rejection_rate,
    separation_auc,
    sign_concordance,
    simulate_outliers,
    simulate_regime_change,
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


def test_persistence_and_sign_concordance_on_known_patterns():
    trend = np.array([[0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 2.5, 2.7]])
    zigzag = np.array([[0.0, 2.0, 0.0, 2.0], [0.0, -1.0, 0.0, -1.0]])
    assert persistence(trend) > 0
    assert persistence(zigzag) == pytest.approx(-1.0)
    assert sign_concordance(trend) == pytest.approx(1.0)
    assert sign_concordance(zigzag) == pytest.approx(-1.0)


def test_pooled_sigma_matches_fit_linear():
    rng = np.random.default_rng(4)
    base = _synthetic_cohort(rng)
    radius = 10.0 + 0.01 * base.days + rng.standard_normal(base.days.shape)
    cohort = base.with_radius(radius)
    assert pooled_sigma(cohort, cohort.radius) == pytest.approx(fit_linear(cohort).sigma)


def test_rivals_predict_opposite_persistence():
    rng = np.random.default_rng(5)
    base = _synthetic_cohort(rng, segments=80)
    fit = fit_linear(base.with_radius(10.0 + 0.01 * base.days))
    draws = Draws.sample(base, 200, rng)
    steady = simulate_outliers(base, fit, draws, rate=0.0, noise=1.0)
    outliers = simulate_outliers(base, fit, draws, rate=0.2, noise=1.0)
    regime = simulate_regime_change(base, fit, draws, amplitude=6.0, noise=1.0)
    assert persistence(outliers).mean() < persistence(steady).mean()
    assert persistence(regime).mean() > persistence(steady).mean()


def test_calibration_hits_noise_and_bunching_targets():
    rng = np.random.default_rng(6)
    base = _synthetic_cohort(rng, segments=80)
    fit = fit_linear(base.with_radius(10.0 + 0.01 * base.days))
    draws = Draws.sample(base, 150, rng)
    truth = simulate_regime_change(base, fit, draws, amplitude=4.0, noise=1.0)
    target_bunching = float(bunching(truth).mean())
    target_sigma = float(pooled_sigma(base, truth).mean())
    result = calibrate(
        base,
        lambda a, s: simulate_regime_change(base, fit, draws, amplitude=a, noise=s),
        np.linspace(0.0, 8.0, 17),
        target_bunching=target_bunching,
        target_sigma=target_sigma,
    )
    assert result.reproduces
    assert result.parameter == pytest.approx(4.0, abs=1.0)
    assert result.noise == pytest.approx(1.0, rel=0.2)


def test_separation_auc_and_verdict():
    high = np.linspace(1.0, 2.0, 50)
    low = np.linspace(-2.0, -1.0, 50)
    assert separation_auc(high, low) == pytest.approx(1.0)
    assert separation_auc(high, high) == pytest.approx(0.5)
    assert persistence_verdict(1.5, high, low)["verdict"] == "switching favored"
    assert persistence_verdict(-1.5, high, low)["verdict"] == "outliers favored"
    assert persistence_verdict(0.0, high, low)["verdict"] == "neither rival explains the data"
    assert persistence_verdict(1.5, high, high)["verdict"].startswith("inconclusive")

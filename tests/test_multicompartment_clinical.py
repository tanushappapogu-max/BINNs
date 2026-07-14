import numpy as np
import pytest

from gbm_pinn.multicompartment_clinical import (
    MultiCompartmentClinicalConfig,
    _latent_seeded_initial_state,
    _metrics,
    _numpy_treatment_exposure,
    _sample_multicompartment_data,
    _scenario_disagreement,
)
from gbm_pinn.treatment import TreatmentWindow


def test_balanced_sampling_keeps_all_present_mri_classes() -> None:
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[1, 1, 1] = 1
    labels[2:4, 2:4, 2:4] = 2
    labels[5:7, 5:7, 5:7] = 3
    brain = np.ones_like(labels, dtype=bool)

    coordinates, targets = _sample_multicompartment_data(
        (labels,),
        brain,
        np.array([0.0]),
        (1.0, 1.0, 1.0),
        40,
        (0.2, 0.3, 0.2, 0.3),
        0.3,
        np.random.default_rng(9),
    )

    assert coordinates.shape == (40, 4)
    assert targets.shape == (40, 3)
    assert np.all(targets.numpy().sum(axis=0) > 0)
    assert np.any(np.all(targets.numpy() == 0, axis=1))


def test_numpy_treatment_exposure_uses_absolute_day_and_decay() -> None:
    exposure = _numpy_treatment_exposure((TreatmentWindow(10.0, 20.0, 2.0, 5.0),), 15.0)

    assert exposure(0.0) == 2.0
    np.testing.assert_allclose(exposure(10.0), 2.0 * np.exp(-1.0))


def test_numpy_treatment_exposure_caps_overlapping_shared_effects() -> None:
    exposure = _numpy_treatment_exposure(
        (TreatmentWindow(0.0, 20.0), TreatmentWindow(5.0, 15.0)), 0.0
    )

    assert exposure(10.0) == 1.0


def test_metrics_reports_skill_against_persistence() -> None:
    target = np.array([[1.0, 1.0], [0.0, 0.0]])
    prediction = target.copy()
    persistence = np.zeros_like(target)

    result = _metrics(prediction, target, persistence, np.ones_like(target, bool), 0.5)

    assert result["forecast_dice"] == 1.0
    assert result["dice_skill_over_persistence"] > 0
    assert result["beats_persistence"] is True


def test_metrics_marks_absent_target_as_unevaluable() -> None:
    empty = np.zeros((2, 2))

    result = _metrics(empty, empty, empty, np.ones_like(empty, bool), 0.5)

    assert result["evaluable"] is False
    assert result["forecast_dice"] is None


def test_scenario_disagreement_does_not_require_future_target() -> None:
    mechanistic = np.array([[1.0, 1.0], [0.0, 0.0]])
    persistence = np.array([[1.0, 0.0], [0.0, 0.0]])

    result = _scenario_disagreement(
        mechanistic, persistence, np.ones_like(mechanistic, bool), 0.5
    )

    assert result["scenario_agreement_dice"] == pytest.approx(2 / 3)
    assert result["disagreement_voxels"] == 1
    assert result["disagreement_fraction_of_brain"] == 0.25


def test_latent_initialization_keeps_only_history_supported_hidden_seeds() -> None:
    first = np.zeros((7, 7, 7), dtype=np.int16)
    latest = np.zeros_like(first)
    first[1, 1, 1] = 2
    latest[5, 5, 5] = 3
    latent = np.full((*first.shape, 3), 0.05, dtype=np.float32)

    viable, edema, necrotic = _latent_seeded_initial_state(
        latent,
        (first, latest),
        np.ones_like(first, bool),
        (1.0, 1.0, 1.0),
        0.3,
        1.5,
        (0.1, 0.1, 0.1),
    )

    np.testing.assert_allclose(viable[1, 1, 1], 0.05)
    assert viable[5, 5, 5] == 1.0
    assert viable[3, 3, 3] == 0.0
    assert edema[3, 3, 3] == 0.0
    assert necrotic[3, 3, 3] == 0.0


def test_hidden_latent_seeds_are_capped_below_mri_detection() -> None:
    labels = np.zeros((3, 3, 3), dtype=np.int16)
    labels[1, 1, 1] = 2
    latest = np.zeros_like(labels)
    latent = np.full((*labels.shape, 3), 0.8, dtype=np.float32)

    viable, edema, necrotic = _latent_seeded_initial_state(
        latent,
        (labels, latest),
        np.ones_like(labels, bool),
        (1.0, 1.0, 1.0),
        0.3,
        2.0,
        (0.1, 0.2, 0.05),
    )

    np.testing.assert_allclose(viable.max(), 0.1)
    np.testing.assert_allclose(edema.max(), 0.2)
    np.testing.assert_allclose(necrotic.max(), 0.05)


def test_latest_cavity_outside_fixed_brain_is_excluded_from_solver_domain() -> None:
    brain = np.zeros((3, 3, 3), dtype=bool)
    brain[1, 1, 1] = True
    latest = np.zeros_like(brain, dtype=np.int16)
    latest[0, 0, 0] = 4

    forward_cavity = (latest == 4) & brain

    assert not forward_cavity.any()


def test_clinical_config_accepts_rolling_scan_origin(tmp_path) -> None:
    config = MultiCompartmentClinicalConfig(
        patient_directory=tmp_path,
        scan_days=(0.0, 7.0, 14.0, 21.0, 28.0, 35.0),
        observation_count=3,
        forecast_index=3,
        scan_start_index=2,
    )

    assert config.scan_start_index == 2


def test_clinical_config_rejects_rolling_origin_without_forecast(tmp_path) -> None:
    with pytest.raises(ValueError, match="held-out forecast"):
        MultiCompartmentClinicalConfig(
            patient_directory=tmp_path,
            scan_days=(0.0, 7.0, 14.0, 21.0, 28.0),
            observation_count=3,
            forecast_index=3,
            scan_start_index=2,
        )

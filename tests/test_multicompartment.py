import numpy as np
import pytest

from gbm_pinn.multicompartment import MultiCompartmentParameters, mri_surrogate_channels


def test_treatment_death_transfers_viable_density_to_necrosis() -> None:
    parameters = MultiCompartmentParameters(
        proliferation_rate=0.0,
        edema_generation_rate=0.0,
        edema_clearance_rate=0.0,
        necrosis_clearance_rate=0.0,
    )

    viable_dt, edema_dt, necrotic_dt = parameters.reaction(
        viable=np.array([0.5]),
        edema=np.array([0.2]),
        necrotic=np.array([0.1]),
        treatment_cell_kill_rate=0.04,
    )

    np.testing.assert_allclose(viable_dt, [-0.02])
    np.testing.assert_allclose(edema_dt, [0.0])
    np.testing.assert_allclose(necrotic_dt, [0.02])
    np.testing.assert_allclose(viable_dt + necrotic_dt, [0.0])


def test_antiangiogenic_treatment_suppresses_edema_generation() -> None:
    parameters = MultiCompartmentParameters(
        proliferation_rate=0.02,
        edema_generation_rate=0.001,
        edema_clearance_rate=0.01,
        necrosis_clearance_rate=0.005,
    )

    viable_dt, edema_dt, _ = parameters.reaction(
        viable=0.2,
        edema=0.8,
        necrotic=0.1,
        treatment_edema_leakage_suppression=1.0,
    )

    assert viable_dt > 0
    assert edema_dt < 0


def test_spontaneous_necrosis_transfers_crowded_viable_density() -> None:
    parameters = MultiCompartmentParameters(
        proliferation_rate=0.0,
        edema_generation_rate=0.0,
        edema_clearance_rate=0.0,
        necrosis_clearance_rate=0.0,
        spontaneous_necrosis_rate=0.04,
        necrosis_threshold=0.6,
        necrosis_transition_width=0.01,
    )

    viable_dt, _, necrotic_dt = parameters.reaction(0.7, 0.0, 0.0)

    assert viable_dt < -0.027
    assert necrotic_dt > 0.027
    np.testing.assert_allclose(viable_dt + necrotic_dt, 0.0)


def test_edema_source_is_saturating_in_viable_density() -> None:
    parameters = MultiCompartmentParameters(
        proliferation_rate=0.0,
        edema_generation_rate=0.03,
        edema_clearance_rate=0.0,
        necrosis_clearance_rate=0.0,
        edema_half_saturation=0.1,
    )

    low_source = parameters.reaction(0.1, 0.0, 0.0)[1]
    high_source = parameters.reaction(0.9, 0.0, 0.0)[1]

    np.testing.assert_allclose(low_source, 0.015)
    assert high_source < 2 * low_source


def test_mri_surrogate_keeps_edema_distinct_from_viable_density() -> None:
    channels = mri_surrogate_channels(viable=np.array([0.2]), edema=0.5, necrotic=0.1)

    np.testing.assert_allclose(channels["enhancing_tissue"], [0.2])
    np.testing.assert_allclose(channels["flair_abnormality"], [0.6])
    np.testing.assert_allclose(channels["nonenhancing_or_necrotic_core"], [0.1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"proliferation_rate": -0.1},
        {"edema_generation_rate": -0.1},
        {"edema_clearance_rate": -0.1},
        {"necrosis_clearance_rate": -0.1},
        {"edema_half_saturation": 0.0},
        {"spontaneous_necrosis_rate": -0.1},
        {"necrosis_transition_width": 0.0},
        {"carrying_capacity": 0.0},
    ],
)
def test_multicompartment_parameters_reject_nonphysical_values(kwargs) -> None:
    values = {
        "proliferation_rate": 0.01,
        "edema_generation_rate": 0.01,
        "edema_clearance_rate": 0.01,
        "necrosis_clearance_rate": 0.01,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        MultiCompartmentParameters(**values)

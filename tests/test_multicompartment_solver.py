import numpy as np

from gbm_pinn.multicompartment import MultiCompartmentParameters
from gbm_pinn.multicompartment_solver import MultiCompartmentSolver, RadiationFraction


def test_uniform_treatment_moves_viable_density_into_necrosis() -> None:
    shape = (8, 7)
    mask = np.ones(shape, dtype=bool)
    parameters = MultiCompartmentParameters(
        proliferation_rate=0.0,
        edema_generation_rate=0.0,
        edema_clearance_rate=0.0,
        necrosis_clearance_rate=0.0,
    )
    solver = MultiCompartmentSolver(
        np.zeros(shape),
        np.zeros(shape),
        mask,
        parameters,
    )

    result = solver.simulate(
        np.full(shape, 0.5),
        np.zeros(shape),
        np.zeros(shape),
        np.array([0.1]),
        maximum_time_step=0.1,
        treatment_cell_kill=lambda _: 0.04,
    )

    np.testing.assert_allclose(result.viable[0], 0.498)
    np.testing.assert_allclose(result.necrotic[0], 0.002)
    np.testing.assert_allclose(result.viable[0] + result.necrotic[0], 0.5)


def test_cavity_remains_empty_for_every_compartment() -> None:
    shape = (10, 9)
    mask = np.ones(shape, dtype=bool)
    cavity = np.zeros(shape, dtype=bool)
    cavity[4:6, 4:6] = True
    parameters = MultiCompartmentParameters(0.01, 0.01, 0.01, 0.01)
    solver = MultiCompartmentSolver(
        np.full(shape, 0.1),
        np.full(shape, 0.2),
        mask,
        parameters,
        cavity_mask=cavity,
    )

    result = solver.simulate(
        np.full(shape, 0.2),
        np.full(shape, 0.3),
        np.full(shape, 0.1),
        np.array([1.0]),
    )

    assert np.all(result.viable[0][cavity] == 0)
    assert np.all(result.edema[0][cavity] == 0)
    assert np.all(result.necrotic[0][cavity] == 0)


def test_edema_diffuses_with_no_flux_mass_conservation_without_reaction() -> None:
    shape = (12, 11)
    mask = np.ones(shape, dtype=bool)
    parameters = MultiCompartmentParameters(0.0, 0.0, 0.0, 0.0)
    solver = MultiCompartmentSolver(
        np.zeros(shape),
        np.full(shape, 0.2),
        mask,
        parameters,
    )
    initial_edema = np.zeros(shape)
    initial_edema[5, 5] = 0.5

    result = solver.simulate(
        np.zeros(shape),
        initial_edema,
        np.zeros(shape),
        np.array([0.5]),
    )

    assert result.edema[0, 5, 5] < initial_edema[5, 5]
    np.testing.assert_allclose(result.edema[0].sum(), initial_edema.sum(), atol=1e-12)


def test_radiation_fraction_applies_linear_quadratic_survival_as_jump() -> None:
    shape = (5, 4)
    mask = np.ones(shape, dtype=bool)
    parameters = MultiCompartmentParameters(0.0, 0.0, 0.0, 0.0)
    solver = MultiCompartmentSolver(np.zeros(shape), np.zeros(shape), mask, parameters)
    initial_viable = np.full(shape, 0.5)
    survival = np.exp(-(0.2 * 2.0 + 0.03 * 2.0**2))

    result = solver.simulate(
        initial_viable,
        np.zeros(shape),
        np.zeros(shape),
        np.array([0.5, 1.0]),
        radiation_fractions=(RadiationFraction(0.75, 2.0, 0.2, 0.03),),
    )

    np.testing.assert_allclose(result.viable[0], initial_viable)
    np.testing.assert_allclose(result.viable[1], initial_viable * survival)
    np.testing.assert_allclose(result.necrotic[1], initial_viable * (1.0 - survival))
    np.testing.assert_allclose(result.viable[1] + result.necrotic[1], initial_viable)

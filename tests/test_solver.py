import numpy as np

from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.solver import FiniteVolumeSolver
from gbm_pinn.synthetic import gaussian_density


def make_solver(
    shape: tuple[int, int] = (24, 24),
    *,
    diffusivity: float = 0.1,
    proliferation_rate: float = 0.0,
    cavity_mask: np.ndarray | None = None,
) -> FiniteVolumeSolver:
    return FiniteVolumeSolver(
        diffusivity=np.full(shape, diffusivity),
        brain_mask=np.ones(shape, dtype=bool),
        cavity_mask=cavity_mask,
        parameters=ReactionDiffusionParameters(proliferation_rate=proliferation_rate),
    )


def test_uniform_density_is_diffusive_equilibrium() -> None:
    solver = make_solver()
    density = np.full((24, 24), 0.35)

    derivative = solver.spatial_derivative(density)

    np.testing.assert_allclose(derivative, 0.0)


def test_zero_flux_diffusion_conserves_total_mass() -> None:
    solver = make_solver(diffusivity=0.2)
    density = gaussian_density((24, 24), standard_deviation=2.0)

    result = solver.simulate(density, np.array([0.0, 1.0, 3.0]))

    total_mass = result.density.sum(axis=(1, 2))
    np.testing.assert_allclose(total_mass, total_mass[0], rtol=1e-12, atol=1e-12)


def test_uniform_logistic_growth_matches_analytical_solution() -> None:
    rate = 0.08
    initial_value = 0.1
    solver = make_solver(diffusivity=0.0, proliferation_rate=rate)
    times = np.array([0.0, 2.0, 5.0])

    result = solver.simulate(np.full((24, 24), initial_value), times, maximum_time_step=0.002)

    expected = 1.0 / (1.0 + ((1.0 - initial_value) / initial_value) * np.exp(-rate * times))
    np.testing.assert_allclose(result.density[:, 0, 0], expected, rtol=2e-5, atol=2e-6)


def test_weak_allee_growth_is_slower_than_logistic_growth_at_low_density() -> None:
    shape = (24, 24)
    density = np.full(shape, 0.05)
    logistic = make_solver(diffusivity=0.0, proliferation_rate=0.08)
    weak_allee = FiniteVolumeSolver(
        diffusivity=np.zeros(shape),
        brain_mask=np.ones(shape, dtype=bool),
        parameters=ReactionDiffusionParameters(
            proliferation_rate=0.08,
            growth_law="weak_allee",
            allee_parameter=0.05,
        ),
    )

    logistic_result = logistic.simulate(density, np.array([10.0]), maximum_time_step=0.01)
    allee_result = weak_allee.simulate(density, np.array([10.0]), maximum_time_step=0.01)

    assert np.all(allee_result.density[-1] < logistic_result.density[-1])


def test_cavity_remains_empty() -> None:
    cavity = np.zeros((24, 24), dtype=bool)
    cavity[10:14, 10:14] = True
    solver = make_solver(diffusivity=0.2, proliferation_rate=0.1, cavity_mask=cavity)
    density = gaussian_density((24, 24), standard_deviation=3.0)

    result = solver.simulate(density, np.array([0.0, 2.0]))

    np.testing.assert_array_equal(result.density[:, cavity], 0.0)


def test_treatment_reduces_uniform_density() -> None:
    solver = make_solver(diffusivity=0.0, proliferation_rate=0.05)
    density = np.full((24, 24), 0.2)

    untreated = solver.simulate(density, np.array([2.0]), maximum_time_step=0.01)
    treated = solver.simulate(
        density,
        np.array([2.0]),
        maximum_time_step=0.01,
        treatment=lambda _time: 0.2,
    )

    assert np.all(treated.density[-1] < untreated.density[-1])

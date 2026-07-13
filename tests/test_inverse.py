import numpy as np
import pytest

from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.inverse import fit_homogeneous_parameters
from gbm_pinn.solver import FiniteVolumeSolver
from gbm_pinn.synthetic import gaussian_density


def test_recovers_known_homogeneous_parameters() -> None:
    shape = (16, 16)
    true_diffusivity = 0.12
    true_proliferation = 0.06
    brain_mask = np.ones(shape, dtype=bool)
    initial_density = gaussian_density(shape, standard_deviation=2.0, peak=0.4)
    observation_times = np.array([1.0, 2.5, 4.0])
    solver = FiniteVolumeSolver(
        diffusivity=np.full(shape, true_diffusivity),
        brain_mask=brain_mask,
        parameters=ReactionDiffusionParameters(proliferation_rate=true_proliferation),
    )
    observations = solver.simulate(initial_density, observation_times).density

    estimate = fit_homogeneous_parameters(
        initial_density,
        observation_times,
        observations,
        brain_mask,
        diffusivity_bounds=(0.02, 0.3),
        proliferation_bounds=(0.01, 0.15),
        initial_guess=(0.2, 0.1),
    )

    assert estimate.converged
    assert estimate.diffusivity == pytest.approx(true_diffusivity, rel=1e-4)
    assert estimate.proliferation_rate == pytest.approx(true_proliferation, rel=1e-4)
    assert estimate.residual_sum_squares < 1e-12


def test_rejects_observation_mask_outside_brain() -> None:
    shape = (8, 8)
    brain_mask = np.ones(shape, dtype=bool)
    brain_mask[0, 0] = False
    observation_mask = np.ones(shape, dtype=bool)

    with pytest.raises(ValueError, match="contained within brain_mask"):
        fit_homogeneous_parameters(
            np.zeros(shape),
            np.array([1.0]),
            np.zeros((1, *shape)),
            brain_mask,
            diffusivity_bounds=(0.01, 0.2),
            proliferation_bounds=(0.01, 0.2),
            observation_mask=observation_mask,
        )

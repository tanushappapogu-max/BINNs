import numpy as np
import pytest
import torch

from gbm_pinn.experiment import (
    SyntheticExperimentConfig,
    _cavity_mask,
    _dice_at_threshold,
    _relative_volume_error,
    _sample_boundaries,
    _sample_collocation,
    _sample_interface,
)


def test_overlap_metrics_for_identical_masks() -> None:
    density = np.array([[0.0, 0.2], [0.3, 0.0]])

    assert _dice_at_threshold(density, density, 0.1) == 1.0
    assert _relative_volume_error(density, density, 0.1) == 0.0


def test_overlap_metrics_for_disjoint_masks() -> None:
    prediction = np.array([[0.2, 0.0], [0.0, 0.0]])
    target = np.array([[0.0, 0.0], [0.0, 0.2]])

    assert _dice_at_threshold(prediction, target, 0.1) == 0.0
    assert _relative_volume_error(prediction, target, 0.1) == 0.0


def test_experiment_configuration_requires_ordered_times() -> None:
    with pytest.raises(ValueError, match="observation_times"):
        SyntheticExperimentConfig(observation_times=(0.0, 1.0), forecast_time=1.0)


def test_experiment_configuration_rejects_negative_noise() -> None:
    with pytest.raises(ValueError, match="observation_noise_standard_deviation"):
        SyntheticExperimentConfig(observation_noise_standard_deviation=-0.01)


def test_experiment_configuration_rejects_tissue_ratio_below_one() -> None:
    with pytest.raises(ValueError, match="white_to_gray_diffusivity_ratio"):
        SyntheticExperimentConfig(white_to_gray_diffusivity_ratio=0.9)


def test_experiment_configuration_rejects_allee_parameter_for_logistic_growth() -> None:
    with pytest.raises(ValueError, match="allee_parameter"):
        SyntheticExperimentConfig(allee_parameter=0.1)


def test_experiment_configuration_rejects_fitted_allee_parameter_without_allee_law() -> None:
    with pytest.raises(ValueError, match="fitted_allee_parameter"):
        SyntheticExperimentConfig(fitted_allee_parameter=0.1)


def test_experiment_configuration_rejects_initial_parameter_outside_bounds() -> None:
    with pytest.raises(ValueError, match="initial_proliferation_rate"):
        SyntheticExperimentConfig(
            proliferation_bounds=(0.05, 0.3),
            initial_proliferation_rate=0.3,
        )


def test_cavity_is_excluded_from_collocation_points() -> None:
    config = SyntheticExperimentConfig(cavity_radius=0.2, collocation_points=100)

    coordinates = _sample_collocation(config, np.random.default_rng(3)).numpy()

    squared_distance = (coordinates[:, 0] - 0.5) ** 2 + (coordinates[:, 1] - 0.5) ** 2
    assert np.all(squared_distance > 0.2**2)


def test_cavity_mask_contains_domain_center() -> None:
    config = SyntheticExperimentConfig(grid_size=11, cavity_radius=0.2)
    grid = np.linspace(0.0, 1.0, 11)

    mask = _cavity_mask(grid, config)

    assert mask[5, 5]
    assert not mask[0, 0]


def test_piecewise_interface_samples_lie_on_configured_circle() -> None:
    config = SyntheticExperimentConfig(
        cavity_radius=0.15,
        cavity_interface_radius=0.3,
        interface_points=40,
    )

    coordinates, normals = _sample_interface(config, np.random.default_rng(6))

    assert coordinates is not None
    assert normals is not None
    radial_distance = np.sqrt(
        (coordinates[:, 0].numpy() - 0.5) ** 2 + (coordinates[:, 1].numpy() - 0.5) ** 2
    )
    np.testing.assert_allclose(radial_distance, 0.3, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(normals.numpy(), axis=1), 1.0, atol=1e-7)


def test_interface_requires_positive_cavity_radius() -> None:
    with pytest.raises(ValueError, match="cavity_radius"):
        SyntheticExperimentConfig(cavity_interface_radius=0.3)


def test_causal_sample_pools_include_initial_time() -> None:
    config = SyntheticExperimentConfig(
        cavity_radius=0.15,
        cavity_interface_radius=0.3,
        collocation_points=5,
        boundary_points=5,
        interface_points=5,
    )
    rng = np.random.default_rng(9)

    collocation = _sample_collocation(config, rng)
    boundary, _ = _sample_boundaries(config, rng)
    interface, _ = _sample_interface(config, rng)

    assert torch.any(collocation[:, 2] == 0.0)
    assert torch.any(boundary[:, 2] == 0.0)
    assert interface is not None
    assert torch.any(interface[:, 2] == 0.0)

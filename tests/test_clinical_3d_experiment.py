import numpy as np

from gbm_pinn.clinical_3d_experiment import (
    _sample_volume_boundary,
    _sample_volume_data,
    _sample_volume_interior,
)


def test_three_dimensional_sampling_builds_finite_space_time_coordinates() -> None:
    brain = np.zeros((12, 11, 10), dtype=bool)
    brain[1:11, 1:10, 1:9] = True
    first = np.zeros_like(brain, dtype=np.int16)
    second = np.zeros_like(first)
    first[4:7, 4:7, 4:7] = 2
    second[3:8, 3:8, 3:8] = 3
    rng = np.random.default_rng(8)

    data, density = _sample_volume_data(
        (first, second),
        brain,
        np.array([0.0, 30.0]),
        (1.0, 1.0, 1.0),
        0.3,
        40,
        0.25,
        rng,
    )
    collocation = _sample_volume_interior(brain, (1.0, 1.0, 1.0), 60.0, 50, rng)
    boundary, normals = _sample_volume_boundary(brain, (1.0, 1.0, 1.0), 60.0, 30, rng)

    assert data.shape == (80, 4)
    assert density.shape == (80, 1)
    assert collocation.shape == (50, 4)
    assert boundary.shape == (30, 4)
    assert normals.shape == (30, 3)
    assert np.all(np.isfinite(normals.numpy()))
    np.testing.assert_allclose(np.linalg.norm(normals.numpy(), axis=1), 1.0, atol=1e-6)

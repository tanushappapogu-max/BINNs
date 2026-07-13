import numpy as np
import pytest

from gbm_pinn.experiment import (
    SyntheticExperimentConfig,
    _dice_at_threshold,
    _relative_volume_error,
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

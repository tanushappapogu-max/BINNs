import numpy as np
import pytest
import torch

from gbm_pinn.shared_forecaster import (
    PreparedTransition,
    SharedResidualForecaster,
    dice_score,
    stratified_sample_indices,
    uniform_sample_indices,
)


def test_residual_forecaster_returns_one_logit_per_voxel() -> None:
    model = SharedResidualForecaster(5, hidden_width=8)

    logits = model(torch.zeros((7, 5)), torch.zeros(7))

    assert logits.shape == (7, 1)


def test_stratified_sampling_is_reproducible_and_fixed_size() -> None:
    transition = PreparedTransition(
        "T1",
        "P1",
        np.zeros((8, 2), np.float32),
        np.array([0, 0, 0, 1, 1, 1, 0, 0], np.float32),
        np.array([0, 1, 0, 1, 0, 1, 0, 0], np.float32),
        (2, 2, 2),
    )

    first = stratified_sample_indices(transition, 20, np.random.default_rng(4))
    second = stratified_sample_indices(transition, 20, np.random.default_rng(4))

    assert np.array_equal(first, second)
    assert first.size == 20


def test_uniform_sampling_does_not_duplicate_when_population_is_large_enough() -> None:
    transition = PreparedTransition(
        "T1",
        "P1",
        np.zeros((8, 2), np.float32),
        np.zeros(8, np.float32),
        np.zeros(8, np.float32),
        (2, 2, 2),
    )

    indices = uniform_sample_indices(transition, 6, np.random.default_rng(2))

    assert len(np.unique(indices)) == 6


@pytest.mark.parametrize(
    ("prediction", "target", "expected"),
    [([0, 0], [0, 0], 1.0), ([1, 0], [1, 1], 2 / 3)],
)
def test_dice_score(prediction, target, expected) -> None:
    assert dice_score(np.array(prediction), np.array(target)) == pytest.approx(expected)

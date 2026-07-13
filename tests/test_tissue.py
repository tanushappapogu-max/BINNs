import numpy as np
import pytest
import torch

from gbm_pinn.pinn import PINNConfig
from gbm_pinn.tissue import TissueAwareTumorPINN


def make_tissue_model() -> TissueAwareTumorPINN:
    probability = np.tile(np.linspace(0.0, 1.0, 5), (5, 1))
    return TissueAwareTumorPINN(
        coordinate_lower_bounds=torch.tensor([0.0, 0.0, 0.0]),
        coordinate_upper_bounds=torch.tensor([1.0, 1.0, 2.0]),
        white_matter_probability=probability,
        white_to_gray_diffusivity_ratio=4.0,
        config=PINNConfig(
            hidden_width=8,
            hidden_layers=2,
            diffusivity_bounds=(0.01, 0.2),
            initial_diffusivity=0.05,
            proliferation_bounds=(0.01, 0.2),
            initial_proliferation_rate=0.1,
        ),
    )


def test_tissue_probability_interpolates_diffusivity() -> None:
    model = make_tissue_model()
    coordinates = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.5, 1.0]])

    diffusivity = model.diffusivity_at(coordinates).detach().ravel()

    assert float(diffusivity[0]) == pytest.approx(0.05)
    assert float(diffusivity[1]) == pytest.approx(0.2)


def test_tissue_diffusivity_gradient_matches_linear_map() -> None:
    model = make_tissue_model()
    coordinates = torch.tensor([[0.25, 0.5, 1.0], [0.75, 0.5, 1.0]])

    gradient = model.diffusivity_gradient_at(coordinates).detach()

    torch.testing.assert_close(gradient[:, 0], torch.full((2,), 0.15))
    torch.testing.assert_close(gradient[:, 1], torch.zeros(2))


def test_tissue_map_rejects_invalid_probability() -> None:
    probability = np.zeros((5, 5))
    probability[2, 2] = 1.1

    with pytest.raises(ValueError, match="values in"):
        TissueAwareTumorPINN(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 2.0]),
            probability,
            white_to_gray_diffusivity_ratio=4.0,
        )

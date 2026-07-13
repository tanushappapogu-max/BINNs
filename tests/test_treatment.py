import pytest
import torch

from gbm_pinn.pinn import PINNConfig, TrainingConfig, TumorPINN, fit_pinn, pde_residual
from gbm_pinn.treatment import TreatmentAwareTumorPINN, TreatmentWindow


def test_treatment_window_has_active_and_decaying_exposure() -> None:
    window = TreatmentWindow(start_day=10.0, end_day=20.0, intensity=2.0, decay_days=5.0)

    exposure = window.exposure_at(torch.tensor([[5.0], [15.0], [20.0], [25.0]]))

    assert exposure[0].item() == 0.0
    assert exposure[1].item() == 2.0
    assert exposure[2].item() == 2.0
    assert exposure[3].item() == pytest.approx(2.0 * torch.exp(torch.tensor(-1.0)).item())


def test_treatment_aware_model_adds_loss_to_pde_residual() -> None:
    torch.manual_seed(4)
    config = PINNConfig(
        hidden_width=8,
        hidden_layers=2,
        initial_diffusivity=0.1,
        initial_proliferation_rate=0.1,
    )
    untreated = TumorPINN(torch.zeros(3), torch.ones(3), config)
    treated = TreatmentAwareTumorPINN(
        torch.zeros(3),
        torch.ones(3),
        (TreatmentWindow(0.0, 1.0),),
        config,
        treatment_response_bounds=(0.0, 0.2),
        initial_treatment_response=0.05,
    )
    treated.network.load_state_dict(untreated.network.state_dict())
    coordinates = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.8, 0.6]])

    expected_difference = treated.treatment_rate_at(coordinates) * treated(coordinates)

    torch.testing.assert_close(
        pde_residual(treated, coordinates) - pde_residual(untreated, coordinates),
        expected_difference,
    )


def test_fit_pinn_optimizes_treatment_response_parameter() -> None:
    torch.manual_seed(7)
    model = TreatmentAwareTumorPINN(
        torch.zeros(3),
        torch.ones(3),
        (TreatmentWindow(0.0, 1.0),),
        PINNConfig(hidden_width=8, hidden_layers=2),
    )
    coordinates = torch.rand(24, 3)
    density = torch.zeros(24, 1)
    initial = model.raw_treatment_response.detach().clone()

    fit_pinn(
        model,
        coordinates,
        density,
        coordinates,
        config=TrainingConfig(epochs=2),
    )

    assert not torch.equal(model.raw_treatment_response.detach(), initial)

import pytest
import torch

from gbm_pinn.cavity import CavityAwareTumorPINN, PiecewiseCavityTumorPINN
from gbm_pinn.pinn import PINNConfig, TrainingConfig, fit_pinn, normal_flux


def make_cavity_model() -> CavityAwareTumorPINN:
    return CavityAwareTumorPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 2.0]),
        cavity_center=(0.5, 0.5),
        cavity_radius=0.2,
        config=PINNConfig(hidden_width=8, hidden_layers=2),
    )


def test_cavity_model_uses_squared_radial_feature() -> None:
    model = make_cavity_model()
    coordinates = torch.tensor([[0.7, 0.5, 1.0], [0.9, 0.5, 1.0]])

    features = model.coordinate_features(coordinates)

    assert features.shape == (2, 4)
    assert float(features[0, 2]) == pytest.approx(-1.0, abs=1e-7)
    assert float(features[1, 2]) > -1.0


def test_cavity_model_predicts_valid_density() -> None:
    model = make_cavity_model()

    density = model(torch.rand(10, 3))

    assert density.shape == (10, 1)
    assert torch.all((density >= 0.0) & (density <= 1.0))


def test_cavity_feature_map_enforces_zero_normal_flux() -> None:
    model = make_cavity_model()
    angles = torch.linspace(0.0, 2.0 * torch.pi, 9)[:-1]
    radial = torch.column_stack((torch.cos(angles), torch.sin(angles)))
    coordinates = torch.column_stack(
        (
            0.5 + 0.2 * radial[:, 0],
            0.5 + 0.2 * radial[:, 1],
            torch.ones(angles.shape[0]),
        )
    )

    flux = normal_flux(model, coordinates, -radial)

    torch.testing.assert_close(flux, torch.zeros_like(flux), atol=1e-7, rtol=0.0)


def make_piecewise_model() -> PiecewiseCavityTumorPINN:
    return PiecewiseCavityTumorPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 2.0]),
        cavity_center=(0.5, 0.5),
        cavity_radius=0.2,
        interface_radius=0.35,
        config=PINNConfig(hidden_width=8, hidden_layers=2),
    )


def test_piecewise_model_has_two_neural_fields() -> None:
    model = make_piecewise_model()

    field_parameters = list(model.field_parameters())

    assert len(field_parameters) == len(list(model.network.parameters())) + len(
        list(model.near_network.parameters())
    )


def test_piecewise_near_field_enforces_zero_cavity_flux() -> None:
    model = make_piecewise_model()
    angles = torch.linspace(0.0, 2.0 * torch.pi, 9)[:-1]
    radial = torch.column_stack((torch.cos(angles), torch.sin(angles)))
    coordinates = torch.column_stack(
        (
            0.5 + 0.2 * radial[:, 0],
            0.5 + 0.2 * radial[:, 1],
            torch.ones(angles.shape[0]),
        )
    )

    flux = normal_flux(model, coordinates, -radial)

    torch.testing.assert_close(flux, torch.zeros_like(flux), atol=1e-6, rtol=0.0)


def test_piecewise_interface_residual_has_expected_shape() -> None:
    model = make_piecewise_model()
    angles = torch.linspace(0.0, 2.0 * torch.pi, 9)[:-1]
    normals = torch.column_stack((torch.cos(angles), torch.sin(angles)))
    coordinates = torch.column_stack(
        (
            0.5 + 0.35 * normals[:, 0],
            0.5 + 0.35 * normals[:, 1],
            torch.ones(angles.shape[0]),
        )
    )

    density_jump, flux_jump = model.interface_residual(coordinates, normals)

    assert density_jump.shape == (8, 1)
    assert flux_jump.shape == (8, 1)
    assert torch.all(torch.isfinite(density_jump))
    assert torch.all(torch.isfinite(flux_jump))


def test_piecewise_training_includes_interface_loss() -> None:
    torch.manual_seed(12)
    model = make_piecewise_model()
    data_coordinates = torch.tensor(
        [[0.75, 0.5, 0.0], [0.9, 0.5, 1.0], [0.5, 0.9, 2.0]], dtype=torch.float32
    )
    data_density = torch.full((3, 1), 0.2)
    collocation_coordinates = torch.tensor(
        [[0.72, 0.5, 0.5], [0.85, 0.5, 1.0], [0.5, 0.88, 1.5]], dtype=torch.float32
    )
    angles = torch.linspace(0.0, 2.0 * torch.pi, 9)[:-1]
    interface_normals = torch.column_stack((torch.cos(angles), torch.sin(angles)))
    interface_coordinates = torch.column_stack(
        (
            0.5 + 0.35 * interface_normals[:, 0],
            0.5 + 0.35 * interface_normals[:, 1],
            torch.ones(angles.shape[0]),
        )
    )

    result = fit_pinn(
        model,
        data_coordinates,
        data_density,
        collocation_coordinates,
        interface_coordinates=interface_coordinates,
        interface_normals=interface_normals,
        config=TrainingConfig(epochs=2),
    )

    assert len(result.interface_loss) == 2
    assert all(torch.isfinite(torch.tensor(result.interface_loss)))

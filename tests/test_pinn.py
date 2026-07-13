import pytest
import torch
from torch import nn

from gbm_pinn.pinn import (
    PINNConfig,
    TrainingConfig,
    TumorPINN,
    fit_pinn,
    pde_residual,
    resolve_torch_device,
)


def make_model() -> TumorPINN:
    return TumorPINN(
        coordinate_lower_bounds=torch.tensor([0.0, 0.0, 0.0]),
        coordinate_upper_bounds=torch.tensor([1.0, 1.0, 2.0]),
        config=PINNConfig(
            hidden_width=8,
            hidden_layers=2,
            diffusivity_bounds=(0.01, 0.3),
            proliferation_bounds=(0.01, 0.2),
            initial_diffusivity=0.1,
            initial_proliferation_rate=0.05,
        ),
    )


def test_output_and_parameters_stay_in_physical_ranges() -> None:
    model = make_model()
    coordinates = torch.rand(12, 3)
    coordinates[:, 2] *= 2.0

    density = model(coordinates)

    assert density.shape == (12, 1)
    assert torch.all((density >= 0.0) & (density <= 1.0))
    assert 0.01 < float(model.diffusivity.detach()) < 0.3
    assert 0.01 < float(model.proliferation_rate.detach()) < 0.2


class AnalyticSolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("diffusivity", torch.tensor(0.25))
        self.register_buffer("proliferation_rate", torch.tensor(0.0))
        self.config = PINNConfig(
            diffusivity_bounds=(0.1, 0.4),
            proliferation_bounds=(0.0, 0.2),
            initial_diffusivity=0.25,
            initial_proliferation_rate=0.1,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        time = coordinates[:, 2:3]
        return x**2 + y**2 + time

    def diffusivity_at(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.diffusivity.expand(coordinates.shape[0], 1)

    def diffusivity_gradient_at(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.zeros((coordinates.shape[0], 2))


def test_pde_residual_matches_analytic_solution() -> None:
    coordinates = torch.rand(10, 3)

    residual = pde_residual(AnalyticSolution(), coordinates)

    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=0.0)


class VariableDiffusivitySolution(AnalyticSolution):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, 0:1] ** 2

    def diffusivity_at(self, coordinates: torch.Tensor) -> torch.Tensor:
        return 1.0 + coordinates[:, 0:1]

    def diffusivity_gradient_at(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.column_stack(
            (torch.ones(coordinates.shape[0]), torch.zeros(coordinates.shape[0]))
        )


def test_pde_residual_includes_diffusivity_gradient() -> None:
    coordinates = torch.rand(10, 3)

    residual = pde_residual(VariableDiffusivitySolution(), coordinates)

    expected = -2.0 - 4.0 * coordinates[:, 0:1]
    torch.testing.assert_close(residual, expected, atol=1e-6, rtol=0.0)


class WeakAlleeConstantSolution(AnalyticSolution):
    def __init__(self) -> None:
        super().__init__()
        self.diffusivity.zero_()
        self.proliferation_rate.fill_(0.4)
        self.config = PINNConfig(
            diffusivity_bounds=(0.1, 0.4),
            proliferation_bounds=(0.0, 0.5),
            initial_diffusivity=0.25,
            initial_proliferation_rate=0.4,
            growth_law="weak_allee",
            allee_parameter=0.1,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, 0:1] ** 2 + 0.5


def test_pde_residual_uses_weak_allee_reaction() -> None:
    coordinates = torch.rand(10, 3)
    coordinates[:, 0] = 0.0

    residual = pde_residual(WeakAlleeConstantSolution(), coordinates)

    torch.testing.assert_close(residual, torch.full_like(residual, -0.06))


def test_training_smoke_test_returns_finite_history() -> None:
    torch.manual_seed(4)
    model = make_model()
    data_coordinates = torch.rand(16, 3)
    data_coordinates[:, 2] *= 2.0
    data_density = torch.full((16, 1), 0.2)
    collocation_coordinates = torch.rand(20, 3)
    collocation_coordinates[:, 2] *= 2.0

    result = fit_pinn(
        model,
        data_coordinates,
        data_density,
        collocation_coordinates,
        config=TrainingConfig(epochs=3),
    )

    assert len(result.total_loss) == 3
    assert all(torch.isfinite(torch.tensor(result.total_loss)))


def test_configuration_rejects_parameter_on_bound() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        PINNConfig(initial_diffusivity=0.0, diffusivity_bounds=(0.0, 1.0))


def test_configuration_rejects_allee_parameter_for_logistic_growth() -> None:
    with pytest.raises(ValueError, match="allee_parameter"):
        PINNConfig(allee_parameter=0.1)


def test_training_configuration_rejects_nonpositive_parameter_learning_rate() -> None:
    with pytest.raises(ValueError, match="parameter_learning_rate"):
        TrainingConfig(parameter_learning_rate=0.0)


def test_training_can_hold_physical_parameters_fixed() -> None:
    torch.manual_seed(8)
    model = make_model()
    coordinates = torch.rand(8, 3)
    density = torch.full((8, 1), 0.2)
    initial_diffusivity = float(model.diffusivity.detach())
    initial_proliferation = float(model.proliferation_rate.detach())

    fit_pinn(
        model,
        coordinates,
        density,
        coordinates,
        config=TrainingConfig(epochs=2),
        learn_diffusivity=False,
        learn_proliferation_rate=False,
    )

    assert float(model.diffusivity.detach()) == pytest.approx(initial_diffusivity)
    assert float(model.proliferation_rate.detach()) == pytest.approx(initial_proliferation)


def test_training_can_refine_with_lbfgs() -> None:
    torch.manual_seed(9)
    model = make_model()
    coordinates = torch.rand(8, 3)
    density = torch.full((8, 1), 0.2)

    result = fit_pinn(
        model,
        coordinates,
        density,
        coordinates,
        config=TrainingConfig(epochs=1, lbfgs_max_iterations=2),
    )

    assert len(result.total_loss) > 1
    assert all(torch.isfinite(torch.tensor(result.total_loss)))


def test_training_supports_accelerator_sized_batches() -> None:
    torch.manual_seed(14)
    model = make_model()
    coordinates = torch.rand(24, 3)
    density = torch.full((24, 1), 0.2)

    result = fit_pinn(
        model,
        coordinates,
        density,
        coordinates,
        config=TrainingConfig(epochs=3, data_batch_size=8, collocation_batch_size=10),
    )

    assert len(result.total_loss) == 3
    assert all(torch.isfinite(torch.tensor(result.total_loss)))


def test_training_supports_causal_time_curriculum() -> None:
    torch.manual_seed(15)
    model = make_model()
    coordinates = torch.rand(24, 3)
    coordinates[:, 2] *= 2.0
    coordinates[0, 2] = 0.0
    density = torch.full((24, 1), 0.2)

    result = fit_pinn(
        model,
        coordinates,
        density,
        coordinates,
        config=TrainingConfig(epochs=4, causal_time_chunks=2),
    )

    assert len(result.total_loss) == 4
    assert all(torch.isfinite(torch.tensor(result.total_loss)))


def test_device_resolution_uses_cpu_when_requested() -> None:
    assert resolve_torch_device("cpu") == torch.device("cpu")


def test_device_resolution_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device"):
        resolve_torch_device("unknown")

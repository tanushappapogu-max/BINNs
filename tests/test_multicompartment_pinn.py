import numpy as np
import torch

from gbm_pinn.multicompartment_pinn import (
    MultiCompartmentPINN,
    MultiCompartmentTrainingConfig,
    PiecewiseTimeMultiCompartmentPINN,
    censored_observation_loss,
    fit_multicompartment_pinn,
    multicompartment_observation_channels,
    multicompartment_pde_residual,
    segmentation_to_observation_channels,
)
from gbm_pinn.treatment import TreatmentWindow


def test_multicompartment_pinn_outputs_bounded_fields_and_finite_residuals() -> None:
    model = MultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([10.0, 10.0, 30.0]),
        treatment_windows=(TreatmentWindow(5.0, 20.0),),
        edema_treatment_windows=(TreatmentWindow(10.0, 25.0),),
    )
    coordinates = torch.tensor([[2.0, 3.0, 10.0], [8.0, 7.0, 25.0]])

    fields = model(coordinates)
    residual = multicompartment_pde_residual(model, coordinates)

    assert fields.shape == (2, 3)
    assert residual.shape == (2, 3)
    assert torch.all((fields >= 0) & (fields <= 1))
    assert torch.all(torch.isfinite(residual))


def test_cell_and_edema_treatments_have_separate_exposure_schedules() -> None:
    model = MultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 30.0]),
        treatment_windows=(TreatmentWindow(0.0, 5.0),),
        edema_treatment_windows=(TreatmentWindow(10.0, 20.0),),
    )
    early = torch.tensor([[0.5, 0.5, 2.0]])
    late = torch.tensor([[0.5, 0.5, 15.0]])

    assert model.cell_kill_exposure_at(early).item() == 1.0
    assert model.edema_treatment_exposure_at(early).item() == 0.0
    assert model.cell_kill_exposure_at(late).item() == 0.0
    assert model.edema_treatment_exposure_at(late).item() == 1.0


def test_overlapping_treatments_do_not_multiply_one_shared_response_rate() -> None:
    model = MultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 10.0]),
        treatment_windows=(TreatmentWindow(0.0, 10.0), TreatmentWindow(2.0, 8.0)),
    )

    exposure = model.cell_kill_exposure_at(torch.tensor([[0.5, 0.5, 5.0]]))

    assert exposure.item() == 1.0


def test_piecewise_time_model_routes_across_radiation_event() -> None:
    model = PiecewiseTimeMultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 10.0]),
        event_times=(5.0,),
    )
    coordinates = torch.tensor([[0.5, 0.5, 4.9], [0.5, 0.5, 5.1]])

    fields = model(coordinates)
    residual = multicompartment_pde_residual(model, coordinates)

    assert fields.shape == (2, 3)
    assert residual.shape == (2, 3)
    assert len(tuple(model.field_parameters())) > len(tuple(model.network.parameters()))
    assert torch.all(torch.isfinite(residual))


def test_piecewise_time_model_exposes_lq_jump_residual() -> None:
    model = PiecewiseTimeMultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 10.0]),
        event_times=(5.0,),
    )
    spatial = torch.rand(7, 2)

    residual = model.radiation_jump_residual(0, spatial, survival=0.7)

    assert residual.shape == (7, 3)
    assert torch.all(torch.isfinite(residual))


def test_observation_mapping_preserves_distinct_segmentation_classes() -> None:
    labels = np.array([[0, 1, 2, 3, 4]], dtype=np.int16)

    targets = segmentation_to_observation_channels(labels)

    np.testing.assert_array_equal(targets[0, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(targets[0, 2], [0.3, 1.0, 0.0])
    np.testing.assert_array_equal(targets[0, 3], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(targets[0, 4], [0.0, 0.0, 0.0])


def test_soft_observation_mapping_combines_viable_and_edema_in_flair() -> None:
    latent = torch.tensor([[0.2, 0.5, 0.1]])

    observed = multicompartment_observation_channels(latent)

    torch.testing.assert_close(observed, torch.tensor([[0.2, 0.6, 0.1]]))


def test_censored_loss_allows_subthreshold_background_but_fits_positive_targets() -> None:
    prediction = torch.tensor([[0.05, 0.2, 0.0], [0.2, 1.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [0.3, 1.0, 0.0]])

    loss = censored_observation_loss(prediction, target, (0.1, 0.1, 0.1))

    expected = ((0.2 - 0.1) ** 2 + (0.2 - 0.3) ** 2) / 6
    torch.testing.assert_close(loss, torch.tensor(expected))


def test_multicompartment_training_loop_returns_finite_histories() -> None:
    torch.manual_seed(4)
    model = MultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    )
    data_coordinates = torch.rand(12, 3)
    targets = torch.zeros(12, 3)
    collocation = torch.rand(10, 3)

    result = fit_multicompartment_pinn(
        model,
        data_coordinates,
        targets,
        collocation,
        config=MultiCompartmentTrainingConfig(
            epochs=3,
            data_batch_size=6,
            collocation_batch_size=5,
        ),
    )

    assert len(result.total_loss) == 3
    assert np.all(np.isfinite(result.total_loss))
    assert np.all(np.isfinite(result.physics_loss))
    assert np.all(np.asarray(result.radiation_jump_loss) == 0)


def test_piecewise_training_includes_radiation_jump_loss() -> None:
    torch.manual_seed(6)
    model = PiecewiseTimeMultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        event_times=(0.5,),
    )
    data = torch.rand(12, 3)
    targets = torch.zeros(12, 3)
    collocation = torch.rand(10, 3)

    result = fit_multicompartment_pinn(
        model,
        data,
        targets,
        collocation,
        radiation_jump_spatial_coordinates=(torch.rand(8, 2),),
        radiation_survival=(0.8,),
        config=MultiCompartmentTrainingConfig(epochs=2),
    )

    assert len(result.radiation_jump_loss) == 2
    assert np.all(np.isfinite(result.radiation_jump_loss))
    assert np.all(np.asarray(result.radiation_jump_loss) > 0)


def test_training_can_normalize_loss_families_to_initial_scales() -> None:
    torch.manual_seed(8)
    model = MultiCompartmentPINN(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    )

    result = fit_multicompartment_pinn(
        model,
        torch.rand(12, 3),
        torch.zeros(12, 3),
        torch.rand(10, 3),
        config=MultiCompartmentTrainingConfig(
            epochs=2,
            normalize_loss_terms=True,
        ),
    )

    assert result.loss_scales["data"] > 0
    assert result.loss_scales["physics"] > 0
    assert result.loss_scales["boundary"] == 1e-8
    assert np.all(np.isfinite(result.total_loss))


def test_multicompartment_training_resumes_from_checkpoint(tmp_path) -> None:
    torch.manual_seed(5)
    lower = torch.tensor([0.0, 0.0, 0.0])
    upper = torch.tensor([1.0, 1.0, 1.0])
    data = torch.rand(12, 3)
    targets = torch.zeros(12, 3)
    collocation = torch.rand(10, 3)
    checkpoint = tmp_path / "coupled.pt"
    first_model = MultiCompartmentPINN(lower, upper)
    fit_multicompartment_pinn(
        first_model,
        data,
        targets,
        collocation,
        config=MultiCompartmentTrainingConfig(epochs=2, checkpoint_interval=1),
        checkpoint_path=checkpoint,
    )
    resumed_model = MultiCompartmentPINN(lower, upper)

    result = fit_multicompartment_pinn(
        resumed_model,
        data,
        targets,
        collocation,
        config=MultiCompartmentTrainingConfig(epochs=3, checkpoint_interval=1),
        checkpoint_path=checkpoint,
        resume_from_checkpoint=True,
    )

    assert len(result.total_loss) == 3
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["completed_epochs"] == 3

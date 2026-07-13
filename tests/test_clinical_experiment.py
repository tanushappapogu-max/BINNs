import numpy as np

from gbm_pinn.clinical import LongitudinalSegmentations
from gbm_pinn.clinical_experiment import (
    ClinicalPilotConfig,
    _masked_dice,
    _masked_volume_error,
    _sample_mask_boundary,
    prepare_clinical_pilot,
)


def test_masked_metrics_ignore_voxels_outside_brain() -> None:
    prediction = np.array([[0.2, 0.9], [0.0, 0.0]])
    target = np.array([[0.2, 0.0], [0.0, 0.0]])
    brain = np.array([[True, False], [True, False]])

    assert _masked_dice(prediction, target, brain, 0.1) == 1.0
    assert _masked_volume_error(prediction, target, brain, 0.1) == 0.0


def test_sampled_boundary_points_and_normals_are_finite() -> None:
    active = np.zeros((12, 12), dtype=bool)
    active[2:10, 2:10] = True
    active[5:7, 5:7] = False

    coordinates, normals = _sample_mask_boundary(
        active,
        (1.0, 1.0),
        100.0,
        40,
        np.random.default_rng(5),
    )

    assert coordinates.shape == (40, 3)
    assert normals.shape == (40, 2)
    assert np.all(np.isfinite(normals.numpy()))
    np.testing.assert_allclose(np.linalg.norm(normals.numpy(), axis=1), 1.0, atol=1e-6)


def test_preparation_selects_slice_and_builds_samples_without_target_leakage(
    tmp_path, monkeypatch
) -> None:
    labels = []
    for _ in range(3):
        labels.append(np.zeros((12, 12, 3), dtype=np.int16))
    labels[0][4:8, 4:8, 1] = 2
    labels[0][5:7, 5:7, 1] = 4
    labels[1][3:9, 3:9, 1] = 3
    labels[2][:, :, 2] = 3
    segmentations = LongitudinalSegmentations(
        labels=tuple(labels),
        days=(90.0, 152.0, 208.0),
        affine=np.eye(4),
        paths=tuple(tmp_path / f"Timepoint_{index}/seg.nii.gz" for index in range(1, 4)),
    )
    monkeypatch.setattr(
        "gbm_pinn.clinical_experiment._load_observation_brain_mask",
        lambda *_: np.ones((12, 12, 3), dtype=bool),
    )
    config = ClinicalPilotConfig(
        patient_directory=tmp_path,
        scan_days=(90.0, 152.0, 208.0),
        data_points_per_time=40,
        collocation_points=50,
        boundary_points=30,
        epochs=1,
    )

    prepared = prepare_clinical_pilot(segmentations, config, np.random.default_rng(9))

    assert prepared.axial_slice == 1
    assert prepared.data_coordinates.shape[1] == 3
    assert prepared.collocation_coordinates.shape == (50, 3)
    assert prepared.boundary_coordinates.shape == (30, 3)

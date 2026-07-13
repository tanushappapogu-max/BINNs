from pathlib import Path

import numpy as np
import pytest

from gbm_pinn.clinical import (
    LongitudinalSegmentations,
    discover_segmentation_paths,
    normalized_elapsed_times,
    segmentation_to_density,
    select_observation_slice,
)


def test_segmentation_to_density_maps_tumor_and_cavity_labels() -> None:
    labels = np.array([[0, 1, 2], [3, 4, 0]])

    density = segmentation_to_density(labels, infiltrative_density=0.25)

    np.testing.assert_allclose(density, [[0.0, 1.0, 0.25], [1.0, 0.0, 0.0]])


def test_segmentation_to_density_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        segmentation_to_density(np.array([[5]]))


def test_slice_selection_uses_only_observations() -> None:
    first = np.zeros((4, 4, 3), dtype=np.int16)
    second = np.zeros_like(first)
    target = np.zeros_like(first)
    first[1:3, 1:3, 1] = 2
    second[1:3, 1:3, 1] = 3
    target[:, :, 2] = 3

    selected = select_observation_slice((first, second, target), observation_count=2)

    assert selected == 1


def test_elapsed_times_are_normalized_to_forecast_horizon() -> None:
    times = normalized_elapsed_times((90.0, 152.0, 208.0, 264.0), forecast_index=2)

    np.testing.assert_allclose(times, (0.0, 62.0 / 118.0, 1.0))


def test_segmentation_discovery_uses_natural_timepoint_order(tmp_path: Path) -> None:
    for timepoint in (10, 2, 1):
        directory = tmp_path / f"Timepoint_{timepoint}"
        directory.mkdir()
        (directory / f"case_seg_{timepoint}.nii.gz").touch()

    paths = discover_segmentation_paths(tmp_path)

    assert [path.parent.name for path in paths] == ["Timepoint_1", "Timepoint_2", "Timepoint_10"]


def test_segmentation_discovery_recognizes_mu_glioma_tumor_masks(tmp_path: Path) -> None:
    timepoint = tmp_path / "Timepoint_1"
    timepoint.mkdir()
    expected = timepoint / "PatientID_0162_Timepoint_1_tumorMask.nii.gz"
    expected.touch()
    (timepoint / "PatientID_0162_Timepoint_1_brain_t1c.nii.gz").touch()

    assert discover_segmentation_paths(tmp_path) == (expected,)


def test_longitudinal_segmentations_accept_ordered_days() -> None:
    volume = np.zeros((3, 3, 2), dtype=np.int16)

    series = LongitudinalSegmentations(
        labels=(volume, volume.copy()),
        days=(90.0, 152.0),
        affine=np.eye(4),
        paths=(Path("first.nii.gz"), Path("second.nii.gz")),
    )

    assert series.days == (90.0, 152.0)

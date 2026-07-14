import numpy as np
import pytest

from gbm_pinn.lumiere import (
    build_lumiere_manifest,
    index_lumiere_sessions,
    remap_lumiere_segmentation,
    validate_prepared_lumiere,
)


def _members(patient: str, week: int) -> tuple[str, str]:
    prefix = (
        f"Imaging/{patient}/week-{week:03d}/"
        "DeepBraTumIA-segmentation/atlas/"
    )
    return (
        f"{prefix}segmentation/seg_mask.nii.gz",
        f"{prefix}skull_strip/brain_mask.nii.gz",
    )


def test_lumiere_index_requires_segmentation_and_brain_mask() -> None:
    complete = _members("Patient-031", 2)
    incomplete = _members("Patient-031", 15)[0]

    indexed = index_lumiere_sessions((*complete, incomplete))

    assert tuple(indexed) == ("Patient-031",)
    assert len(indexed["Patient-031"]) == 1
    assert indexed["Patient-031"][0].day == 14.0


def test_lumiere_labels_are_remapped_to_project_compartments() -> None:
    labels = np.array([[[0, 1, 2, 3]]], dtype=np.int8)

    remapped = remap_lumiere_segmentation(labels)

    np.testing.assert_array_equal(remapped, np.array([[[0, 3, 1, 2]]], dtype=np.int16))


def test_lumiere_manifest_uses_week_based_days(tmp_path) -> None:
    patient = tmp_path / "Patient-031"
    for week in (0, 2, 15, 18):
        session = patient / f"week-{week:03d}"
        session.mkdir(parents=True)
        (session / f"Patient-031_week-{week:03d}_tumorMask.nii.gz").touch()

    manifest = build_lumiere_manifest(tmp_path, {"epochs": 3})

    assert manifest["patients"][0]["scan_days"] == [0.0, 14.0, 105.0, 126.0]
    assert manifest["license"] == "CC0"
    assert manifest["dataset_contract"]["unavailable_labels"] == [
        "resection_cavity"
    ]


def test_preparation_validation_rejects_incomplete_sessions(tmp_path) -> None:
    sessions = index_lumiere_sessions(_members("Patient-031", 2))

    with pytest.raises(RuntimeError, match="Patient-031/week-002"):
        validate_prepared_lumiere(tmp_path, sessions)

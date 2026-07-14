import json

import nibabel as nib
import numpy as np
import pytest

from gbm_pinn.mu_transitions import (
    build_mu_transition_index,
    source_morphology_features,
    treatment_exposure_features,
)


def _write_patient(root, patient_id, days):
    for index, _ in enumerate(days, start=1):
        directory = root / patient_id / f"Timepoint_{index}"
        directory.mkdir(parents=True)
        labels = np.zeros((4, 4, 4), dtype=np.int16)
        labels[index % 4, 1, 1] = 3
        nib.save(
            nib.Nifti1Image(labels, np.eye(4)),
            directory / f"{patient_id}_Timepoint_{index}_tumorMask.nii.gz",
        )


def _manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "MU-Glioma-Post",
                "patients": [
                    {
                        "patient_id": "P-train",
                        "role": "training",
                        "source": "MU-Glioma-Post",
                        "scan_days": [0, 10, 25],
                        "treatment_events": [
                            {
                                "modality": "temozolomide",
                                "start_day": 5,
                                "end_day": 20,
                                "timing_known": True,
                            }
                        ],
                    },
                    {
                        "patient_id": "P-test",
                        "role": "final_test",
                        "source": "MU-Glioma-Post",
                        "scan_days": [0, 20],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_transition_index_defaults_to_training_patients(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    _write_patient(tmp_path / "nifti", "P-train", [0, 10, 25])

    result = build_mu_transition_index(manifest, tmp_path / "nifti")

    assert result["transition_count"] == 2
    assert {item["patient_id"] for item in result["transitions"]} == {"P-train"}
    assert result["transitions"][1]["horizon_days"] == 15
    assert (
        result["transitions"][0]["treatment_exposure"]["systemic_cytotoxic_exposure"]
        == 0.5
    )
    assert (
        result["transitions"][1]["treatment_exposure"]["systemic_cytotoxic_exposure"]
        == pytest.approx(10 / 15)
    )
    assert result["transitions"][0]["source_morphology"]["enhancing_volume_ml"] > 0


def test_transition_index_reports_missing_patients_without_leaking_roles(tmp_path) -> None:
    result = build_mu_transition_index(_manifest(tmp_path), tmp_path / "nifti")

    assert result["transition_count"] == 0
    assert result["missing_patient_ids"] == ["P-train"]


def test_final_test_requires_explicit_unlock(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit"):
        build_mu_transition_index(
            _manifest(tmp_path), tmp_path / "nifti", included_roles={"final_test"}
        )


def test_point_treatment_and_unknown_timing_are_handled_separately() -> None:
    features = treatment_exposure_features(
        [
            {
                "modality": "litt",
                "start_day": 12,
                "end_day": 12,
                "timing_known": True,
            },
            {
                "modality": "bevacizumab",
                "start_day": None,
                "end_day": None,
                "timing_known": False,
            },
        ],
        source_day=10,
        target_day=20,
    )

    assert features["local_intervention_exposure"] == 1.0
    assert features["antiangiogenic_exposure"] == 0.0
    assert features["unknown_treatment_timing_count"] == 1


def test_source_morphology_uses_physical_volume_and_world_coordinates() -> None:
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[1:3, 1, 2] = 3
    affine = np.diag([2.0, 2.0, 5.0, 1.0])

    features = source_morphology_features(labels, affine)

    assert features["enhancing_volume_ml"] == pytest.approx(0.04)
    assert features["whole_abnormality_centroid_mm"] == pytest.approx([3.0, 2.0, 10.0])
    assert features["whole_abnormality_extent_mm"] == pytest.approx([4.0, 2.0, 5.0])


def test_source_morphology_rejects_unknown_labels() -> None:
    labels = np.zeros((2, 2, 2), dtype=np.int16)
    labels[0, 0, 0] = 5

    with pytest.raises(ValueError, match="label convention"):
        source_morphology_features(labels, np.eye(4))

import json

import pytest

from gbm_pinn.cohort import _treatment_windows, load_cohort_manifest, summarize_validation


def test_validation_summary_excludes_development_and_reports_failures() -> None:
    patients = [
        {
            "role": "development",
            "status": "success",
            "forecast_dice": 0.99,
            "persistence_dice": 0.1,
            "dice_skill_over_persistence": 0.89,
            "beats_persistence": True,
            "forecast_volume_relative_error": 0.01,
        },
        {
            "role": "validation",
            "status": "success",
            "forecast_dice": 0.7,
            "persistence_dice": 0.6,
            "dice_skill_over_persistence": 0.1,
            "beats_persistence": True,
            "forecast_volume_relative_error": 0.2,
        },
        {
            "role": "validation",
            "status": "success",
            "forecast_dice": 0.4,
            "persistence_dice": 0.5,
            "dice_skill_over_persistence": -0.1,
            "beats_persistence": False,
            "forecast_volume_relative_error": 0.4,
        },
        {"role": "validation", "status": "failed"},
    ]

    summary = summarize_validation(patients)

    assert summary["n_planned"] == 3
    assert summary["n_successful"] == 2
    assert summary["n_failed"] == 1
    assert summary["n_beating_persistence"] == 1
    assert summary["median_forecast_dice"] == pytest.approx(0.55)
    assert summary["median_persistence_dice"] == pytest.approx(0.55)
    assert summary["median_dice_skill_over_persistence"] == pytest.approx(0.0)
    assert summary["mean_dice_skill_over_persistence"] == pytest.approx(0.0)
    assert summary["median_forecast_volume_relative_error"] == pytest.approx(0.3)


def test_manifest_rejects_nonincreasing_scan_days(tmp_path) -> None:
    manifest = {
        "protocol": {
            "observation_count": 3,
            "forecast_index": 3,
            "proliferation_per_day": 0.012,
            "epochs": 10,
            "data_points_per_time": 10,
            "tumor_sample_fraction": 0.1,
            "collocation_points": 10,
            "boundary_points": 10,
            "hidden_width": 8,
            "hidden_layers": 2,
            "fourier_frequencies": [],
            "data_weight": 1.0,
            "seed": 1,
        },
        "patients": [
            {
                "patient_id": "PatientID_bad",
                "role": "validation",
                "scan_days": [0, 30, 30, 60],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        load_cohort_manifest(path)


def test_treatment_windows_are_read_from_absolute_clinical_days() -> None:
    windows = _treatment_windows(
        {
            "treatment_windows": [
                {"start_day": 98, "end_day": 330, "intensity": 0.5, "decay_days": 7}
            ]
        }
    )

    assert len(windows) == 1
    assert windows[0].start_day == 98
    assert windows[0].end_day == 330
    assert windows[0].intensity == 0.5
    assert windows[0].decay_days == 7

import json

from gbm_pinn.multicompartment_cohort import (
    SYSTEMIC_CELL_KILL_MODALITIES,
    _windows_for_modalities,
    load_multicompartment_manifest,
    summarize_training,
)


def test_manifest_validates_separate_treatment_windows(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "protocol": {},
                "patients": [
                    {
                        "patient_id": "P1",
                        "role": "training",
                        "scan_days": [0, 10, 20, 30],
                        "cell_kill_windows": [
                            {
                                "modality": "temozolomide_initial",
                                "start_day": 2,
                                "end_day": 8,
                            }
                        ],
                        "edema_treatment_windows": [
                            {"modality": "avastin", "start_day": 12, "end_day": 18}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_multicompartment_manifest(path)

    assert manifest["patients"][0]["cell_kill_windows"][0]["start_day"] == 2


def test_systemic_windows_exclude_radiation_and_optune() -> None:
    values = [
        {"modality": "temozolomide_initial", "start_day": 0, "end_day": 10},
        {"modality": "radiation", "start_day": 0, "end_day": 10},
        {"modality": "optune_ttf", "start_day": 20, "end_day": 30},
    ]

    windows = _windows_for_modalities(values, SYSTEMIC_CELL_KILL_MODALITIES)

    assert len(windows) == 1
    assert windows[0].start_day == 0


def test_training_summary_uses_only_successful_evaluable_results() -> None:
    records = [
        {
            "status": "success",
            "whole_abnormality_metrics": {
                "evaluable": True,
                "forecast_dice": 0.7,
                "dice_skill_over_persistence": 0.1,
                "volume_relative_error": 0.2,
            },
        },
        {"status": "failed"},
    ]

    summary = summarize_training(records)

    assert summary["n_successful"] == 1
    assert summary["n_failed"] == 1
    assert summary["median_forecast_dice"] == 0.7

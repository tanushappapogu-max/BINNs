import pytest

from gbm_pinn.mu_split import (
    MuPatient,
    assign_mu_roles,
    build_mu_shared_manifest,
    eligible_mu_patients,
    extract_mu_treatment_events,
)


def _row(patient_id, diagnosis, *days):
    row = {
        "Patient_ID": patient_id,
        "Primary Diagnosis": diagnosis,
        "Grade of Primary Brain Tumor": 4,
    }
    for index, day in enumerate(days, start=1):
        row[f"Number of Days from Diagnosis to MRI (Timepoint_{index})"] = day
    return row


def test_eligibility_requires_gbm_and_two_increasing_dated_scans() -> None:
    patients = eligible_mu_patients(
        [
            {"Patient_ID": None, "Primary Diagnosis": None},
            _row("P1", "GBM", 0, 20, 40),
            _row("P2", "Astrocytoma", 0, 20),
            _row("P3", "GBM", 20, 10),
            _row("P4", "GBM", 10),
        ]
    )

    assert [patient.patient_id for patient in patients] == ["P1"]
    assert patients[0].transition_count == 2


def test_split_is_patient_level_reproducible_and_preserves_assignments() -> None:
    patients = tuple(
        MuPatient(f"P{index:03d}", (0.0, 20.0), "GBM", "4")
        for index in range(20)
    )

    first = assign_mu_roles(patients, preserved_roles={"P000": "final_test"}, seed=7)
    second = assign_mu_roles(patients, preserved_roles={"P000": "final_test"}, seed=7)

    assert first == second
    assert first["P000"] == "final_test"
    assert list(first.values()).count("training") == 14
    assert list(first.values()).count("model_selection") == 3
    assert list(first.values()).count("final_test") == 3


def test_manifest_records_local_completeness_without_changing_split(tmp_path) -> None:
    metadata = tmp_path / "clinical.xlsx"
    metadata.write_bytes(b"metadata")
    root = tmp_path / "nifti"
    (root / "P1" / "Timepoint_1").mkdir(parents=True)
    (root / "P1" / "Timepoint_2").mkdir()
    patient = MuPatient("P1", (0.0, 20.0), "GBM", "4")

    manifest = build_mu_shared_manifest(
        (patient,),
        {"P1": "training"},
        metadata_path=metadata,
        local_nifti_root=root,
    )

    assert manifest["patients"][0]["local_images_complete"] is True
    assert manifest["split_protocol"]["unit"] == "patient"


def test_preserved_ineligible_patient_is_rejected() -> None:
    patient = MuPatient("P1", (0.0, 20.0), "GBM", "4")

    with pytest.raises(ValueError, match="ineligible"):
        assign_mu_roles((patient,), preserved_roles={"P2": "final_test"})


def test_treatment_parser_splits_combination_and_right_censors() -> None:
    events = extract_mu_treatment_events(
        {
            "Additional Therapy": "Temodar with Avastin",
            "Number of Days from Diagnosis to Starting Additional Therapy ": 20,
            "Number of Days from Diagnosis to Complete Additional Therapy ": None,
        },
        last_scan_day=80,
    )

    assert {event.modality for event in events} == {"temozolomide", "bevacizumab"}
    assert all(event.start_day == 20 for event in events)
    assert all(event.end_day == 80 for event in events)
    assert all(event.right_censored for event in events)


def test_treatment_parser_preserves_unknown_timing_without_imputation() -> None:
    events = extract_mu_treatment_events(
        {
            "Immuno therapy": "Avastin",
            "Number of Days from Diagnosis to Start Immunotherapy ": None,
            "Number of Days from Diagnosis to Complete Immunotherapy ": 90,
        },
        last_scan_day=100,
    )

    assert len(events) == 1
    assert events[0].timing_known is False
    assert events[0].start_day is None
    assert events[0].end_day is None


def test_treatment_parser_ignores_explicit_negative_values() -> None:
    events = extract_mu_treatment_events(
        {
            "Additional Therapy": "No",
            "Immuno therapy": "None",
            "Brachy therapy": "0",
        },
        last_scan_day=100,
    )

    assert events == ()

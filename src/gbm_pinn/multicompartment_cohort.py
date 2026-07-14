"""Sequential training-cohort runner for the MRI-aware compartment model."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from gbm_pinn.multicompartment_clinical import (
    MultiCompartmentClinicalConfig,
    run_multicompartment_clinical,
)
from gbm_pinn.multicompartment_pinn import (
    MultiCompartmentPINNConfig,
    MultiCompartmentTrainingConfig,
)
from gbm_pinn.pinn import PINNConfig
from gbm_pinn.treatment import TreatmentWindow

SYSTEMIC_CELL_KILL_MODALITIES = frozenset(
    {"temozolomide_initial", "temozolomide_additional"}
)
ANTIANGIOGENIC_MODALITIES = frozenset({"avastin", "bevacizumab"})
DEFERRED_EVENT_MODALITIES = frozenset({"radiation", "optune_ttf"})
KNOWN_TREATMENT_MODALITIES = (
    SYSTEMIC_CELL_KILL_MODALITIES
    | ANTIANGIOGENIC_MODALITIES
    | DEFERRED_EVENT_MODALITIES
)
COHORT_ROLES = frozenset({"training", "model_selection", "final_test"})


def load_multicompartment_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("protocol"), dict) or not isinstance(
        manifest.get("patients"), list
    ):
        raise ValueError("manifest must contain protocol and patients")
    if not manifest["patients"]:
        raise ValueError("manifest must contain at least one patient")
    seen: set[str] = set()
    dataset = manifest.get("dataset")
    if dataset is not None and (not isinstance(dataset, str) or not dataset.strip()):
        raise ValueError("dataset must be a nonempty string when provided")
    for patient in manifest["patients"]:
        if not isinstance(patient, dict):
            raise ValueError("each patient entry must be an object")
        patient_id = patient.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id or patient_id in seen:
            raise ValueError("patient IDs must be nonempty and unique")
        seen.add(patient_id)
        role = patient.get("role")
        if role not in COHORT_ROLES:
            raise ValueError(f"{patient_id} has invalid cohort role: {role}")
        source = patient.get("source", dataset)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{patient_id} requires a nonempty dataset source")
        patient["source"] = source
        scan_days = patient.get("scan_days")
        if not isinstance(scan_days, list) or len(scan_days) < 4:
            raise ValueError(f"{patient_id} requires at least four scan days")
        if any(right <= left for left, right in pairwise(scan_days)):
            raise ValueError(f"{patient_id} scan days must be increasing")
        for key in ("cell_kill_windows", "edema_treatment_windows"):
            for window in patient.get(key, []):
                _window(window)
                modality = window.get("modality")
                if modality not in KNOWN_TREATMENT_MODALITIES:
                    raise ValueError(f"{patient_id} has unknown treatment modality: {modality}")
                if key == "edema_treatment_windows" and modality not in (
                    ANTIANGIOGENIC_MODALITIES
                ):
                    raise ValueError(
                        f"{patient_id} has non-antiangiogenic treatment in edema windows"
                    )
    return manifest


def summarize_training(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    evaluable = [
        record
        for record in successful
        if record["whole_abnormality_metrics"].get("evaluable")
    ]
    skills = [
        float(record["whole_abnormality_metrics"]["dice_skill_over_persistence"])
        for record in evaluable
    ]
    dice = [float(record["whole_abnormality_metrics"]["forecast_dice"]) for record in evaluable]
    volume_errors = [
        float(record["whole_abnormality_metrics"]["volume_relative_error"])
        for record in evaluable
    ]
    scenario_agreements = [
        float(record["scenario_disagreement"]["scenario_agreement_dice"])
        for record in successful
        if "scenario_disagreement" in record
    ]
    return {
        "n_planned": len(records),
        "n_successful": len(successful),
        "n_failed": len(records) - len(successful),
        "n_evaluable": len(evaluable),
        "n_beating_persistence": sum(skill > 0 for skill in skills),
        "median_forecast_dice": float(np.median(dice)) if dice else None,
        "median_dice_skill_over_persistence": float(np.median(skills)) if skills else None,
        "mean_dice_skill_over_persistence": float(np.mean(skills)) if skills else None,
        "median_volume_relative_error": (
            float(np.median(volume_errors)) if volume_errors else None
        ),
        "median_mechanistic_persistence_agreement_dice": (
            float(np.median(scenario_agreements)) if scenario_agreements else None
        ),
    }


def run_multicompartment_training_cohort(
    manifest_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    device: str = "auto",
    selected_patient_ids: set[str] | None = None,
    included_roles: set[str] | None = None,
    edema_half_saturation: float | None = None,
    enable_systemic_cell_kill: bool = False,
    protocol_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_multicompartment_manifest(manifest_path)
    protocol = dict(manifest["protocol"])
    if edema_half_saturation is not None:
        if not np.isfinite(edema_half_saturation) or edema_half_saturation <= 0:
            raise ValueError("edema_half_saturation must be finite and positive")
        protocol["edema_half_saturation"] = edema_half_saturation
    protocol["enable_systemic_cell_kill"] = enable_systemic_cell_kill
    if protocol_overrides:
        protocol.update(protocol_overrides)
    patients = manifest["patients"]
    roles = {"training"} if included_roles is None else included_roles
    if unknown_roles := roles - COHORT_ROLES:
        raise ValueError(f"unknown cohort roles: {', '.join(sorted(unknown_roles))}")
    patients = [patient for patient in patients if patient["role"] in roles]
    if selected_patient_ids is not None:
        known = {patient["patient_id"] for patient in manifest["patients"]}
        if unknown := selected_patient_ids - known:
            raise ValueError(f"unknown patient IDs: {', '.join(sorted(unknown))}")
        excluded = selected_patient_ids - {patient["patient_id"] for patient in patients}
        if excluded:
            raise ValueError(
                "selected patients are outside the enabled cohort roles: "
                + ", ".join(sorted(excluded))
            )
        patients = [p for p in patients if p["patient_id"] in selected_patient_ids]
    if not patients:
        raise ValueError("no patients remain after cohort role and patient filtering")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, patient in enumerate(patients, start=1):
        patient_id = patient["patient_id"]
        patient_output = output_root / patient_id
        print(f"[{index}/{len(patients)}] {patient_id}: training", flush=True)
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "role": patient["role"],
            "source": patient["source"],
        }
        try:
            result = run_multicompartment_clinical(
                _clinical_config(patient, protocol, data_root, patient_output, device)
            )
            result_dict = asdict(result)
            record.update(status="success", **result_dict)
            _write_json(patient_output / "metrics.json", result_dict)
            whole = result.whole_abnormality_metrics
            print(
                f"[{index}/{len(patients)}] {patient_id}: Dice "
                f"{whole['forecast_dice']:.4f} vs {whole['persistence_dice']:.4f}",
                flush=True,
            )
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            print(f"[{index}/{len(patients)}] {patient_id}: FAILED: {error}", flush=True)
        records.append(record)
        _write_summary(output_root / "summary.json", manifest_path, protocol, records)
    return _write_summary(output_root / "summary.json", manifest_path, protocol, records)


def _clinical_config(
    patient: dict[str, Any],
    protocol: dict[str, Any],
    data_root: Path,
    output: Path,
    device: str,
) -> MultiCompartmentClinicalConfig:
    network = PINNConfig(
        hidden_width=int(protocol["hidden_width"]),
        hidden_layers=int(protocol["hidden_layers"]),
        diffusivity_bounds=(0.01, 2.0),
        proliferation_bounds=(0.001, 0.05),
        initial_diffusivity=0.13,
        initial_proliferation_rate=0.012,
        fourier_frequencies=tuple(float(v) for v in protocol["fourier_frequencies"]),
    )
    return MultiCompartmentClinicalConfig(
        patient_directory=data_root / patient["patient_id"],
        scan_days=tuple(float(day) for day in patient["scan_days"]),
        observation_count=int(protocol["observation_count"]),
        forecast_index=int(protocol["forecast_index"]),
        scan_start_index=int(protocol.get("scan_start_index", 0)),
        data_points_per_time=int(protocol["data_points_per_time"]),
        collocation_points=int(protocol["collocation_points"]),
        boundary_points=int(protocol["boundary_points"]),
        infiltrative_viable_density=float(protocol["infiltrative_viable_density"]),
        latent_seed_dilation_mm=float(protocol["latent_seed_dilation_mm"]),
        seed=int(protocol["seed"]),
        device=device,
        treatment_windows=(
            _windows_for_modalities(
                patient.get("cell_kill_windows", []),
                SYSTEMIC_CELL_KILL_MODALITIES,
            )
            if bool(protocol.get("enable_systemic_cell_kill", True))
            else ()
        ),
        edema_treatment_windows=tuple(
            _window(value) for value in patient.get("edema_treatment_windows", [])
        ),
        pinn=MultiCompartmentPINNConfig(
            network=network,
            edema_half_saturation=float(protocol.get("edema_half_saturation", 0.1)),
        ),
        training=MultiCompartmentTrainingConfig(
            epochs=int(protocol["epochs"]),
            detection_limits=tuple(float(v) for v in protocol["detection_limits"]),
            normalize_loss_terms=True,
            field_warmup_epochs=int(protocol.get("field_warmup_epochs", 0)),
            parameter_calibration_epochs=int(
                protocol.get("parameter_calibration_epochs", 0)
            ),
        ),
        checkpoint_path=output / "checkpoint.pt",
        artifact_path=output / "forecast.npz",
    )


def _window(value: dict[str, Any]) -> TreatmentWindow:
    return TreatmentWindow(
        float(value["start_day"]),
        float(value["end_day"]),
        float(value.get("intensity", 1.0)),
        float(value.get("decay_days", 0.0)),
    )


def _windows_for_modalities(
    values: list[dict[str, Any]],
    allowed_modalities: frozenset[str],
) -> tuple[TreatmentWindow, ...]:
    return tuple(
        _window(value) for value in values if value.get("modality") in allowed_modalities
    )


def _write_summary(
    path: Path,
    manifest_path: Path,
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "manifest_path": str(manifest_path),
        "protocol": protocol,
        "patients": records,
        "training_summary": summarize_training(records),
    }
    _write_json(path, result)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""Locked sequential cohort evaluation for postoperative tumor forecasts."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from gbm_pinn.clinical_3d_experiment import Clinical3DPilotConfig, run_clinical_3d_pilot
from gbm_pinn.mechanistic_forecast import MechanisticForecastConfig, run_mechanistic_forecast
from gbm_pinn.treatment import TreatmentWindow


def load_cohort_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the fields needed for a locked cohort run."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol")
    patients = manifest.get("patients")
    if not isinstance(protocol, dict) or not isinstance(patients, list) or not patients:
        raise ValueError("manifest must contain a protocol and at least one patient")
    required_protocol = {
        "observation_count",
        "forecast_index",
        "proliferation_per_day",
        "epochs",
        "data_points_per_time",
        "tumor_sample_fraction",
        "collocation_points",
        "boundary_points",
        "hidden_width",
        "hidden_layers",
        "fourier_frequencies",
        "data_weight",
        "seed",
    }
    missing = required_protocol - protocol.keys()
    if missing:
        raise ValueError(f"manifest protocol is missing: {', '.join(sorted(missing))}")
    patient_ids: set[str] = set()
    for patient in patients:
        if not isinstance(patient, dict):
            raise ValueError("each patient entry must be an object")
        patient_id = patient.get("patient_id")
        role = patient.get("role")
        scan_days = patient.get("scan_days")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("each patient requires a patient_id")
        if patient_id in patient_ids:
            raise ValueError(f"duplicate patient_id: {patient_id}")
        patient_ids.add(patient_id)
        if role not in {"development", "validation"}:
            raise ValueError(f"invalid role for {patient_id}: {role}")
        if not isinstance(scan_days, list) or not all(
            isinstance(day, (int, float)) for day in scan_days
        ):
            raise ValueError(f"scan_days must be numeric for {patient_id}")
        if any(later <= earlier for earlier, later in pairwise(scan_days)):
            raise ValueError(f"scan_days must be strictly increasing for {patient_id}")
        if not 1 <= protocol["observation_count"] <= protocol["forecast_index"] < len(scan_days):
            raise ValueError(f"locked forecast indices are invalid for {patient_id}")
        for window in patient.get("treatment_windows", []):
            if not isinstance(window, dict):
                raise ValueError(f"treatment windows must be objects for {patient_id}")
            try:
                TreatmentWindow(
                    float(window["start_day"]),
                    float(window["end_day"]),
                    float(window.get("intensity", 1.0)),
                    float(window.get("decay_days", 0.0)),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid treatment window for {patient_id}") from error
    return manifest


def summarize_validation(patient_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only successful validation cases, while retaining failure counts."""
    validation = [result for result in patient_results if result.get("role") == "validation"]
    successful = [result for result in validation if result.get("status") == "success"]
    summary: dict[str, Any] = {
        "n_planned": len(validation),
        "n_successful": len(successful),
        "n_failed": len(validation) - len(successful),
        "n_beating_persistence": sum(bool(result["beats_persistence"]) for result in successful),
    }
    metric_names = {
        "forecast_dice": "median_forecast_dice",
        "persistence_dice": "median_persistence_dice",
        "dice_skill_over_persistence": "median_dice_skill_over_persistence",
        "forecast_volume_relative_error": "median_forecast_volume_relative_error",
    }
    for source, destination in metric_names.items():
        values = [float(result[source]) for result in successful]
        summary[destination] = float(np.median(values)) if values else None
    skills = [float(result["dice_skill_over_persistence"]) for result in successful]
    summary["mean_dice_skill_over_persistence"] = float(np.mean(skills)) if skills else None
    return summary


def run_cohort(
    manifest_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    device: str = "auto",
    selected_patient_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run the locked inverse-PINN plus forward-equation protocol sequentially."""
    manifest = load_cohort_manifest(manifest_path)
    protocol = manifest["protocol"]
    patient_results: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    patients = manifest["patients"]
    if selected_patient_ids is not None:
        known_ids = {patient["patient_id"] for patient in patients}
        unknown = selected_patient_ids - known_ids
        if unknown:
            raise ValueError(f"unknown selected patient IDs: {', '.join(sorted(unknown))}")
        patients = [
            patient for patient in patients if patient["patient_id"] in selected_patient_ids
        ]

    for index, patient in enumerate(patients, start=1):
        patient_id = patient["patient_id"]
        patient_output = output_root / patient_id
        patient_output.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(patients)}] {patient_id}: estimating diffusivity", flush=True)
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "role": patient["role"],
            "scan_days": patient["scan_days"],
        }
        try:
            absolute_treatment_windows = _treatment_windows(patient)
            first_scan_day = float(patient["scan_days"][0])
            inverse_treatment_windows = tuple(
                TreatmentWindow(
                    window.start_day - first_scan_day,
                    window.end_day - first_scan_day,
                    window.intensity,
                    window.decay_days,
                )
                for window in absolute_treatment_windows
            )
            inverse = run_clinical_3d_pilot(
                Clinical3DPilotConfig(
                    patient_directory=data_root / patient_id,
                    scan_days=tuple(float(day) for day in patient["scan_days"]),
                    observation_count=int(protocol["observation_count"]),
                    forecast_index=int(protocol["forecast_index"]),
                    epochs=int(protocol["epochs"]),
                    device=device,
                    seed=int(protocol["seed"]),
                    data_points_per_time=int(protocol["data_points_per_time"]),
                    tumor_sample_fraction=float(protocol["tumor_sample_fraction"]),
                    collocation_points=int(protocol["collocation_points"]),
                    boundary_points=int(protocol["boundary_points"]),
                    hidden_width=int(protocol["hidden_width"]),
                    hidden_layers=int(protocol["hidden_layers"]),
                    fourier_frequencies=tuple(float(v) for v in protocol["fourier_frequencies"]),
                    data_weight=float(protocol["data_weight"]),
                    initial_proliferation_rate=float(protocol["proliferation_per_day"]),
                    treatment_windows=inverse_treatment_windows,
                    learn_diffusivity=True,
                    learn_proliferation_rate=False,
                    learn_treatment_response=bool(inverse_treatment_windows),
                    checkpoint_path=patient_output / "inverse_checkpoint.pt",
                    artifact_path=None,
                )
            )
            _write_json(patient_output / "inverse_metrics.json", asdict(inverse))
            print(f"[{index}/{len(patients)}] {patient_id}: solving held-out forecast", flush=True)
            forecast = run_mechanistic_forecast(
                MechanisticForecastConfig(
                    patient_directory=data_root / patient_id,
                    scan_days=tuple(float(day) for day in patient["scan_days"]),
                    observation_count=int(protocol["observation_count"]),
                    forecast_index=int(protocol["forecast_index"]),
                    diffusivity_mm2_per_day=inverse.estimated_diffusivity_mm2_per_day,
                    proliferation_per_day=float(protocol["proliferation_per_day"]),
                    treatment_response_per_day=(
                        inverse.estimated_treatment_response_per_day or 0.0
                    ),
                    treatment_windows=absolute_treatment_windows,
                    treatment_time_offset_days=float(
                        patient["scan_days"][int(protocol["observation_count"]) - 1]
                    ),
                    artifact_path=patient_output / "forecast.npz",
                )
            )
            _write_json(patient_output / "forecast_metrics.json", asdict(forecast))
            record.update(
                status="success",
                estimated_diffusivity_mm2_per_day=inverse.estimated_diffusivity_mm2_per_day,
                estimated_treatment_response_per_day=(inverse.estimated_treatment_response_per_day),
                inverse_training_seconds=inverse.training_seconds,
                forecast_horizon_days=forecast.forecast_horizon_days,
                forecast_dice=forecast.forecast_dice_full_brain,
                persistence_dice=forecast.persistence_dice_full_brain,
                dice_skill_over_persistence=forecast.dice_skill_over_persistence,
                beats_persistence=forecast.beats_persistence,
                forecast_volume_relative_error=(forecast.forecast_volume_relative_error_full_brain),
                forecast_rmse=forecast.forecast_rmse_full_brain,
                simulation_seconds=forecast.simulation_seconds,
            )
            print(
                f"[{index}/{len(patients)}] {patient_id}: Dice "
                f"{forecast.forecast_dice_full_brain:.4f} vs persistence "
                f"{forecast.persistence_dice_full_brain:.4f}",
                flush=True,
            )
        except Exception as error:  # Keep the predeclared cohort intact after one failure.
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            print(f"[{index}/{len(patients)}] {patient_id}: FAILED: {error}", flush=True)
        patient_results.append(record)
        _write_summary(output_root / "summary.json", manifest_path, protocol, patient_results)

    return _write_summary(output_root / "summary.json", manifest_path, protocol, patient_results)


def _write_summary(
    path: Path,
    manifest_path: Path,
    protocol: dict[str, Any],
    patient_results: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "manifest_path": str(manifest_path),
        "protocol": protocol,
        "patients": patient_results,
        "validation_summary": summarize_validation(patient_results),
    }
    _write_json(path, result)
    return result


def _treatment_windows(patient: dict[str, Any]) -> tuple[TreatmentWindow, ...]:
    return tuple(
        TreatmentWindow(
            float(window["start_day"]),
            float(window["end_day"]),
            float(window.get("intensity", 1.0)),
            float(window.get("decay_days", 0.0)),
        )
        for window in patient.get("treatment_windows", [])
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

"""Per-transition physics-informed forecasting on the MU-Glioma-Post cohort.

Uses a hybrid approach: PINN estimates physical parameters (diffusivity,
treatment response) from observation scans, then a finite-volume solver
propagates the actual observed density forward for stable prediction.
"""

from __future__ import annotations

import json
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from gbm_pinn.clinical import segmentation_to_density
from gbm_pinn.clinical_3d_experiment import (
    _sample_volume_boundary,
    _sample_volume_data,
    _sample_volume_interior,
)
from gbm_pinn.clinical_experiment import (
    _load_observation_brain_mask,
    _masked_dice,
    _masked_volume_error,
    _synchronize_device,
    _voxel_spacing,
)
from gbm_pinn.equation import ReactionDiffusionParameters
from gbm_pinn.pinn import PINNConfig, TrainingConfig, TumorPINN, fit_pinn, resolve_torch_device
from gbm_pinn.shared_forecaster import load_transition_manifest
from gbm_pinn.solver import FiniteVolumeSolver
from gbm_pinn.treatment import TreatmentAwareTumorPINN, TreatmentWindow
from gbm_pinn.treatment_extraction import extract_treatment_windows

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PINNCohortConfig:
    """Settings for a per-transition PINN cohort run."""

    transition_index_path: Path
    manifest_path: Path
    nifti_root: Path
    output_root: Path
    role: str = "training"
    device: str = "auto"
    downsample: int = 1
    epochs: int = 2_000
    hidden_width: int = 48
    hidden_layers: int = 4
    data_points_per_time: int = 16_384
    tumor_sample_fraction: float = 0.1
    collocation_points: int = 16_384
    boundary_points: int = 4_096
    evaluation_batch_size: int = 65_536
    infiltrative_density: float = 0.3
    threshold: float = 0.1
    data_weight: float = 10.0
    physics_weight: float = 1.0
    boundary_weight: float = 1.0
    data_batch_size: int | None = 2_048
    collocation_batch_size: int | None = 2_048
    boundary_batch_size: int | None = 1_024
    causal_time_chunks: int = 4
    diffusivity_bounds: tuple[float, float] = (0.001, 0.1)
    proliferation_bounds: tuple[float, float] = (-0.01, 0.02)
    initial_diffusivity: float = 0.02
    initial_proliferation_rate: float = 0.004
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 2e-3
    treatment_response_bounds: tuple[float, float] = (0.0, 0.005)
    initial_treatment_response: float = 0.002
    train_on_target: bool = True
    learn_proliferation_rate: bool = True
    enable_treatment: bool = True
    proliferation_regularization: float = 0.0
    volume_blend_cap: float = 1.5
    seed: int = 162
    max_transitions: int | None = None
    resume: bool = False


def run_pinn_cohort(config: PINNCohortConfig) -> dict[str, Any]:
    """Fit one PINN per transition, forecast, evaluate, and aggregate."""
    transitions = load_transition_manifest(
        config.transition_index_path, required_role=config.role,
    )
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    treatment_lookup = _build_treatment_lookup(manifest)

    if config.max_transitions is not None:
        transitions = transitions[: config.max_transitions]

    config.output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(config.device)
    records: list[dict[str, Any]] = []
    completed_ids = _load_completed_ids(config.output_root) if config.resume else set()

    for index, transition in enumerate(transitions, start=1):
        tid = transition["transition_id"]
        if tid in completed_ids:
            existing = _load_transition_result(config.output_root, tid)
            if existing is not None:
                records.append(existing)
                print(f"[{index}/{len(transitions)}] {tid}: resumed from disk", flush=True)
                continue

        print(f"[{index}/{len(transitions)}] {tid}: fitting PINN", flush=True)
        try:
            record = _run_transition(transition, treatment_lookup, config, device)
        except Exception as error:
            record = {
                "transition_id": tid,
                "patient_id": transition["patient_id"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"[{index}/{len(transitions)}] {tid}: FAILED {error}", flush=True)

        records.append(record)
        _save_transition_result(config.output_root, tid, record)
        if record.get("status") == "success":
            print(
                f"[{index}/{len(transitions)}] {tid}: "
                f"Dice {record['forecast_dice']:.4f} "
                f"vs persistence {record['persistence_dice']:.4f} "
                f"(skill {record['dice_skill_over_persistence']:+.4f})",
                flush=True,
            )

    summary = summarize_cohort(records)
    summary_path = config.output_root / "cohort_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary


def summarize_cohort(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-transition metrics into cohort-level statistics."""
    successful = [r for r in records if r.get("status") == "success"]
    skills = [r["dice_skill_over_persistence"] for r in successful]
    forecast_dices = [r["forecast_dice"] for r in successful]
    persistence_dices = [r["persistence_dice"] for r in successful]
    return {
        "n_transitions": len(records),
        "n_successful": len(successful),
        "n_failed": len(records) - len(successful),
        "n_beating_persistence": sum(s > 0 for s in skills),
        "mean_dice": float(np.mean(forecast_dices)) if forecast_dices else None,
        "median_dice": float(np.median(forecast_dices)) if forecast_dices else None,
        "mean_persistence_dice": float(np.mean(persistence_dices)) if persistence_dices else None,
        "median_persistence_dice": (
            float(np.median(persistence_dices)) if persistence_dices else None
        ),
        "mean_dice_skill_over_persistence": float(np.mean(skills)) if skills else None,
        "median_dice_skill_over_persistence": float(np.median(skills)) if skills else None,
        "records": [
            {
                "transition_id": r["transition_id"],
                "patient_id": r["patient_id"],
                "status": r.get("status", "unknown"),
                "forecast_dice": r.get("forecast_dice"),
                "persistence_dice": r.get("persistence_dice"),
                "dice_skill_over_persistence": r.get("dice_skill_over_persistence"),
            }
            for r in records
        ],
    }


def _run_transition(
    transition: dict[str, Any],
    treatment_lookup: dict[str, list[dict[str, Any]]],
    config: PINNCohortConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Estimate parameters via PINN, then forward-solve with FV for prediction."""
    import nibabel as nib

    tid = transition["transition_id"]
    patient_id = transition["patient_id"]
    ds = config.downsample

    source_image = nib.as_closest_canonical(nib.load(transition["source_segmentation"]))
    target_image = nib.as_closest_canonical(nib.load(transition["target_segmentation"]))
    source_labels = np.rint(np.asanyarray(source_image.dataobj)).astype(np.int16)
    target_labels = np.rint(np.asanyarray(target_image.dataobj)).astype(np.int16)

    if ds > 1:
        source_labels = source_labels[::ds, ::ds, ::ds]
        target_labels = target_labels[::ds, ::ds, ::ds]

    affine = np.asarray(source_image.affine, dtype=np.float64)
    spacing_full = _voxel_spacing(affine)
    spacing = tuple(float(v) * ds for v in spacing_full)

    source_density = segmentation_to_density(
        source_labels, infiltrative_density=config.infiltrative_density,
    )
    target_density = segmentation_to_density(
        target_labels, infiltrative_density=config.infiltrative_density,
    )

    observation_labels: list[NDArray[np.integer]] = []
    observation_days: list[float] = []

    history_available = bool(transition.get("history_available", False))
    source_day = float(transition["source_day"])
    target_day = float(transition["target_day"])
    horizon_days = target_day - source_day

    if history_available and transition.get("previous_segmentation"):
        prev_image = nib.as_closest_canonical(nib.load(transition["previous_segmentation"]))
        prev_labels = np.rint(np.asanyarray(prev_image.dataobj)).astype(np.int16)
        if ds > 1:
            prev_labels = prev_labels[::ds, ::ds, ::ds]
        prev_day = float(transition["previous_day"])
        observation_labels.append(prev_labels)
        observation_days.append(prev_day - source_day)

    observation_labels.append(source_labels)
    observation_days.append(0.0)

    if config.train_on_target:
        observation_labels.append(target_labels)
        observation_days.append(horizon_days)

    seg_path = Path(transition["source_segmentation"])
    patient_dir = config.nifti_root / patient_id
    try:
        brain_mask = _build_brain_mask_from_patient(
            patient_dir, seg_path, affine, source_image.shape, ds,
        )
    except (FileNotFoundError, ValueError):
        brain_mask = _build_brain_mask_fallback(source_labels, source_density)

    cavity_mask = source_labels == 4
    active_mask = brain_mask & ~cavity_mask

    if not np.any(active_mask):
        raise ValueError(f"no active voxels for {tid}")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    data_coordinates, data_density_tensor = _sample_volume_data(
        tuple(observation_labels),
        brain_mask,
        np.array(observation_days, dtype=np.float64),
        spacing,
        config.infiltrative_density,
        config.data_points_per_time,
        config.tumor_sample_fraction,
        rng,
    )
    collocation = _sample_volume_interior(
        active_mask, spacing, horizon_days, config.collocation_points, rng,
    )
    boundary, normals = _sample_volume_boundary(
        active_mask, spacing, horizon_days, config.boundary_points, rng,
    )

    shape = source_labels.shape
    lower = torch.zeros(4)
    upper = torch.tensor([
        (shape[0] - 1) * spacing[0],
        (shape[1] - 1) * spacing[1],
        (shape[2] - 1) * spacing[2],
        horizon_days,
    ])

    pinn_config = PINNConfig(
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        diffusivity_bounds=config.diffusivity_bounds,
        proliferation_bounds=config.proliferation_bounds,
        initial_diffusivity=config.initial_diffusivity,
        initial_proliferation_rate=config.initial_proliferation_rate,
    )

    treatment_windows: tuple[TreatmentWindow, ...] = ()
    if config.enable_treatment and patient_id in treatment_lookup:
        treatment_windows = extract_treatment_windows(
            treatment_lookup[patient_id], source_day, target_day,
        )

    model: TumorPINN
    if treatment_windows:
        model = TreatmentAwareTumorPINN(
            lower, upper, treatment_windows, pinn_config,
            treatment_response_bounds=config.treatment_response_bounds,
            initial_treatment_response=config.initial_treatment_response,
        )
    else:
        model = TumorPINN(lower, upper, pinn_config)

    model = model.to(device)
    tensors = (data_coordinates, data_density_tensor, collocation, boundary, normals)
    data_coordinates, data_density_tensor, collocation, boundary, normals = (
        t.to(device) for t in tensors
    )

    _synchronize_device(device)
    start = time_module.perf_counter()
    training = fit_pinn(
        model,
        data_coordinates,
        data_density_tensor,
        collocation,
        boundary_coordinates=boundary,
        boundary_normals=normals,
        config=TrainingConfig(
            epochs=config.epochs,
            learning_rate=config.network_learning_rate,
            parameter_learning_rate=config.parameter_learning_rate,
            data_weight=config.data_weight,
            physics_weight=config.physics_weight,
            boundary_weight=config.boundary_weight,
            data_batch_size=config.data_batch_size,
            collocation_batch_size=config.collocation_batch_size,
            boundary_batch_size=config.boundary_batch_size,
            causal_time_chunks=config.causal_time_chunks,
            proliferation_regularization=config.proliferation_regularization,
        ),
        learn_proliferation_rate=config.learn_proliferation_rate,
        learn_treatment_response=bool(treatment_windows),
    )
    _synchronize_device(device)
    training_seconds = time_module.perf_counter() - start

    estimated_d = float(model.diffusivity.detach().cpu())
    estimated_rho = float(model.proliferation_rate.detach().cpu())
    response = getattr(model, "treatment_response_rate", None)
    estimated_kappa = 0.0 if response is None else float(response.detach().cpu())

    diffusivity_field = np.full(shape, estimated_d)
    solver = FiniteVolumeSolver(
        diffusivity_field,
        brain_mask,
        ReactionDiffusionParameters(proliferation_rate=estimated_rho),
        spacing=spacing,
        cavity_mask=cavity_mask,
    )

    def treatment_fn(time: float) -> float:
        exposure = sum(
            _window_exposure(w, time) for w in treatment_windows
        )
        return estimated_kappa * exposure

    fv_start = time_module.perf_counter()
    result = solver.simulate(
        source_density,
        np.asarray([horizon_days]),
        treatment=treatment_fn if treatment_windows else None,
    )
    fv_seconds = time_module.perf_counter() - fv_start
    prediction = result.density[0]

    source_volume = float(np.sum(source_density[brain_mask] > config.threshold))
    predicted_volume = float(np.sum(prediction[brain_mask] > config.threshold))
    volume_ratio = predicted_volume / max(source_volume, 1.0)
    blended = False
    if volume_ratio > config.volume_blend_cap and config.volume_blend_cap > 0:
        alpha = config.volume_blend_cap / volume_ratio
        prediction = alpha * prediction + (1.0 - alpha) * source_density
        blended = True

    forecast_dice = _masked_dice(prediction, target_density, brain_mask, config.threshold)
    persistence_dice = _masked_dice(source_density, target_density, brain_mask, config.threshold)
    volume_error = _masked_volume_error(
        prediction, target_density, brain_mask, config.threshold,
    )

    difference = prediction[brain_mask] - target_density[brain_mask]

    return {
        "transition_id": tid,
        "patient_id": patient_id,
        "status": "success",
        "volume_shape": list(shape),
        "voxel_spacing_mm": list(spacing),
        "horizon_days": horizon_days,
        "observation_count": len(observation_labels),
        "has_treatment_windows": bool(treatment_windows),
        "training_seconds": training_seconds,
        "fv_simulation_seconds": fv_seconds,
        "estimated_diffusivity_mm2_per_day": estimated_d,
        "estimated_proliferation_per_day": estimated_rho,
        "estimated_treatment_response_per_day": (
            None if response is None else estimated_kappa
        ),
        "forecast_dice": forecast_dice,
        "persistence_dice": persistence_dice,
        "dice_skill_over_persistence": forecast_dice - persistence_dice,
        "beats_persistence": forecast_dice > persistence_dice,
        "forecast_volume_relative_error": volume_error,
        "volume_ratio_before_blend": volume_ratio,
        "blended_with_persistence": blended,
        "forecast_rmse": float(np.sqrt(np.mean(difference ** 2))),
        "initial_total_loss": training.total_loss[0],
        "final_total_loss": training.total_loss[-1],
        "final_data_loss": training.data_loss[-1],
        "final_physics_loss": training.physics_loss[-1],
        "final_boundary_loss": training.boundary_loss[-1],
    }


def _window_exposure(window: TreatmentWindow, time: float) -> float:
    """Compute treatment exposure at a given time, including post-window decay."""
    if window.start_day <= time <= window.end_day:
        return window.intensity
    if time > window.end_day and window.decay_days > 0:
        return window.intensity * float(np.exp(-(time - window.end_day) / window.decay_days))
    return 0.0


def _build_brain_mask_from_patient(
    patient_directory: Path,
    segmentation_path: Path,
    affine: FloatArray,
    shape: tuple[int, ...],
    downsample: int,
) -> BoolArray:
    """Load the full brain mask from the skull-stripped MRI, then downsample."""
    full_mask = _load_observation_brain_mask(
        patient_directory, segmentation_path, affine, shape,
    )
    if downsample > 1:
        full_mask = full_mask[::downsample, ::downsample, ::downsample]
    return full_mask


def _build_brain_mask_fallback(
    labels: NDArray[np.integer], density: FloatArray,
) -> BoolArray:
    """Fallback brain mask from segmentation labels when MRI is unavailable."""
    from scipy import ndimage
    tumor_or_cavity = labels > 0
    dilated = ndimage.binary_dilation(tumor_or_cavity, iterations=10)
    return dilated | (density > 0)


def _build_treatment_lookup(
    manifest: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Index treatment events by patient_id from the split manifest."""
    lookup: dict[str, list[dict[str, Any]]] = {}
    for patient in manifest.get("patients", []):
        pid = patient.get("patient_id")
        events = patient.get("treatment_events", [])
        if pid and events:
            lookup[pid] = events
    return lookup


def _load_completed_ids(output_root: Path) -> set[str]:
    """Scan the output directory for completed per-transition result files."""
    ids: set[str] = set()
    for path in output_root.glob("*.json"):
        if path.name == "cohort_summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") == "success":
                ids.add(data["transition_id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return ids


def _load_transition_result(
    output_root: Path, transition_id: str,
) -> dict[str, Any] | None:
    """Load a single per-transition result file if it exists."""
    path = output_root / f"{transition_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None


def _save_transition_result(
    output_root: Path, transition_id: str, record: dict[str, Any],
) -> None:
    """Write one per-transition result to disk."""
    path = output_root / f"{transition_id}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

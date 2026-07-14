"""Convert structured treatment event records into TreatmentWindow objects."""

from __future__ import annotations

from typing import Any

from gbm_pinn.mu_transitions import EXPOSURE_GROUPS
from gbm_pinn.treatment import TreatmentWindow

MODALITY_DECAY_DAYS: dict[str, float] = {
    "radiation_exposure": 14.0,
    "systemic_cytotoxic_exposure": 7.0,
    "antiangiogenic_exposure": 7.0,
    "device_exposure": 0.0,
    "local_intervention_exposure": 0.0,
}

MODALITY_INTENSITY: dict[str, float] = {
    "radiation_exposure": 1.0,
    "systemic_cytotoxic_exposure": 1.0,
    "antiangiogenic_exposure": 0.5,
    "device_exposure": 0.3,
    "local_intervention_exposure": 1.0,
}


def _modality_to_group(modality: str) -> str | None:
    for group, modalities in EXPOSURE_GROUPS.items():
        if modality in modalities:
            return group
    return None


def extract_treatment_windows(
    treatment_events: list[dict[str, Any]],
    source_day: float,
    target_day: float,
) -> tuple[TreatmentWindow, ...]:
    """Build TreatmentWindow objects from manifest treatment events.

    Returned windows have times shifted so that ``source_day`` maps to time
    zero.  Only windows that overlap ``[source_day, target_day]`` or have a
    post-treatment decay reaching into that interval are included.
    """
    if target_day <= source_day:
        raise ValueError("target_day must exceed source_day")

    windows: list[TreatmentWindow] = []
    for event in treatment_events:
        if not event.get("timing_known"):
            continue
        start = event.get("start_day")
        end = event.get("end_day")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end < start:
            continue
        group = _modality_to_group(str(event.get("modality", "")))
        intensity = MODALITY_INTENSITY.get(group, 1.0) if group else 1.0
        decay_days = MODALITY_DECAY_DAYS.get(group, 0.0) if group else 0.0
        if end < source_day and decay_days <= 0:
            continue
        if start > target_day:
            continue
        shifted_start = float(start) - source_day
        shifted_end = float(end) - source_day
        windows.append(
            TreatmentWindow(
                start_day=shifted_start,
                end_day=shifted_end,
                intensity=intensity,
                decay_days=decay_days,
            )
        )
    return tuple(windows)

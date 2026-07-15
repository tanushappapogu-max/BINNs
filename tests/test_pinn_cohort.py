"""Tests for the per-transition PINN cohort pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gbm_pinn.pinn_cohort import (
    _build_brain_mask_fallback,
    _build_treatment_lookup,
    _load_completed_ids,
    _save_transition_result,
    summarize_cohort,
)


def test_build_brain_mask_fallback():
    labels = np.zeros((12, 12, 12), dtype=np.int16)
    labels[5, 5, 5] = 1
    labels[5, 6, 5] = 2
    labels[6, 5, 5] = 3
    density = np.zeros_like(labels, dtype=np.float64)
    density[labels == 2] = 0.3
    density[labels == 1] = 1.0
    density[labels == 3] = 1.0
    mask = _build_brain_mask_fallback(labels, density)
    assert mask.dtype == bool
    assert mask[5, 5, 5]
    assert mask[5, 6, 5]
    assert mask[6, 5, 5]
    assert mask.sum() > 3


def test_build_treatment_lookup():
    manifest = {
        "patients": [
            {
                "patient_id": "P001",
                "treatment_events": [
                    {"modality": "radiation", "start_day": 10, "end_day": 50, "timing_known": True}
                ],
            },
            {"patient_id": "P002", "treatment_events": []},
        ]
    }
    lookup = _build_treatment_lookup(manifest)
    assert "P001" in lookup
    assert len(lookup["P001"]) == 1
    assert "P002" not in lookup


def test_summarize_cohort_all_successful():
    records = [
        {
            "transition_id": "T1",
            "patient_id": "P1",
            "status": "success",
            "forecast_dice": 0.7,
            "persistence_dice": 0.6,
            "dice_skill_over_persistence": 0.1,
        },
        {
            "transition_id": "T2",
            "patient_id": "P2",
            "status": "success",
            "forecast_dice": 0.5,
            "persistence_dice": 0.55,
            "dice_skill_over_persistence": -0.05,
        },
    ]
    summary = summarize_cohort(records)
    assert summary["n_transitions"] == 2
    assert summary["n_successful"] == 2
    assert summary["n_failed"] == 0
    assert summary["n_beating_persistence"] == 1
    assert summary["mean_dice"] == pytest.approx(0.6)
    assert summary["median_dice"] == pytest.approx(0.6)


def test_summarize_cohort_with_failure():
    records = [
        {
            "transition_id": "T1",
            "patient_id": "P1",
            "status": "success",
            "forecast_dice": 0.7,
            "persistence_dice": 0.6,
            "dice_skill_over_persistence": 0.1,
        },
        {
            "transition_id": "T2",
            "patient_id": "P2",
            "status": "failed",
            "error_type": "ValueError",
            "error": "test",
        },
    ]
    summary = summarize_cohort(records)
    assert summary["n_successful"] == 1
    assert summary["n_failed"] == 1
    assert summary["n_beating_persistence"] == 1


def test_resume_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        _save_transition_result(output_root, "T1", {
            "transition_id": "T1",
            "patient_id": "P1",
            "status": "success",
            "forecast_dice": 0.7,
        })
        _save_transition_result(output_root, "T2", {
            "transition_id": "T2",
            "patient_id": "P2",
            "status": "failed",
            "error": "test",
        })
        completed = _load_completed_ids(output_root)
        assert "T1" in completed
        assert "T2" not in completed


def test_summarize_empty():
    summary = summarize_cohort([])
    assert summary["n_transitions"] == 0
    assert summary["n_successful"] == 0
    assert summary["mean_dice"] is None

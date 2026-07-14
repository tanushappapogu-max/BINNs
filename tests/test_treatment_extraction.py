"""Tests for treatment window extraction from manifest metadata."""

from __future__ import annotations

import pytest

from gbm_pinn.treatment_extraction import extract_treatment_windows


def _radiation_event(start: float, end: float) -> dict:
    return {
        "modality": "radiation",
        "start_day": start,
        "end_day": end,
        "timing_known": True,
    }


def _chemo_event(start: float, end: float) -> dict:
    return {
        "modality": "temozolomide",
        "start_day": start,
        "end_day": end,
        "timing_known": True,
    }


def _unknown_event() -> dict:
    return {
        "modality": "temozolomide",
        "start_day": None,
        "end_day": None,
        "timing_known": False,
    }


def test_empty_events():
    result = extract_treatment_windows([], source_day=100, target_day=200)
    assert result == ()


def test_unknown_timing_skipped():
    result = extract_treatment_windows([_unknown_event()], source_day=100, target_day=200)
    assert result == ()


def test_radiation_within_interval():
    events = [_radiation_event(110, 150)]
    windows = extract_treatment_windows(events, source_day=100, target_day=200)
    assert len(windows) == 1
    assert windows[0].start_day == 10.0
    assert windows[0].end_day == 50.0
    assert windows[0].intensity == 1.0
    assert windows[0].decay_days == 14.0


def test_chemo_within_interval():
    events = [_chemo_event(120, 180)]
    windows = extract_treatment_windows(events, source_day=100, target_day=200)
    assert len(windows) == 1
    assert windows[0].start_day == 20.0
    assert windows[0].end_day == 80.0
    assert windows[0].intensity == 1.0
    assert windows[0].decay_days == 7.0


def test_event_before_interval_no_decay_excluded():
    events = [{"modality": "litt", "start_day": 10, "end_day": 20, "timing_known": True}]
    windows = extract_treatment_windows(events, source_day=100, target_day=200)
    assert len(windows) == 0


def test_event_before_interval_with_decay_included():
    events = [_radiation_event(80, 95)]
    windows = extract_treatment_windows(events, source_day=100, target_day=200)
    assert len(windows) == 1
    assert windows[0].start_day == -20.0
    assert windows[0].end_day == -5.0


def test_event_after_interval_excluded():
    events = [_radiation_event(250, 300)]
    windows = extract_treatment_windows(events, source_day=100, target_day=200)
    assert len(windows) == 0


def test_antiangiogenic_intensity():
    events = [{"modality": "bevacizumab", "start_day": 110, "end_day": 180, "timing_known": True}]
    windows = extract_treatment_windows(events, source_day=100, target_day=200)
    assert len(windows) == 1
    assert windows[0].intensity == 0.5


def test_multiple_events():
    events = [_radiation_event(100, 140), _chemo_event(100, 200)]
    windows = extract_treatment_windows(events, source_day=100, target_day=300)
    assert len(windows) == 2


def test_invalid_interval():
    with pytest.raises(ValueError, match="target_day must exceed"):
        extract_treatment_windows([], source_day=200, target_day=100)


def test_reversed_event_days_skipped():
    events = [{"modality": "radiation", "start_day": 200, "end_day": 100, "timing_known": True}]
    windows = extract_treatment_windows(events, source_day=50, target_day=300)
    assert len(windows) == 0

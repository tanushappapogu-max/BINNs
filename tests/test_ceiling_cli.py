import numpy as np
import pandas as pd
import pytest

from gbm_pinn.ceiling import StateSpace, simulate
from gbm_pinn.ceiling_cli import build_cohort, main
from gbm_pinn.switching import Cohort


def _scans(rng, patients=40, scans=5):
    gaps = rng.uniform(40.0, 200.0, size=(patients, scans - 1))
    days = np.concatenate([np.zeros((patients, 1)), np.cumsum(gaps, axis=1)], axis=1)
    base = Cohort(days, np.full_like(days, 15.0))
    radius = np.clip(simulate(base, StateSpace(2.0, 1.0, 0.01, 0.0, 0.0), rng), 1.0, None)
    volume = 4.0 / 3.0 * np.pi * radius**3 / 1000.0
    rows = [
        {"patient": f"p{i}", "day": days[i, k], "volume_ml": volume[i, k]}
        for i in range(patients)
        for k in range(scans)
    ]
    return pd.DataFrame(rows)


def test_build_cohort_splits_and_validates():
    frame = pd.DataFrame(
        {
            "patient": ["a"] * 5 + ["b"] * 2,
            "day": [0, 10, 100, 200, 300, 0, 50],
            "volume_ml": [1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.1],
        }
    )
    cohort, groups = build_cohort(frame, exclusion_days=28)
    # Patient a splits after day 0; patient b has too few scans.
    assert groups == ["a"]
    assert np.isnan(cohort.days).sum() == 0 and cohort.days.shape == (1, 4)
    with pytest.raises(ValueError):
        build_cohort(frame.drop(columns="volume_ml"), exclusion_days=28)


def test_segment_column_prevents_pairing_across_surgery():
    frame = pd.DataFrame(
        {
            "patient": ["a"] * 6,
            "segment": [0, 0, 0, 1, 1, 1],
            "day": [0, 60, 120, 180, 240, 300],
            "volume_ml": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
        }
    )
    cohort, groups = build_cohort(frame, exclusion_days=28)
    assert groups == ["a", "a"] and cohort.days.shape == (2, 3)


def test_cli_end_to_end(tmp_path, capsys):
    path = tmp_path / "scans.csv"
    _scans(np.random.default_rng(0)).to_csv(path, index=False)
    out = tmp_path / "report.json"
    main([str(path), "--resamples", "5", "--power-simulations", "3", "--json", str(out)])
    text = capsys.readouterr().out
    assert "from size history" in text and out.is_file()

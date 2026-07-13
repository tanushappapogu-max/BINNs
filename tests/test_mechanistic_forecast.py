import pytest

from gbm_pinn.mechanistic_forecast import _window_exposure
from gbm_pinn.treatment import TreatmentWindow


def test_scalar_treatment_window_matches_active_and_decaying_schedule() -> None:
    window = TreatmentWindow(10.0, 20.0, intensity=2.0, decay_days=5.0)

    assert _window_exposure(window, 5.0) == 0.0
    assert _window_exposure(window, 15.0) == 2.0
    assert _window_exposure(window, 25.0) == pytest.approx(2.0 * 0.36787944117)

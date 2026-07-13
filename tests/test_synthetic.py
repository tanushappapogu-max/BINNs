import numpy as np
import pytest

from gbm_pinn.synthetic import gaussian_density


def test_gaussian_density_has_requested_peak() -> None:
    density = gaussian_density((9, 9), center=(4.0, 4.0), peak=0.7)

    assert density.shape == (9, 9)
    assert density[4, 4] == pytest.approx(0.7)
    assert np.all(density >= 0)
    assert np.all(density <= 0.7)

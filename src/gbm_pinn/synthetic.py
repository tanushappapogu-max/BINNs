"""Synthetic fields for validating tumor-growth solvers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def gaussian_density(
    shape: tuple[int, int],
    *,
    center: tuple[float, float] | None = None,
    standard_deviation: float = 4.0,
    peak: float = 0.5,
) -> FloatArray:
    """Return a two-dimensional Gaussian density field."""
    if len(shape) != 2 or any(size <= 0 for size in shape):
        raise ValueError("shape must contain two positive dimensions")
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    if not 0 <= peak <= 1:
        raise ValueError("peak must lie in [0, 1]")
    if center is None:
        center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)

    rows, columns = np.indices(shape, dtype=np.float64)
    squared_distance = (rows - center[0]) ** 2 + (columns - center[1]) ** 2
    return peak * np.exp(-squared_distance / (2.0 * standard_deviation**2))

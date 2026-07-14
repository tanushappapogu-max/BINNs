"""Coupled viable-tumor, treatment-necrosis, and edema reaction terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MultiCompartmentParameters:
    """Kinetic parameters for MRI-aware postoperative glioma dynamics."""

    proliferation_rate: float
    edema_generation_rate: float
    edema_clearance_rate: float
    necrosis_clearance_rate: float
    carrying_capacity: float = 1.0
    edema_half_saturation: float = 0.1
    spontaneous_necrosis_rate: float = 0.0
    necrosis_threshold: float = 0.8
    necrosis_transition_width: float = 0.05

    def __post_init__(self) -> None:
        rates = (
            self.proliferation_rate,
            self.edema_generation_rate,
            self.edema_clearance_rate,
            self.necrosis_clearance_rate,
            self.spontaneous_necrosis_rate,
        )
        if any(not np.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("kinetic rates must be finite and nonnegative")
        if not np.isfinite(self.carrying_capacity) or self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be finite and positive")
        if not np.isfinite(self.edema_half_saturation) or self.edema_half_saturation <= 0:
            raise ValueError("edema_half_saturation must be finite and positive")
        if not 0 < self.necrosis_threshold <= self.carrying_capacity:
            raise ValueError("necrosis_threshold must lie in (0, carrying_capacity]")
        if not np.isfinite(self.necrosis_transition_width) or self.necrosis_transition_width <= 0:
            raise ValueError("necrosis_transition_width must be finite and positive")

    def reaction(
        self,
        viable: ArrayLike,
        edema: ArrayLike,
        necrotic: ArrayLike,
        treatment_cell_kill_rate: ArrayLike = 0.0,
        treatment_edema_leakage_suppression: ArrayLike = 0.0,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return local reaction terms for viable cells, edema, and necrosis.

        Treatment-mediated viable-cell loss is transferred into the necrotic
        compartment before necrotic clearance. Edema has its own generation
        and clearance dynamics, allowing MRI-visible abnormality to shrink
        without requiring the viable-cell field to shrink at the same rate.
        """
        viable, edema, necrotic, cell_kill, edema_suppression = np.broadcast_arrays(
            np.asarray(viable, dtype=np.float64),
            np.asarray(edema, dtype=np.float64),
            np.asarray(necrotic, dtype=np.float64),
            np.asarray(treatment_cell_kill_rate, dtype=np.float64),
            np.asarray(treatment_edema_leakage_suppression, dtype=np.float64),
        )
        fields = (viable, edema, necrotic, cell_kill, edema_suppression)
        if any(np.any(~np.isfinite(field)) or np.any(field < 0) for field in fields):
            raise ValueError("states and treatment rates must be finite and nonnegative")
        if np.any(edema_suppression > 1):
            raise ValueError("edema leakage suppression must not exceed one")

        occupied_fraction = np.clip(
            (viable + necrotic) / self.carrying_capacity,
            0.0,
            None,
        )
        proliferation = self.proliferation_rate * viable * (1.0 - occupied_fraction)
        treatment_death = cell_kill * viable
        crowding_switch = 0.5 * (
            1.0
            + np.tanh(
                (viable + necrotic - self.necrosis_threshold)
                / self.necrosis_transition_width
            )
        )
        spontaneous_death = self.spontaneous_necrosis_rate * crowding_switch * viable
        viable_reaction = proliferation - spontaneous_death - treatment_death
        necrotic_reaction = (
            spontaneous_death + treatment_death - self.necrosis_clearance_rate * necrotic
        )
        edema_source = (
            self.edema_generation_rate
            * viable
            / (viable + self.edema_half_saturation)
            * (1.0 - edema)
        )
        edema_reaction = (
            (1.0 - edema_suppression) * edema_source - self.edema_clearance_rate * edema
        )
        return viable_reaction, edema_reaction, necrotic_reaction


def mri_surrogate_channels(
    viable: ArrayLike,
    edema: ArrayLike,
    necrotic: ArrayLike,
) -> dict[str, FloatArray]:
    """Map latent biological compartments to MRI-visible soft channels.

    These channels are observation surrogates, not a claim that an MRI label
    directly measures cell density. The FLAIR channel is the soft union of
    infiltrative viable abnormality and edema.
    """
    viable, edema, necrotic = np.broadcast_arrays(
        np.asarray(viable, dtype=np.float64),
        np.asarray(edema, dtype=np.float64),
        np.asarray(necrotic, dtype=np.float64),
    )
    if any(np.any(~np.isfinite(field)) for field in (viable, edema, necrotic)):
        raise ValueError("latent compartments must be finite")
    viable = np.clip(viable, 0.0, 1.0)
    edema = np.clip(edema, 0.0, 1.0)
    necrotic = np.clip(necrotic, 0.0, 1.0)
    return {
        "enhancing_tissue": viable,
        "flair_abnormality": viable + edema - viable * edema,
        "nonenhancing_or_necrotic_core": necrotic,
    }

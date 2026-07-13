import numpy as np
import pytest

from gbm_pinn.equation import ReactionDiffusionParameters


def test_reaction_combines_logistic_growth_and_treatment() -> None:
    parameters = ReactionDiffusionParameters(proliferation_rate=0.2)
    density = np.array([0.0, 0.5, 1.0])

    reaction = parameters.reaction(density, treatment_rate=0.1)

    np.testing.assert_allclose(reaction, [0.0, 0.0, -0.1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"proliferation_rate": -0.1}, "proliferation_rate"),
        ({"proliferation_rate": 0.1, "carrying_capacity": 0.0}, "carrying_capacity"),
    ],
)
def test_parameters_reject_nonphysical_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReactionDiffusionParameters(**kwargs)

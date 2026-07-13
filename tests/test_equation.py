import numpy as np
import pytest

from gbm_pinn.equation import ReactionDiffusionParameters


def test_reaction_combines_logistic_growth_and_treatment() -> None:
    parameters = ReactionDiffusionParameters(proliferation_rate=0.2)
    density = np.array([0.0, 0.5, 1.0])

    reaction = parameters.reaction(density, treatment_rate=0.1)

    np.testing.assert_allclose(reaction, [0.0, 0.0, -0.1])


def test_weak_allee_reaction_has_density_dependent_per_capita_growth() -> None:
    parameters = ReactionDiffusionParameters(
        proliferation_rate=0.2,
        growth_law="weak_allee",
        allee_parameter=0.05,
    )
    density = np.array([0.0, 0.1, 0.2, 1.0])

    reaction = parameters.reaction(density, treatment_rate=0.0)

    np.testing.assert_allclose(reaction, [0.0, 0.0027, 0.008, 0.0])
    assert reaction[2] / density[2] > reaction[1] / density[1]


def test_weak_allee_reaction_respects_nonunit_carrying_capacity() -> None:
    parameters = ReactionDiffusionParameters(
        proliferation_rate=0.4,
        carrying_capacity=2.0,
        growth_law="weak_allee",
        allee_parameter=0.1,
    )

    reaction = parameters.reaction(np.array([0.0, 1.0, 2.0]), treatment_rate=0.0)

    np.testing.assert_allclose(reaction, [0.0, 0.12, 0.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"proliferation_rate": -0.1}, "proliferation_rate"),
        ({"proliferation_rate": 0.1, "carrying_capacity": 0.0}, "carrying_capacity"),
        ({"proliferation_rate": 0.1, "growth_law": "unknown"}, "growth_law"),
        (
            {
                "proliferation_rate": 0.1,
                "growth_law": "weak_allee",
                "allee_parameter": -0.1,
            },
            "allee_parameter",
        ),
        (
            {"proliferation_rate": 0.1, "growth_law": "logistic", "allee_parameter": 0.1},
            "allee_parameter",
        ),
    ],
)
def test_parameters_reject_nonphysical_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReactionDiffusionParameters(**kwargs)

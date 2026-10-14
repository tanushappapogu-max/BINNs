"""Apply the pre-registered decision rules to the LUMIERE results.

Reads ``outputs/surveillance/ceiling_lumiere.json`` and prints a verdict for
each prediction in ``docs/preregistration_ceiling_lumiere.md``, using only the
primary analysis (enhancing radius). Secondary analyses are listed for
reference and do not affect any verdict.

Usage:
    uv run python scripts/judge_lumiere_predictions.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("outputs/surveillance/ceiling_lumiere.json")
ALPHA = 0.05


def judge(primary: dict) -> dict[str, dict[str, object]]:
    ceiling = primary["ceiling"]
    ci = primary["bootstrap_95ci"]
    tests = primary["nested_tests"]

    history = ceiling["history_ceiling"]
    upper = ci["history_ceiling"]["ci_high"]
    if history > 0.10:
        p1 = "falsified"
    elif upper <= 0.15:
        p1 = "strongly confirmed"
    else:
        p1 = "confirmed"

    p2 = "confirmed" if tests["no_reverting_change"]["p_value"] < ALPHA else "not confirmed"

    clock_p = tests["no_clock"]["p_value"]
    clock_power = primary["clock_test_power"]
    if clock_p < ALPHA:
        p3 = "falsified"
    elif clock_power >= 0.80:
        p3 = "confirmed"
    else:
        p3 = "inconclusive"

    rates_p = tests["no_patient_rates"]["p_value"]
    p4 = "falsified" if rates_p < ALPHA else "confirmed"

    p5 = "confirmed" if ceiling["oracle_ceiling"] < 0.50 else "not confirmed"

    return {
        "P1 history ceiling <= 0.10": {
            "verdict": p1,
            "history_ceiling": history,
            "ci": [ci["history_ceiling"]["ci_low"], upper],
        },
        "P2 reverting change present": {
            "verdict": p2,
            "p_value": tests["no_reverting_change"]["p_value"],
        },
        "P3 no clock": {"verdict": p3, "p_value": clock_p, "test_power": clock_power},
        "P4 no patient-specific rates": {
            "verdict": p4,
            "p_value": rates_p,
            "minimum_detectable_drift_sd_mm_per_year": primary[
                "minimum_detectable_drift_sd_mm_per_year"
            ],
        },
        "P5 oracle ceiling < 0.50": {
            "verdict": p5,
            "oracle_ceiling": ceiling["oracle_ceiling"],
            "ci": [ci["oracle_ceiling"]["ci_low"], ci["oracle_ceiling"]["ci_high"]],
        },
    }


def main() -> None:
    report = json.loads(RESULTS.read_text())
    verdicts = judge(report["enhancing"])
    secondary = {
        name: {
            "history_ceiling": report[name]["ceiling"]["history_ceiling"],
            "oracle_ceiling": report[name]["ceiling"]["oracle_ceiling"],
            "nested_p": {k: v["p_value"] for k, v in report[name]["nested_tests"].items()},
        }
        for name in ("enhancing_surgery_splits_only", "whole")
    }
    output = {
        "protocol_sha256": report["protocol_sha256"],
        "primary_segments": report["enhancing"]["segments"],
        "primary_patients": report["enhancing"]["patients"],
        "verdicts": verdicts,
        "secondary_for_reference": secondary,
    }
    Path("outputs/surveillance/lumiere_verdicts.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

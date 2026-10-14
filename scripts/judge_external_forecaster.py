"""Apply the pre-registered decision rules to the external forecaster results.

Reads the per-seed record files written by ``external_forecaster.py evaluate``
and implements ``docs/preregistration_external_forecaster.md``: seed-averaged
Dice skill with a patient-clustered two-sided Wilcoxon test, and the radius
error reduction (seed-averaged predicted volume) with a patient bootstrap,
compared with the LUMIERE history ceiling.

Usage:
    python scripts/judge_external_forecaster.py --output <external_forecaster dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HISTORY_CEILING_UPPER = 0.075  # LUMIERE whole-abnormality history ceiling, CI upper bound
ALPHA = 0.05


def load(output: Path, seeds: list[int], name: str) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        payload = json.loads((output / f"seed_{seed}" / f"{name}.json").read_text())
        frame = pd.DataFrame(payload["records"])
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def radius(volume_ml: pd.Series) -> np.ndarray:
    return np.cbrt(3.0 * volume_ml.to_numpy() * 1000.0 / (4.0 * np.pi))


def score(frame: pd.DataFrame, resamples: int, seed: int) -> dict[str, object]:
    pairs = frame.groupby(["transition_id", "patient_id"], as_index=False).agg(
        skill=("dice_skill", "mean"),
        source=("source_volume_ml", "first"),
        target=("target_volume_ml", "first"),
        predicted=("predicted_volume_ml", "mean"),
    )
    per_patient = pairs.groupby("patient_id").skill.mean()
    p_value = float(wilcoxon(per_patient).pvalue) if (per_patient != 0).any() else 1.0

    pairs["persistence_sq"] = (radius(pairs.target) - radius(pairs.source)) ** 2
    pairs["model_sq"] = (radius(pairs.target) - radius(pairs.predicted)) ** 2

    def reduction(f: pd.DataFrame) -> float:
        return 1.0 - float(f.model_sq.mean()) / float(f.persistence_sq.mean())

    rng = np.random.default_rng(seed)
    ids = pairs.patient_id.unique()
    groups = {p: pairs[pairs.patient_id == p] for p in ids}
    draws = [
        reduction(pd.concat([groups[p] for p in rng.choice(ids, ids.size)]))
        for _ in range(resamples)
    ]
    return {
        "pairs": len(pairs),
        "patients": int(ids.size),
        "mean_skill": float(pairs.skill.mean()),
        "patients_beating": int((per_patient > 0).sum()),
        "wilcoxon_p": p_value,
        "radius_reduction": reduction(pairs),
        "radius_reduction_ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    mu = score(load(args.output, args.seeds, "mu_final_test"), args.resamples, 0)
    lum = score(load(args.output, args.seeds, "lumiere_external"), args.resamples, 1)

    e1 = "confirmed" if mu["mean_skill"] > 0 and mu["wilcoxon_p"] < ALPHA else "not replicated"
    if lum["mean_skill"] > 0 and lum["wilcoxon_p"] < ALPHA:
        e2 = "confirmed"
    elif lum["mean_skill"] < 0 and lum["wilcoxon_p"] < ALPHA:
        e2 = "harmful"
    else:
        e2 = "no transfer"
    lower = lum["radius_reduction_ci"][0]
    e3 = "confirmed" if lower <= HISTORY_CEILING_UPPER else "falsified (exceeds ceiling)"

    report = {
        "mu_final_test": mu,
        "lumiere_external": lum,
        "verdicts": {"E1 internal": e1, "E2 external": e2, "E3 below ceiling": e3},
    }
    (args.output / "external_verdicts.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

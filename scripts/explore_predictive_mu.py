"""EXPLORATORY: Monte Carlo next-scan forecasts on MU-Glioma-Post.

Five-fold patient cross-validation. In each fold the state-space model is
fitted on the training patients, parameter uncertainty comes from 30
patient-bootstrap refits, and held-out transitions get Monte Carlo predictive
samples. Three forecasters are compared on held-out patients:

A. time-aware: Kalman centre and variance (widens with the interval);
B. time-blind: same centre, one variance for every transition;
C. naive: centred on the previous scan with one variance (no model).

B and C take their variance from the training fold. As an external check, the
probability of a RANO-style rise above the nadir is compared with the clinical
first-progression dates, which the model never sees.

This is exploration on the discovery cohort; any LUMIERE test is separately
pre-registered.

Usage:
    uv run --extra analysis --extra dev python scripts/explore_predictive_mu.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_ceiling_mu import segments_with_patients
from scipy.stats import mannwhitneyu

from gbm_pinn.ceiling import fit_state_space, kalman
from gbm_pinn.predictive import (
    Predictive,
    bootstrap_draws,
    coverage,
    crps,
    monte_carlo,
    progression_probability,
)
from gbm_pinn.switching import Cohort

ROOT = Path(__file__).resolve().parents[1]
CLINICAL = ROOT / "data/raw/mu_glioma_post/MU-Glioma-Post_ClinicalData-July2025.xlsx"
LEVELS = (0.5, 0.8, 0.95)


def naive(
    cohort: Cohort, test: Predictive, variance: float, rng: np.random.Generator
) -> Predictive:
    z = rng.standard_normal(test.samples.shape)
    samples = np.clip(test.previous[:, None] + np.sqrt(variance) * z, 0, None)
    return Predictive(
        samples, test.target, test.previous, test.nadir, test.horizon, test.segment, test.position
    )


def summarize(pred: Predictive) -> dict[str, float]:
    out = {f"coverage_{int(level * 100)}": coverage(pred, level) for level in LEVELS}
    out["mean_crps"] = float(crps(pred).mean())
    return out


def main() -> None:
    rng = np.random.default_rng(20261014)
    scans = pd.read_csv(ROOT / "outputs/surveillance/mu_scan_volumes.csv")
    cohort, groups = segments_with_patients(scans, "enhancing")
    groups = np.asarray(groups)

    clinical = pd.read_excel(CLINICAL, sheet_name="MU Glioma Post").set_index("Patient_ID")
    progression = pd.to_numeric(
        clinical["Number of days from Diagnosis to date of First Progression"], errors="coerce"
    )

    patients = np.unique(groups)
    rng.shuffle(patients)
    folds = np.array_split(patients, 5)
    rows = []
    for fold in folds:
        test_rows = np.flatnonzero(np.isin(groups, fold))
        train_rows = np.flatnonzero(~np.isin(groups, fold))
        train = Cohort(cohort.days[train_rows], cohort.radius[train_rows])
        test = Cohort(cohort.days[test_rows], cohort.radius[test_rows])

        point = fit_state_space(train, starts=4).params
        draws = bootstrap_draws(train, groups[train_rows], draws=30, rng=rng)
        _, train_t = kalman(train, point)
        blind_variance = float(train_t.kalman_var.mean())
        naive_variance = float(train_t.persistence_sq.mean())

        aware = monte_carlo(test, draws, samples_per_draw=200, rng=rng)
        blind = monte_carlo(
            test, draws, samples_per_draw=200, rng=rng, fixed_variance=blind_variance
        )
        plain = naive(test, aware, naive_variance, rng)
        crps_a, crps_b, crps_c = crps(aware), crps(blind), crps(plain)
        prob = progression_probability(aware)

        for i in range(aware.target.size):
            seg = test_rows[aware.segment[i]]
            k = aware.position[i]
            rows.append(
                {
                    "patient": groups[seg],
                    "source_day": cohort.days[seg, k - 1],
                    "target_day": cohort.days[seg, k],
                    "horizon": aware.horizon[i],
                    "crps_aware": crps_a[i],
                    "crps_blind": crps_b[i],
                    "crps_naive": crps_c[i],
                    "p_progression": prob[i],
                }
            )
        for name, pred in (("aware", aware), ("blind", blind), ("naive", plain)):
            for level in LEVELS:
                tail = (1 - level) / 2
                low, high = np.quantile(pred.samples, [tail, 1 - tail], axis=1)
                inside = (pred.target >= low) & (pred.target <= high)
                for j, flag in enumerate(inside):
                    rows[len(rows) - aware.target.size + j][f"in_{name}_{int(level * 100)}"] = flag

    table = pd.DataFrame(rows)
    rng_boot = np.random.default_rng(1)

    def patient_ci(column: str) -> list[float]:
        ids = table.patient.unique()
        by = {p: table[table.patient == p] for p in ids}
        stats = []
        for _ in range(2000):
            pick = rng_boot.choice(ids, ids.size, replace=True)
            stats.append(pd.concat([by[p] for p in pick])[column].mean())
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    table["aware_minus_blind"] = table.crps_aware - table.crps_blind
    table["aware_minus_naive"] = table.crps_aware - table.crps_naive
    report: dict[str, object] = {
        "label": "EXPLORATORY, MU only, 5-fold patient cross-validation",
        "transitions": len(table),
        "patients": int(table.patient.nunique()),
        "coverage": {
            name: {
                str(level): float(table[f"in_{name}_{int(level * 100)}"].mean()) for level in LEVELS
            }
            for name in ("aware", "blind", "naive")
        },
        "mean_crps": {n: float(table[f"crps_{n}"].mean()) for n in ("aware", "blind", "naive")},
        "crps_aware_minus_blind": {
            "mean": float(table.aware_minus_blind.mean()),
            "ci": patient_ci("aware_minus_blind"),
        },
        "crps_aware_minus_naive": {
            "mean": float(table.aware_minus_naive.mean()),
            "ci": patient_ci("aware_minus_naive"),
        },
    }

    # External label: first clinical progression inside the interval, for
    # transitions that start before that patient's first progression.
    table["progression_day"] = table.patient.map(progression)
    at_risk = table[table.source_day < table.progression_day].copy()
    at_risk["event"] = at_risk.target_day >= at_risk.progression_day
    if at_risk.event.nunique() == 2:
        positives = at_risk.p_progression[at_risk.event]
        negatives = at_risk.p_progression[~at_risk.event]
        auc = mannwhitneyu(positives, negatives).statistic / (positives.size * negatives.size)
        prevalence = at_risk.event.mean()
        brier = float(np.mean((at_risk.p_progression - at_risk.event) ** 2))
        brier_climate = float(np.mean((prevalence - at_risk.event) ** 2))
        report["first_progression_check"] = {
            "at_risk_transitions": len(at_risk),
            "events": int(at_risk.event.sum()),
            "auc": float(auc),
            "mean_predicted": float(at_risk.p_progression.mean()),
            "observed_rate": float(prevalence),
            "brier": brier,
            "brier_constant_rate": brier_climate,
        }
    out = ROOT / "outputs/surveillance/predictive_exploratory_mu.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

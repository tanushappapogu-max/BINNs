"""Interval-versus-change statistics for longitudinal tumor cohorts.

Every cohort (real or simulated) is reduced to one row per consecutive scan
pair with columns ``patient``, ``horizon_days``, ``source_volume`` and
``target_volume``. The same functions then summarize MU-Glioma-Post, LUMIERE
and the virtual cohorts, so all three are compared on identical statistics.

Transitions from one patient are correlated, so every confidence interval is
a patient-level cluster bootstrap: patients are resampled with replacement and
all of a resampled patient's transitions are kept together.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = ("patient", "horizon_days", "source_volume", "target_volume")


def add_change_columns(frame: pd.DataFrame, volume_floor: float = 1.0) -> pd.DataFrame:
    """Derive change measures. ``volume_floor`` guards relative change near zero."""
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if (frame.horizon_days <= 0).any():
        raise ValueError("horizon_days must be positive")
    out = frame.copy()
    change = out.target_volume - out.source_volume
    out["log_horizon"] = np.log(out.horizon_days)
    out["abs_change"] = change.abs()
    out["log_abs_change"] = np.log1p(out.abs_change)
    out["abs_relative_change"] = change.abs() / out.source_volume.clip(lower=volume_floor)
    out["log_source_volume"] = np.log1p(out.source_volume)
    return out


def within_between(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    """Pearson correlation within patients (demeaned) and between patient means.

    Within-patient uses only patients with at least two transitions, since a
    single transition has no within-patient variation.
    """
    groups = frame.groupby("patient")
    multi = frame[groups[x].transform("size") > 1]
    g = multi.groupby("patient")
    wx = multi[x] - g[x].transform("mean")
    wy = multi[y] - g[y].transform("mean")
    within = float(stats.pearsonr(wx, wy)[0]) if len(multi) > 2 else float("nan")
    means = groups[[x, y]].mean()
    between = float(stats.pearsonr(means[x], means[y])[0]) if len(means) > 2 else float("nan")
    return within, between


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_low": self.low, "ci_high": self.high}


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Percentile 95% interval from resampling whole patients."""
    rng = np.random.default_rng(seed)
    patients = frame.patient.unique()
    by_patient = {p: frame[frame.patient == p] for p in patients}
    draws = []
    for _ in range(resamples):
        picked = rng.choice(patients, size=patients.size, replace=True)
        # Relabel duplicates so a patient drawn twice counts as two clusters.
        sample = pd.concat(
            [by_patient[p].assign(patient=f"{p}#{k}") for k, p in enumerate(picked)],
            ignore_index=True,
        )
        value = statistic(sample)
        if np.isfinite(value):
            draws.append(value)
    low, high = np.percentile(draws, [2.5, 97.5])
    return Estimate(float(statistic(frame)), float(low), float(high))


def pearson(x: str, y: str) -> Callable[[pd.DataFrame], float]:
    return lambda f: float(stats.pearsonr(f[x], f[y])[0])


def spearman(x: str, y: str) -> Callable[[pd.DataFrame], float]:
    return lambda f: float(stats.spearmanr(f[x], f[y])[0])


def within(x: str, y: str) -> Callable[[pd.DataFrame], float]:
    return lambda f: within_between(f, x, y)[0]


def between(x: str, y: str) -> Callable[[pd.DataFrame], float]:
    return lambda f: within_between(f, x, y)[1]


def mixed_model(frame: pd.DataFrame, formula: str) -> dict[str, float]:
    """Random-intercept-per-patient linear model; returns the log-horizon term."""
    import statsmodels.formula.api as smf

    fit = smf.mixedlm(formula, frame, groups=frame["patient"]).fit(reml=True)
    low, high = fit.conf_int().loc["log_horizon"]
    return {
        "coefficient": float(fit.params["log_horizon"]),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(fit.pvalues["log_horizon"]),
    }


def summarize(
    frame: pd.DataFrame, *, resamples: int = 2000, seed: int = 0, short_interval_days: int = 28
) -> dict[str, object]:
    """The full interval-versus-change panel used for every cohort."""
    f = add_change_columns(frame)
    boot = {"resamples": resamples, "seed": seed}
    short = f.horizon_days <= short_interval_days
    panel: dict[str, object] = {
        "transitions": len(f),
        "patients": int(f.patient.nunique()),
        "median_horizon_days": float(f.horizon_days.median()),
        "correlations": {
            "pearson_log_horizon_abs_change": cluster_bootstrap(
                f, pearson("log_horizon", "abs_change"), **boot
            ).as_dict(),
            "spearman_horizon_abs_change": cluster_bootstrap(
                f, spearman("horizon_days", "abs_change"), **boot
            ).as_dict(),
            "spearman_horizon_abs_relative_change": cluster_bootstrap(
                f, spearman("horizon_days", "abs_relative_change"), **boot
            ).as_dict(),
            "within_patient_log_horizon_log_abs_change": cluster_bootstrap(
                f, within("log_horizon", "log_abs_change"), **boot
            ).as_dict(),
            "between_patient_log_horizon_log_abs_change": cluster_bootstrap(
                f, between("log_horizon", "log_abs_change"), **boot
            ).as_dict(),
            "within_patient_log_horizon_abs_relative_change": cluster_bootstrap(
                f, within("log_horizon", "abs_relative_change"), **boot
            ).as_dict(),
        },
        "noise_floor": {
            "short_interval_days": short_interval_days,
            "short_transitions": int(short.sum()),
            "median_abs_change_short": float(f.abs_change[short].median()) if short.any() else None,
            "median_abs_change_other": float(f.abs_change[~short].median()),
            "median_abs_relative_change_short": (
                float(f.abs_relative_change[short].median()) if short.any() else None
            ),
            "median_abs_relative_change_other": float(f.abs_relative_change[~short].median()),
        },
    }
    try:
        panel["mixed_model_log_abs_change"] = mixed_model(
            f, "log_abs_change ~ log_horizon + log_source_volume"
        )
        panel["mixed_model_abs_relative_change"] = mixed_model(
            f, "abs_relative_change ~ log_horizon + log_source_volume"
        )
    except Exception as error:  # report rather than abort the panel
        panel["mixed_model_error"] = repr(error)
    return panel

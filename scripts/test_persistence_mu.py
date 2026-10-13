"""Pre-registered persistence test on MU-Glioma-Post (Amendment 1).

Two rivals, steady growth with outlier-contaminated noise and an abrupt regime
change, are each calibrated to reproduce the observed bunching and residual
noise. Their predictions for the persistence of consecutive radius changes,
which neither was calibrated to, are then compared with the observed value.

Requires ``outputs/surveillance/mu_scan_volumes.csv`` from
``scripts/test_switching_mu.py``.

Usage:
    uv run --extra analysis --extra dev python scripts/test_persistence_mu.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from test_switching_mu import build_cohort

from gbm_pinn.switching import (
    Cohort,
    Draws,
    bunching,
    calibrate,
    fit_linear,
    persistence,
    persistence_verdict,
    sign_concordance,
    simulate_outliers,
    simulate_regime_change,
    simulate_steady,
)

PROTOCOL = Path("docs/preregistration_switching.md")
CALIBRATION_DRAWS = 300
PREDICTIVE_DRAWS = 2000
OUTLIER_GRID = np.round(np.arange(0.0, 0.51, 0.02), 2)


def analyse(cohort: Cohort, rng: np.random.Generator) -> dict[str, object]:
    fit = fit_linear(cohort)
    observed_bunching = float(bunching(cohort.radius))
    observed = float(persistence(cohort.radius))
    targets = {"target_bunching": observed_bunching, "target_sigma": fit.sigma}

    calibration_draws = Draws.sample(cohort, CALIBRATION_DRAWS, rng)
    outlier_fit = calibrate(
        cohort,
        lambda rate, s: simulate_outliers(cohort, fit, calibration_draws, rate=rate, noise=s),
        OUTLIER_GRID,
        **targets,
    )
    regime_fit = calibrate(
        cohort,
        lambda a, s: simulate_regime_change(cohort, fit, calibration_draws, amplitude=a, noise=s),
        np.linspace(0.0, 4.0 * fit.sigma, 33),
        **targets,
    )

    predictive = Draws.sample(cohort, PREDICTIVE_DRAWS, rng)
    outliers = simulate_outliers(
        cohort, fit, predictive, rate=outlier_fit.parameter, noise=outlier_fit.noise
    )
    regime = simulate_regime_change(
        cohort, fit, predictive, amplitude=regime_fit.parameter, noise=regime_fit.noise
    )
    steady = simulate_steady(cohort, fit, PREDICTIVE_DRAWS, rng, sigma=None)

    p_outliers, p_regime, p_steady = persistence(outliers), persistence(regime), persistence(steady)
    return {
        "segments": int(cohort.days.shape[0]),
        "consecutive_pairs": int(np.sum(~np.isnan(np.diff(cohort.radius, axis=-1)[:, 1:]))),
        "observed_bunching": observed_bunching,
        "observed_sigma_mm": fit.sigma,
        "observed_persistence": observed,
        "observed_sign_concordance": float(sign_concordance(cohort.radius)),
        "outlier_calibration": outlier_fit.as_dict(),
        "regime_calibration": regime_fit.as_dict(),
        "predicted_persistence_mean": {
            "steady": float(p_steady.mean()),
            "outliers": float(p_outliers.mean()),
            "regime_change": float(p_regime.mean()),
        },
        "predicted_sign_concordance_mean": {
            "steady": float(sign_concordance(steady).mean()),
            "outliers": float(sign_concordance(outliers).mean()),
            "regime_change": float(sign_concordance(regime).mean()),
        },
        "steady_p_upper": float((1 + np.sum(p_steady >= observed)) / (PREDICTIVE_DRAWS + 1)),
        "steady_p_lower": float((1 + np.sum(p_steady <= observed)) / (PREDICTIVE_DRAWS + 1)),
        "decision": persistence_verdict(observed, p_regime, p_outliers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surveillance"))
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()

    scans = pd.read_csv(args.output_dir / "mu_scan_volumes.csv")
    rng = np.random.default_rng(args.seed)
    report = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "primary_enhancing": analyse(build_cohort(scans, "enhancing"), rng),
        "secondary_core": analyse(build_cohort(scans, "core"), rng),
    }
    (args.output_dir / "persistence_test_mu.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

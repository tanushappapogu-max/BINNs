"""Run the synthetic PINN forecast and serialize its metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gbm_pinn.experiment import SyntheticExperimentConfig, run_synthetic_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--observation-times", type=float, nargs="+", default=(0.0, 0.4))
    parser.add_argument("--forecast-time", type=float, default=1.0)
    parser.add_argument("--initial-diffusivity", type=float, default=0.01)
    parser.add_argument("--initial-proliferation-rate", type=float, default=0.2)
    parser.add_argument("--fix-diffusivity", action="store_true")
    parser.add_argument("--fix-proliferation-rate", action="store_true")
    parser.add_argument("--lbfgs-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = run_synthetic_forecast(
        SyntheticExperimentConfig(
            epochs=arguments.epochs,
            seed=arguments.seed,
            observation_times=tuple(arguments.observation_times),
            forecast_time=arguments.forecast_time,
            initial_diffusivity=arguments.initial_diffusivity,
            initial_proliferation_rate=arguments.initial_proliferation_rate,
            learn_diffusivity=not arguments.fix_diffusivity,
            learn_proliferation_rate=not arguments.fix_proliferation_rate,
            lbfgs_max_iterations=arguments.lbfgs_iterations,
        )
    )
    serialized = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

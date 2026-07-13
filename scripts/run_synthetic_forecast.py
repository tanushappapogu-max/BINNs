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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--observation-times", type=float, nargs="+", default=(0.0, 0.4))
    parser.add_argument("--forecast-time", type=float, default=1.0)
    parser.add_argument("--initial-diffusivity", type=float, default=0.01)
    parser.add_argument("--initial-proliferation-rate", type=float, default=0.2)
    parser.add_argument("--initial-peak", type=float, default=0.6)
    parser.add_argument("--initial-standard-deviation", type=float, default=0.125)
    parser.add_argument("--diffusivity-upper-bound", type=float, default=0.02)
    parser.add_argument("--proliferation-upper-bound", type=float, default=0.8)
    parser.add_argument("--fix-diffusivity", action="store_true")
    parser.add_argument("--fix-proliferation-rate", action="store_true")
    parser.add_argument("--lbfgs-iterations", type=int, default=0)
    parser.add_argument("--data-batch-size", type=int)
    parser.add_argument("--collocation-batch-size", type=int)
    parser.add_argument("--boundary-batch-size", type=int)
    parser.add_argument("--interface-batch-size", type=int)
    parser.add_argument("--causal-time-chunks", type=int, default=1)
    parser.add_argument("--data-points-per-time", type=int, default=256)
    parser.add_argument("--observation-noise", type=float, default=0.0)
    parser.add_argument("--true-diffusivity", type=float, default=0.005)
    parser.add_argument("--true-proliferation-rate", type=float, default=0.4)
    parser.add_argument("--growth-law", choices=("logistic", "weak_allee"), default="logistic")
    parser.add_argument("--allee-parameter", type=float, default=0.0)
    parser.add_argument("--fitted-growth-law", choices=("logistic", "weak_allee"))
    parser.add_argument("--fitted-allee-parameter", type=float, default=0.0)
    parser.add_argument("--tissue-ratio", type=float, default=1.0)
    parser.add_argument("--cavity-radius", type=float, default=0.0)
    parser.add_argument("--cavity-interface-radius", type=float, default=0.0)
    parser.add_argument("--interface-points", type=int, default=128)
    parser.add_argument("--interface-weight", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--collocation-points", type=int, default=512)
    parser.add_argument("--boundary-points", type=int, default=128)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument("--physics-weight", type=float, default=5.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = run_synthetic_forecast(
        SyntheticExperimentConfig(
            epochs=arguments.epochs,
            seed=arguments.seed,
            device=arguments.device,
            observation_times=tuple(arguments.observation_times),
            forecast_time=arguments.forecast_time,
            initial_diffusivity=arguments.initial_diffusivity,
            initial_proliferation_rate=arguments.initial_proliferation_rate,
            initial_peak=arguments.initial_peak,
            initial_standard_deviation=arguments.initial_standard_deviation,
            diffusivity_bounds=(0.001, arguments.diffusivity_upper_bound),
            proliferation_bounds=(0.05, arguments.proliferation_upper_bound),
            learn_diffusivity=not arguments.fix_diffusivity,
            learn_proliferation_rate=not arguments.fix_proliferation_rate,
            lbfgs_max_iterations=arguments.lbfgs_iterations,
            data_batch_size=arguments.data_batch_size,
            collocation_batch_size=arguments.collocation_batch_size,
            boundary_batch_size=arguments.boundary_batch_size,
            interface_batch_size=arguments.interface_batch_size,
            causal_time_chunks=arguments.causal_time_chunks,
            data_points_per_time=arguments.data_points_per_time,
            observation_noise_standard_deviation=arguments.observation_noise,
            true_diffusivity=arguments.true_diffusivity,
            true_proliferation_rate=arguments.true_proliferation_rate,
            growth_law=arguments.growth_law,
            allee_parameter=arguments.allee_parameter,
            fitted_growth_law=arguments.fitted_growth_law,
            fitted_allee_parameter=arguments.fitted_allee_parameter,
            white_to_gray_diffusivity_ratio=arguments.tissue_ratio,
            cavity_radius=arguments.cavity_radius,
            cavity_interface_radius=arguments.cavity_interface_radius,
            interface_points=arguments.interface_points,
            interface_weight=arguments.interface_weight,
            grid_size=arguments.grid_size,
            collocation_points=arguments.collocation_points,
            boundary_points=arguments.boundary_points,
            hidden_width=arguments.hidden_width,
            data_weight=arguments.data_weight,
            physics_weight=arguments.physics_weight,
            boundary_weight=arguments.boundary_weight,
        )
    )
    serialized = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

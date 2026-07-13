"""Run the real-patient MU-Glioma-Post forecasting pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gbm_pinn.clinical_experiment import ClinicalPilotConfig, run_clinical_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_directory", type=Path)
    parser.add_argument("--scan-days", type=float, nargs="+", required=True)
    parser.add_argument("--observation-count", type=int, default=2)
    parser.add_argument("--forecast-index", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=162)
    parser.add_argument("--data-points-per-time", type=int, default=2_048)
    parser.add_argument("--collocation-points", type=int, default=4_096)
    parser.add_argument("--boundary-points", type=int, default=1_024)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--network-learning-rate", type=float, default=1e-3)
    parser.add_argument("--parameter-learning-rate", type=float, default=2e-3)
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/clinical/checkpoint.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact", type=Path, default=Path("outputs/clinical/forecast.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/clinical/metrics.json"))
    arguments = parser.parse_args()

    result = run_clinical_pilot(
        ClinicalPilotConfig(
            patient_directory=arguments.patient_directory,
            scan_days=tuple(arguments.scan_days),
            observation_count=arguments.observation_count,
            forecast_index=arguments.forecast_index,
            epochs=arguments.epochs,
            device=arguments.device,
            seed=arguments.seed,
            data_points_per_time=arguments.data_points_per_time,
            collocation_points=arguments.collocation_points,
            boundary_points=arguments.boundary_points,
            hidden_width=arguments.hidden_width,
            hidden_layers=arguments.hidden_layers,
            network_learning_rate=arguments.network_learning_rate,
            parameter_learning_rate=arguments.parameter_learning_rate,
            data_weight=arguments.data_weight,
            physics_weight=arguments.physics_weight,
            boundary_weight=arguments.boundary_weight,
            checkpoint_interval=arguments.checkpoint_interval,
            checkpoint_path=arguments.checkpoint,
            resume_from_checkpoint=arguments.resume,
            artifact_path=arguments.artifact,
        )
    )
    serialized = json.dumps(asdict(result), indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

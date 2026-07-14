"""Run one real-patient compartment-PINN forecast from the command line."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gbm_pinn.multicompartment_clinical import (
    MultiCompartmentClinicalConfig,
    run_multicompartment_clinical,
)
from gbm_pinn.multicompartment_pinn import (
    MultiCompartmentPINNConfig,
    MultiCompartmentTrainingConfig,
)
from gbm_pinn.pinn import PINNConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-directory", type=Path, required=True)
    parser.add_argument("--scan-days", type=float, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=162)
    parser.add_argument("--data-points-per-time", type=int, default=8_192)
    parser.add_argument("--collocation-points", type=int, default=16_384)
    parser.add_argument("--boundary-points", type=int, default=4_096)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--edema-half-saturation", type=float, default=0.1)
    parser.add_argument("--no-normalize-loss-terms", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    network = PINNConfig(
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        diffusivity_bounds=(0.01, 2.0),
        proliferation_bounds=(0.001, 0.05),
        initial_diffusivity=0.13,
        initial_proliferation_rate=0.012,
        fourier_frequencies=(1.0, 2.0, 4.0, 8.0),
    )
    result = run_multicompartment_clinical(
        MultiCompartmentClinicalConfig(
            patient_directory=args.patient_directory,
            scan_days=tuple(args.scan_days),
            device=args.device,
            seed=args.seed,
            data_points_per_time=args.data_points_per_time,
            collocation_points=args.collocation_points,
            boundary_points=args.boundary_points,
            pinn=MultiCompartmentPINNConfig(
                network=network,
                edema_half_saturation=args.edema_half_saturation,
            ),
            training=MultiCompartmentTrainingConfig(
                epochs=args.epochs,
                normalize_loss_terms=not args.no_normalize_loss_terms,
            ),
            checkpoint_path=args.output_directory / "checkpoint.pt",
            resume_from_checkpoint=args.resume,
            artifact_path=args.output_directory / "forecast.npz",
        )
    )
    metrics_path = args.output_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

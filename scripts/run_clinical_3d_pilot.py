"""Run the full-volume MU-Glioma-Post forecasting pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gbm_pinn.clinical_3d_experiment import Clinical3DPilotConfig, run_clinical_3d_pilot
from gbm_pinn.treatment import TreatmentWindow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_directory", type=Path)
    parser.add_argument("--scan-days", type=float, nargs="+", required=True)
    parser.add_argument("--observation-count", type=int, default=2)
    parser.add_argument("--forecast-index", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=162)
    parser.add_argument("--data-points-per-time", type=int, default=8_192)
    parser.add_argument("--tumor-sample-fraction", type=float, default=0.1)
    parser.add_argument("--collocation-points", type=int, default=16_384)
    parser.add_argument("--boundary-points", type=int, default=4_096)
    parser.add_argument("--evaluation-batch-size", type=int, default=65_536)
    parser.add_argument("--hidden-width", type=int, default=48)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--fourier-frequencies", type=float, nargs="*", default=[])
    parser.add_argument("--data-weight", type=float, default=10.0)
    parser.add_argument(
        "--treatment-window",
        type=float,
        nargs=4,
        action="append",
        metavar=("START", "END", "INTENSITY", "DECAY"),
        default=[],
    )
    parser.add_argument("--hold-diffusivity-fixed", action="store_true")
    parser.add_argument("--hold-proliferation-fixed", action="store_true")
    parser.add_argument("--hold-treatment-response-fixed", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/clinical_3d/checkpoint.pt")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact", type=Path, default=Path("outputs/clinical_3d/forecast.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/clinical_3d/metrics.json"))
    arguments = parser.parse_args()
    treatment_windows = tuple(TreatmentWindow(*values) for values in arguments.treatment_window)
    result = run_clinical_3d_pilot(
        Clinical3DPilotConfig(
            patient_directory=arguments.patient_directory,
            scan_days=tuple(arguments.scan_days),
            observation_count=arguments.observation_count,
            forecast_index=arguments.forecast_index,
            epochs=arguments.epochs,
            device=arguments.device,
            seed=arguments.seed,
            data_points_per_time=arguments.data_points_per_time,
            tumor_sample_fraction=arguments.tumor_sample_fraction,
            collocation_points=arguments.collocation_points,
            boundary_points=arguments.boundary_points,
            evaluation_batch_size=arguments.evaluation_batch_size,
            hidden_width=arguments.hidden_width,
            hidden_layers=arguments.hidden_layers,
            fourier_frequencies=tuple(arguments.fourier_frequencies),
            data_weight=arguments.data_weight,
            treatment_windows=treatment_windows,
            learn_diffusivity=not arguments.hold_diffusivity_fixed,
            learn_proliferation_rate=not arguments.hold_proliferation_fixed,
            learn_treatment_response=not arguments.hold_treatment_response_fixed,
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

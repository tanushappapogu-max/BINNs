"""Run a 3D mechanistic rollout from the final observed clinical mask."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gbm_pinn.mechanistic_forecast import MechanisticForecastConfig, run_mechanistic_forecast
from gbm_pinn.treatment import TreatmentWindow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_directory", type=Path)
    parser.add_argument("--scan-days", type=float, nargs="+", required=True)
    parser.add_argument("--observation-count", type=int, required=True)
    parser.add_argument("--forecast-index", type=int, required=True)
    parser.add_argument("--diffusivity", type=float, required=True)
    parser.add_argument("--proliferation", type=float, required=True)
    parser.add_argument("--treatment-response", type=float, default=0.0)
    parser.add_argument("--treatment-time-offset", type=float, default=0.0)
    parser.add_argument(
        "--treatment-window",
        type=float,
        nargs=4,
        action="append",
        metavar=("START", "END", "INTENSITY", "DECAY"),
        default=[],
    )
    parser.add_argument("--maximum-time-step", type=float)
    parser.add_argument("--artifact", type=Path, default=Path("outputs/mechanistic/forecast.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mechanistic/metrics.json"))
    arguments = parser.parse_args()
    result = run_mechanistic_forecast(
        MechanisticForecastConfig(
            patient_directory=arguments.patient_directory,
            scan_days=tuple(arguments.scan_days),
            observation_count=arguments.observation_count,
            forecast_index=arguments.forecast_index,
            diffusivity_mm2_per_day=arguments.diffusivity,
            proliferation_per_day=arguments.proliferation,
            treatment_response_per_day=arguments.treatment_response,
            treatment_windows=tuple(
                TreatmentWindow(*values) for values in arguments.treatment_window
            ),
            treatment_time_offset_days=arguments.treatment_time_offset,
            maximum_time_step=arguments.maximum_time_step,
            artifact_path=arguments.artifact,
        )
    )
    serialized = json.dumps(asdict(result), indent=2, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

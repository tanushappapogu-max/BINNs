"""Run the frozen multi-compartment training cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_pinn.multicompartment_cohort import run_multicompartment_training_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--patient", action="append", dest="patients")
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        choices=("training", "model_selection", "final_test"),
        help="cohort role to run; defaults to training only",
    )
    parser.add_argument("--edema-half-saturation", type=float)
    parser.add_argument("--enable-systemic-cell-kill", action="store_true")
    parser.add_argument("--field-warmup-epochs", type=int)
    parser.add_argument("--parameter-calibration-epochs", type=int)
    parser.add_argument("--observation-count", type=int)
    parser.add_argument("--forecast-index", type=int)
    parser.add_argument("--scan-start-index", type=int)
    args = parser.parse_args()
    protocol_overrides = {}
    if args.field_warmup_epochs is not None:
        protocol_overrides["field_warmup_epochs"] = args.field_warmup_epochs
    if args.parameter_calibration_epochs is not None:
        protocol_overrides["parameter_calibration_epochs"] = (
            args.parameter_calibration_epochs
        )
    if args.observation_count is not None:
        protocol_overrides["observation_count"] = args.observation_count
    if args.forecast_index is not None:
        protocol_overrides["forecast_index"] = args.forecast_index
    if args.scan_start_index is not None:
        protocol_overrides["scan_start_index"] = args.scan_start_index
    result = run_multicompartment_training_cohort(
        args.manifest,
        args.data_root,
        args.output_root,
        device=args.device,
        selected_patient_ids=set(args.patients) if args.patients else None,
        included_roles=set(args.roles) if args.roles else None,
        edema_half_saturation=args.edema_half_saturation,
        enable_systemic_cell_kill=args.enable_systemic_cell_kill,
        protocol_overrides=protocol_overrides,
    )
    print(json.dumps(result["training_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

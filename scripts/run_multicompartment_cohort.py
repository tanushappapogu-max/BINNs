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
    parser.add_argument("--edema-half-saturation", type=float)
    parser.add_argument("--enable-systemic-cell-kill", action="store_true")
    args = parser.parse_args()
    result = run_multicompartment_training_cohort(
        args.manifest,
        args.data_root,
        args.output_root,
        device=args.device,
        selected_patient_ids=set(args.patients) if args.patients else None,
        edema_half_saturation=args.edema_half_saturation,
        enable_systemic_cell_kill=args.enable_systemic_cell_kill,
    )
    print(json.dumps(result["training_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Run per-transition PINN forecasting on MU-Glioma-Post transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_pinn.pinn_cohort import PINNCohortConfig, run_pinn_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nifti-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--role", default="training")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = PINNCohortConfig(
        transition_index_path=args.transition_index,
        manifest_path=args.manifest,
        nifti_root=args.nifti_root,
        output_root=args.output_root,
        role=args.role,
        device=args.device,
        downsample=args.downsample,
        epochs=args.epochs,
        max_transitions=args.max_transitions,
        resume=args.resume,
    )
    summary = run_pinn_cohort(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

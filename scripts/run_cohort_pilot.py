"""Run the locked MU-Glioma-Post development and validation cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_pinn.cohort import run_cohort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/derived/mu_glioma_post/cohort_manifest.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mu_glioma_post/nifti"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cohort"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--patient", action="append", default=None)
    arguments = parser.parse_args()
    result = run_cohort(
        arguments.manifest,
        arguments.data_root,
        arguments.output_root,
        device=arguments.device,
        selected_patient_ids=None if arguments.patient is None else set(arguments.patient),
    )
    print(json.dumps(result["validation_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

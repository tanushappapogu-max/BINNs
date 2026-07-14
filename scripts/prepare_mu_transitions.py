"""Validate and index locally available MU patient-to-patient transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbm_pinn.mu_transitions import build_mu_transition_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nifti-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        choices=("training", "model_selection", "final_test"),
    )
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    index = build_mu_transition_index(
        args.manifest,
        args.nifti_root,
        included_roles=set(args.roles) if args.roles else None,
        allow_final_test=args.allow_final_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: index[key] for key in (
                "patient_count",
                "locally_complete_patient_count",
                "transition_count",
                "missing_patient_ids",
            )},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

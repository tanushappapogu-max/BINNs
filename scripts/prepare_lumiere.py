"""Index and selectively extract aligned LUMIERE masks without downloading the 32 GB ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gbm_pinn.lumiere import (
    LUMIERE_ARCHIVE_URL,
    build_lumiere_manifest,
    index_lumiere_sessions,
    remap_lumiere_segmentation,
    validate_prepared_lumiere,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive-url", default=LUMIERE_ARCHIVE_URL)
    parser.add_argument("--patient", action="append", dest="patients")
    parser.add_argument("--max-patients", type=int, default=1)
    parser.add_argument("--minimum-sessions", type=int, default=4)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.max_patients <= 0 or args.minimum_sessions < 4:
        raise ValueError("max-patients must be positive and minimum-sessions at least four")
    try:
        import nibabel as nib
        from remotezip import RemoteZip
    except ImportError as error:
        raise ImportError("install the 'datasets' extra to prepare LUMIERE") from error

    args.output_root.mkdir(parents=True, exist_ok=True)
    with RemoteZip(args.archive_url) as archive:
        indexed = index_lumiere_sessions(archive.namelist())
        eligible = {
            patient: sessions
            for patient, sessions in indexed.items()
            if len(sessions) >= args.minimum_sessions
        }
        if args.patients:
            unknown = set(args.patients) - set(eligible)
            if unknown:
                raise ValueError(f"unknown or ineligible patients: {', '.join(sorted(unknown))}")
            selected = {patient: eligible[patient] for patient in args.patients}
        else:
            ranked = sorted(eligible.items(), key=lambda item: (-len(item[1]), item[0]))
            selected = dict(ranked[: args.max_patients])
        print(
            json.dumps(
                {
                    "archive_patients": len(indexed),
                    "eligible_patients": len(eligible),
                    "selected": {
                        patient: [session.week for session in sessions]
                        for patient, sessions in selected.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.list_only:
            return
        for patient, sessions in selected.items():
            for session in sessions:
                destination = args.output_root / patient / f"week-{session.week:03d}"
                destination.mkdir(parents=True, exist_ok=True)
                segmentation_path = destination / (
                    f"{patient}_week-{session.week:03d}_tumorMask.nii.gz"
                )
                brain_path = destination / f"{patient}_week-{session.week:03d}_brain_mask.nii.gz"
                if segmentation_path.is_file() and brain_path.is_file():
                    continue
                _write_remapped_image(
                    archive.read(session.segmentation_member),
                    segmentation_path,
                    nib,
                    remap=True,
                )
                _write_remapped_image(
                    archive.read(session.brain_mask_member),
                    brain_path,
                    nib,
                    remap=False,
                )

    validate_prepared_lumiere(args.output_root, selected)

    if args.manifest is not None:
        manifest = build_lumiere_manifest(
            args.output_root,
            _default_protocol(),
            minimum_sessions=args.minimum_sessions,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_remapped_image(data: bytes, destination: Path, nib, *, remap: bool) -> None:
    temporary = destination.with_name(f".{destination.stem}.download.nii.gz")
    temporary.write_bytes(data)
    try:
        image = nib.as_closest_canonical(nib.load(temporary))
        labels = np.asanyarray(image.dataobj)
        output = remap_lumiere_segmentation(labels) if remap else (labels > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(output, image.affine, image.header), destination)
    finally:
        temporary.unlink(missing_ok=True)


def _default_protocol() -> dict[str, object]:
    return {
        "observation_count": 3,
        "forecast_index": 3,
        "epochs": 300,
        "data_points_per_time": 2048,
        "collocation_points": 4096,
        "boundary_points": 1024,
        "hidden_width": 48,
        "hidden_layers": 4,
        "fourier_frequencies": [1.0, 2.0, 4.0, 8.0],
        "infiltrative_viable_density": 0.3,
        "detection_limits": [0.1, 0.1, 0.1],
        "latent_seed_dilation_mm": 10.0,
        "edema_half_saturation": 0.3,
        "seed": 162,
    }


if __name__ == "__main__":
    main()

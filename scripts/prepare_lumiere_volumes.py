"""Per-session LUMIERE tumor volumes and RANO ratings, read straight from figshare.

Reads only the DeepBraTumIA atlas-space segmentation of every session from the
32 GB imaging ZIP using HTTP range requests, so nothing large is downloaded.
Unlike ``prepare_lumiere.py`` this keeps surgery-day sessions, whose folders
carry a suffix (``week-000-1`` pre-operative, ``week-000-2`` post-operative),
because they mark the interventions the surveillance analysis must split at.

DeepBraTumIA labels: 1 enhancing, 2 necrotic or non-enhancing core, 3 edema.
After surgery the necrotic label also covers the resection cavity, so the
core volume is not a clean tumor measure; enhancing tumor is the primary one.

Usage:
    uv run --extra datasets --extra dev python scripts/prepare_lumiere_volumes.py
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from remotezip import RemoteZip

from gbm_pinn.lumiere import LUMIERE_ARCHIVE_URL

SEGMENTATION = re.compile(
    r"^Imaging/(?P<patient>Patient-\d+)/(?P<session>week-(?P<week>\d+)(?:-(?P<part>\d+))?)/"
    r"DeepBraTumIA-segmentation/atlas/segmentation/seg_mask\.nii\.gz$"
)
RATINGS = Path("data/raw/lumiere_clinical/LUMIERE-ExpertRating-v202211.csv")


def label_volumes(data: bytes) -> dict[str, float]:
    image = nib.Nifti1Image.from_bytes(gzip.decompress(data))
    labels = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    unexpected = set(np.unique(labels).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise ValueError(f"unexpected DeepBraTumIA labels {sorted(unexpected)}")
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return {
        "enhancing_ml": float((labels == 1).sum() * voxel_ml),
        "core_ml": float(np.isin(labels, (1, 2)).sum() * voxel_ml),
        "whole_ml": float(np.isin(labels, (1, 2, 3)).sum() * voxel_ml),
    }


def load_ratings() -> pd.DataFrame:
    ratings = pd.read_csv(RATINGS)
    rating_column = next(c for c in ratings.columns if c.startswith("Rating (according"))
    return pd.DataFrame(
        {
            "patient": ratings["Patient"].str.strip(),
            "session": ratings["Date"].str.strip(),
            "rano": ratings[rating_column].astype(str).str.strip(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path("outputs/surveillance/lumiere_scan_volumes.csv")
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with RemoteZip(LUMIERE_ARCHIVE_URL) as archive:
        members = [(m, name) for name in archive.namelist() if (m := SEGMENTATION.match(name))]
        for count, (match, name) in enumerate(sorted(members, key=lambda item: item[1]), 1):
            rows.append(
                {
                    "patient": match["patient"],
                    "session": match["session"],
                    "week": int(match["week"]),
                    "surgery_part": int(match["part"]) if match["part"] else 0,
                    **label_volumes(archive.read(name)),
                }
            )
            if count % 50 == 0:
                print(f"{count}/{len(members)} sessions", flush=True)

    frame = pd.DataFrame(rows).merge(load_ratings(), on=["patient", "session"], how="left")
    frame.to_csv(args.output, index=False)
    print(
        f"wrote {len(frame)} sessions from {frame.patient.nunique()} patients; "
        f"{frame.rano.isna().sum()} sessions without a RANO rating"
    )


if __name__ == "__main__":
    main()

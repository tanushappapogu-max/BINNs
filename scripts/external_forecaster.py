"""External validation of the physics-informed forecaster: train on MU, test on LUMIERE.

Three subcommands, run in order:

``prepare``
    Writes deduplicated, treatment-free MU transition indices (training,
    model selection, final test) and a LUMIERE external index. Duplicate MU
    patients are removed (0062 duplicates 0060, 0132 duplicates 0113) and
    transitions touching the byte-identical timepoint 5 of 0018 and 0032 are
    dropped. Treatment exposure is removed because LUMIERE has none; on MU
    this was shown not to change skill. The LUMIERE transitions are exactly the
    consecutive pairs of the pre-registered ceiling analysis (surgery splits,
    28-day rule, segments of at least three scans), so the forecaster and the
    ceiling are scored on the same pairs.

``train``
    Trains with ``gbm_pinn.unet_forecaster.train`` on MU training, selecting
    the epoch on MU model selection. Nothing from LUMIERE is used.

``evaluate``
    Scores the selected model, unchanged, on MU final test and on LUMIERE:
    Dice skill over persistence with the patient-clustered test from
    ``summarize_records``, and the reduction in squared error of
    equivalent-sphere radius relative to persistence, which is directly
    comparable with the forecastability ceiling.

Paths inside the indices are relative to ``--data-root``, so the same files
work locally and on Colab with Google Drive.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_ceiling_lumiere import EXCLUSION_DAYS, surgery_segments
from torch.utils.data import DataLoader

from gbm_pinn.unet_forecaster import (
    ParameterNet,
    ReactionDiffusionSimulator,
    TrainConfig,
    TumorTransitionDataset,
    _resolve_device,
    evaluate,
    train,
)

MU_DERIVED = Path("data/derived/mu_glioma_post")
EXTERNAL = Path("data/derived/external")
LUMIERE_MASKS = Path("data/raw/lumiere_masks")
DUPLICATE_PATIENTS = {"PatientID_0062", "PatientID_0132"}
DUPLICATE_TIMEPOINTS = {("PatientID_0018", 5), ("PatientID_0032", 5)}
ROLES = {"training": "training", "model_selection": "model_selection", "final_test": "final_test"}


def _timepoints(t: dict) -> tuple[int, int]:
    """Source and target timepoints, from fields or from an ID like ``..._T4_to_T5``."""
    if "source_timepoint" in t and "target_timepoint" in t:
        return int(t["source_timepoint"]), int(t["target_timepoint"])
    match = re.search(r"_T(\d+)_to_T(\d+)$", t["transition_id"])
    if match is None:
        raise ValueError(f"cannot read timepoints of {t['transition_id']}")
    return int(match[1]), int(match[2])


def _clean_mu(transitions: list[dict]) -> list[dict]:
    kept = []
    for t in transitions:
        if t["patient_id"] in DUPLICATE_PATIENTS:
            continue
        source, target = _timepoints(t)
        touched = {(t["patient_id"], source), (t["patient_id"], target)}
        if touched & DUPLICATE_TIMEPOINTS:
            continue
        kept.append({**t, "treatment_exposure": {}})
    return kept


def lumiere_transitions(volumes: pd.DataFrame) -> list[dict]:
    """Consecutive pairs from the pre-registered ceiling segments."""
    out = []
    for patient, rows in volumes.groupby("patient"):
        for segment_index, part in enumerate(surgery_segments(rows)):
            part = part.reset_index(drop=True)
            days = part.week.to_numpy(dtype=float) * 7.0
            cuts = np.flatnonzero(np.diff(days) <= EXCLUSION_DAYS) + 1
            for piece, index in enumerate(np.split(np.arange(len(part)), cuts)):
                if len(index) < 3 or not np.all(np.diff(days[index]) > 0):
                    continue
                for a, b in pairwise(index):
                    source, target = part.session[a], part.session[b]
                    out.append(
                        {
                            "transition_id": f"{patient}_{source}_to_{target}",
                            "patient_id": patient,
                            "source": "LUMIERE",
                            "role": "external_test",
                            "segment": f"{patient}_{segment_index}_{piece}",
                            "source_day": float(days[a]),
                            "target_day": float(days[b]),
                            "source_segmentation": str(
                                LUMIERE_MASKS / patient / f"{source}_tumorMask.nii.gz"
                            ),
                            "target_segmentation": str(
                                LUMIERE_MASKS / patient / f"{target}_tumorMask.nii.gz"
                            ),
                            "treatment_exposure": {},
                        }
                    )
    return out


def prepare(args: argparse.Namespace) -> None:
    root = args.data_root
    (root / EXTERNAL).mkdir(parents=True, exist_ok=True)
    for name, role in ROLES.items():
        index = root / MU_DERIVED / f"shared_{name}_transitions.json"
        if not index.is_file():
            raise SystemExit(f"missing MU index {index}; copy data/derived/mu_glioma_post to Drive")
        payload = json.loads(index.read_text())
        cleaned = _clean_mu(payload["transitions"])
        out = {**payload, "transitions": cleaned, "transition_count": len(cleaned), "roles": [role]}
        (root / EXTERNAL / f"mu_{name}_clean.json").write_text(json.dumps(out, indent=2))
        print(f"MU {name}: {len(payload['transitions'])} -> {len(cleaned)} transitions")
    volumes = pd.read_csv(args.lumiere_volumes)
    lumiere = lumiere_transitions(volumes)
    missing = [
        t["transition_id"]
        for t in lumiere
        if not (root / t["source_segmentation"]).is_file()
        or not (root / t["target_segmentation"]).is_file()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} LUMIERE masks missing under {root}, e.g. {missing[:3]}")
    payload = {"dataset": "LUMIERE", "roles": ["external_test"], "transitions": lumiere}
    (root / EXTERNAL / "lumiere_external.json").write_text(json.dumps(payload, indent=2))
    patients = len({t["patient_id"] for t in lumiere})
    print(f"LUMIERE external: {len(lumiere)} transitions from {patients} patients")


def _shape(extent: tuple[int, ...], downsample: int, multiple: int = 8) -> tuple[int, int, int]:
    """Smallest padded shape holding the downsampled volume, divisible by ``multiple``."""
    return tuple(int(math.ceil(math.ceil(n / downsample) / multiple) * multiple) for n in extent)  # type: ignore[return-value]


def _config(args: argparse.Namespace, seed: int) -> TrainConfig:
    root = args.data_root
    return TrainConfig(
        transition_index_path=root / EXTERNAL / "mu_training_clean.json",
        manifest_path=root / MU_DERIVED / "shared_split_manifest.json",
        val_transition_index_path=root / EXTERNAL / "mu_model_selection_clean.json",
        data_root=root,
        output_root=args.output / f"seed_{seed}",
        device=args.device,
        downsample=args.downsample,
        target_shape=_shape((240, 240, 155), args.downsample),
        base_filters=16,
        n_steps=30,
        dt=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        param_reg_weight=0.1,
        time_scaled=False,
        use_cavity_domain=False,
        num_workers=args.num_workers,
        seed=seed,
    )


def run_training(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        done = args.output / f"seed_{seed}" / "results.json"
        if done.is_file():
            # Colab sessions can disconnect; a rerun skips finished seeds.
            print(f"seed {seed}: already trained ({done}), skipping", flush=True)
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = train(_config(args, seed))
        print(f"seed {seed}: model-selection skill {result['mean_skill']:+.4f}", flush=True)


def _radius_mm(volume_ml: np.ndarray) -> np.ndarray:
    return np.cbrt(3.0 * np.asarray(volume_ml) * 1000.0 / (4.0 * np.pi))


def radius_error_reduction(records: list[dict], resamples: int, seed: int) -> dict[str, object]:
    frame = pd.DataFrame(records)
    src = _radius_mm(frame.source_volume_ml)
    tgt = _radius_mm(frame.target_volume_ml)
    prd = _radius_mm(frame.predicted_volume_ml)
    frame["persistence_sq"] = (tgt - src) ** 2
    frame["model_sq"] = (tgt - prd) ** 2

    def reduction(f: pd.DataFrame) -> float:
        return 1.0 - float(f.model_sq.mean()) / float(f.persistence_sq.mean())

    rng = np.random.default_rng(seed)
    patients = frame.patient_id.unique()
    groups = {p: frame[frame.patient_id == p] for p in patients}
    draws = [
        reduction(pd.concat([groups[p] for p in rng.choice(patients, patients.size)]))
        for _ in range(resamples)
    ]
    return {
        "radius_mse_reduction_vs_persistence": reduction(frame),
        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def run_evaluation(args: argparse.Namespace) -> None:
    root = args.data_root
    sets = {
        "mu_final_test": (
            json.loads((root / EXTERNAL / "mu_final_test_clean.json").read_text())["transitions"],
            _shape((240, 240, 155), args.downsample),
        ),
        "lumiere_external": (
            json.loads((root / EXTERNAL / "lumiere_external.json").read_text())["transitions"],
            _shape((182, 218, 182), args.downsample),
        ),
    }
    summary: dict[str, object] = {"seeds": args.seeds, "downsample": args.downsample}
    for seed in args.seeds:
        cfg = _config(args, seed)
        device = _resolve_device(args.device)
        net = ParameterNet(n_treatment=4, base_filters=cfg.base_filters).to(device)
        state = torch.load(
            cfg.output_root / "best_model.pt", map_location=device, weights_only=True
        )
        net.load_state_dict(state)
        simulator = ReactionDiffusionSimulator(n_steps=cfg.n_steps, dt=cfg.dt).to(device)
        for name, (transitions, shape) in sets.items():
            dataset = TumorTransitionDataset(
                transitions, root, downsample=args.downsample, target_shape=shape
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
            result = evaluate(net, simulator, loader, device, cfg, save_records=True)
            records = result.pop("records")
            result.update(radius_error_reduction(records, args.resamples, seed))
            out = args.output / f"seed_{seed}" / f"{name}.json"
            out.write_text(json.dumps({**result, "records": records}, indent=2))
            summary[f"seed_{seed}/{name}"] = {
                k: result.get(k)
                for k in (
                    "n_total",
                    "n_patients",
                    "mean_skill",
                    "n_beating_persistence",
                    "wilcoxon_p_patient_clustered",
                    "mean_growth_dice",
                    "radius_mse_reduction_vs_persistence",
                    "ci",
                )
            }
            print(f"seed {seed} {name}: {summary[f'seed_{seed}/{name}']}", flush=True)
    (args.output / "external_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--lumiere-volumes",
        type=Path,
        default=Path("outputs/surveillance/lumiere_scan_volumes.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/external_forecaster"))
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    {"prepare": prepare, "train": run_training, "evaluate": run_evaluation}[args.command](args)


if __name__ == "__main__":
    main()

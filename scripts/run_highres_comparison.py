"""Two-arm high-resolution forecaster comparison for the Colab run.

Runs the reaction-diffusion forecaster with and without PDE time integration
and prints the contrast. Keeping the experiment here rather than in notebook
cells means a fresh ``git clone`` always executes the current logic, so the
Colab notebook can stay a thin launcher that never goes stale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gbm_pinn.unet_forecaster import TrainConfig, train


def _paths(drive_root: Path) -> dict[str, Path]:
    derived = drive_root / "data" / "derived" / "mu_glioma_post"
    return {
        "train": derived / "shared_training_transitions.json",
        "val": derived / "shared_model_selection_transitions.json",
        "manifest": derived / "shared_split_manifest.json",
        "output": drive_root / "data" / "outputs" / "unet_pinn",
    }


def _show(name: str, r: dict) -> None:
    print(
        f"{name:16s} skill={r['mean_skill']:+.4f}  "
        f"beat={r['n_beating_persistence']}/{r['n_total']}  "
        f"growth_dice={r['mean_growth_dice']:.4f}  "
        f"patients={r['n_patients_beating']}/{r['n_patients']}  "
        f"p={r.get('wilcoxon_p_patient_clustered', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"{'':16s} corr(pred change, gap) = "
        f"{r.get('pred_change_vs_gap_correlation', float('nan')):+.3f}   "
        f"[data: {r.get('true_change_vs_gap_correlation', float('nan')):+.3f}]",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    p = _paths(args.drive_root)
    for key in ("train", "val", "manifest"):
        assert p[key].exists(), f"missing: {p[key]}"
    print("All paths verified", flush=True)

    side = args.downsample
    shape = (240 // side, 240 // side, 160 // side)

    def run(time_scaled: bool, tag: str) -> dict:
        cfg = TrainConfig(
            transition_index_path=p["train"],
            manifest_path=p["manifest"],
            val_transition_index_path=p["val"],
            data_root=args.drive_root,
            output_root=p["output"] / tag,
            device=args.device,
            downsample=args.downsample,
            target_shape=shape,
            base_filters=16,
            n_steps=30,
            dt=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            param_reg_weight=0.1,
            time_scaled=time_scaled,
            use_cavity_domain=False,
        )
        print(f"\n===== {tag} (time_scaled={time_scaled}) =====", flush=True)
        return train(cfg)

    results = {
        "time_integrated": run(True, "time_integrated"),
        "time_blind": run(False, "time_blind"),
    }

    print("\n" + "=" * 78, flush=True)
    _show("time-integrated", results["time_integrated"])
    _show("time-blind", results["time_blind"])
    print("=" * 78, flush=True)
    print("corr>0 tracks tumor biology; corr<0 means the model learned the scan schedule.",
          flush=True)


if __name__ == "__main__":
    main()

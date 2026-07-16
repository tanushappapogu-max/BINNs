"""Physics-informed 3D U-Net for tumor growth prediction.

Trains on the full training cohort to learn tumor evolution patterns,
with a physics loss (reaction-diffusion PDE residual) that keeps
predictions biologically plausible.
"""

from __future__ import annotations

import json
import time as time_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from gbm_pinn.clinical import segmentation_to_density
from gbm_pinn.clinical_experiment import _masked_dice, _masked_volume_error
from gbm_pinn.shared_forecaster import load_transition_manifest

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# 3D U-Net Architecture
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """Compact 3D U-Net for tumor density prediction.

    Input channels:
        0: source tumor density
        1: normalized time horizon (days / 365)
        2: radiation exposure flag
        3: systemic chemo exposure flag
        4: antiangiogenic exposure flag
        5: device (TTFields) exposure flag

    Output: single channel predicted density change (residual from source).
    """

    def __init__(self, in_channels: int = 6, base_filters: int = 8, dropout: float = 0.3) -> None:
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(dropout)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)

        self.out_conv = nn.Conv3d(f, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.dropout(self.pool(e3)))

        d3 = self.up3(e4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(self.dropout(d3))

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(self.dropout(d2))

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff = [s - u for s, u in zip(skip.shape[2:], upsampled.shape[2:])]
        padding = []
        for d in reversed(diff):
            padding.extend([d // 2, d - d // 2])
        if any(p != 0 for p in padding):
            upsampled = F.pad(upsampled, padding)
        return torch.cat([upsampled, skip], dim=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TumorTransitionDataset(Dataset):
    """Loads transitions lazily, returns padded/downsampled volumes."""

    def __init__(
        self,
        transitions: list[dict[str, Any]],
        data_root: Path,
        downsample: int = 2,
        target_shape: tuple[int, int, int] = (120, 120, 80),
        infiltrative_density: float = 0.3,
        augment: bool = False,
    ) -> None:
        self.transitions = transitions
        self.data_root = data_root
        self.downsample = downsample
        self.target_shape = target_shape
        self.infiltrative_density = infiltrative_density
        self.augment = augment

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import nibabel as nib

        t = self.transitions[idx]
        ds = self.downsample

        source_img = nib.as_closest_canonical(
            nib.load(self.data_root / t["source_segmentation"]),
        )
        target_img = nib.as_closest_canonical(
            nib.load(self.data_root / t["target_segmentation"]),
        )

        source_labels = np.rint(np.asanyarray(source_img.dataobj)).astype(np.int16)
        target_labels = np.rint(np.asanyarray(target_img.dataobj)).astype(np.int16)

        if ds > 1:
            source_labels = source_labels[::ds, ::ds, ::ds]
            target_labels = target_labels[::ds, ::ds, ::ds]

        source_density = segmentation_to_density(
            source_labels, infiltrative_density=self.infiltrative_density,
        ).astype(np.float32)
        target_density = segmentation_to_density(
            target_labels, infiltrative_density=self.infiltrative_density,
        ).astype(np.float32)

        source_density = self._pad_to_target(source_density)
        target_density = self._pad_to_target(target_density)

        horizon_days = float(t["target_day"]) - float(t["source_day"])
        horizon_normalized = horizon_days / 365.0

        te = t.get("treatment_exposure", {})

        horizon_vol = np.full(self.target_shape, horizon_normalized, dtype=np.float32)
        radiation_vol = np.full(
            self.target_shape,
            float(te.get("radiation_exposure", 0.0) > 0),
            dtype=np.float32,
        )
        chemo_vol = np.full(
            self.target_shape,
            float(te.get("systemic_cytotoxic_exposure", 0.0) > 0),
            dtype=np.float32,
        )
        antiangio_vol = np.full(
            self.target_shape,
            float(te.get("antiangiogenic_exposure", 0.0) > 0),
            dtype=np.float32,
        )
        device_vol = np.full(
            self.target_shape,
            float(te.get("device_exposure", 0.0) > 0),
            dtype=np.float32,
        )

        input_tensor = torch.from_numpy(np.stack([
            source_density, horizon_vol, radiation_vol,
            chemo_vol, antiangio_vol, device_vol,
        ], axis=0))

        target_tensor = torch.from_numpy(target_density[np.newaxis])
        source_tensor = torch.from_numpy(source_density[np.newaxis])

        if self.augment:
            for dim in range(3):
                if np.random.random() > 0.5:
                    input_tensor = torch.flip(input_tensor, [dim + 1])
                    target_tensor = torch.flip(target_tensor, [dim + 1])
                    source_tensor = torch.flip(source_tensor, [dim + 1])

        return {
            "input": input_tensor,
            "target": target_tensor,
            "source": source_tensor,
            "horizon_days": torch.tensor(horizon_days, dtype=torch.float32),
            "transition_id": t["transition_id"],
        }

    def _pad_to_target(self, volume: np.ndarray) -> np.ndarray:
        ts = self.target_shape
        padded = np.zeros(ts, dtype=volume.dtype)
        slices = tuple(slice(0, min(v, t)) for v, t in zip(volume.shape, ts))
        src_slices = tuple(slice(0, min(v, t)) for v, t in zip(volume.shape, ts))
        padded[slices] = volume[src_slices]
        return padded


# ---------------------------------------------------------------------------
# Physics Loss
# ---------------------------------------------------------------------------

def physics_loss(
    prediction: torch.Tensor,
    source: torch.Tensor,
    horizon_days: torch.Tensor,
    spacing: float = 2.0,
) -> torch.Tensor:
    """Reaction-diffusion PDE residual as a soft constraint.

    Computes the PDE in normalized time (years) so the residual has
    magnitude comparable to the data loss.
    """
    batch_size = prediction.shape[0]
    u = prediction
    dt_years = (horizon_days / 365.0).view(batch_size, 1, 1, 1, 1).clamp(min=1.0 / 365.0)

    du_dt = (u - source) / dt_years

    laplacian = torch.zeros_like(u)
    for dim in range(3):
        laplacian += (
            torch.roll(u, 1, dims=dim + 2)
            + torch.roll(u, -1, dims=dim + 2)
            - 2.0 * u
        ) / (spacing ** 2)

    D = 0.02
    rho = 0.004
    reaction = rho * u * (1.0 - u)
    pde_residual = du_dt - D * laplacian - reaction

    tumor_mask = (u > 0.05) | (source > 0.05)
    if tumor_mask.any():
        return torch.mean(pde_residual[tumor_mask] ** 2)
    return torch.mean(pde_residual ** 2)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    transition_index_path: Path
    manifest_path: Path
    val_transition_index_path: Path
    data_root: Path = Path(".")
    output_root: Path = Path("outputs/unet_pinn")
    downsample: int = 2
    target_shape: tuple[int, int, int] = (120, 120, 80)
    base_filters: int = 8
    dropout: float = 0.3
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    physics_weight: float = 0.01
    infiltrative_density: float = 0.3
    threshold: float = 0.1
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def train(config: TrainConfig) -> dict[str, Any]:
    """Train the physics-informed U-Net and evaluate on validation set."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = _resolve_device(config.device)
    config.output_root.mkdir(parents=True, exist_ok=True)

    train_transitions = load_transition_manifest(
        config.transition_index_path, required_role="training",
    )
    val_transitions = load_transition_manifest(
        config.val_transition_index_path, required_role="model_selection",
    )

    print(f"Training: {len(train_transitions)} transitions", flush=True)
    print(f"Validation: {len(val_transitions)} transitions", flush=True)
    print(f"Device: {device}", flush=True)

    train_dataset = TumorTransitionDataset(
        train_transitions, config.data_root,
        downsample=config.downsample,
        target_shape=config.target_shape,
        infiltrative_density=config.infiltrative_density,
        augment=True,
    )
    val_dataset = TumorTransitionDataset(
        val_transitions, config.data_root,
        downsample=config.downsample,
        target_shape=config.target_shape,
        infiltrative_density=config.infiltrative_density,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config.num_workers,
    )

    model = UNet3D(
        in_channels=6, base_filters=config.base_filters, dropout=config.dropout,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs,
    )

    best_val_skill = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            source = batch["source"].to(device)
            horizon = batch["horizon_days"].to(device)

            residual = model(inp)
            predicted = source + residual
            predicted = predicted.clamp(0.0, 1.0)

            tumor_mask = (source > 0.05) | (target > 0.05)
            if tumor_mask.any():
                tumor_weight = torch.ones_like(predicted)
                tumor_weight[tumor_mask] = 10.0
                data_loss = (tumor_weight * (predicted - target) ** 2).mean()
            else:
                data_loss = F.mse_loss(predicted, target)
            phys_loss = physics_loss(
                predicted, source, horizon,
                spacing=float(config.downsample),
            )
            total_loss = data_loss + config.physics_weight * phys_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_phys_loss += phys_loss.item()
            n_batches += 1

        scheduler.step()

        avg_data = epoch_data_loss / max(n_batches, 1)
        avg_phys = epoch_phys_loss / max(n_batches, 1)

        if epoch % 2 == 0 or epoch == 1:
            val_results = evaluate(model, val_loader, device, config)
            skill = val_results["mean_skill"]
            beating = val_results["n_beating_persistence"]
            total = val_results["n_total"]

            print(
                f"Epoch {epoch:3d} | "
                f"data_loss={avg_data:.5f} phys_loss={avg_phys:.5f} | "
                f"val_skill={skill:+.4f} beating={beating}/{total}",
                flush=True,
            )

            history.append({
                "epoch": epoch,
                "data_loss": avg_data,
                "physics_loss": avg_phys,
                "val_mean_skill": skill,
                "val_mean_dice": val_results["mean_dice"],
                "val_beating": beating,
            })

            if skill > best_val_skill:
                best_val_skill = skill
                torch.save(model.state_dict(), config.output_root / "best_model.pt")
                print(f"  -> new best model (skill={skill:+.4f})", flush=True)

    model.load_state_dict(torch.load(config.output_root / "best_model.pt", weights_only=True))
    final_results = evaluate(model, val_loader, device, config, save_per_transition=True)
    final_results["training_history"] = history
    final_results["model_parameters"] = param_count

    results_path = config.output_root / "results.json"
    results_path.write_text(
        json.dumps(final_results, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nFinal Results:", flush=True)
    print(f"  Mean Dice: {final_results['mean_dice']:.4f}", flush=True)
    print(f"  Mean Skill: {final_results['mean_skill']:+.4f}", flush=True)
    print(
        f"  Beating Persistence: "
        f"{final_results['n_beating_persistence']}/{final_results['n_total']}",
        flush=True,
    )

    return final_results


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    save_per_transition: bool = False,
) -> dict[str, Any]:
    """Run model on validation set and compute Dice metrics."""
    model.eval()
    records = []

    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input"].to(device)
            source = batch["source"].to(device)

            residual = model(inp)
            predicted = (source + residual).clamp(0.0, 1.0)

            pred_np = predicted.squeeze().cpu().numpy()
            target_np = batch["target"].squeeze().numpy()
            source_np = batch["source"].squeeze().numpy()
            tid = batch["transition_id"][0]

            brain_mask = (source_np > 0) | (target_np > 0) | (pred_np > 0)
            if not np.any(brain_mask):
                brain_mask = np.ones_like(source_np, dtype=bool)

            forecast_dice = float(_masked_dice(
                pred_np, target_np, brain_mask, config.threshold,
            ))
            persistence_dice = float(_masked_dice(
                source_np, target_np, brain_mask, config.threshold,
            ))

            records.append({
                "transition_id": tid,
                "forecast_dice": forecast_dice,
                "persistence_dice": persistence_dice,
                "dice_skill": forecast_dice - persistence_dice,
            })

    skills = [r["dice_skill"] for r in records]
    dices = [r["forecast_dice"] for r in records]

    result: dict[str, Any] = {
        "n_total": len(records),
        "n_beating_persistence": sum(1 for s in skills if s > 0),
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_skill": float(np.mean(skills)) if skills else 0.0,
        "median_skill": float(np.median(skills)) if skills else 0.0,
    }

    if save_per_transition:
        result["records"] = records

    return result

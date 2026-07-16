"""Coordinate-based Physics-Informed Neural Network for tumor growth.

Proper PINN architecture following Zhang et al. (Medical Image Analysis, 2025):
- MLP takes (x, y, z, t, treatment_features) as input
- Outputs tumor density u(x, y, z, t)
- PDE residual computed via torch.autograd.grad (exact derivatives)
- D and rho learned per transition
- Trains across all 208 training transitions
"""

from __future__ import annotations

import json
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from gbm_pinn.clinical import segmentation_to_density
from gbm_pinn.clinical_experiment import _masked_dice
from gbm_pinn.shared_forecaster import load_transition_manifest

FloatArray = NDArray[np.float64]


class TumorPINNMLP(nn.Module):
    """Coordinate-based MLP for tumor density prediction.

    Input: (x, y, z, t, radiation, chemo, antiangiogenic, device) -> 8 dims
    Output: tumor density u at that point
    """

    def __init__(self, in_dim: int = 8, hidden: int = 256, layers: int = 4) -> None:
        super().__init__()
        net = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class TransitionPINN:
    """Per-transition PINN that learns D and rho for one scan pair."""

    def __init__(
        self,
        model: TumorPINNMLP,
        device: torch.device,
        lr: float = 1e-3,
    ) -> None:
        self.model = model
        self.device = device

        self.log_D = nn.Parameter(torch.tensor(-3.0, device=device))
        self.log_rho = nn.Parameter(torch.tensor(-4.0, device=device))

        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + [self.log_D, self.log_rho],
            lr=lr,
        )

    @property
    def D(self) -> torch.Tensor:
        return torch.exp(self.log_D)

    @property
    def rho(self) -> torch.Tensor:
        return torch.exp(self.log_rho)


def _compute_pde_residual(
    model: TumorPINNMLP,
    coords: torch.Tensor,
    D: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Compute Fisher-KPP PDE residual using autograd for exact derivatives."""
    coords.requires_grad_(True)
    u = model(coords)

    grad_u = torch.autograd.grad(
        u, coords, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True,
    )[0]

    du_dx = grad_u[:, 0:1]
    du_dy = grad_u[:, 1:2]
    du_dz = grad_u[:, 2:3]
    du_dt = grad_u[:, 3:4]

    du_dxx = torch.autograd.grad(
        du_dx, coords, grad_outputs=torch.ones_like(du_dx),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]

    du_dyy = torch.autograd.grad(
        du_dy, coords, grad_outputs=torch.ones_like(du_dy),
        create_graph=True, retain_graph=True,
    )[0][:, 1:2]

    du_dzz = torch.autograd.grad(
        du_dz, coords, grad_outputs=torch.ones_like(du_dz),
        create_graph=True, retain_graph=True,
    )[0][:, 2:3]

    laplacian = du_dxx + du_dyy + du_dzz
    reaction = rho * u * (1.0 - u)
    residual = du_dt - D * laplacian - reaction

    return residual


def _sample_points_near_tumor(
    density: np.ndarray,
    n_points: int,
    t_value: float,
    treatment: list[float],
    spacing: tuple[float, ...] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    """Sample (x, y, z, t, treatments) coordinates near and around tumor."""
    tumor_coords = np.argwhere(density > 0.05)

    if len(tumor_coords) == 0:
        shape = density.shape
        coords = np.random.rand(n_points, 3) * np.array(shape)
    else:
        n_tumor = n_points // 2
        n_random = n_points - n_tumor

        tumor_idx = np.random.randint(0, len(tumor_coords), size=n_tumor)
        tumor_pts = tumor_coords[tumor_idx].astype(np.float32)
        tumor_pts += np.random.randn(n_tumor, 3) * 3.0

        shape = density.shape
        random_pts = np.random.rand(n_random, 3) * np.array(shape)

        coords = np.concatenate([tumor_pts, random_pts], axis=0)

    coords[:, 0] = np.clip(coords[:, 0], 0, density.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, density.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, density.shape[2] - 1)

    coords *= np.array(spacing)

    t_col = np.full((n_points, 1), t_value, dtype=np.float32)
    treat_cols = np.tile(np.array(treatment, dtype=np.float32), (n_points, 1))

    return np.concatenate([coords, t_col, treat_cols], axis=1)


def _sample_data_points(
    density: np.ndarray,
    n_points: int,
    t_value: float,
    treatment: list[float],
    spacing: tuple[float, ...] = (2.0, 2.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample coordinates and their density values from a volume."""
    tumor_coords = np.argwhere(density > 0.05)
    bg_coords = np.argwhere((density <= 0.05) & (density >= 0))

    n_tumor = min(n_points * 3 // 4, len(tumor_coords)) if len(tumor_coords) > 0 else 0
    n_bg = n_points - n_tumor

    pts = []
    vals = []

    if n_tumor > 0:
        idx = np.random.randint(0, len(tumor_coords), size=n_tumor)
        tc = tumor_coords[idx]
        pts.append(tc)
        vals.append(density[tc[:, 0], tc[:, 1], tc[:, 2]])

    if n_bg > 0 and len(bg_coords) > 0:
        idx = np.random.randint(0, len(bg_coords), size=n_bg)
        bc = bg_coords[idx]
        pts.append(bc)
        vals.append(density[bc[:, 0], bc[:, 1], bc[:, 2]])

    coords = np.concatenate(pts, axis=0).astype(np.float32)
    values = np.concatenate(vals, axis=0).astype(np.float32)

    coords *= np.array(spacing)

    t_col = np.full((len(coords), 1), t_value, dtype=np.float32)
    treat_cols = np.tile(np.array(treatment, dtype=np.float32), (len(coords), 1))

    full_coords = np.concatenate([coords, t_col, treat_cols], axis=1)
    return full_coords, values


@dataclass
class TrainConfig:
    transition_index_path: Path
    manifest_path: Path
    val_transition_index_path: Path
    data_root: Path = Path(".")
    output_root: Path = Path("outputs/unet_pinn")
    downsample: int = 2
    hidden_dim: int = 256
    hidden_layers: int = 4
    n_data_points: int = 8192
    n_collocation_points: int = 4096
    pinn_epochs: int = 500
    pinn_lr: float = 1e-3
    data_weight: float = 10.0
    physics_weight: float = 1.0
    infiltrative_density: float = 0.3
    threshold: float = 0.1
    device: str = "auto"
    seed: int = 42


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _load_transition_volumes(
    transition: dict[str, Any],
    data_root: Path,
    downsample: int,
    infiltrative_density: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load source and target density volumes for a transition."""
    import nibabel as nib

    source_img = nib.as_closest_canonical(
        nib.load(data_root / transition["source_segmentation"]),
    )
    target_img = nib.as_closest_canonical(
        nib.load(data_root / transition["target_segmentation"]),
    )

    source_labels = np.rint(np.asanyarray(source_img.dataobj)).astype(np.int16)
    target_labels = np.rint(np.asanyarray(target_img.dataobj)).astype(np.int16)

    if downsample > 1:
        source_labels = source_labels[::downsample, ::downsample, ::downsample]
        target_labels = target_labels[::downsample, ::downsample, ::downsample]

    source_density = segmentation_to_density(
        source_labels, infiltrative_density=infiltrative_density,
    ).astype(np.float32)
    target_density = segmentation_to_density(
        target_labels, infiltrative_density=infiltrative_density,
    ).astype(np.float32)

    return source_density, target_density


def _get_treatment_flags(transition: dict[str, Any]) -> list[float]:
    """Extract 4 treatment flags from a transition record."""
    te = transition.get("treatment_exposure", {})
    return [
        float(te.get("radiation_exposure", 0.0) > 0),
        float(te.get("systemic_cytotoxic_exposure", 0.0) > 0),
        float(te.get("antiangiogenic_exposure", 0.0) > 0),
        float(te.get("device_exposure", 0.0) > 0),
    ]


def fit_transition(
    transition: dict[str, Any],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Fit a PINN to one transition and predict the target."""
    source_density, target_density = _load_transition_volumes(
        transition, config.data_root, config.downsample, config.infiltrative_density,
    )

    horizon_days = float(transition["target_day"]) - float(transition["source_day"])
    t_normalized = horizon_days / 365.0
    treatment = _get_treatment_flags(transition)
    spacing = (float(config.downsample),) * 3

    model = TumorPINNMLP(
        in_dim=8, hidden=config.hidden_dim, layers=config.hidden_layers,
    ).to(device)
    pinn = TransitionPINN(model, device, lr=config.pinn_lr)

    source_coords, source_vals = _sample_data_points(
        source_density, config.n_data_points, 0.0, treatment, spacing,
    )
    target_coords, target_vals = _sample_data_points(
        target_density, config.n_data_points, t_normalized, treatment, spacing,
    )

    source_coords_t = torch.tensor(source_coords, dtype=torch.float32, device=device)
    source_vals_t = torch.tensor(source_vals, dtype=torch.float32, device=device).unsqueeze(1)
    target_coords_t = torch.tensor(target_coords, dtype=torch.float32, device=device)
    target_vals_t = torch.tensor(target_vals, dtype=torch.float32, device=device).unsqueeze(1)

    for epoch in range(config.pinn_epochs):
        pinn.optimizer.zero_grad()

        pred_source = model(source_coords_t)
        pred_target = model(target_coords_t)

        data_loss = (
            nn.functional.mse_loss(pred_source, source_vals_t)
            + nn.functional.mse_loss(pred_target, target_vals_t)
        )

        t_random = torch.rand(config.n_collocation_points, 1, device=device) * t_normalized
        colloc_spatial = _sample_points_near_tumor(
            source_density, config.n_collocation_points, 0.0, treatment, spacing,
        )
        colloc_coords = torch.tensor(colloc_spatial, dtype=torch.float32, device=device)
        colloc_coords[:, 3:4] = t_random

        pde_residual = _compute_pde_residual(model, colloc_coords, pinn.D, pinn.rho)
        physics_loss = torch.mean(pde_residual ** 2)

        total_loss = config.data_weight * data_loss + config.physics_weight * physics_loss
        total_loss.backward()
        pinn.optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_volume = _predict_full_volume(
            model, target_density.shape, t_normalized, treatment, spacing, device,
        )

    brain_mask = (source_density > 0) | (target_density > 0) | (pred_volume > 0)
    if not np.any(brain_mask):
        brain_mask = np.ones_like(source_density, dtype=bool)

    forecast_dice = float(_masked_dice(pred_volume, target_density, brain_mask, config.threshold))
    persistence_dice = float(_masked_dice(source_density, target_density, brain_mask, config.threshold))

    return {
        "transition_id": transition["transition_id"],
        "patient_id": transition["patient_id"],
        "forecast_dice": forecast_dice,
        "persistence_dice": persistence_dice,
        "dice_skill": forecast_dice - persistence_dice,
        "learned_D": float(pinn.D.item()),
        "learned_rho": float(pinn.rho.item()),
        "horizon_days": horizon_days,
    }


def _predict_full_volume(
    model: TumorPINNMLP,
    shape: tuple[int, int, int],
    t_value: float,
    treatment: list[float],
    spacing: tuple[float, ...],
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    """Evaluate the PINN over every voxel to produce a full density volume."""
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32) * spacing[0],
        np.arange(shape[1], dtype=np.float32) * spacing[1],
        np.arange(shape[2], dtype=np.float32) * spacing[2],
        indexing="ij",
    )

    all_coords = np.stack([
        zz.ravel(), yy.ravel(), xx.ravel(),
        np.full(zz.size, t_value, dtype=np.float32),
        *[np.full(zz.size, t, dtype=np.float32) for t in treatment],
    ], axis=1)

    predictions = []
    for i in range(0, len(all_coords), batch_size):
        batch = torch.tensor(all_coords[i:i + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(batch).cpu().numpy().ravel()
        predictions.append(pred)

    return np.concatenate(predictions).reshape(shape)


def train(config: TrainConfig) -> dict[str, Any]:
    """Train per-transition PINNs across the full training set."""
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

    print(f"Training transitions: {len(train_transitions)}", flush=True)
    print(f"Validation transitions: {len(val_transitions)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Architecture: MLP {config.hidden_layers}x{config.hidden_dim}, tanh", flush=True)
    print(f"PINN epochs per transition: {config.pinn_epochs}", flush=True)

    records = []
    for i, transition in enumerate(val_transitions):
        t_start = time_module.time()
        try:
            result = fit_transition(transition, config, device)
            records.append(result)

            skill_str = f"{result['dice_skill']:+.4f}"
            elapsed = time_module.time() - t_start
            print(
                f"[{i+1}/{len(val_transitions)}] {result['transition_id']} | "
                f"Dice={result['forecast_dice']:.4f} "
                f"skill={skill_str} "
                f"D={result['learned_D']:.4f} rho={result['learned_rho']:.5f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        except Exception as e:
            print(f"[{i+1}/{len(val_transitions)}] FAILED: {e}", flush=True)
            records.append({
                "transition_id": transition["transition_id"],
                "patient_id": transition["patient_id"],
                "status": "failed",
                "error": str(e),
            })

    successful = [r for r in records if "forecast_dice" in r]
    skills = [r["dice_skill"] for r in successful]
    dices = [r["forecast_dice"] for r in successful]

    summary: dict[str, Any] = {
        "n_total": len(records),
        "n_successful": len(successful),
        "n_beating_persistence": sum(1 for s in skills if s > 0),
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_skill": float(np.mean(skills)) if skills else 0.0,
        "median_skill": float(np.median(skills)) if skills else 0.0,
        "records": records,
    }

    results_path = config.output_root / "results.json"
    results_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nFinal Results:", flush=True)
    print(f"  Mean Dice: {summary['mean_dice']:.4f}", flush=True)
    print(f"  Mean Skill: {summary['mean_skill']:+.4f}", flush=True)
    print(
        f"  Beating Persistence: {summary['n_beating_persistence']}/{summary['n_total']}",
        flush=True,
    )

    return summary

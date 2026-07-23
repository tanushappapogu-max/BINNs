"""Physics-Informed forward simulator for tumor growth forecasting.

A learned reaction-diffusion model: a CNN reads the source scan and treatment
context and predicts patient-specific PDE parameters (diffusion D,
proliferation rho, treatment death kappa). A differentiable Fisher-KPP
simulator then evolves the true source density forward to the target time.

The physics IS the forecaster; the network learns the parameters from data
across the whole training cohort and generalizes to unseen validation
patients. Starting from the true source guarantees the prediction begins at
the persistence baseline and only departs from it through learned dynamics.
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
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from gbm_pinn.clinical import segmentation_to_density
from gbm_pinn.clinical_experiment import _masked_dice
from gbm_pinn.shared_forecaster import load_transition_manifest

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Parameter network: source scan + treatment -> (D, rho, kappa)
# ---------------------------------------------------------------------------

class ParameterNet(nn.Module):
    """Predicts patient-specific reaction-diffusion parameters.

    Encodes the source density with a small 3D CNN, concatenates treatment
    flags and the normalized horizon, and regresses three positive, bounded
    PDE parameters.
    """

    # Upper bounds on each parameter. Exposed so the regularizer can normalize
    # by them: penalizing raw magnitudes makes the penalty strength depend on
    # the bounds, which silently changes the objective when they are widened.
    D_MAX, RHO_MAX, KAPPA_MAX = 0.15, 0.5, 0.5
    D_MAX_RATE, RHO_MAX_RATE, KAPPA_MAX_RATE = 1.5, 4.0, 4.0

    def __init__(
        self, n_treatment: int = 4, base_filters: int = 16, in_channels: int = 1,
        per_year_rates: bool = False,
    ) -> None:
        super().__init__()
        f = base_filters
        # Under time scaling the simulator multiplies by the elapsed interval,
        # so the parameters represent per-year rates and need wider bounds.
        self.bounds = (
            (self.D_MAX_RATE, self.RHO_MAX_RATE, self.KAPPA_MAX_RATE) if per_year_rates
            else (self.D_MAX, self.RHO_MAX, self.KAPPA_MAX)
        )
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, f, 3, stride=2, padding=1),
            nn.InstanceNorm3d(f),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(f, f * 2, 3, stride=2, padding=1),
            nn.InstanceNorm3d(f * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(f * 2, f * 4, 3, stride=2, padding=1),
            nn.InstanceNorm3d(f * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(f * 4 + n_treatment + 1, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 3),
        )

        # Start with near-zero parameters so the initial prediction is the
        # persistence baseline; training then learns how to depart from it.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(
        self, source: torch.Tensor, treatment: torch.Tensor, horizon: torch.Tensor,
        mri: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_input = source if mri is None else torch.cat([source, mri], dim=1)
        features = self.encoder(encoder_input).flatten(1)
        context = torch.cat([features, treatment, horizon.unsqueeze(1)], dim=1)
        raw = self.head(context)

        d_max, rho_max, kappa_max = self.bounds
        D = d_max * torch.sigmoid(raw[:, 0])
        rho = rho_max * torch.sigmoid(raw[:, 1])
        kappa = kappa_max * torch.sigmoid(raw[:, 2])
        return D, rho, kappa

    def scale_free_penalty(self, D: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Parameter penalty normalized by the bounds, so its strength does not
        change when the bounds do."""
        d_max, rho_max, _ = self.bounds
        return ((D / d_max) ** 2).mean() + ((rho / rho_max) ** 2).mean()


class SpatialDiffusivityNet(nn.Module):
    """Predicts a voxelwise diffusivity multiplier from the source imaging.

    A single global diffusivity forces tumor to spread isotropically at one
    rate. Real spread follows tissue structure, and the infiltration that seeds
    recurrence is visible in FLAIR before it is segmentable. This produces a
    field in (0, 2) that modulates the global rate, so image structure can
    steer *where* the tumor grows while the PDE still governs *how*.
    """

    def __init__(self, in_channels: int = 5, f: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, f, 3, padding=1),
            nn.InstanceNorm3d(f),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(f, f, 3, padding=1),
            nn.InstanceNorm3d(f),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(f, 1, 1),
        )
        # Zero init makes the field start uniformly at 1.0, so training begins
        # from the scalar-diffusivity model and departs from it only as needed.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.net(x))


# ---------------------------------------------------------------------------
# Differentiable Fisher-KPP simulator
# ---------------------------------------------------------------------------

class ReactionDiffusionSimulator(nn.Module):
    """Evolves density via du/dt = D*laplacian(u) + rho*u*(1-u) - kappa*u.

    Uses an explicit finite-difference scheme with a fixed 3D Laplacian
    kernel. Fully differentiable so gradients flow back to the parameter
    network through every timestep.
    """

    def __init__(self, n_steps: int = 30, dt: float = 0.2) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.dt = dt

        # Face-neighbour stencil only; the centre weight is applied separately
        # so the operator can be restricted to the valid domain.
        neighbors = torch.zeros(1, 1, 3, 3, 3)
        for idx in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
            neighbors[0, 0, idx[0], idx[1], idx[2]] = 1.0
        self.register_buffer("neighbor_kernel", neighbors)

    # Face-neighbour offsets used by the variable-coefficient operator.
    _SHIFTS = ((1, 2), (-1, 2), (1, 3), (-1, 3), (1, 4), (-1, 4))

    def _divergence_flux(self, u: torch.Tensor, d_field: torch.Tensor) -> torch.Tensor:
        """Discrete div(D grad u) with face-averaged diffusivity.

        Summing D_face * (u_neighbour - u_centre) over the six faces is the
        conservative form. A spatially varying D lets image-derived structure
        steer where the tumor spreads, which a single global scalar cannot.
        """
        total = torch.zeros_like(u)
        for shift, dim in self._SHIFTS:
            u_n = torch.roll(u, shift, dims=dim)
            d_n = torch.roll(d_field, shift, dims=dim)
            total = total + 0.5 * (d_field + d_n) * (u_n - u)
        return total

    def _laplacian(self, u: torch.Tensor, domain: torch.Tensor | None) -> torch.Tensor:
        """Discrete Laplacian, restricted to `domain` with no-flux boundaries.

        Summing only over in-domain neighbours makes the flux across any face
        touching excluded tissue zero, which is the no-flux condition at the
        resection cavity wall. With no domain this reduces to the standard
        7-point stencil.
        """
        if domain is None:
            neighbor_sum = F.conv3d(u, self.neighbor_kernel, padding=1)
            return neighbor_sum - 6.0 * u

        neighbor_sum = F.conv3d(u * domain, self.neighbor_kernel, padding=1)
        neighbor_count = F.conv3d(domain, self.neighbor_kernel, padding=1)
        return (neighbor_sum - neighbor_count * u) * domain

    def forward(
        self,
        u0: torch.Tensor,
        D: torch.Tensor,
        rho: torch.Tensor,
        kappa: torch.Tensor,
        domain: torch.Tensor | None = None,
        horizon: torch.Tensor | None = None,
        d_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u = u0
        D = D.view(-1, 1, 1, 1, 1)
        rho = rho.view(-1, 1, 1, 1, 1)
        kappa = kappa.view(-1, 1, 1, 1, 1)

        # The network predicts rates per year; integrating them over the real
        # elapsed interval is what makes a three-day gap and a three-year gap
        # produce different amounts of growth. Without this the simulator runs
        # the same fixed amount of time for every transition.
        if horizon is not None:
            years = horizon.view(-1, 1, 1, 1, 1).clamp(min=1.0 / 365.0)
            D = D * years
            rho = rho * years
            kappa = kappa * years
            # Explicit diffusion is stable only while D*dt <= 1/6 in grid units.
            D = D.clamp(max=1.0 / (6.0 * self.dt))

        # A spatial field replaces the scalar diffusivity entirely; the global
        # scalar sets its overall magnitude.
        diffusivity = None if d_field is None else D * d_field

        for _ in range(self.n_steps):
            if diffusivity is None:
                spread = D * self._laplacian(u, domain)
            else:
                spread = self._divergence_flux(u, diffusivity)
            reaction = rho * u * (1.0 - u)
            death = kappa * u
            u = u + self.dt * (spread + reaction - death)
            u = u.clamp(0.0, 1.0)
            if domain is not None:
                u = u * domain

        return u


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TumorTransitionDataset(Dataset):
    """Loads source/target density volumes plus treatment and horizon."""

    def __init__(
        self,
        transitions: list[dict[str, Any]],
        data_root: Path,
        downsample: int = 2,
        target_shape: tuple[int, int, int] = (120, 120, 80),
        infiltrative_density: float = 0.3,
        augment: bool = False,
        use_mri: bool = False,
    ) -> None:
        self.transitions = transitions
        self.data_root = data_root
        self.downsample = downsample
        self.target_shape = target_shape
        self.infiltrative_density = infiltrative_density
        self.augment = augment
        self.use_mri = use_mri

    # Source-timepoint intensity modalities, in fixed channel order.
    MRI_MODALITIES = ("brain_t1c", "brain_t1n", "brain_t2f", "brain_t2w")

    def _load_mri(self, seg_path: str) -> np.ndarray:
        """Load, downsample, normalize and pad the four source modalities."""
        import nibabel as nib

        ds = self.downsample
        channels = []
        for modality in self.MRI_MODALITIES:
            path = self.data_root / seg_path.replace("tumorMask", modality)
            vol = np.asanyarray(
                nib.as_closest_canonical(nib.load(path)).dataobj,
            ).astype(np.float32)
            if ds > 1:
                vol = vol[::ds, ::ds, ::ds]
            brain = vol > 0
            if brain.any():
                mean, std = vol[brain].mean(), vol[brain].std()
                vol = np.where(brain, (vol - mean) / (std + 1e-6), 0.0)
            channels.append(self._pad(vol.astype(np.float32)))
        return np.stack(channels, axis=0)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import nibabel as nib

        t = self.transitions[idx]
        ds = self.downsample

        source_img = nib.as_closest_canonical(nib.load(self.data_root / t["source_segmentation"]))
        target_img = nib.as_closest_canonical(nib.load(self.data_root / t["target_segmentation"]))

        source_labels = np.rint(np.asanyarray(source_img.dataobj)).astype(np.int16)
        target_labels = np.rint(np.asanyarray(target_img.dataobj)).astype(np.int16)

        if ds > 1:
            source_labels = source_labels[::ds, ::ds, ::ds]
            target_labels = target_labels[::ds, ::ds, ::ds]

        source_density = self._pad(segmentation_to_density(
            source_labels, infiltrative_density=self.infiltrative_density,
        ).astype(np.float32))
        target_density = self._pad(segmentation_to_density(
            target_labels, infiltrative_density=self.infiltrative_density,
        ).astype(np.float32))

        # Label 4 is the resection cavity. It carries zero density, so without
        # an explicit domain it is indistinguishable from brain tissue and the
        # simulator would diffuse tumor straight through it.
        domain = self._pad((source_labels != 4).astype(np.float32))

        horizon_days = float(t["target_day"]) - float(t["source_day"])
        te = t.get("treatment_exposure", {})
        treatment = np.array([
            float(te.get("radiation_exposure", 0.0) > 0),
            float(te.get("systemic_cytotoxic_exposure", 0.0) > 0),
            float(te.get("antiangiogenic_exposure", 0.0) > 0),
            float(te.get("device_exposure", 0.0) > 0),
        ], dtype=np.float32)

        source_t = torch.from_numpy(source_density[np.newaxis])
        target_t = torch.from_numpy(target_density[np.newaxis])
        domain_t = torch.from_numpy(domain[np.newaxis])
        mri_t = (
            torch.from_numpy(self._load_mri(t["source_segmentation"]))
            if self.use_mri else torch.empty(0)
        )

        if self.augment:
            for dim in range(3):
                if np.random.random() > 0.5:
                    source_t = torch.flip(source_t, [dim + 1])
                    target_t = torch.flip(target_t, [dim + 1])
                    domain_t = torch.flip(domain_t, [dim + 1])
                    if self.use_mri:
                        mri_t = torch.flip(mri_t, [dim + 1])

        return {
            "source": source_t,
            "target": target_t,
            "domain": domain_t,
            "mri": mri_t,
            "treatment": torch.from_numpy(treatment),
            "horizon": torch.tensor(horizon_days / 365.0, dtype=torch.float32),
            "transition_id": t["transition_id"],
            "patient_id": t["patient_id"],
        }

    def _pad(self, volume: np.ndarray) -> np.ndarray:
        ts = self.target_shape
        padded = np.zeros(ts, dtype=volume.dtype)
        slices = tuple(slice(0, min(v, s)) for v, s in zip(volume.shape, ts))
        padded[slices] = volume[slices]
        return padded


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - soft Dice, per-sample and averaged. Directly targets the metric.

    Operates on raw density so background voxels, which are exactly zero,
    contribute nothing to either the intersection or the union.
    """
    pred_f = pred.flatten(1)
    target_f = target.flatten(1)
    intersection = (pred_f * target_f).sum(dim=1)
    union = (pred_f ** 2).sum(dim=1) + (target_f ** 2).sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    transition_index_path: Path
    manifest_path: Path
    val_transition_index_path: Path
    data_root: Path = Path(".")
    output_root: Path = Path("outputs/unet_pinn")
    downsample: int = 4
    target_shape: tuple[int, int, int] = (60, 60, 40)
    base_filters: int = 16
    n_steps: int = 30
    dt: float = 0.2
    epochs: int = 25
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mse_weight: float = 0.5
    # With the bound-normalized penalty, skill is flat across reg in [0, 0.2]
    # and only degrades once it dominates; a small value keeps the GBM-scale
    # prior on D and rho without suppressing them.
    param_reg_weight: float = 0.1
    # Integrating the PDE over the true elapsed interval is the physically
    # correct formulation and is required to forecast at an arbitrary horizon.
    # It scores lower on this cohort not because the physics is wrong but
    # because scan intervals are clinically driven: unstable patients are
    # imaged sooner, so a horizon supplied as a free feature lets the network
    # predict the scheduling process (corr of predicted change with gap goes
    # negative) rather than the biology. Reported as the central comparison,
    # not silently switched.
    time_scaled: bool = False
    # No-flux at the resection cavity is the physically standard treatment, but
    # the cavity collapses over follow-up and recurrence arises at its margin,
    # so freezing the baseline cavity as forbidden blocks real predictions and
    # measurably degrades every metric. Kept for ablation, off by default.
    use_cavity_domain: bool = False
    infiltrative_density: float = 0.3
    threshold: float = 0.1
    device: str = "auto"
    num_workers: int = 2
    seed: int = 42


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(config: TrainConfig) -> dict[str, Any]:
    """Train the parameter network + simulator across the training cohort."""
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
    print(f"Simulator: {config.n_steps} steps, dt={config.dt}", flush=True)

    train_ds = TumorTransitionDataset(
        train_transitions, config.data_root, downsample=config.downsample,
        target_shape=config.target_shape, infiltrative_density=config.infiltrative_density,
        augment=True,
    )
    val_ds = TumorTransitionDataset(
        val_transitions, config.data_root, downsample=config.downsample,
        target_shape=config.target_shape, infiltrative_density=config.infiltrative_density,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers)

    param_net = ParameterNet(
        n_treatment=4, base_filters=config.base_filters,
        per_year_rates=config.time_scaled,
    ).to(device)
    simulator = ReactionDiffusionSimulator(n_steps=config.n_steps, dt=config.dt).to(device)

    param_count = sum(p.numel() for p in param_net.parameters() if p.requires_grad)
    print(f"Parameter network: {param_count:,} params", flush=True)

    optimizer = torch.optim.AdamW(
        param_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_skill = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        param_net.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            treatment = batch["treatment"].to(device)
            horizon = batch["horizon"].to(device)

            domain = batch["domain"].to(device) if config.use_cavity_domain else None
            D, rho, kappa = param_net(source, treatment, horizon)
            predicted = simulator(
                source, D, rho, kappa, domain,
                horizon if config.time_scaled else None,
            )

            # Penalize large diffusion/proliferation. Measured rates for GBM
            # are small, and without this prior the fit drives both to their
            # bounds long after validation skill has peaked.
            param_penalty = param_net.scale_free_penalty(D, rho)

            loss = (
                soft_dice_loss(predicted, target)
                + config.mse_weight * F.mse_loss(predicted, target)
                + config.param_reg_weight * param_penalty
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(param_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate every epoch: the skill optimum is narrow and a coarser
        # cadence can step straight over the best checkpoint.
        val = evaluate(param_net, simulator, val_loader, device, config)
        print(
            f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
            f"skill={val['mean_skill']:+.4f} beating={val['n_beating_persistence']}/{val['n_total']} "
            f"| growth_dice={val['mean_growth_dice']:.4f} "
            f"| hi_change skill={val.get('high_change_mean_skill', 0):+.4f} "
            f"({val.get('high_change_beating', 0)}/{val.get('high_change_n', 0)})",
            flush=True,
        )
        history.append({
            "epoch": epoch, "loss": avg_loss,
            "val_dice": val["mean_dice"], "val_skill": val["mean_skill"],
            "val_beating": val["n_beating_persistence"],
        })

        if val["mean_skill"] > best_skill:
            best_skill = val["mean_skill"]
            torch.save(param_net.state_dict(), config.output_root / "best_model.pt")
            print(f"  -> new best (skill={best_skill:+.4f})", flush=True)

    param_net.load_state_dict(torch.load(config.output_root / "best_model.pt", weights_only=True))
    final = evaluate(param_net, simulator, val_loader, device, config, save_records=True)
    final["training_history"] = history
    final["model_parameters"] = param_count

    (config.output_root / "results.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nFinal Results:", flush=True)
    print(f"  Mean Dice: {final['mean_dice']:.4f}", flush=True)
    print(f"  Mean Skill: {final['mean_skill']:+.4f}", flush=True)
    print(f"  Beating Persistence: {final['n_beating_persistence']}/{final['n_total']}", flush=True)

    return final


def evaluate(
    param_net: nn.Module,
    simulator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    save_records: bool = False,
) -> dict[str, Any]:
    """Predict on the validation set and compute Dice / skill over persistence."""
    param_net.eval()
    records = []

    with torch.no_grad():
        for batch in val_loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            treatment = batch["treatment"].to(device)
            horizon = batch["horizon"].to(device)

            domain = batch["domain"].to(device) if config.use_cavity_domain else None
            D, rho, kappa = param_net(source, treatment, horizon)
            predicted = simulator(
                source, D, rho, kappa, domain,
                horizon if config.time_scaled else None,
            )

            pred_np = predicted.squeeze().cpu().numpy()
            target_np = target.squeeze().cpu().numpy()
            source_np = source.squeeze().cpu().numpy()

            brain_mask = (source_np > 0) | (target_np > 0) | (pred_np > 0)
            if not np.any(brain_mask):
                brain_mask = np.ones_like(source_np, dtype=bool)

            forecast_dice = float(_masked_dice(pred_np, target_np, brain_mask, config.threshold))
            persistence_dice = float(_masked_dice(source_np, target_np, brain_mask, config.threshold))

            src_bin = source_np > config.threshold
            tgt_bin = target_np > config.threshold
            prd_bin = pred_np > config.threshold

            # Whole-tumor Dice is dominated by tissue that simply does not move,
            # which persistence reproduces for free. Scoring the newly-appearing
            # tumor isolates the part of the forecast that is actually a
            # forecast: persistence predicts no new tumor, so it scores zero here
            # whenever growth occurs.
            growth_true = tgt_bin & ~src_bin
            growth_pred = prd_bin & ~src_bin
            shrink_true = src_bin & ~tgt_bin
            shrink_pred = src_bin & ~prd_bin

            def _dice(a: np.ndarray, b: np.ndarray) -> float:
                denom = int(a.sum()) + int(b.sum())
                if denom == 0:
                    return float("nan")
                return 2.0 * float((a & b).sum()) / denom

            union = int((src_bin | tgt_bin).sum())
            change_fraction = (
                float((src_bin ^ tgt_bin).sum()) / union if union > 0 else 0.0
            )

            # Tumor burden. Dice asks whether the shape landed in the right
            # place; volume asks how much tumor there will be, which is the
            # quantity followed clinically.
            ml_per_voxel = (config.downsample ** 3) / 1000.0
            src_vol = float(src_bin.sum()) * ml_per_voxel
            tgt_vol = float(tgt_bin.sum()) * ml_per_voxel
            pred_vol = float(prd_bin.sum()) * ml_per_voxel
            true_change = tgt_vol - src_vol
            pred_change = pred_vol - src_vol

            records.append({
                "transition_id": batch["transition_id"][0],
                "patient_id": batch["patient_id"][0],
                "forecast_dice": forecast_dice,
                "persistence_dice": persistence_dice,
                "dice_skill": forecast_dice - persistence_dice,
                "growth_dice": _dice(growth_pred, growth_true),
                "shrink_dice": _dice(shrink_pred, shrink_true),
                "growth_voxels_true": int(growth_true.sum()),
                "change_fraction": change_fraction,
                "source_volume_ml": src_vol,
                "target_volume_ml": tgt_vol,
                "predicted_volume_ml": pred_vol,
                "true_volume_change_ml": true_change,
                "predicted_volume_change_ml": pred_change,
                "volume_abs_error_ml": abs(pred_change - true_change),
                "persistence_abs_error_ml": abs(true_change),
                "horizon_days": float(horizon.item()) * 365.0,
                "D": float(D.item()),
                "rho": float(rho.item()),
                "kappa": float(kappa.item()),
            })

    skills = [r["dice_skill"] for r in records]
    dices = [r["forecast_dice"] for r in records]
    growth = [r["growth_dice"] for r in records if not np.isnan(r["growth_dice"])]

    result: dict[str, Any] = {
        "n_total": len(records),
        "n_beating_persistence": sum(1 for s in skills if s > 0),
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_skill": float(np.mean(skills)) if skills else 0.0,
        "median_skill": float(np.median(skills)) if skills else 0.0,
        "mean_growth_dice": float(np.mean(growth)) if growth else 0.0,
        "n_growth_scored": len(growth),
        "mean_D": float(np.mean([r["D"] for r in records])) if records else 0.0,
        "mean_rho": float(np.mean([r["rho"] for r in records])) if records else 0.0,
        "mean_kappa": float(np.mean([r["kappa"] for r in records])) if records else 0.0,
    }
    result.update(summarize_records(records))

    if save_records:
        result["records"] = records

    return result


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Patient-clustered significance and stratification by observed change.

    Transitions are not independent: consecutive scan pairs share a patient, so
    per-transition counts overstate the evidence. Skill is therefore averaged
    within patient before testing. Results are also split by how much the tumor
    actually moved, since persistence is unbeatable where nothing changed.
    """
    if not records:
        return {}

    by_patient: dict[str, list[float]] = {}
    for r in records:
        by_patient.setdefault(r["patient_id"], []).append(r["dice_skill"])
    patient_skills = [float(np.mean(v)) for v in by_patient.values()]

    summary: dict[str, Any] = {
        "n_patients": len(by_patient),
        "patient_mean_skill": float(np.mean(patient_skills)),
        "n_patients_beating": sum(1 for s in patient_skills if s > 0),
    }

    if "true_volume_change_ml" in records[0]:
        true_dv = np.array([r["true_volume_change_ml"] for r in records])
        pred_dv = np.array([r["predicted_volume_change_ml"] for r in records])
        model_err = np.abs(pred_dv - true_dv)
        # Persistence forecasts zero volume change, so its error is |true|.
        pers_err = np.abs(true_dv)
        moved = true_dv != 0
        summary.update({
            "volume_mae_ml": float(model_err.mean()),
            "persistence_volume_mae_ml": float(pers_err.mean()),
            "volume_mae_reduction_ml": float(pers_err.mean() - model_err.mean()),
            "n_volume_better_than_persistence": int((model_err < pers_err).sum()),
            "direction_accuracy": (
                float((np.sign(pred_dv[moved]) == np.sign(true_dv[moved])).mean())
                if moved.any() else 0.0
            ),
            "n_direction_scored": int(moved.sum()),
        })
        if len(records) > 2 and true_dv.std() > 0 and pred_dv.std() > 0:
            summary["volume_change_correlation"] = float(np.corrcoef(true_dv, pred_dv)[0, 1])

        # Central diagnostic: does predicted growth track the elapsed interval?
        # A model integrating real biology predicts more change over longer
        # gaps (positive). One given the horizon as a free feature can instead
        # learn the confounded scan schedule, where long gaps mark stable
        # patients, and go negative -- the opposite of the physics.
        if "horizon_days" in records[0]:
            gap = np.array([r["horizon_days"] for r in records])
            if len(records) > 2 and gap.std() > 0 and pred_dv.std() > 0:
                summary["pred_change_vs_gap_correlation"] = float(np.corrcoef(gap, pred_dv)[0, 1])
            if len(records) > 2 and gap.std() > 0 and true_dv.std() > 0:
                summary["true_change_vs_gap_correlation"] = float(np.corrcoef(gap, true_dv)[0, 1])

    try:
        from scipy.stats import wilcoxon

        if len(patient_skills) >= 6 and any(s != 0 for s in patient_skills):
            stat, p = wilcoxon(patient_skills)
            summary["wilcoxon_p_patient_clustered"] = float(p)
    except Exception:
        pass

    if all("change_fraction" in r for r in records):
        changes = sorted(r["change_fraction"] for r in records)
        cut = changes[max(0, int(len(changes) * 2 / 3) - 1)]
        movers = [r for r in records if r["change_fraction"] > cut]
        stable = [r for r in records if r["change_fraction"] <= cut]
        for name, group in (("high_change", movers), ("low_change", stable)):
            if not group:
                continue
            g_skill = [r["dice_skill"] for r in group]
            g_growth = [
                r["growth_dice"] for r in group
                if not np.isnan(r.get("growth_dice", float("nan")))
            ]
            summary[f"{name}_n"] = len(group)
            summary[f"{name}_mean_skill"] = float(np.mean(g_skill))
            summary[f"{name}_beating"] = sum(1 for s in g_skill if s > 0)
            summary[f"{name}_mean_growth_dice"] = (
                float(np.mean(g_growth)) if g_growth else 0.0
            )

    return summary

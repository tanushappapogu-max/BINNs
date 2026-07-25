"""Biologically-Informed Neural Network for learning reaction-diffusion terms.

Given noisy observations of a density field u(x, t), a BINN learns the governing
one-dimensional reaction-diffusion PDE

    u_t = d/dx( D(u) u_x ) + R(u)

where the diffusivity D(u) and growth R(u) are represented by small neural
networks rather than assumed functional forms. A solution surrogate u(x, t) is
fit to the data while the learned terms are constrained to satisfy the PDE via
automatic differentiation.

This module is validated on synthetic data with a known ground-truth PDE
(see tests). It reproduces a documented property of this model class: the
growth term is recovered far more reliably than the diffusion term, because
diffusion is weakly constrained when growth dominates the dynamics. The API
reports both so that the weaker identifiability of D(u) is visible rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _mlp(sizes: list[int], out_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class BINN(nn.Module):
    """Solution surrogate plus learned diffusion and reaction terms.

    Coordinates are supplied to ``solution`` already normalized to roughly the
    unit range; the physics residual denormalizes with ``x_scale``/``t_scale``
    so derivatives are in physical units.
    """

    def __init__(
        self,
        x_scale: float,
        t_scale: float,
        solution_width: int = 64,
        solution_depth: int = 3,
        term_width: int = 32,
    ) -> None:
        super().__init__()
        self.x_scale = float(x_scale)
        self.t_scale = float(t_scale)
        sol_sizes = [2] + [solution_width] * solution_depth + [1]
        self.solution = _mlp(sol_sizes, nn.Sigmoid())
        # Diffusivity is non-negative; growth may take either sign.
        self.diffusivity = _mlp([1, term_width, term_width, 1], nn.Softplus())
        self.growth = _mlp([1, term_width, term_width, 1])

    def u(self, xt_normalized: torch.Tensor) -> torch.Tensor:
        return self.solution(xt_normalized)

    def pde_residual(self, xt_physical: torch.Tensor) -> torch.Tensor:
        """Residual u_t - d/dx(D(u) u_x) - R(u) at physical coordinates."""
        coords = xt_physical.detach().clone().requires_grad_(True)
        xt_n = torch.stack(
            [coords[:, 0] / self.x_scale, coords[:, 1] / self.t_scale], dim=1
        )
        u = self.solution(xt_n)
        grad = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True)[0]
        u_x, u_t = grad[:, 0:1], grad[:, 1:2]
        flux = self.diffusivity(u) * u_x
        dflux_dx = torch.autograd.grad(
            flux, coords, torch.ones_like(flux), create_graph=True
        )[0][:, 0:1]
        return u_t - dflux_dx - self.growth(u)


@dataclass
class BINNConfig:
    iterations: int = 4000
    batch_size: int = 2048
    learning_rate: float = 2e-3
    physics_weight: float = 0.5
    device: str = "auto"
    seed: int = 0


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def fit_binn(
    coords: FloatArray,
    values: FloatArray,
    x_scale: float,
    t_scale: float,
    config: BINNConfig | None = None,
) -> BINN:
    """Fit a BINN to density observations.

    ``coords`` is (N, 2) physical (x, t); ``values`` is (N,) observed density.
    """
    config = config or BINNConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _resolve_device(config.device)

    model = BINN(x_scale=x_scale, t_scale=t_scale).to(device)
    xt = torch.tensor(coords, dtype=torch.float32, device=device)
    xt_n = torch.stack([xt[:, 0] / x_scale, xt[:, 1] / t_scale], dim=1)
    y = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n = xt.shape[0]
    for _ in range(config.iterations):
        idx = torch.randint(0, n, (min(config.batch_size, n),), device=device)
        opt.zero_grad()
        data_loss = torch.mean((model.u(xt_n[idx]) - y[idx]) ** 2)
        phys_loss = torch.mean(model.pde_residual(xt[idx]) ** 2)
        (data_loss + config.physics_weight * phys_loss).backward()
        opt.step()

    return model


def evaluate_terms(
    model: BINN,
    u_grid: FloatArray,
    true_diffusivity: FloatArray,
    true_growth: FloatArray,
) -> dict[str, float]:
    """Correlation and affine-aligned relative L2 of the recovered terms.

    Learned terms are identifiable only up to a gauge (a constant can move
    between the flux and reaction terms), so shape recovery is reported after
    a least-squares affine alignment in addition to raw correlation.
    """
    device = next(model.parameters()).device
    ug = torch.tensor(u_grid, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        d_pred = model.diffusivity(ug).cpu().numpy().ravel()
        r_pred = model.growth(ug).cpu().numpy().ravel()

    def _affine(pred: FloatArray, true: FloatArray) -> FloatArray:
        design = np.vstack([pred, np.ones_like(pred)]).T
        coef, *_ = np.linalg.lstsq(design, true, rcond=None)
        return coef[0] * pred + coef[1]

    def _rel_l2(a: FloatArray, b: FloatArray) -> float:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))

    return {
        "diffusion_corr": float(np.corrcoef(d_pred, true_diffusivity)[0, 1]),
        "diffusion_rel_l2": _rel_l2(_affine(d_pred, true_diffusivity), true_diffusivity),
        "growth_corr": float(np.corrcoef(r_pred, true_growth)[0, 1]),
        "growth_rel_l2": _rel_l2(_affine(r_pred, true_growth), true_growth),
    }


def solution_r2(model: BINN, coords: FloatArray, clean_values: FloatArray) -> float:
    """R^2 of the solution surrogate against noise-free density."""
    device = next(model.parameters()).device
    xt = torch.tensor(coords, dtype=torch.float32, device=device)
    xt_n = torch.stack([xt[:, 0] / model.x_scale, xt[:, 1] / model.t_scale], dim=1)
    with torch.no_grad():
        pred = model.u(xt_n).cpu().numpy().ravel()
    ss_res = float(np.sum((pred - clean_values) ** 2))
    ss_tot = float(np.sum((clean_values - clean_values.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def simulate_reaction_diffusion(
    diffusivity, growth, *, length: float = 10.0, n_x: int = 200,
    duration: float = 6.0, n_t: int = 6000, n_snapshots: int = 10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Explicit finite-difference reference solver on a no-flux domain.

    Returns (snapshots, times, x). ``diffusivity`` and ``growth`` are callables
    of the density array. Used to generate ground-truth data for validation.
    """
    dx = length / (n_x - 1)
    dt = duration / n_t
    x = np.linspace(0.0, length, n_x)
    u = np.exp(-((x - length / 2) ** 2) / 0.5)
    snapshots, times = [u.copy()], [0.0]
    save_every = max(n_t // n_snapshots, 1)
    for step in range(1, n_t + 1):
        d_face = 0.5 * (diffusivity(u)[1:] + diffusivity(u)[:-1])
        flux = d_face * (u[1:] - u[:-1]) / dx
        div = np.zeros_like(u)
        div[1:-1] = (flux[1:] - flux[:-1]) / dx
        div[0] = flux[0] / dx
        div[-1] = -flux[-1] / dx
        u = np.clip(u + dt * (div + growth(u)), 0.0, 1.5)
        if step % save_every == 0:
            snapshots.append(u.copy())
            times.append(step * dt)
    return np.array(snapshots), np.array(times), x

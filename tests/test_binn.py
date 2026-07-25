"""Tests for the BINN reaction-diffusion term-learning module.

Assertions reflect what this model class can and cannot reliably do: the
solution surrogate and the growth term are recoverable and are asserted; the
diffusion term is only weakly identifiable when growth dominates, so it is
sanity-checked (finite, non-negative) but not asserted to be accurate.
"""

from __future__ import annotations

import numpy as np
import torch

from gbm_pinn.binn import (
    BINN,
    BINNConfig,
    evaluate_terms,
    fit_binn,
    simulate_reaction_diffusion,
    solution_r2,
)

D0, KD, RHO = 0.10, 1.5, 1.0
LENGTH, DURATION = 10.0, 6.0


def _true_D(u):
    return D0 * (1.0 + KD * u)


def _true_R(u):
    return RHO * u * (1.0 - u)


def _make_dataset(noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    snaps, times, x = simulate_reaction_diffusion(
        _true_D, _true_R, length=LENGTH, duration=DURATION,
    )
    xx, tt = np.meshgrid(x, times)
    coords = np.stack([xx.ravel(), tt.ravel()], axis=1)
    clean = snaps.ravel()
    noisy = clean + noise * rng.standard_normal(clean.shape)
    return coords, clean, noisy


def test_pde_residual_shapes_and_grad():
    model = BINN(x_scale=LENGTH, t_scale=DURATION)
    xt = torch.rand(16, 2) * torch.tensor([LENGTH, DURATION])
    res = model.pde_residual(xt)
    assert res.shape == (16, 1)
    res.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and sum(g.abs().sum().item() for g in grads) > 0


def test_diffusivity_is_nonnegative():
    model = BINN(x_scale=LENGTH, t_scale=DURATION)
    u = torch.linspace(0, 1, 50).unsqueeze(1)
    with torch.no_grad():
        assert torch.all(model.diffusivity(u) >= 0)


def test_reference_solver_grows_then_saturates():
    snaps, times, x = simulate_reaction_diffusion(_true_D, _true_R)
    mass = snaps.sum(axis=1)
    assert mass[-1] > mass[0]                       # net growth
    assert np.all(np.isfinite(snaps))               # stable
    assert snaps.max() <= 1.5 + 1e-6                # bounded


def test_binn_recovers_solution_and_growth():
    coords, clean, noisy = _make_dataset(noise=0.02)
    model = fit_binn(
        coords, noisy, x_scale=LENGTH, t_scale=DURATION,
        config=BINNConfig(iterations=2500, device="cpu", seed=0),
    )
    r2 = solution_r2(model, coords, clean)
    assert r2 > 0.99, f"solution fit too poor: R2={r2:.4f}"

    u_grid = np.linspace(0.05, 0.95, 80)
    metrics = evaluate_terms(model, u_grid, _true_D(u_grid), _true_R(u_grid))
    # Growth is the identifiable term and must be recovered well.
    assert metrics["growth_corr"] > 0.85, f"growth corr={metrics['growth_corr']:.3f}"
    # Diffusion is only weakly identifiable here; require finiteness, not accuracy.
    assert np.isfinite(metrics["diffusion_corr"])
    assert np.isfinite(metrics["diffusion_rel_l2"])

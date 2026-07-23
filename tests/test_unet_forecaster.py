"""Tests for the reaction-diffusion forecaster's physics and parameter net."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gbm_pinn.unet_forecaster import (
    ParameterNet,
    ReactionDiffusionSimulator,
    SpatialDiffusivityNet,
    soft_dice_loss,
    summarize_records,
)


def _blob(shape=(1, 1, 20, 20, 16)):
    u = torch.zeros(shape)
    u[:, :, 8:12, 8:12, 6:10] = 1.0
    return u


def test_uniform_field_matches_scalar_diffusion():
    sim = ReactionDiffusionSimulator(n_steps=20, dt=0.2)
    u = _blob()
    D = torch.tensor([0.1])
    zero = torch.tensor([0.0])
    scalar = sim(u, D, zero, zero)
    field = sim(u, D, zero, zero, d_field=torch.ones_like(u))
    assert torch.allclose(scalar, field, atol=1e-5)


def test_pure_diffusion_conserves_mass():
    sim = ReactionDiffusionSimulator(n_steps=10, dt=0.1)
    u = torch.zeros(1, 1, 20, 20, 20)
    u[:, :, 9:11, 9:11, 9:11] = 0.5
    out = sim(u, torch.tensor([0.05]), torch.tensor([0.0]), torch.tensor([0.0]),
              d_field=torch.ones_like(u))
    assert abs(float(out.sum()) - float(u.sum())) < 1e-3


def test_zero_parameters_reproduce_persistence():
    sim = ReactionDiffusionSimulator(n_steps=30, dt=0.2)
    u = _blob()
    zero = torch.tensor([0.0])
    out = sim(u, zero, zero, zero)
    assert torch.allclose(out, u, atol=1e-6)


def test_reaction_grows_and_death_shrinks():
    sim = ReactionDiffusionSimulator(n_steps=30, dt=0.2)
    # Half-saturated so the logistic reaction rho*u*(1-u) is nonzero in the
    # interior; a fully saturated blob (u=1) does not grow, which is correct.
    u = _blob() * 0.5
    zero = torch.tensor([0.0])
    grown = sim(u, zero, torch.tensor([0.4]), zero)
    shrunk = sim(u, zero, zero, torch.tensor([0.4]))
    assert float(grown.sum()) > float(u.sum())
    assert float(shrunk.sum()) < float(u.sum())


def test_no_flux_domain_blocks_diffusion_into_excluded_region():
    sim = ReactionDiffusionSimulator(n_steps=30, dt=0.2)
    u = torch.zeros(1, 1, 20, 20, 12)
    u[:, :, 8:12, 8:12, 4:8] = 1.0
    domain = torch.ones(1, 1, 20, 20, 12)
    domain[:, :, 12:16, :, :] = 0.0
    D, r, k = torch.tensor([0.12]), torch.tensor([0.4]), torch.tensor([0.0])
    leak = sim(u, D, r, k, None)[:, :, 12:16, :, :].sum()
    walled = sim(u, D, r, k, domain)[:, :, 12:16, :, :].sum()
    assert float(leak) > 0.1
    assert float(walled) == 0.0


def test_time_scaling_makes_growth_increase_with_horizon():
    sim = ReactionDiffusionSimulator(n_steps=30, dt=0.2)
    u = _blob((3, 1, 24, 24, 16))
    D = torch.tensor([0.3, 0.3, 0.3])
    r = torch.tensor([1.0, 1.0, 1.0])
    k = torch.tensor([0.0, 0.0, 0.0])
    horizon = torch.tensor([3 / 365.0, 90 / 365.0, 1095 / 365.0])
    out = sim(u, D, r, k, None, horizon)
    masses = [float(out[i].sum()) for i in range(3)]
    assert masses[0] < masses[1] < masses[2]
    # A three-day gap should barely move the tumor.
    assert abs(masses[0] - float(u[0].sum())) < 5.0


def test_parameter_net_starts_at_persistence():
    net = ParameterNet(base_filters=8)
    source = _blob()
    treatment = torch.zeros(1, 4)
    horizon = torch.tensor([0.5])
    with torch.no_grad():
        D, rho, kappa = net(source, treatment, horizon)
        sim = ReactionDiffusionSimulator(n_steps=30, dt=0.2)
        out = sim(source, D, rho, kappa)
        dice = float(2 * (out * source).sum() / ((out ** 2).sum() + (source ** 2).sum()))
    assert dice > 0.95


def test_per_year_rates_widen_bounds():
    narrow = ParameterNet(per_year_rates=False)
    wide = ParameterNet(per_year_rates=True)
    assert wide.bounds[0] > narrow.bounds[0]
    assert wide.bounds[1] > narrow.bounds[1]


def test_scale_free_penalty_is_bound_invariant():
    """Same relative magnitude gives the same penalty regardless of bounds."""
    narrow = ParameterNet(per_year_rates=False)
    wide = ParameterNet(per_year_rates=True)
    half_narrow_D = torch.tensor([narrow.bounds[0] / 2])
    half_narrow_rho = torch.tensor([narrow.bounds[1] / 2])
    half_wide_D = torch.tensor([wide.bounds[0] / 2])
    half_wide_rho = torch.tensor([wide.bounds[1] / 2])
    pn = narrow.scale_free_penalty(half_narrow_D, half_narrow_rho)
    pw = wide.scale_free_penalty(half_wide_D, half_wide_rho)
    assert abs(float(pn) - float(pw)) < 1e-6


def test_mri_conditioning_changes_parameters():
    net = ParameterNet(in_channels=5)
    # The head is zero-initialized so an untrained net ignores every input
    # (it starts at persistence). Move it off zero to represent a trained net.
    with torch.no_grad():
        net.head[-1].weight.normal_(0, 0.1)
    source = _blob()
    treatment = torch.zeros(1, 4)
    horizon = torch.tensor([0.5])
    a = net(source, treatment, horizon, torch.randn(1, 4, 20, 20, 16))
    b = net(source, treatment, horizon, torch.randn(1, 4, 20, 20, 16))
    assert not torch.allclose(a[0], b[0])


def test_spatial_field_starts_uniform_and_flows_gradients():
    net = SpatialDiffusivityNet(in_channels=5)
    x = torch.randn(2, 5, 24, 24, 16)
    field = net(x)
    assert abs(float(field.mean().detach()) - 1.0) < 1e-4
    sim = ReactionDiffusionSimulator(n_steps=20, dt=0.2)
    u = _blob((2, 1, 24, 24, 16))
    tgt = torch.zeros(2, 1, 24, 24, 16)
    tgt[:, :, 8:12, 4:10, 6:10] = 1.0
    out = sim(u, torch.tensor([0.5, 0.5]), torch.tensor([0.3, 0.3]),
              torch.tensor([0.0, 0.0]), None, None, net(x))
    soft_dice_loss(out, tgt).backward()
    grad = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    assert grad > 1e-6


def test_soft_dice_loss_ignores_zero_background():
    a = torch.zeros(1, 1, 8, 8, 8)
    a[:, :, 2:5, 2:5, 2:5] = 1.0
    identical = soft_dice_loss(a, a.clone())
    assert float(identical) < 1e-4


def test_summarize_reports_gap_correlation():
    records = [
        {"patient_id": "P1", "dice_skill": 0.02, "horizon_days": 30.0,
         "true_volume_change_ml": 5.0, "predicted_volume_change_ml": 1.0},
        {"patient_id": "P1", "dice_skill": 0.01, "horizon_days": 400.0,
         "true_volume_change_ml": 2.0, "predicted_volume_change_ml": -1.0},
        {"patient_id": "P2", "dice_skill": -0.01, "horizon_days": 100.0,
         "true_volume_change_ml": 8.0, "predicted_volume_change_ml": 3.0},
        {"patient_id": "P3", "dice_skill": 0.03, "horizon_days": 60.0,
         "true_volume_change_ml": 4.0, "predicted_volume_change_ml": 2.0},
    ]
    summary = summarize_records(records)
    assert "pred_change_vs_gap_correlation" in summary
    assert -1.0 <= summary["pred_change_vs_gap_correlation"] <= 1.0

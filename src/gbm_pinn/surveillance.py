"""Ground-truth simulation of informative surveillance scheduling.

Real longitudinal cohorts are imaged on a clinical schedule, not at random
times. This module builds a virtual cohort with known reaction-diffusion
dynamics and re-images the *same* patients under different schedules. Because
the tumors are identical across schedules, any change in fitted parameters or
in model rankings is caused by the schedule alone.

Each virtual patient follows

    du/dt = D * laplacian(u) + rho * u * (1 - u) - kappa(t) * u

on a 2D disk with no-flux walls. ``kappa(t)`` holds the tumor near stability
(treatment response) until a latent progression time, after which it drops to
zero and the tumor grows at ``rho``.

Two scheduling mechanisms are modelled, each with its own strength:

* **Risk-adapted planning** (``gamma``): the planned next interval shrinks as
  the patient's latent growth rate at the source scan rises.
* **Event triggering** (``trigger_probability``): if the tumor grows past a
  symptomatic threshold before the planned scan, an unscheduled scan follows
  after a short lag. The interval then ends *because* of the change, which
  no source-time covariate can anticipate.

Every schedule is rescaled so its median interval equals
``base_interval_days``, so scan frequency stays fixed while informativeness
varies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CohortConfig:
    """Virtual-cohort settings. Units are millimetres and days."""

    n_patients: int = 240
    grid_size: int = 64
    spacing_mm: float = 2.0
    brain_radius_mm: float = 60.0
    follow_up_days: int = 720
    max_scans: int = 8
    rho_median: float = 0.02
    rho_log_sigma: float = 0.4
    diffusivity_median: float = 0.1
    diffusivity_log_sigma: float = 0.4
    progression_mean_days: float = 300.0
    stable_margin_low: float = -0.1
    stable_margin_high: float = 0.3
    base_interval_days: float = 90.0
    interval_log_noise: float = 0.35
    min_interval_days: int = 21
    max_interval_days: int = 270
    trigger_growth_fraction: float = 0.25
    trigger_lag_mean_days: float = 10.0
    threshold: float = 0.1


@dataclass(frozen=True, slots=True)
class VirtualPatients:
    """Patient-level ground truth and fixed random draws shared by every schedule."""

    rho: FloatArray
    diffusivity: FloatArray
    stable_margin: FloatArray
    progression_day: FloatArray
    initial_density: FloatArray
    schedule_noise: FloatArray
    trigger_draw: FloatArray
    trigger_lag: FloatArray

    def net_growth(self, day: FloatArray | float, index: int | slice = slice(None)) -> FloatArray:
        """Latent net exponential growth rate at low density."""
        progressing = np.asarray(day) >= self.progression_day[index]
        stable = -self.stable_margin[index] * self.rho[index]
        return np.where(progressing, self.rho[index], stable)


@dataclass(frozen=True, slots=True)
class Transitions:
    """Consecutive scan pairs observed under one schedule."""

    patient: NDArray[np.int64]
    source_day: NDArray[np.int64]
    horizon: NDArray[np.int64]
    triggered: NDArray[np.bool_]
    source: FloatArray
    target: FloatArray

    def __len__(self) -> int:
        return int(self.patient.shape[0])

    def subset(self, keep: NDArray[np.bool_]) -> Transitions:
        return Transitions(
            self.patient[keep],
            self.source_day[keep],
            self.horizon[keep],
            self.triggered[keep],
            self.source[keep],
            self.target[keep],
        )


@dataclass(frozen=True, slots=True)
class Schedule:
    """Scan days per patient and whether each scan was event-triggered."""

    days: list[list[int]]
    triggered: list[list[bool]]

    def intervals(self) -> NDArray[np.int64]:
        gaps = [np.diff(d) for d in self.days if len(d) > 1]
        return np.concatenate(gaps) if gaps else np.zeros(0, dtype=np.int64)


def brain_domain(config: CohortConfig) -> FloatArray:
    """Disk-shaped tissue mask on the simulation grid."""
    centre = (config.grid_size - 1) / 2.0
    axis = (np.arange(config.grid_size) - centre) * config.spacing_mm
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return (np.hypot(xx, yy) <= config.brain_radius_mm).astype(np.float64)


def sample_patients(config: CohortConfig, rng: np.random.Generator) -> VirtualPatients:
    """Draw patient kinetics, progression times, residual tumors, and schedule noise."""
    n = config.n_patients
    rho = config.rho_median * np.exp(config.rho_log_sigma * rng.standard_normal(n))
    diffusivity = config.diffusivity_median * np.exp(
        config.diffusivity_log_sigma * rng.standard_normal(n)
    )
    margin = rng.uniform(config.stable_margin_low, config.stable_margin_high, n)
    progression = rng.exponential(config.progression_mean_days, n)

    centre = (config.grid_size - 1) / 2.0
    axis = (np.arange(config.grid_size) - centre) * config.spacing_mm
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    initial = np.empty((n, config.grid_size, config.grid_size))
    for i in range(n):
        radius = rng.uniform(0.0, 25.0)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        width = rng.uniform(4.0, 8.0)
        amplitude = rng.uniform(0.3, 0.6)
        initial[i] = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width**2))
    initial *= brain_domain(config)

    shape = (n, config.max_scans)
    return VirtualPatients(
        rho,
        diffusivity,
        margin,
        progression,
        initial,
        schedule_noise=rng.standard_normal(shape),
        trigger_draw=rng.uniform(size=shape),
        trigger_lag=rng.exponential(config.trigger_lag_mean_days, shape),
    )


class _Stepper:
    """Batched explicit Fisher-KPP integrator with no-flux domain walls."""

    def __init__(self, domain: FloatArray, spacing_mm: float, device: torch.device) -> None:
        self.domain = torch.as_tensor(domain, dtype=torch.float32, device=device)[None, None]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.kernel = kernel.to(device)[None, None]
        self.neighbours = F.conv2d(self.domain, self.kernel, padding=1)
        self.inv_h2 = 1.0 / spacing_mm**2

    def laplacian(self, u: torch.Tensor) -> torch.Tensor:
        total = F.conv2d(u * self.domain, self.kernel, padding=1)
        return (total - self.neighbours * u) * self.domain * self.inv_h2

    def step(
        self, u: torch.Tensor, d: torch.Tensor, rho: torch.Tensor, kappa: torch.Tensor
    ) -> torch.Tensor:
        u = u + d * self.laplacian(u) + rho * u * (1.0 - u) - kappa * u
        return u.clamp(0.0, 1.0) * self.domain


def _tensor(values: FloatArray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).view(-1, 1, 1, 1)


def _integrate(
    patients: VirtualPatients,
    config: CohortConfig,
    device: torch.device,
    record: dict[int, list[int]] | None,
) -> tuple[FloatArray, dict[tuple[int, int], FloatArray]]:
    """Daily integration. Returns tumor area per day and any requested snapshots."""
    stepper = _Stepper(brain_domain(config), config.spacing_mm, device)
    u = torch.as_tensor(patients.initial_density, dtype=torch.float32, device=device)[:, None]
    rho = _tensor(patients.rho, device)
    d = _tensor(patients.diffusivity, device)
    stable_kappa = _tensor(patients.rho * (1.0 + patients.stable_margin), device)
    progression = _tensor(patients.progression_day, device)

    days = config.follow_up_days + 1
    area = np.empty((u.shape[0], days))
    snapshots: dict[tuple[int, int], FloatArray] = {}
    pixel = config.spacing_mm**2
    with torch.no_grad():
        for day in range(days):
            area[:, day] = (u[:, 0] >= config.threshold).sum(dim=(1, 2)).cpu().numpy() * pixel
            for i in (record or {}).get(day, []):
                snapshots[(i, day)] = u[i, 0].cpu().numpy().astype(np.float64)
            kappa = torch.where(day >= progression, torch.zeros_like(stable_kappa), stable_kappa)
            u = stepper.step(u, d, rho, kappa)
    return area, snapshots


def tumor_area(patients: VirtualPatients, config: CohortConfig, device: torch.device) -> FloatArray:
    """Thresholded tumor area (mm^2) for every patient on every day of follow-up."""
    return _integrate(patients, config, device, None)[0]


def scan_schedule(
    patients: VirtualPatients,
    area: FloatArray,
    config: CohortConfig,
    *,
    gamma: float = 0.0,
    trigger_probability: float = 0.0,
    scale: float | None = None,
) -> Schedule:
    """Scan days under risk-adapted planning and/or event-triggered imaging."""
    if scale is None:
        scale = _calibrate_scale(
            patients, area, config, gamma=gamma, trigger_probability=trigger_probability
        )
    reference = np.concatenate([patients.rho, -patients.stable_margin * patients.rho])
    mean, spread = float(reference.mean()), float(reference.std())

    all_days: list[list[int]] = []
    all_triggered: list[list[bool]] = []
    for i in range(patients.rho.shape[0]):
        day, days, triggered = 0, [0], [False]
        for k in range(config.max_scans - 1):
            z = (float(patients.net_growth(day, i)) - mean) / spread
            planned = (
                scale
                * config.base_interval_days
                * np.exp(-gamma * z + config.interval_log_noise * patients.schedule_noise[i, k])
            )
            interval = int(
                np.clip(round(planned), config.min_interval_days, config.max_interval_days)
            )
            event = False
            if patients.trigger_draw[i, k] < trigger_probability:
                limit = min(day + interval, config.follow_up_days)
                window = area[i, day + 1 : limit + 1]
                crossed = np.flatnonzero(
                    window >= area[i, day] * (1.0 + config.trigger_growth_fraction)
                )
                if crossed.size:
                    early = int(crossed[0]) + 1 + round(patients.trigger_lag[i, k])
                    early = max(early, 7)
                    if early < interval:
                        interval, event = early, True
            day += interval
            if day > config.follow_up_days:
                break
            days.append(day)
            triggered.append(event)
        all_days.append(days)
        all_triggered.append(triggered)
    return Schedule(all_days, all_triggered)


def _calibrate_scale(
    patients: VirtualPatients,
    area: FloatArray,
    config: CohortConfig,
    *,
    gamma: float,
    trigger_probability: float,
) -> float:
    """Bisect the planned-interval multiplier so the median interval hits the base."""
    low, high = -3.0, 3.0
    for _ in range(25):
        mid = 0.5 * (low + high)
        schedule = scan_schedule(
            patients,
            area,
            config,
            gamma=gamma,
            trigger_probability=trigger_probability,
            scale=float(np.exp(mid)),
        )
        gaps = schedule.intervals()
        if gaps.size and np.median(gaps) < config.base_interval_days:
            low = mid
        else:
            high = mid
    return float(np.exp(0.5 * (low + high)))


def observe(
    patients: VirtualPatients, schedule: Schedule, config: CohortConfig, device: torch.device
) -> Transitions:
    """Image every patient on its scan days and pair consecutive scans."""
    record: dict[int, list[int]] = {}
    for i, days in enumerate(schedule.days):
        for day in days:
            record.setdefault(day, []).append(i)
    _, scans = _integrate(patients, config, device, record)

    patient, source_day, horizon, triggered, source, target = [], [], [], [], [], []
    for i, days in enumerate(schedule.days):
        for k in range(len(days) - 1):
            start, end = days[k], days[k + 1]
            patient.append(i)
            source_day.append(start)
            horizon.append(end - start)
            triggered.append(schedule.triggered[i][k + 1])
            source.append(scans[(i, start)])
            target.append(scans[(i, end)])
    return Transitions(
        np.asarray(patient),
        np.asarray(source_day),
        np.asarray(horizon),
        np.asarray(triggered, dtype=bool),
        np.stack(source),
        np.stack(target),
    )


def forecast(
    transitions: Transitions,
    rate: float,
    diffusivity: float,
    config: CohortConfig,
    device: torch.device,
    *,
    fixed_horizon: int | None = None,
) -> FloatArray:
    """Shared-parameter Fisher-KPP forecast from each source scan.

    With ``fixed_horizon=None`` the model is time-integrated: it evolves each
    source over its true elapsed interval. Otherwise every transition is
    evolved for the same number of days, which is the time-blind model.
    """
    stepper = _Stepper(brain_domain(config), config.spacing_mm, device)
    horizons = (
        transitions.horizon
        if fixed_horizon is None
        else np.full(len(transitions), fixed_horizon, dtype=np.int64)
    )
    u = torch.as_tensor(transitions.source, dtype=torch.float32, device=device)[:, None]
    out = torch.empty_like(u)
    h = torch.as_tensor(horizons, device=device).view(-1, 1, 1, 1)
    d = torch.tensor(diffusivity, device=device)
    # Negative net rates act as pure decay so the logistic term never pushes
    # density above the observed level in a shrinking tumor.
    growth = torch.tensor(max(rate, 0.0), device=device)
    decay = torch.tensor(max(-rate, 0.0), device=device)
    with torch.no_grad():
        for day in range(int(horizons.max()) + 1):
            out = torch.where(h == day, u, out)
            u = stepper.step(u, d, growth, decay)
    return out[:, 0].cpu().numpy().astype(np.float64)


def dice(prediction: FloatArray, target: FloatArray, threshold: float) -> FloatArray:
    """Per-transition Dice of thresholded masks. Two empty masks score 1."""
    p = prediction >= threshold
    t = target >= threshold
    axes = tuple(range(1, p.ndim))
    overlap = (p & t).sum(axis=axes)
    total = p.sum(axis=axes) + t.sum(axis=axes)
    return np.where(total == 0, 1.0, 2.0 * overlap / np.maximum(total, 1))


@dataclass(frozen=True, slots=True)
class Fit:
    rate: float
    train_dice: float


def fit_rate(
    transitions: Transitions,
    diffusivity: float,
    config: CohortConfig,
    device: torch.device,
    *,
    fixed_horizon: int | None = None,
    bounds: tuple[float, float] = (-0.02, 0.06),
    coarse: int = 17,
    fine: int = 11,
) -> Fit:
    """Coarse-to-fine grid search for the shared net rate maximizing mean Dice."""

    def score(rate: float) -> float:
        pred = forecast(transitions, rate, diffusivity, config, device, fixed_horizon=fixed_horizon)
        return float(dice(pred, transitions.target, config.threshold).mean())

    grid = np.linspace(*bounds, coarse)
    scores = [score(r) for r in grid]
    best = int(np.argmax(scores))
    step = grid[1] - grid[0]
    refine = np.linspace(grid[best] - step, grid[best] + step, fine)
    refine_scores = [score(r) for r in refine]
    k = int(np.argmax(refine_scores))
    return Fit(float(refine[k]), float(refine_scores[k]))


def area_change(transitions: Transitions, config: CohortConfig) -> FloatArray:
    """Absolute change in thresholded tumor area (mm^2) per transition."""
    pixel = config.spacing_mm**2
    src = (transitions.source >= config.threshold).sum(axis=(1, 2)) * pixel
    tgt = (transitions.target >= config.threshold).sum(axis=(1, 2)) * pixel
    return np.abs(tgt - src).astype(np.float64)


def config_dict(config: CohortConfig) -> dict[str, float | int]:
    return asdict(config)

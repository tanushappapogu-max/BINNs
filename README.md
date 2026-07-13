# BINNs

BINNs is a research codebase for modeling and forecasting glioblastoma growth after surgical resection. It combines biological reaction-diffusion equations with physics-informed neural networks, or PINNs.

The model represents tumor cells as a continuous density field that changes across space and time. It learns from early tumor observations while being constrained by equations for cell migration, proliferation, and treatment response. The system supports controlled synthetic experiments and a two-dimensional longitudinal MRI pilot.

## What we are building

The project is being developed as a pipeline with four main parts:

1. A numerical simulator that generates tumor growth from known biological parameters.
2. A PINN that learns tumor evolution from early observations and the governing equation.
3. A surgery-aware representation that excludes the resection cavity and enforces physical boundary conditions around it.
4. An evaluation system that hides a future tumor field, predicts it, and compares the prediction with the known answer and a persistence baseline.

## Biological model

The primary baseline is a tissue-aware logistic reaction-diffusion equation:

```math
\frac{\partial u(\mathbf{x},t)}{\partial t}
=
\nabla \cdot \left[D(\mathbf{x})\nabla u(\mathbf{x},t)\right]
+
\rho\,u(\mathbf{x},t)\left[1-u(\mathbf{x},t)\right]
-
\kappa\,q(t)\,u(\mathbf{x},t)
```

In this equation:

- $u(x,t)$ is normalized viable tumor-cell density between 0 and 1.
- $D(x)$ is the effective tumor-cell diffusion coefficient.
- $\rho$ is the tumor-cell proliferation rate.
- $q(t)$ is known treatment exposure derived from the treatment schedule.
- $\kappa$ is the treatment-response rate.
- $x$ is spatial position.
- $t$ is time.

The diffusion field can vary across tissue. White matter and gray matter can therefore have different effective migration rates.

The code also supports a weak-Allee growth law:

```math
\frac{\partial u(\mathbf{x},t)}{\partial t}
=
\nabla \cdot \left[D(\mathbf{x})\nabla u(\mathbf{x},t)\right]
+
\rho\,u(\mathbf{x},t)
\left[u(\mathbf{x},t)+\beta\right]
\left[1-u(\mathbf{x},t)\right]
```

where $\beta$ controls low-density growth behavior. This form is useful for studying sparse residual tumor cells after surgery.

## Modeling surgery

The resection cavity is treated as an excluded internal region. Tumor density is set to zero inside the cavity, and tumor cells cannot diffuse through the cavity wall.

The cavity boundary uses the zero-flux condition

```math
\left.
\mathbf{n}\cdot\left[D(\mathbf{x})\nabla u(\mathbf{x},t)\right]
\right|_{\partial\Omega_{\mathrm{cavity}}}
=0
```

where $\mathbf n$ is the boundary normal.

The piecewise cavity PINN uses two neural fields:

- a geometry-aware field for the tissue directly surrounding the cavity;
- a Cartesian field for tissue farther from the cavity.

The fields meet at an artificial interface. Training enforces continuity of tumor density and diffusive flux across that interface.

## How the PINN is trained

The PINN receives spatial and temporal coordinates and predicts normalized tumor density. Its loss combines several constraints:

- Observation loss compares predictions with known early tumor fields.
- Physics loss measures violation of the reaction-diffusion equation.
- Boundary loss enforces zero normal flux at anatomical boundaries.
- Interface loss connects the near-cavity and far-field neural representations.

Physical parameters such as $D$ and $\rho$ can be fixed or learned. Learned parameters are constrained to configured intervals.

Treatment windows can represent active therapy and an optional exponentially decaying post-treatment effect. The schedule supplies $q(t)$, while the bounded response coefficient $\kappa$ can be fixed or learned.

Training uses Adam followed by optional L-BFGS refinement. Data, collocation, boundary, and interface points can be minibatched independently for accelerator memory control.

## Synthetic hidden-future experiment

The included experiment performs a controlled forecasting test:

1. The finite-volume solver generates tumor fields at several times.
2. Only the early fields are supplied to the PINN.
3. The future field is withheld during training.
4. The PINN predicts the withheld time.
5. The prediction is compared with the reference field.

The result includes:

- Dice score for tumor-shape overlap;
- RMSE and MAE for density-field error;
- relative tumor-volume error;
- recovered diffusion and proliferation parameters;
- data, physics, boundary, and interface losses;
- resolved execution device, training time, and CUDA peak memory when available.

## Repository layout

```text
src/gbm_pinn/
  equation.py     Reaction parameters and growth laws
  solver.py       Finite-volume reference solver
  synthetic.py    Synthetic initial tumor fields
  inverse.py      Numerical parameter estimation
  pinn.py         PINN architecture, residuals, and training
  tissue.py       Tissue-dependent diffusion model
  cavity.py       Single-field and piecewise cavity PINNs
  treatment.py    Treatment schedules and treatment-aware PINN
  clinical.py     Longitudinal NIfTI loading and label preprocessing
  clinical_experiment.py  Held-out real-patient pilot
  clinical_3d_experiment.py  Full-volume 3D PINN pilot
  mechanistic_forecast.py  Last-observation-anchored 3D rollout
  experiment.py   End-to-end synthetic forecast experiment

scripts/
  run_synthetic_forecast.py
  run_clinical_pilot.py
  run_clinical_3d_pilot.py
  run_mechanistic_forecast.py

tests/
  Unit and integration tests for equations, solvers, PINNs, tissue, cavities, and experiments
```

## Installation

Python 3.11 through 3.13 is supported. Python 3.12 is recommended.

Using `uv`:

```bash
uv venv --python 3.12
uv sync --extra dev --extra ml
```

Using `pip` in an activated virtual environment:

```bash
python -m pip install -e ".[dev,ml]"
```

The optional `imaging` dependency group installs NIfTI support for the clinical pilot:

```bash
uv sync --extra dev --extra ml --extra imaging
```

## Running the tests

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Running a postoperative cavity forecast

This command runs the validated two-dimensional CPU configuration:

```bash
uv run python scripts/run_synthetic_forecast.py \
  --device cpu \
  --cavity-radius 0.15 \
  --cavity-interface-radius 0.35 \
  --interface-points 256 \
  --interface-weight 5 \
  --grid-size 41 \
  --observation-times 0 1 \
  --forecast-time 2 \
  --epochs 3000 \
  --lbfgs-iterations 500 \
  --seed 17 \
  --data-points-per-time 512 \
  --collocation-points 1024 \
  --boundary-points 256 \
  --hidden-width 64 \
  --data-weight 10 \
  --physics-weight 5 \
  --boundary-weight 5
```

Use `--device mps` on a compatible Apple Silicon Mac or `--device cuda` on a CUDA system. `--device auto` selects CUDA first, then MPS, then CPU.

To save the metrics as JSON, add:

```bash
--output path/to/result.json
```

## Running a longitudinal MRI pilot

The clinical runner expects one patient directory containing naturally ordered timepoint folders. Each timepoint must include a tumor segmentation and at least one aligned skull-stripped NIfTI image.

```bash
uv run python scripts/run_clinical_pilot.py /path/to/PatientID_XXXX \
  --scan-days 90 152 208 \
  --observation-count 2 \
  --forecast-index 2 \
  --epochs 2000 \
  --device mps \
  --checkpoint outputs/clinical/checkpoint.pt \
  --artifact outputs/clinical/forecast.npz \
  --output outputs/clinical/metrics.json
```

The last observation is also evaluated as a persistence forecast. The result reports whether the PINN beats that baseline.

## Running a full-volume 3D pilot

The 3D runner trains on complete longitudinal volumes through sampled observation, physics, and anatomical-boundary points. Full-volume evaluation is divided into bounded batches.

```bash
uv run python scripts/run_clinical_3d_pilot.py /path/to/PatientID_XXXX \
  --scan-days 90 152 208 264 \
  --observation-count 3 \
  --forecast-index 3 \
  --device mps \
  --fourier-frequencies 1 2 4 8 \
  --checkpoint outputs/clinical_3d/checkpoint.pt \
  --artifact outputs/clinical_3d/forecast.npz \
  --output outputs/clinical_3d/metrics.json
```

The mechanistic runner starts exactly from the last observed 3D tumor field and evolves the reaction-diffusion equation using fixed or PINN-estimated coefficients.

```bash
uv run python scripts/run_mechanistic_forecast.py /path/to/PatientID_XXXX \
  --scan-days 90 152 208 264 \
  --observation-count 3 \
  --forecast-index 3 \
  --diffusivity 0.02 \
  --proliferation 0.012 \
  --artifact outputs/mechanistic/forecast.npz \
  --output outputs/mechanistic/metrics.json
```

## Current scope

The clinical runner is a research experiment, not a diagnostic or treatment-planning tool.

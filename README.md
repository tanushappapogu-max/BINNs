# BINNs

BINNs is a research codebase for modeling and forecasting glioblastoma growth after surgical resection. It combines biological reaction-diffusion equations with physics-informed neural networks, or PINNs.

The model represents tumor cells as a continuous density field that changes across space and time. It learns from early tumor observations while being constrained by equations for cell migration and proliferation. The current system focuses on predicting a future tumor-density field around a simulated surgical cavity.

## What we are building

The project is being developed as a pipeline with four main parts:

1. A numerical simulator that generates tumor growth from known biological parameters.
2. A PINN that learns tumor evolution from early observations and the governing equation.
3. A surgery-aware representation that excludes the resection cavity and enforces physical boundary conditions around it.
4. An evaluation system that hides a future tumor field, predicts it, and compares the prediction with the known answer.

The current implementation operates on two-dimensional synthetic tumor fields. The longer-term system will use three-dimensional postoperative MRI data, patient anatomy, and longitudinal scans.

## Biological model

The primary baseline is a tissue-aware logistic reaction-diffusion equation:

$$
\frac{\partial u}{\partial t}
=
\nabla\cdot\left(D(x)\nabla u\right)
+\rho u(1-u).
$$

In this equation:

- $u(x,t)$ is normalized viable tumor-cell density between 0 and 1.
- $D(x)$ is the effective tumor-cell diffusion coefficient.
- $\rho$ is the tumor-cell proliferation rate.
- $x$ is spatial position.
- $t$ is time.

The diffusion field can vary across tissue. White matter and gray matter can therefore have different effective migration rates.

The code also supports a weak-Allee growth law:

$$
\frac{\partial u}{\partial t}
=
\nabla\cdot\left(D(x)\nabla u\right)
+\rho u(u+\beta)(1-u),
$$

where $\beta$ controls low-density growth behavior. This form is useful for studying sparse residual tumor cells after surgery.

## Modeling surgery

The resection cavity is treated as an excluded internal region. Tumor density is set to zero inside the cavity, and tumor cells cannot diffuse through the cavity wall.

The cavity boundary uses the zero-flux condition

$$
\mathbf n\cdot D\nabla u=0,
$$

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
  experiment.py   End-to-end synthetic forecast experiment

scripts/
  run_synthetic_forecast.py

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

The optional `imaging` dependency group installs NIfTI support for future MRI workflows:

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

## Current scope

The code currently validates the mathematical and machine-learning pipeline using synthetic normalized tumor-density fields. It is not a clinical prediction tool and should not be used for diagnosis or treatment decisions.

Real-patient forecasting will require three-dimensional MRI preprocessing, longitudinal registration, tumor segmentation, an MRI observation model, treatment information, and external clinical validation.

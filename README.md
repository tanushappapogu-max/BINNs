# BINNs

Mechanistic and physics-informed models for forecasting postoperative glioblastoma growth.

The initial implementation contains a finite-volume reference solver for the tissue-aware Fisher–KPP equation

$$
\frac{\partial u}{\partial t}
=
\nabla\cdot(D(x)\nabla u)
+\rho u(1-u)
-k(x,t)u,
$$

where $u$ is normalized viable tumor-cell density, $D$ is spatial diffusivity, $\rho$ is the proliferation rate, and $k$ is an effective treatment rate. Brain and resection-cavity boundaries use zero normal flux.

## Development setup

Python 3.11–3.13 is supported. Python 3.12 is recommended.

```bash
uv venv --python 3.12
uv sync --extra dev
uv run pytest
```

## Implemented components

- Validated numerical reference solver
- Spatially varying diffusivity
- Brain and cavity masks
- Synthetic initial conditions
- Unit-tested conservation and growth behavior

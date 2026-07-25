# Project status — shelved

This project is paused. It contains two independent, checked results and a
working codebase, kept for possible later continuation.

## What is here

**1. GBM post-surgical regrowth forecaster (clinical).**
A physics-informed reaction-diffusion forecaster on MU-Glioma-Post
(`src/gbm_pinn/unet_forecaster.py`). Main finding, documented in
`docs/results.md`: scan intervals are length-biased (elapsed time is
anti-correlated with observed change, r = -0.285), which penalizes the
biologically-faithful time-integrated model under standard Dice; the effect is
not a between-interval artifact (interval-stratified null), and inverse-intensity
correction is infeasible here (interval unpredictable from covariates, R^2 ~ 0).
Held-out test skill over persistence +0.0129 (patient-clustered p = 0.026).

**2. BINN reaction-diffusion term learning (in-vitro / synthetic).**
`src/gbm_pinn/binn.py` learns D(u) and R(u) as neural networks. Validated on
synthetic data (solution R^2 > 0.99, growth corr > 0.85). Identifiability study
(scratchpad): diffusion recovery is structurally capped at ~0.8 correlation and
collapses below ~15 timepoints; data quality does not push it past the ceiling.

## If resumed, open next steps

- GBM: replicate the length-bias finding on a second cohort (LUMIERE / BraTS
  longitudinal) to move from single-cohort observation to phenomenon.
- BINN: calibrated uncertainty that reports diffusion non-identifiability
  faithfully; attempt on real assay data (needs density-field preprocessing).

## Tests

`PYTHONPATH=src pytest tests/` — 178 passing at time of shelving.

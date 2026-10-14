# Pre-registration: external test of the physics-informed forecaster

Written 2026-10-14, before the forecaster was trained for this test or
evaluated on any LUMIERE outcome. Pipeline checks run so far: index
construction, one training epoch at 4 mm, and evaluation of that one-epoch
model on 3 LUMIERE pairs to confirm shapes and output fields; no metric from
those checks was examined.

## Question

Does the MU-trained physics-informed forecaster (3D CNN predicting D, ρ, κ for
a differentiable Fisher–KPP solver) beat persistence on (a) MU's untouched
final-test patients and (b) an independent hospital, LUMIERE? And does an
image-based model forecast tumor size better than the size-history ceiling
measured on the same LUMIERE pairs?

## Data

- **Training / selection:** MU shared training (202 pairs) and model selection
  (39 pairs), after removing duplicate patients (0062 = 0060, 0132 = 0113) and
  pairs touching the byte-identical timepoint 5 of 0018 and 0032. Treatment
  exposure is removed (LUMIERE has none; on MU zeroing it did not change
  skill).
- **MU final test:** 35 pairs, 17 patients (no duplicates were present).
- **LUMIERE external:** 313 pairs, 61 patients: the consecutive pairs of the
  pre-registered ceiling segments (surgery splits, 28-day rule, segments of at
  least three scans), built from the deduplicated session table (three
  sessions are listed twice in LUMIERE's rating table; see
  `docs/preregistration_ceiling_lumiere.md`). LUMIERE masks are DeepBraTumIA
  atlas-space (MNI, 1 mm), remapped to MU labels; the network's global pooling
  accepts the different grid, padded to a size divisible by 8.

## Model and training (fixed)

`scripts/external_forecaster.py` with `gbm_pinn.unet_forecaster`:
2 mm (downsample 2), 25 epochs, batch size 2, learning rate 3e-4, parameter
regularization 0.1, 30 solver steps of dt 0.2, time-blind (no PDE time
scaling), no cavity domain, seeds 42, 43, 44. The epoch with the best MU
model-selection skill is kept. LUMIERE is never used for training or selection.

## Primary quantities

For each pair, Dice skill = forecast Dice − persistence Dice (threshold 0.1 on
density; whole abnormality), averaged over the three seeds. Patient-clustered
test: per-patient mean of the seed-averaged skill, two-sided Wilcoxon
signed-rank against zero.

Radius error: equivalent-sphere radius of the thresholded source, target and
predicted volumes. Predicted volume is averaged over seeds before conversion.
Reduction = 1 − mean (target − predicted)² / mean (target − source)², with a
2000-resample patient bootstrap 95% CI.

## Predictions and decision rules

**E1, internal replication (MU final test).** Confirmed if mean seed-averaged
skill > 0 and p < 0.05. Otherwise not replicated. (Original report: +0.0129,
p = 0.026, with treatment flags and duplicates present in training.)

**E2, external transfer (LUMIERE).** Confirmed if mean seed-averaged skill > 0
and p < 0.05. "No transfer" if p ≥ 0.05. "Harmful" if mean < 0 and p < 0.05.

**E3, images versus the size-history ceiling (LUMIERE).** The LUMIERE
whole-abnormality history ceiling from the confirmatory run is 0.016
[−0.041, 0.075]; the oracle ceiling is 0.406. Prediction: the forecaster does
**not** exceed the history ceiling. Confirmed if the lower bound of the
forecaster's radius-error reduction CI is at most 0.075. Falsified (images
carry forecastable information beyond size history) if that lower bound is
above 0.075.

Verdicts are applied mechanically by `scripts/judge_external_forecaster.py`.

## Reported but not decisive

Per-seed results; growth-region Dice; MU final-test radius-error reduction.

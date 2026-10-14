# Pre-registration: forecastability ceiling, confirmatory test on LUMIERE

Written 2026-10-13, before any tumor-change statistic was computed on LUMIERE.
At the time of writing, the LUMIERE per-session volumes had been extracted
(`outputs/surveillance/lumiere_scan_volumes.csv`) and checked only for data
integrity: missing values, empty masks, surgery-day session labels, and the
number and length of segments the rules below produce (79 segments from 64
patients for the primary analysis). No model was fitted and no change, ceiling,
correlation or test statistic was computed on LUMIERE.

MU-Glioma-Post is the discovery cohort. Its results
(`outputs/surveillance/ceiling_mu.json`, enhancing tumor) were:

| Quantity | MU estimate [95% patient-bootstrap CI] |
|---|---|
| History ceiling | 0.036 [0.002, 0.098] |
| Oracle ceiling | 0.230 [0.092, 0.418] |
| Reverting component needed | p = 0.005 |
| Clock component needed | p = 0.31 (test power 0.60) |
| Patient-specific rates needed | p = 0.49; 80% power reached at drift SD 10 mm/yr |

## Model and code

`gbm_pinn.ceiling.analyze_cohort`, unchanged from the version that produced
the MU results, run by `scripts/analyze_ceiling_lumiere.py`: equivalent-sphere
radius y at irregular days follows

    x_{k+1} = x_k + v h_k + w_k,  w_k ~ N(0, q0 + q1 h_k);  y_k = x_k + e_k,  e_k ~ N(0, r)

with v ~ N(μ, τ²) per segment, fitted by maximum likelihood (Kalman filter).

## Data and trajectories

- LUMIERE, all 91 patients, DeepBraTumIA atlas-space segmentations.
- Day = week × 7.
- Every post-operative session (a later part of a surgery day, or rated
  Post-Op) starts a new segment. Sessions before a patient's first
  post-operative scan (untreated tumor) are dropped. Pre-operative scans
  before a later surgery stay as the last scan of the preceding segment.
- Segments are then split at intervals of 28 days or less, as in MU.
- Segments with at least 3 scans are analysed.
- **Primary measure:** enhancing tumor radius. Tumor core is not used, because
  after surgery DeepBraTumIA's necrotic label also covers the resection cavity.
- Secondary: enhancing radius with surgery splits only (no 28-day rule); whole
  abnormality radius.
- 300 patient-bootstrap resamples; 100 simulations per power estimate; seed
  20261016.

## Predictions and how each is judged

**P1, primary: forecasting from size history removes little of persistence's
error.** Confirmed if the history ceiling point estimate is at most 0.10.
Strongly confirmed if, in addition, its bootstrap upper bound is at most 0.15.
Falsified if the point estimate exceeds 0.10.

**P2: part of scan-to-scan change reverts.** Confirmed if dropping the
reverting component is rejected (p < 0.05).

**P3: change does not accumulate with elapsed time (no clock).**
- Confirmed: clock test power ≥ 0.80 and dropping the clock term is not
  rejected (p ≥ 0.05).
- Falsified: dropping the clock term is rejected (p < 0.05).
- Inconclusive: power < 0.80 and p ≥ 0.05.

**P4: no detectable between-patient differences in steady growth rate.**
- Confirmed: dropping patient-specific rates is not rejected (p ≥ 0.05). The
  minimum detectable drift SD at 80% power is then reported as the size of
  rate differences the data rule out.
- Falsified: p < 0.05 (patient-specific rates detected).
- Design expectation, not a test: longer LUMIERE trajectories give a minimum
  detectable drift SD at or below MU's 10 mm/yr.

**P5: even a perfect model of the current tumor could not remove most of
persistence's error.** Confirmed if the oracle ceiling is below 0.50.

## What would change the conclusions

- P1 falsified: the low ceiling is specific to MU, and the report says so.
- P3 or P4 falsified: the corresponding MU hint does not generalize.
- Secondary analyses are reported alongside but do not change any verdict.

## Exploratory, not pre-registered

Relating the reverting component to RANO ratings (for example, whether large
excursions that revert coincide with PD followed by SD or PR, the pattern of
pseudoprogression) will be reported as exploratory.

## Outcome

Run 2026-10-14 after the protocol was committed (b30bc34). Results in
`outputs/surveillance/ceiling_lumiere.json`; verdicts applied mechanically by
`scripts/judge_lumiere_predictions.py` (`outputs/surveillance/lumiere_verdicts.json`).
Primary analysis: enhancing radius, 75 segments, 61 patients, 312 transitions.

| Prediction | Verdict | Result |
|---|---|---|
| P1 history ceiling ≤ 0.10 | **strongly confirmed** | 0.049 [−0.003, 0.114] |
| P2 reverting change present | **confirmed** | p = 0.006 |
| P3 no clock | **falsified** | clock needed, p = 3.4e-5; test power 0.82 |
| P4 no patient-specific rates | **confirmed** | p = 0.50; minimum detectable drift SD 10 mm/yr |
| P5 oracle ceiling < 0.50 | **confirmed** | 0.233 [0.012, 0.380] |

Fitted primary model: reverting SD 2.5 mm [0, 3.7]; interval-independent
lasting change SD 0.0 mm [0, 3.2]; clock-like lasting change SD 4.0 mm per
√90 days [2.3, 5.2]; common drift 3.5 mm/yr [2.0, 4.9]; between-patient drift
SD at the floor.

Reading, stated plainly: in LUMIERE, lasting change does accumulate with
elapsed time (P3 falsified), but as diffusion-like variation whose direction
is not predictable, around a slow common drift with no measurable
between-patient differences in steady rate. That is why the forecastability
ceiling stays low (P1) even though a clock is present. The MU hint of "no
clock" does not generalize; MU's clock test had power 0.60.

Secondary analyses (not decisive): surgery splits only gives the same verdicts
(clock p = 1e-5, ceiling 0.066). Whole abnormality shows no clock (p = 0.15) but
that test had power 0.02, and over half of its change reverts (reverting share
0.54), consistent with edema fluctuating independently of tumor growth.

# Pre-registration: is post-treatment GBM change bunched beyond steady growth?

Written 2026-10-13, before the bunching statistic below was computed on any real
data. At the time of writing, the interval-versus-change correlations in
`outputs/surveillance/mu_panel.json` had been seen; the bunching statistic,
power analysis, and every number this test produces had not.

## Question

After treatment, does glioblastoma change steadily (as reaction-diffusion
models assume) or in bursts concentrated in a few intervals ("switching")?
This is kill criterion 1 of the switching hypothesis.

## Data

- MU-Glioma-Post, the 115 GBM patients of the shared split manifest.
- Duplicates removed: PatientID_0062 (same patient as 0060) and PatientID_0132
  (same as 0113) are dropped. Timepoint 5 of PatientID_0018 and PatientID_0032
  (byte-identical tumor masks) is dropped from both.
- Scan day for timepoint *k*: the clinical spreadsheet column "Number of days
  from Diagnosis to *k*th MRI".

## Measurement

- **Primary:** enhancing tumor (label 3). Edema is excluded because it moves
  with steroids and antiangiogenic therapy independently of tumor cells.
- **Secondary:** tumor core (labels 1+3) and whole abnormality (1+2+3).
- Each volume is converted to an equivalent-sphere radius
  r = (3V / 4π)^(1/3) in mm.

## Trajectories

A patient's scans are split into separate segments at every interval of 28
days or less. Such intervals bracket surgery or other interventions (see
`outputs/surveillance/`), which change the tumor for non-biological reasons.
Only segments with at least 3 scans are analysed.

## Null hypothesis H0: steady growth plus measurement noise

r(t) = a + v·t + ε, with ε ~ N(0, σ²) independent across scans. Each segment
has its own a and v (any sign); σ is pooled across segments from the OLS
residuals. Simulated radii are clipped at zero, as observed radii are.

Rationale: a Fisher-KPP tumor converges to a travelling wave whose radius
grows at constant velocity, which is the standard "velocity of radial
expansion" used clinically. Fitting σ from residuals under H0 absorbs any
switching structure into the noise, which makes the test conservative.

## Primary statistic

Bunching per segment: B = max_k |Δr_k| / Σ_k |Δr_k|, the share of the
segment's total radius change that falls in its single largest interval.
The cohort statistic is the mean of B over segments with nonzero total change.

One-sided p-value from 5000 parametric-bootstrap cohorts simulated under H0 at
the observed scan days.

## Is the test able to see switching? (power and calibration)

- **Power:** 200 synthetic cohorts at the observed scan days and σ, where
  each segment is flat until a switch time τ ~ Uniform(first, last scan) and
  then changes linearly, with the slope set so the net change over the segment
  matches its H0 fit. Each synthetic cohort is tested exactly like the real one
  (refit, 1000 bootstrap draws).
- **Type I:** 200 synthetic cohorts drawn from H0 itself, tested the same way.

## Decision rule (fixed in advance)

| Outcome | Condition | Consequence |
|---|---|---|
| Supported | p < 0.05, power ≥ 0.80, type-I rate ≤ 0.10 | Passes kill criterion 1; proceed to criterion 2 (treatment) |
| Killed | p ≥ 0.05 and power ≥ 0.80 | Steady growth plus noise explains MU; drop the hypothesis for MU |
| Inconclusive | power < 0.80 | MU cannot discriminate; decision deferred to LUMIERE |

## Reported but not decisive

Tumor core and whole-abnormality volumes; excluding patients with multiple
surgeries; excluding segments containing a zero enhancing volume; dropping
each patient's first interval (chemoradiation and pseudoprogression period);
null with σ halved; direction (growth or shrinkage) of each segment's largest
interval.

## Outcome of the test above

Run 2026-10-13 (`outputs/surveillance/switching_test_mu.json`). Observed
bunching 0.651 against a null mean of 0.611 (p = 0.006; type-I rate 0.045),
significant in every secondary analysis. Power against the switching
alternative was 0.115, so the pre-registered verdict is **inconclusive**.
Steady growth with Gaussian noise is rejected, but the test cannot say whether
the excess bunching comes from abrupt regime changes or from heavy-tailed
measurement error (occasional badly segmented scans).

## Amendment 1: persistence test (written before it was computed)

Written 2026-10-13 after the outcome above, before any statistic in this
amendment was computed on real data.

### Idea

Two explanations can each produce the observed bunching, and they make
opposite predictions about consecutive intervals:

- **Regime change** (switching): once a tumor's trajectory changes course it
  keeps going, so consecutive radius changes tend to share a direction.
- **Measurement outliers**: a badly segmented scan pushes the radius away and
  the next scan snaps back, so consecutive changes tend to cancel.

Each rival is calibrated to reproduce the observed bunching and residual noise
level, then judged on a statistic it was not calibrated to.

### Statistic

Persistence P: the Pearson correlation of consecutive radius changes
(Δr_k, Δr_{k+1}), pooled over all consecutive interval pairs within segments.
Secondary: sign concordance, the mean of sign(Δr_k)·sign(Δr_{k+1}).

Same data, segments and primary measure (enhancing radius) as above.

### Rival models

Both keep each segment's H0 fit (start radius and net change) and the observed
scan days, and both use common random numbers during calibration.

- **Outlier rival:** steady growth plus contaminated noise. Each scan's error
  is N(0, s²) with probability 1 − π and N(0, (4s)²) with probability π.
  Grid π ∈ {0, 0.02, …, 0.50}.
- **Regime-change rival:** each segment is flat until τ ~ Uniform(first scan,
  last scan), then linear, with total change equal to its H0 net change plus
  an extra amplitude A in the same direction, and Gaussian noise N(0, s²).
  Grid A ∈ 33 evenly spaced values from 0 to 4σ̂.

Calibration: for each grid value, s is set by bisection so that the mean
refitted pooled σ over 300 simulated cohorts equals the observed σ̂. The grid
value whose mean simulated bunching is closest to the observed bunching is
selected. A rival that cannot come within 0.005 of the observed bunching
anywhere on its grid is reported as unable to reproduce the data.

Each calibrated rival and H0 then generate 2000 cohorts, giving predictive
distributions of P.

### Decision rule (fixed in advance)

1. **Discriminability:** the AUC separating the two rivals' P distributions
   must be at least 0.80; otherwise the verdict is **inconclusive**.
2. A rival is **compatible** if the observed P lies within its central 95%
   predictive interval.
3. Verdict:
   - regime change compatible and outliers not: **switching favored**
     (passes this test; criterion 2, treatment timing, comes next);
   - outliers compatible and regime change not: **outliers favored**
     (the switching hypothesis is dropped for MU);
   - both compatible: **undetermined**;
   - neither compatible: **neither rival explains MU**.

Reported but not decisive: sign concordance; the same analysis on tumor-core
radius; the one-sided H0 p-value for P in each direction.

## Outcome of Amendment 1

Run 2026-10-13 (`outputs/surveillance/persistence_test_mu.json`; protocol
sha256 recorded in the output). Enhancing radius, 73 segments, 137 consecutive
interval pairs.

- Observed persistence −0.217, indistinguishable from steady growth (−0.238;
  one-sided p = 0.39).
- Rivals were separable (AUC 0.957). Regime change predicted −0.037
  (95% interval −0.189 to 0.121) and is **incompatible**; outliers predicted
  −0.248 (−0.418 to −0.054) and are compatible.
- **Pre-registered verdict: outliers favored. The switching hypothesis is
  dropped for MU.** Tumor core gives the same verdict.
- Caveat stated in the protocol: the outlier rival could not reproduce the
  observed bunching (0.617 versus 0.651), so it does not explain all of the
  data either.

Post-hoc, not pre-registered (`outputs/surveillance/persistence_posthoc_mu.json`):
the observed persistence sits at the 1st percentile of the regime-change
predictive distribution; heavier-tailed outliers (×6, ×8, ×12) still cannot
reproduce the bunching; a single permanent step jump reproduces the bunching
but not the persistence (0.4th percentile). The data combine bunched change
with consecutive changes that cancel, consistent with large excursions that
revert rather than with sustained regime change.

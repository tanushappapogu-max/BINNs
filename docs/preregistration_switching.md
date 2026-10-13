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

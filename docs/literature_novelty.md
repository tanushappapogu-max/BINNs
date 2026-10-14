# Literature check: what is new and what is not

Searched 2026-10-14 (Consensus, PubMed, web) for prior work on the claims in
this project. "Not found" means not found in these searches, not proven absent.

## Claims and the closest prior work

### 1. A forecastability ceiling for routine post-treatment glioma follow-up

**Not found.** Prior predictability work is synthetic or uses dense research
imaging:

- LaMonica et al., *Cancers* 2025: limits of predictability of reaction-diffusion
  models on **synthetic** rat-brain tumors with regular sampling; predictions
  stayed accurate (Dice and CCC > 0.9) as image quality degraded.
- Hormuth et al., *Scientific Reports* 2021: patient-calibrated models predicted
  enhancing volume at 3 months with median error −2.5% (IQR 10%) in **9
  patients with weekly research mpMRI** during chemoradiation.
- Farhat et al., *Neuro-Oncology* 2022 abstract: weekly mpMRI, 11 patients,
  volume error < 8.6% at end of chemoradiation, < 20% at 3 months.
- Jarrett et al., *Radiation Oncology* 2020: rat C6 gliomas, dense quantitative
  MRI, < 12% volume error.
- Jung and Shin, arXiv 2609.12435 (2026): learned forecaster on MU-Glioma-Post
  ties latest-observation persistence (Dice 0.607 vs 0.607); no analysis of why.

**Scope this forces on our claim:** the ceiling is a property of *routine*
clinical follow-up (sparse, irregular, standard sequences, automated
segmentation), not of tumors in general. Dense, protocolized quantitative
imaging appears to support forecasting. This is consistent with Pash et al.
2025 (more frequent imaging, more information) and is a useful contrast for
the report.

### 2. Patient-specific growth rates are not measurable from routine follow-up

**Partly new; must be worded carefully.**

- Growth-rate heterogeneity is documented in **untreated** tumors: Swanson et
  al. 2008 (glioblastoma velocities "vary widely"); Pallud et al.,
  *Neuro-Oncology* 2013 (407 low-grade gliomas, diametric velocity
  5.8 ± 6.3 mm/yr, prognostic); Rayfield et al., *JCO CCI* 2018 (48 GBMs,
  growth-kinetics clusters predict survival).
- Pallud's spread, ±6.3 mm/yr in diameter, is about ±3 mm/yr in radius,
  **below our 80%-power detection limit of 10 mm/yr.** So our data cannot rule
  out heterogeneity of the size reported for untreated low-grade gliomas.
- Correct wording: *between-patient differences in steady post-treatment growth
  rate larger than about 10 mm/yr are ruled out in both cohorts; smaller
  differences, of the size reported for untreated gliomas, are not resolvable
  from routine follow-up.* "Not measurable", not "not present".

### 3. Measurement repeatability (what the reverting component might be)

- Khalaf et al., *AJNR* 2021: **BraTumIA** (predecessor of DeepBraTumIA, which
  produced the LUMIERE masks), 20 recurrent GBM patients scanned about 2 days
  apart. Repeatability coefficient 31–46% for enhancing tumor and edema volume;
  enhancing volume larger on the later scan, attributed to incomplete
  gadolinium washout.
- Chang et al., *Neuro-Oncology* 2019 (AutoRANO): 54 GBM patients with double
  baseline scans; ICC 0.991 for contrast-enhancing volume.
- Conversion (exploratory, computed 2026-10-14). A repeatability coefficient
  RC = 2.77 × within-subject SD, so RC 31–46% means a single-scan volume
  coefficient of variation of 11–17%, which is 3.7–5.5% in equivalent radius
  (radius scales as the cube root of volume). Two ways to apply it:
  - proportional to each tumor's size, at our median enhancing radii
    (LUMIERE 8.3 mm, MU 11.3 mm): per-scan radius SD 0.31–0.63 mm;
  - as a fixed boundary error at Khalaf's mean enhancing volume (about 24 mL,
    radius about 18 mm): 0.67–0.99 mm.
  Our fitted reverting SD is 2.50 mm (LUMIERE) and 2.43 mm (MU), so
  **same-week segmentation variability explains roughly 2–17% of the
  reverting variance.** Most change that later reverts is not short-term
  segmentation noise: it is transient real change (pseudoprogression or
  pseudoresponse, steroid and contrast effects) or between-visit acquisition
  differences over months. QIN double-baseline data would sharpen this but are
  under NIH controlled access.

### 4. State-space / Kalman methods for tumor growth

- Kostelich et al., *Biology Direct* 2011: ensemble Kalman filter on synthetic
  glioblastoma images.
- Collin et al., *SIAM J. Appl. Math.* 2021: joint state-parameter estimation,
  synthetic.
- **Not found:** a state-space decomposition of real multi-cohort glioma
  trajectories into reverting change, interval-independent change,
  time-accumulating change and patient-specific drift.

## Bottom line for the report

Defensible as new (to the best of these searches): a cohort-level,
pre-registered, two-cohort quantification of how much routine post-treatment
glioma follow-up can be forecast, with a decomposition of observed change and
an explicit detection limit for patient-specific rates. Must cite and contrast:
dense-imaging forecasting successes (Hormuth, Farhat, Jarrett), untreated-rate
heterogeneity (Swanson, Pallud, Rayfield), and repeatability studies (Khalaf,
Chang).

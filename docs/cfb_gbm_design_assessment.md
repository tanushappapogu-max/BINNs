# CFB-GBM as a third cohort: design assessment (no outcomes examined)

Assessed 2026-10-14 using only data-integrity checks, segmentation provenance
and scan intervals. No change statistic, correlation or model fit was computed
on CFB-GBM tumor sizes.

## What is available without the 208 GB image package

- `CFB-GBM_rano_criteria_v04_20260909.tsv`: GTV size (cm³) at t0, t1, t2 for
  165 patients; 107 have all three. Sizes match the PyRadiomics GTV
  VoxelVolume exactly (median ratio 1.000; 5–95% range 0.996–1.004).
- `CFB-GBM_mri_availability_v02_20260129.tsv`: weeks from t0 for each
  timepoint.

## Why it cannot serve as a confirmatory cohort for this project

1. **Measure.** The GTV is the radiotherapy target, which after surgery
   includes the resection cavity, not enhancing tumor alone.
2. **Segmentation provenance changes between timepoints.** At t0, 191 GTVs
   are drawn by radiation oncologists and 70 are model-generated; every t1
   and t2 GTV is model-generated. The dataset paper itself reports a
   boundary-character mismatch between the two. Only 14 of the 107 complete
   patients have a model-generated t0, so the t0→t1 change mixes tumor change
   with a change of method for almost everyone.
3. **Too little temporal structure.** Three scans per patient, the first
   interval spanning chemoradiation. Using only the consistent t1→t2
   interval leaves one transition per patient, with gaps concentrated between
   60 and 98 days (median 84; range 14–210).
4. **Power.** Simulating the pre-specified clock test (Spearman correlation of
   squared radius change with interval, one-sided α = 0.05) at the observed
   t1→t2 gaps with the LUMIERE-fitted variance components gives power 0.30;
   with doubled or quadrupled reverting variance, 0.18 and 0.12.

## Decision

CFB-GBM is not used as a confirmatory cohort. It is reported as considered
and excluded for the design reasons above, decided before any outcome was
examined. It remains a candidate for a future analysis that re-segments all
timepoints with one model from the released images.

# Results

## Dataset and protocol

We use MU-Glioma-Post: 115 patients with post-surgical glioblastoma, each with
multiparametric MRI and expert tumor segmentations at two to six timepoints. The
cohort is split at the patient level with no leakage: 81 training patients (208
consecutive-scan transitions), 17 model-selection patients (39 transitions), and
17 final-test patients (35 transitions); training, model-selection, and test
patient sets are disjoint.

The unit of forecasting is a *transition*: a consecutive scan pair with source
day *S* and target day *T*. The model receives the source segmentation, binary
treatment-exposure flags (radiation, systemic chemotherapy, antiangiogenic,
device), and the elapsed interval *T*−*S*, and predicts the target tumor mask.
Accuracy is the Dice overlap between the predicted and observed target mask
within the brain, thresholded at a density of 0.1. *Skill* is the model Dice
minus the persistence Dice, where persistence copies the source mask forward.
Because transitions from a single patient are correlated, significance is
assessed with a Wilcoxon signed-rank test on per-patient mean skill
(patient-clustered).

The model is physics-informed: a 3D CNN reads the source scan (with treatment
and horizon) and predicts three reaction–diffusion parameters — diffusion *D*,
proliferation *ρ*, and treatment-associated death *κ* — and a differentiable
Fisher–KPP solver evolves the source density forward. The output is initialized
at persistence by construction (zero parameters reproduce the source exactly),
so the prediction departs from the baseline only through learned dynamics.
Unless stated otherwise, results are at 2 mm isotropic resolution.

## 1. Persistence is a strong, rarely-reported baseline

Persistence (no-change) achieves whole-tumor Dice of approximately 0.49–0.52.
Because most post-surgical tumor volume is stable between consecutive scans, this
baseline is difficult to exceed and dominates the overlap metric: an apparent
Dice near 0.55 corresponds to skill of only about +0.02 over persistence. We
therefore report skill over persistence throughout, and separately isolate the
newly-appearing tumor region (Section 3), where persistence scores zero by
construction.

## 2. The forecaster beats persistence modestly, with a significant held-out gain

The forecaster exceeds persistence by a small but reproducible margin. Across
five random seeds (4 mm), it beats persistence on 27.6 ± 0.5 of 39
model-selection transitions (skill +0.0256 ± 0.0006; patient-clustered Wilcoxon
*p* < 0.005 in every seed). On the untouched 17-patient final-test set (35
transitions), a single evaluation of the selected configuration yields skill
+0.0129 (patient-clustered Wilcoxon *p* = 0.026), beating persistence on 23/35
transitions and 12/17 patients. The ≈ +0.012 drop from model-selection to test
is consistent with epoch selection on the model-selection split.

## 3. Growth-region evaluation isolates genuine forecasting

Whole-tumor Dice is dominated by unchanged tissue that persistence reproduces
for free. We therefore additionally score the newly-appearing tumor region
(voxels that are tumor at the target but not the source), where persistence
scores 0.000 by construction whenever growth occurs. The forecaster attains
growth-region Dice of 0.26 on model selection, and this transfers almost intact
to held-out patients (0.260 → 0.256 on test, a 1.5% relative drop). The
newly-appearing-tumor prediction is thus a learned capability, not a fit to the
selection split.

## 4. Accuracy is information-limited, not architecture-limited

Three independent attempts to raise accuracy converge at the same ceiling (4 mm,
3 seeds):

- **Physics vs. plain U-Net.** The physics model attains skill +0.026; an
  equally-sized residual 3D U-Net attains −0.008 (worse than persistence) with
  roughly ten times the seed-to-seed variance. On 208 examples the U-Net
  overfits, whereas the three-parameter PDE cannot, acting as a strong physical
  regularizer.
- **Adding MRI to the parameter network.** Supplying all four source modalities
  leaves accuracy unchanged (+0.026 → +0.026).
- **Spatially-varying diffusivity.** Replacing the scalar *D* with an
  MRI-conditioned voxelwise diffusivity field leaves accuracy unchanged
  (+0.026).

Additional imaging and model capacity do not improve accuracy: the future tumor
configuration is not determined by the present scan on this cohort. The plain
U-Net does benefit from MRI (−0.008 → +0.020), confirming the imaging carries
signal that the scalar-parameter bottleneck cannot exploit — but not enough to
exceed the physics model.

## 5. Treatment flags and cavity boundaries do not help

Zeroing all treatment-exposure channels leaves every metric unchanged (skill
+0.0256 → +0.0252). The binary flags carry no usable signal: 164 of 208
transitions are treated, radiation appears in only 17, and the flags encode
neither dose nor timing relative to the interval.

Enforcing a no-flux boundary at the source resection cavity degrades skill
(+0.026 → +0.016). The cavity collapses over follow-up and recurrence arises at
its margin, so freezing the baseline cavity as forbidden blocks valid
predictions.

## 6. Elapsed time is anti-informative, and enforcing physical time-integration is harmful

In this cohort the elapsed scan interval is weakly *anti*-correlated with the
observed volume change (Pearson *r* = −0.11; on the log interval, *r* = −0.285).
Longer intervals accompany smaller changes — the signature of length-biased
surveillance, in which stable disease is imaged less frequently.

This inverts the assumption underlying reaction–diffusion forecasting, that
change accumulates with elapsed time. We compared two configurations that are
identical except for whether the PDE is integrated over the true interval (2 mm):

| configuration    | skill   | beats persistence | corr(predicted change, interval) |
|------------------|---------|-------------------|----------------------------------|
| time-integrated  | +0.0140 | 29/39             | +0.154                           |
| time-blind       | +0.0212 | 30/39             | −0.290                           |
| observed data    | —       | —                 | +0.015                           |

The time-integrated model, forced to obey the PDE, predicts more change over
longer intervals (+0.154). The time-blind model, given the interval as a free
feature, instead reproduces the cohort's inverse relationship (−0.290).
Enforcing the physically-correct time dependence lowers skill (+0.021 → +0.014).
The biologically-faithful model is penalized precisely because the observed
intervals do not behave like elapsed biological time.

## 7. The skill difference is not a between-interval artifact

We tested whether the time-blind advantage arises from the interval distribution
by stratifying evaluation into interval tertiles (≤ 56, 56–91, > 91 days),
within which the interval is approximately constant. The time-blind advantage
over the time-integrated model is unchanged by stratification (+0.0072 pooled
vs. +0.0072 stratified; skill is flat across strata for both models). The
advantage therefore reflects a per-case difference — the time-integrated model's
forced interval scaling produces worse predictions even at fixed interval —
rather than exploitation of the interval distribution. We attribute the effect
to a harmful inductive bias (imposing elapsed-time accumulation on length-biased
data), not to metric-level gaming of the interval.

## 8. Standard inverse-intensity correction is infeasible on this cohort

Inverse-intensity weighting, the standard correction for informative observation
times, requires the observation intensity to be predictable from measured
covariates. Predicting the interval from source covariates (tumor sub-volumes,
treatment flags, time since surgery, previous interval) yields cross-validated
*R²* ≈ 0 (ridge regression: −0.03; nonparametric *k*-nearest-neighbors: −0.07 to
+0.03). The scan schedule is driven by unobserved clinical state — the
latent-driven regime in which inverse-intensity weighting is known to fail.
Correction on this dataset would require covariates it does not contain
(symptoms, performance status, indication for scan). We therefore bound the
residual bias by design, via the interval-stratified analysis of Section 7,
rather than by weighting.

## Limitations

- **Single cohort.** The length-bias phenomenon and its modeling consequence
  require replication on an independent longitudinal dataset (e.g., LUMIERE,
  BraTS-longitudinal) before being claimed as general.
- **Test scope.** The final-test evaluation covers one selected configuration;
  the two-configuration time comparison is reported on model selection and at
  2 mm resolution.
- **Effect size.** Absolute skill over persistence is small; the contribution is
  the characterization of *why* accuracy is bounded and how observation timing
  distorts evaluation, not a high-accuracy forecaster.
- **Sample size.** The held-out set is 17 patients / 35 transitions; the
  patient-clustered test mitigates but does not remove the resulting uncertainty.

# Run sheet — external forecaster test (Colab)

Everything below was smoke-tested on 2026-10-14: the notebook's own commands
ran end to end against a fake Drive folder (prepare → 3-seed training →
evaluation on MU final test and a 6-pair LUMIERE subset → pre-registered
verdicts), every notebook cell compiles in IPython, training resumes after a
disconnect, and the full LUMIERE download from an empty folder (all 599
sessions and masks) reproduces the deduplicated table exactly and yields a
byte-identical 313-pair external index.

The one step left is yours by choice: commit and push (step 1). Nothing is run
on real data until then.

## 1. Commit and push (required: Colab clones from GitHub)

The pre-registration must be committed **before** the run, so its timestamp
precedes the results.

```bash
cd "/Users/tanush_appapogu/STS Project"
git add docs/preregistration_external_forecaster.md scripts/external_forecaster.py \
  scripts/judge_external_forecaster.py notebooks/colab_external_forecaster.ipynb \
  scripts/prepare_lumiere_volumes.py scripts/prepare_cfb_gbm.py \
  src/gbm_pinn/ceiling_cli.py tests/test_ceiling_cli.py pyproject.toml \
  docs/literature_novelty.md docs/cfb_gbm_design_assessment.md \
  docs/test_retest_assessment.md docs/preregistration_ceiling_lumiere.md \
  docs/RUN_TOMORROW.md
git commit -m "Pre-register external forecaster test; add Colab notebook, ceiling CLI, data prep"
git push
```

## 2. Check Drive has the MU data the old notebook used

`MyDrive/STS_Project_Data/` must contain:

- `data/derived/mu_glioma_post/shared_training_transitions.json`,
  `shared_model_selection_transitions.json`, `shared_final_test_transitions.json`,
  `shared_split_manifest.json`
- `data/raw/mu_glioma_post/nifti/...` (the MU masks)

These are the same files `colab_pinn_cohort.ipynb` used. If the final-test
index is missing, `prepare` stops with a message saying which file.

## 3. Run on Colab

1. colab.research.google.com → File → Open notebook → GitHub →
   `tanushappapogu-max/BINNs` → `notebooks/colab_external_forecaster.ipynb`.
2. Runtime → Change runtime type → **GPU** (T4 is fine).
3. Runtime → Run all. Approve the Drive mount.

Expected time:

| Step | Time |
|---|---|
| LUMIERE masks to Drive (first run only) | ~40 min |
| Training, 3 seeds × 25 epochs at 2 mm | roughly 1.5–3 h on a T4 |
| Evaluation + verdicts | ~10 min |

If Colab disconnects, just **Run all** again: finished seeds are skipped and the
LUMIERE download is skipped once it is on Drive.

## 4. What to send back

The last cell prints `external_verdicts.json`. Paste that output into the
chat, or download `MyDrive/STS_Project_Data/outputs/external_forecaster/` into
the project's `outputs/external_forecaster/`. The verdicts (E1 internal
replication, E2 external transfer, E3 below the size-history ceiling) are
applied mechanically from the pre-registration; nothing needs interpreting
on your side.

## Status of the rest of the middle option

- **MU + LUMIERE:** done (ceiling, pre-registered confirmation).
- **CFB-GBM:** data gathered (`scripts/prepare_cfb_gbm.py`, 437 scans, 165
  patients) but excluded as a confirmatory cohort for design reasons recorded
  before any outcome was examined (`docs/cfb_gbm_design_assessment.md`).
- **Test–retest:** no open GBM test–retest dataset (QIN controlled access,
  RIDER no longer public); published repeatability used instead
  (`docs/test_retest_assessment.md`).
- **Tool:** `gbm-ceiling scans.csv` reports any cohort's forecastability
  ceiling (`src/gbm_pinn/ceiling_cli.py`).

# Test–retest data: why the noise floor comes from published repeatability

Checked 2026-10-14.

The middle-path plan included measuring segmentation repeatability directly on
glioblastoma patients scanned twice a few days apart. No such dataset is openly
downloadable:

- **QIN-GBM-Treatment-Response** (54 patients, double baseline 2–5 days apart):
  licensed under the "NIH Controlled Data Access Policy" on its TCIA page, so it
  needs an approved access request.
- **RIDER Neuro MRI** (19 recurrent GBM patients, repeat scans about 2 days
  apart): not present in TCIA's public collection list
  (`nbia-api/services/v1/getCollectionValues`, queried 2026-10-14), and its
  series query returns nothing.
- **UCSF-ALPTDG** requires a data use agreement (and is two follow-up
  timepoints, not test–retest).

The repeatability of the segmentation family used for LUMIERE is instead taken
from the literature (`docs/literature_novelty.md`, section 3): Khalaf et al.,
AJNR 2021, BraTumIA on recurrent GBM scanned about 2 days apart, repeatability
coefficient 31–46% for enhancing volume; Chang et al., Neuro-Oncology 2019,
AutoRANO on 54 double-baseline GBM patients, ICC 0.991. Converted to
equivalent radius, same-week segmentation variability explains roughly 2–17%
of the reverting variance fitted on MU and LUMIERE.

An access request for QIN-GBM is the natural next step after the submission.

# Pre-registration: cross-device evaluation on DOD-H and HMC

Tag: `prereg-v1` · Written 26 November 2026 · Author: Bora Bayrakci

This file fixes, before any data is seen, how the frozen Goldilocks models will be tested on two
datasets nobody on the project has opened: DOD-H and a 25-recording subset of HMC. Neither dataset
had been downloaded when this file was tagged. The hashes of every frozen model and evaluation file
are in `preregistration_hashes.json`.

Anything done after the tag that isn't described here gets labelled post hoc, in the results file
and in the paper.

## 1. What we already know (exploratory, Sleep-EDF)

The 20 Sleep-EDF people were looked at during the v2 evaluation, so everything in this section is
exploratory. It's here because it decides what the confirmatory test should ask. Numbers come from
`research_outputs/day1/` and `results/ablation_shift/ablation_shift_metrics.json`.

**The recordings are very different.** BOAS test nights run a median 7.7 hours with 14% wake.
Sleep-EDF recordings run a median 23.0 hours with 70% wake.

**The CNN family doesn't recognise Sleep-EDF wake at all.** Wake recall is 0.87 for the plain CNN on
BOAS and 0.20 on Sleep-EDF wake that falls inside the night, 0.12 outside it. So it isn't only
daytime wake. The plain CNN labels 67% of wake, 77% of REM and 85% of N3 as N2. Collapsing onto one
class like that usually means the input looks unlike anything seen in training.

**The CNN should be almost blind to amplitude.** Every convolution is followed by instance
normalisation over time, which removes any per-epoch scale or offset. The per-recording robust
scaling therefore can't change much apart from what the clip at ±20 cuts off. That makes a
normalisation explanation unlikely before we even run A1.

**The v2 feature model is very sensitive to amplitude.** v2's log-power features shift with gain, so
we can rescale them exactly without touching raw data. Multiplying the Sleep-EDF signal by 2 takes v2
from kappa 0.579 and a 235.9-minute sleep-time error to kappa 0.674 and 110.4 minutes, about where
YASA sits (0.669 and 110.3). The same factor on BOAS makes things worse (kappa 0.661 to 0.281), as
it should.

**A label-free gain correction recovers most of that.** One gain per dataset, matching the median
across recordings of the q-th percentile of epoch log standard deviation to the BOAS fitting nights:

| Rule | Gain on Sleep-EDF | Sleep-EDF kappa / sleep-time error | BOAS test kappa | BOAS probability-validation kappa |
| --- | --- | --- | --- | --- |
| None (shipped) | 1.00 | 0.579 / 235.9 min | 0.661 | 0.567 |
| Dataset q50 | 1.02 | 0.587 / 226.7 min | 0.654 | 0.564 |
| Dataset q75 | 1.20 | 0.636 / 174.4 min | 0.664 | 0.561 |
| Dataset q90 | 1.70 | 0.681 / 110.7 min | 0.648 | 0.511 |

We also tried one gain per recording (q50, q75, q90). It helped Sleep-EDF a bit more but cost more on
BOAS, especially on the probability-validation people (down to 0.443). All six rules are in the results file. We
tried these rules on data we'd already seen, so the q75 and q90 numbers above are the reason for
the test below. They are not evidence on their own.

## 2. Hypotheses

**H1, amplitude.** Much of v2's drop on a new device is a gain mismatch, and a dataset-level gain
calibration fixed in advance recovers part of it.

- Primary rule: dataset-level q75, chosen because it costs no more than 0.01 kappa on either BOAS
  subset. Secondary rule: dataset-level q90, which does better on Sleep-EDF but costs 0.056 on the
  probability-validation people.
- Supported on a dataset if v2 + q75 beats uncalibrated v2 by at least 0.03 in person-averaged
  macro-F1, with the 95% paired bootstrap interval of the difference above 0.
- Not supported if the difference interval includes 0. Contradicted if it's entirely below 0.

**H2, CNN wake.** The CNN family's collapse is a failure to recognise wake on PSG frontal
derivations, not something specific to Sleep-EDF's long recordings.

- DOD-H and HMC are night recordings, so they have far less wake to get wrong. That makes wake recall
  the deciding number, not macro-F1.
- Supported if the plain CNN's mean wake recall on a dataset is below 0.50 while v2's is at least 0.60.
- Contradicted if the plain CNN's wake recall is at least 0.70. That would mean the Sleep-EDF
  collapse came from something about Sleep-EDF itself (recording length, daytime behaviour or a
  preprocessing difference), and A6 becomes the first thing to chase.
- Anything in between is reported as mixed, with no winner.

**Secondary, the family gap.** Gap = v2 macro-F1 minus plain CNN macro-F1, person-averaged, with a
95% bootstrap interval. Below 0.05 with an interval that includes 0: the gap closes. Above 0.15 with
an interval that excludes 0.05: the gap persists. Otherwise mixed.

**Recorded before A1 runs.** The architecture argument above predicts that none of the A1
normalisation variants moves the plain CNN's Sleep-EDF macro-F1 by more than 0.03. If one does, the
instance-normalisation argument is wrong and the paper says so.

## 3. Data

**DOD-H** (Guillot et al., IEEE TNSRE 2020). 25 healthy adults, one night each, scored by 5 sleep
technologists.

- Reference: the consensus hypnogram as defined in the DOD paper. Each scorer is also kept, for a
  human ceiling.
- Channels: FP1-F3 and FP2-F4, used as the two channels for the channel-shared models. That mirrors
  the two BOAS headband channels. v2 and the CNN average their channel probabilities, as they do
  on BOAS.
- Windows: the whole recording is primary. The reference night window ±30 min is secondary,
  defined exactly as `yasa_evaluate.windows`.

**HMC** (Alvarez-Estevez and Rijsman, PhysioNet 2022, CC BY 4.0). The 25 recordings listed in
`data/external_hmc/selection.json`, drawn with seed 20261126 from the public file listing only.

- Reference: the technologists' AASM 2.4 scoring in `SNxxx_sleepscoring.edf`.
- Channel: F4-M1, single channel.
- Windows: the whole recording is primary. Lights-off to lights-on from the scoring file is secondary.

**Processing, both datasets.** Read the native rate from the EDF header. Resample with
`scipy.signal.resample_poly` to 256 Hz for v2 features and to 64 Hz for the CNN. Use 30-second epochs
on the scoring grid. Use the same quality rule as v2 (non-finite or flat epochs are invalid).
Movement and unscored epochs are excluded. Map stage 3 and 4 to N3 where needed. Units are
microvolts.

**Opening order.** Download, hash, extract features with labels set aside, run every model, then
open the labels and score once.

## 4. Models

All frozen, hashes in `preregistration_hashes.json`. Nothing gets refitted or reselected on DOD-H or HMC.

1. v2 feature model, as shipped.
2. v2 + dataset-level q75 gain (primary for H1).
3. v2 + dataset-level q90 gain (secondary for H1).
4. YASA 0.7.0 off the shelf, on the same channel.
5. Plain CNN (`results/cnn/`), the model used for H2.
6. The other six CNN variants and the ensemble of seven, reported in full but not tested.

A CNN retrained with recording-independent normalisation is not part of this file. If it's built,
it goes into a `prereg-v2` addendum that's committed before it's evaluated.

## 5. Metrics and statistics

- Primary: person-averaged macro-F1. Also reported: pooled kappa, wake F1, wake recall,
  per-stage F1, absolute and signed sleep-time error.
- Uncertainty: paired person-level bootstrap, 10,000 resamples, as in `confidence_intervals.py`.
- Multiplicity: Holm correction across the four confirmatory tests (H1 and H2 on each dataset).
- Every table states the number of people and whether a result is confirmatory or exploratory.

## 6. The two headlines, written before the results

Only one of these gets used, and the data decide which. A mix chosen after seeing the numbers isn't
allowed.

**If H1 and H2 are both supported:** "A frozen feature model's cross-device loss was mostly a gain
mismatch that one label-free number per device fixes. The convolutional models that scored best on
the training device failed in a different way: they stopped recognising wake on PSG electrodes."

**If they aren't:** "Selecting by in-domain score picked the models that failed worst on new devices.
Neither amplitude calibration nor shorter recordings explained the drop, so in-domain ranking was not
a safe guide to cross-device performance here."

Partial outcomes are reported hypothesis by hypothesis, with the wording of the closer headline
changed to match.

## 7. The 236-minute error (exploratory, separate from the above)

A sleep-window detector is developed on the Sleep-EDF SC subjects numbered 20 and above, which have
the same long daytime recordings. It's tested once on the usual 20 people (subjects 00 to 19).
Those 20 were already looked at, so this stays exploratory however well it does.

## 8. Deviations

Any change to this plan after the tag gets logged here with its date and reason, and the affected
results are labelled post hoc.

- 26 November 2026, before any DOD-H file was opened: Dreem's S3 bucket for DOD-H no longer
  exists. DOD-H is now on Zenodo (record 15900394, DOI 10.5281/zenodo.15900394, v1 of July 2025)
  as one 21.9 GB zip, too big for the laptop's free space. `scripts/fetch_dodh_compact.py` reads
  the zip one record at a time, hashes each .h5, and keeps only the hypnogram and the FP1-F3 and
  FP2-F4 signals named in section 3. This changes where the data comes from and what is stored,
  not what is analysed. The scorer files for the consensus come from the
  `dreem-learning-evaluation` repository, as in the DOD paper.
- 26 November 2026, amendment v1.1, written before any DOD-H or HMC signal or label was opened
  (the DOD-H records sit on the laptop as compact files; none has been read):
  a bootstrap on the already-inspected Sleep-EDF people (`analysis/gain_uncertainty.py`) showed
  that the q75 gain clearly improves kappa (+0.057, 95% interval 0.019 to 0.100) and the
  sleep-time error (-61.5 min, -98.3 to -32.7), but not macro-F1 (+0.027, -0.007 to 0.058).
  Macro-F1 alone was a poor choice for a hypothesis about amplitude, which mostly shifts the
  wake/sleep boundary and so the totals. H1 is therefore assessed two ways, and both are
  reported:
  - **H1 as originally written** (macro-F1, at least 0.03, interval above 0), unchanged.
  - **H1b, added:** v2 + q75 lowers the absolute sleep-time error against uncalibrated v2, with
    the paired 95% interval entirely below 0, and raises kappa with the interval above 0.
  Holm correction now runs over six tests (H1, H1b and H2 on each dataset). The rules, gains,
  models and every other threshold are unchanged.

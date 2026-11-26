# Day 1 (26 November 2026)

Everything in this folder is exploratory. The Sleep-EDF people were looked at during the v2
evaluation, so none of it is a fresh test. It's here to decide what the pre-registered test on
DOD-H and HMC should ask (see `PREREGISTRATION.md`).

## What was checked

| Check | Result |
| --- | --- |
| `reproduce_v2.py` on a clean clone | 7 tests pass; kappa 0.6612737502932162 and sleep-time error 31.8 min, as published |
| `pytest` | 25 tests pass |
| `verify_claims.py` | 102 figures checked, none unsupported; 12 stale hashes, now rebuilt |
| CNN environment | `requirements.txt` plus `requirements_live.txt` install and import fine (jax 0.12.0.1) |
| CNN epoch caches | Not in the repo (812 MB on the laptop). Staged from the laptop for A1, A2 and A6 |

## What we found

Recording composition (`recording_composition.csv`): BOAS test nights have a median of 7.7 hours
and 14% wake. Sleep-EDF recordings have a median of 23.0 hours and 70% wake.

The CNN family on Sleep-EDF (`external_confusion.csv`, `wake_recall_by_window.csv`):

- Wake recall drops from 0.87 on BOAS to 0.20 inside the Sleep-EDF night and 0.12 outside it.
- The plain CNN labels 67% of wake, 77% of REM and 85% of N3 as N2.
- The class-balanced CNN spreads wake over N3 (35%) and REM (25%) instead.

The network uses instance normalisation after every convolution, so it should be close to
scale-invariant. A normalisation explanation looks unlikely even before A1 is run.

v2 on Sleep-EDF (`results/ablation_shift/ablation_shift_metrics.json`): v2 depends strongly on
amplitude.

| Scale applied to the signal | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- |
| Sleep-EDF kappa | 0.213 | 0.430 | 0.579 | 0.662 | 0.674 |
| Sleep-EDF sleep-time error (min) | 697.1 | 403.6 | 235.9 | 131.5 | 110.4 |
| BOAS test kappa | 0.565 | 0.655 | 0.661 | 0.471 | 0.281 |

A label-free gain set once per dataset gets most of this back without using labels. The q90 rule
gets Sleep-EDF to kappa 0.681 and 110.7 min. The q75 rule gets 0.636 and 174.4 min, with no real
cost on BOAS. Every rule tried is in the results file.

## With the epoch caches (run after `prereg-v1` was tagged)

**The shipped CNN result reproduces exactly.** Re-running `cnn_external.py` on the caches gives
predictions identical to `results/cnn/cnn_external_predictions.json` (macro-F1 0.18777).

**A1 and A2 do nothing to the CNN.** That is what the pre-registration predicted from the
instance-normalisation argument. Across the plain, balanced, square-root-balanced and augmented
CNNs, none of the four normalisation windows and none of the four input scales moves macro-F1 by
more than 0.001. The plain CNN stays at macro-F1 0.188, wake F1 0.211 and an 803-minute
sleep-time error in every variant. Normalisation and amplitude are ruled out as causes of the
CNN collapse.

**A6 finds no preprocessing bug.** Both caches have 1920 samples per epoch and 30-second onset
spacing. Sleep-EDF N2 shows a normal spindle peak at 13.75 Hz. The script flags the 8–16 Hz
peak for N1, N2 and N3, but the N1 and N3 flags are artefacts: those spectra fall steadily across
the band, so the "peak" lands on its edge (8.0 or 15.75 Hz). The real difference is in wake:

| Wake, relative power | Delta | Theta | Alpha | Sigma | Beta |
| --- | --- | --- | --- | --- | --- |
| BOAS headband | 0.813 | 0.048 | 0.045 | 0.022 | 0.071 |
| Sleep-EDF Fpz-Cz | 0.843 | 0.079 | 0.025 | 0.013 | 0.038 |

Headband wake carries about twice the beta share of Sleep-EDF wake. That fits forehead muscle
activity on a dry headband, which a PSG Fpz-Cz derivation picks up much less. A CNN that learned
"wake = high-frequency power" would then read quiet PSG wake as sleep, and that's what it does.
The next section tests it.

## Testing the beta explanation (`results/ablation_shift/beta_perturbation.json`)

Each band was removed from the BOAS test nights with a zero-phase band-stop before the
shipped scaling, and the frozen CNNs were run again. Nothing was refitted.

| Band removed | Plain CNN wake recall | Balanced CNN wake recall |
| --- | --- | --- |
| None | 0.868 | 0.734 |
| Theta 4–8 Hz | 0.892 | 0.756 |
| Alpha 8–12 Hz | 0.806 | 0.703 |
| Sigma 12–16 Hz | 0.832 | 0.666 |
| Beta 16–20 Hz | 0.819 | 0.683 |
| Beta 20–24 Hz | 0.852 | 0.721 |
| Beta 24–30 Hz | 0.851 | 0.734 |
| **All of 16–30 Hz** | **0.701** | **0.631** |

No single 4 Hz slice of beta matters more than alpha or sigma. Removing the whole 16–30 Hz
range does, and the lost wake goes to N2 and REM (plain CNN: 10% and 11% of wake, up from 3.5%
each), the same direction as on Sleep-EDF. So the CNN leans on broadband high-frequency power
to call wake, which fits muscle activity more than any single rhythm.

On Sleep-EDF, recordings whose wake carries more beta are recognised better: Spearman rho
0.55 (p = 0.012) for the plain CNN and 0.72 (p < 0.001) for the balanced one, 20 recordings.

This supports the explanation, but only partly. With all of 16–30 Hz removed, BOAS wake recall
is still 0.70, far above the 0.20 on Sleep-EDF, so high-frequency power is one ingredient of
the collapse, not all of it.

## How solid is the gain calibration? (`results/ablation_shift/gain_uncertainty.json`)

Paired person-level bootstrap, 10,000 resamples, Sleep-EDF, change against uncalibrated v2:

| Rule | Gain | Kappa | Macro-F1 | Sleep-time error (min) |
| --- | --- | --- | --- | --- |
| Dataset q75 | 1.20 | +0.057 (0.019 to 0.100) | +0.027 (−0.007 to 0.058) | −61.5 (−98.3 to −32.7) |
| Dataset q90 | 1.70 | +0.102 (0.020 to 0.184) | +0.039 (−0.025 to 0.101) | −125.2 (−212.9 to −39.8) |

The calibration reliably improves kappa and the sleep-time error. Its effect on macro-F1 is
uncertain. That's why `PREREGISTRATION.md` gained H1b (sleep-time error and kappa) as a dated
amendment, before any DOD-H or HMC data was opened.

Estimated from a random half of the recordings (10), the q75 gain lands between 1.07 and 1.37
in 95% of splits, and q90 between 1.52 and 1.91. A dataset of DOD-H's size should give a
usable estimate.

## Still open

| Task | Needs |
| --- | --- |
| Check the Sleep-EDF branch of `build_epoch_cache.py` | The 20 original Sleep-EDF PSG files (subjects 00–19); the BOAS branch already rebuilds sub-1 and sub-6 bit for bit (`cache_rebuild_check.json`) |
| Download DOD-H, HMC subset, raw BOAS, extra Sleep-EDF subjects | `scripts/fetch_external_data.py` and `node acquire.mjs` on a machine that can reach those hosts |

## Re-run

```
python analysis/day1_saved_predictions.py
python analysis/ablation_shift.py
python analysis/gain_uncertainty.py
```

These use only files already in the repo and take a few minutes. The CNN parts need the 64 Hz
epoch caches:

```
python analysis/ablation_shift.py --boas-cache <boas cache> --ext-cache <sleep-edf cache>
python analysis/a6_spectral_check.py --boas-cache <boas cache> --ext-cache <sleep-edf cache>
python analysis/beta_perturbation.py --boas-cache <boas cache> --ext-cache <sleep-edf cache>
```

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
This is a hypothesis for day 2, not a result.

## Still open

| Task | Needs |
| --- | --- |
| Test the beta explanation: per-recording wake beta share against CNN wake recall, and a band-limited input check | Day 2, caches already staged |
| Add the script that built the 64 Hz caches to the repo | Not found in the cache folders; the manifests record its settings (`resample_poly` 256→64 Hz down=4 for BOAS, 100→64 Hz up=16 down=25 for Sleep-EDF) |
| Download DOD-H, HMC subset, raw BOAS, extra Sleep-EDF subjects | `scripts/fetch_external_data.py` and `node acquire.mjs` on a machine that can reach those hosts |

## Re-run

```
python analysis/day1_saved_predictions.py
python analysis/ablation_shift.py
```

Both use only files already in the repo and take a few minutes.

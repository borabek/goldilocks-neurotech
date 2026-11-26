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
| CNN epoch caches | Not in the repo (about 723 MB). The script that builds the 64 Hz caches isn't in the repo either |

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

## Still to do today, and what it needs

| Task | Needs |
| --- | --- |
| A6 spectral check (`analysis/a6_spectral_check.py`) | The 64 Hz epoch caches from the laptop |
| A1 and A2 for the CNN (`analysis/ablation_shift.py --boas-cache ... --ext-cache ...`) | The same caches |
| Add the cache-building script to the repo | The laptop copy |
| Download DOD-H, HMC subset, raw BOAS, extra Sleep-EDF subjects | A machine with access to those hosts |

## Re-run

```
python analysis/day1_saved_predictions.py
python analysis/ablation_shift.py
```

Both use only files already in the repo and take a few minutes.

# Day 2 (27 November 2026): the pre-registered external test

Scored once, as `PREREGISTRATION.md` (tag `prereg-v1`, amendment v1.1) lays it out. The
predictions were committed before scoring (commit `a8ca00d`). Full numbers:
`results/external_test_verdicts.json`, `results/external_dodh/metrics.json`,
`results/external_hmc/metrics.json`.

## Verdicts

| Test | DOD-H (25 people) | HMC (25 people) |
| --- | --- | --- |
| H1, macro-F1, v2 + q75 vs v2 | **Supported**: +0.161 (0.119 to 0.202) | **Contradicted**: −0.088 (−0.127 to −0.049) |
| H1b, sleep-time error and kappa | **Supported**: error −11.3 min (−19.1 to −2.3), kappa +0.207 (0.145 to 0.273); Holm p = 0.0495 | **Contradicted**: error +81.7 min (49.8 to 117.5), kappa −0.088 (−0.135 to −0.044) |
| H2, CNN wake recall < 0.50 with v2 ≥ 0.60 | **Mixed**: CNN 0.503 (0.420 to 0.586), v2 0.142 | **Mixed**: CNN 0.638 (0.561 to 0.710), v2 0.760 |
| Secondary, family gap (v2 − CNN macro-F1) | Mixed: −0.149 (CNN ahead) | Closes: −0.034 (−0.075 to 0.005) |

Intervals are 95% paired person-level bootstrap intervals (10,000 resamples). Holm runs over the
six confirmatory tests; only the two DOD-H H1 tests reach significance.

## What it means

**The gain calibration is a fix for gross gain mismatches, not a general calibration.** DOD-H's
distributed signals are about four times smaller than the BOAS headband (label-free gain 4.06
for q75). Uncalibrated v2 falls apart there (kappa 0.111, wake recall 0.14), and the
calibration recovers a large part (kappa 0.318). On HMC the estimated gain is a modest 1.38,
and applying it makes v2 worse: wake recall rises from 0.76 to 0.91 and the sleep-time error
grows from 49 to 131 minutes. The percentile of epoch log-SD is pulled by things other than
amplitude (montage, population, how much of the night is wake), so it overshoots when the true
mismatch is small. H1 holds where the mismatch is large and fails where it is small.

**The CNN's wake collapse does not replicate on night-only recordings.** On Sleep-EDF the plain
CNN found 20% of wake. Here it finds 50% (DOD-H) and 64% (HMC): poor, but not the collapse.
The pre-registered thresholds sit either side of these values, so H2 is mixed on both. On
DOD-H the reading is complicated by the band-limited signal noted before scoring
(`PREREGISTRATION.md`, deviations).

A later check (prereg-v2.1, replicate seeds) shows the plain CNN's external wake recall depends on
the training seed: 0.48 to 0.58 on DOD-H and 0.61 to 0.76 on HMC across three seeds. On DOD-H
every seed lands in the mixed band. On HMC one of the three seeds (0.762) would cross the 0.70
line that contradicts H2. The verdict above is for the frozen model, as pre-registered, but on HMC
it is not stable across seeds.

**Neither family is close to the off-the-shelf baseline or to people.**

| Primary window, whole recording | DOD-H kappa | HMC kappa |
| --- | --- | --- |
| YASA 0.7.0 | 0.655 | 0.614 |
| Best CNN (balanced) | 0.445 | 0.430 |
| Ensemble of 7 | 0.334 | 0.414 |
| v2 + q75 | 0.318 | 0.243 |
| v2 | 0.111 | 0.331 |
| Each DOD-H scorer vs the other four | 0.690 to 0.789 | — |

Selecting by BOAS score did not pick a model that travels: the best model on both new datasets
is the one that was never trained on the headband.

## Claim gating (exploratory; `results/claim_gating/claim_gating.json`)

Epoch level: AURC of the risk-coverage curve, ranking epochs by the model's top probability
(lower is better; the best possible for the same number of errors in brackets). Recording level,
v2 only: withhold the sleep-time claim when the recording's own label-free gain is outside
1/1.5 to 1.5 (cut-off chosen here).

| Cohort | v2 AURC | CNN AURC | Claims withheld | Large errors (>60 min) blocked | Good claims (≤30 min) withheld | v2 mean error, all → kept (min) |
| --- | --- | --- | --- | --- | --- | --- |
| BOAS test | 0.057 (0.021) | 0.038 (0.013) | 6/20 | 1/4 | 4 | 31.8 → 34.6 |
| Sleep-EDF | 0.100 (0.039) | 0.640 (0.377) | 4/20 | 4/12 | 0 | 235.9 → 140.6 |
| DOD-H | 0.270 (0.148) | 0.268 (0.101) | 24/25 | 7/8 | 7 | 57.5 → 86.0 |
| HMC | 0.427 (0.131) | 0.403 (0.118) | 9/25 | 2/6 | 4 | 48.8 → 32.6 |

Model confidence does not protect a claim under shift. On Sleep-EDF the CNN's most confident
epochs are mostly wrong (AURC 0.64). The recording-level amplitude gate is better than nothing
where the mismatch is real: on Sleep-EDF it blocks 4 of 12 large errors and withholds no good
claim, and on DOD-H it flags 24 of 25 recordings as out of range, which they are. In-domain it is
too noisy: on BOAS it withholds 4 good claims to block 1 large error. A usable gate needs more
than one amplitude statistic.

## A3, recording length (Sleep-EDF; sleep-time error in minutes)

| Model | Whole recording | Label-free night window (v2 sleep span ±30 min) | Reference night window (oracle) |
| --- | --- | --- | --- |
| v2 | 235.9 | 235.9 | 36.8 |
| cnn | 803.2 | 551.2 | 55.7 |
| ensemble_of_7 | 842.6 | 580.8 | 65.5 |
| yasa | 110.3 | 100.5 | 23.1 |

Cropping to a window found without labels barely helps: v2 calls some daytime epochs sleep,
so its own sleep span covers most of the recording. Only the oracle window, which needs the
labels, removes the error. A real fix needs a detector trained on data with daytime wake
(section 7 of the pre-registration).

## A5, labelling scheme (stage share among scored epochs)

| Cohort | W | N1 | N2 | N3 | REM |
| --- | --- | --- | --- | --- | --- |
| boas_test (AASM) | 0.170 | 0.040 | 0.593 | 0.036 | 0.161 |
| sleep_edf (R&K, 3+4 -> N3) | 0.685 | 0.023 | 0.169 | 0.055 | 0.069 |
| dodh (AASM) | 0.125 | 0.059 | 0.486 | 0.140 | 0.190 |
| hmc (AASM) | 0.128 | 0.123 | 0.372 | 0.205 | 0.171 |

BOAS has very little N3 (3.6% of scored epochs, about 4% of sleep) against 14–21% on DOD-H and
HMC. The models trained on it under-call N3 everywhere, including on BOAS itself (v2 sends 41% of
reference N3 to N2). On Sleep-EDF v2 sends 74% of N3 to N2 and the CNN 85%,
while YASA finds 93% of it. So the N3 deficit is a property of the training labels, not
of R&K scoring on Sleep-EDF.

## A4, channel and montage (`results/ablation_shift/a4_montage.json`)

Run on the 20 original Sleep-EDF recordings, downloaded today and checked against the recorded
PhysioNet hashes (40 of 40 match). Before anything else, the Sleep-EDF branch of
`src/build_epoch_cache.py` rebuilt all 20 of the 64 Hz caches behind every Sleep-EDF CNN result
**bit for bit**, which closes the open check from day 1.

| Model and input | Kappa | Macro-F1 | Wake recall | Sleep-time error (min) |
| --- | --- | --- | --- | --- |
| v2 Fpz-Cz | 0.579 | 0.478 | 0.747 | 235.9 |
| v2 Pz-Oz | 0.356 | 0.311 | 0.533 | 423.8 |
| v2 Fpz-Cz + Pz-Oz | 0.622 | 0.470 | 0.841 | 143.4 |
| cnn Fpz-Cz | 0.086 | 0.188 | 0.130 | 803.2 |
| cnn Pz-Oz | 0.207 | 0.234 | 0.372 | 568.9 |
| cnn Fpz-Cz + Pz-Oz | 0.138 | 0.207 | 0.238 | 702.6 |
| cnn_balanced Fpz-Cz | 0.181 | 0.307 | 0.201 | 716.7 |
| cnn_balanced Pz-Oz | 0.267 | 0.402 | 0.391 | 537.1 |
| cnn_balanced Fpz-Cz + Pz-Oz | 0.291 | 0.405 | 0.361 | 573.9 |
| cnn_twochannel [Fpz-Cz, Fpz-Cz] | 0.115 | 0.224 | 0.226 | 698.3 |
| cnn_twochannel [Fpz-Cz, Pz-Oz] | 0.071 | 0.177 | 0.287 | 612.6 |

The montage matters, but it doesn't explain the CNN collapse. Pz-Oz, which sees more occipital
alpha in wake, lifts the plain CNN's wake recall from 0.13 to 0.37, still far from the 0.87 it
gets on the headband. For v2 the two derivations together beat either alone (kappa 0.622,
sleep-time error 143 min). The two-channel CNN does no better with a real second channel than
with Fpz-Cz copied twice.

## The 236-minute error: a sleep-window detector (`results/sleep_window/sleep_window_detector.json`)

Pre-registered in section 7 as exploratory. The detector smooths the model's own sleep/wake calls
over W epochs, merges sleep runs separated by at most G epochs, keeps the longest run and calls
everything outside it wake. No labels. W and G were chosen on the 56 development recordings
(subjects 20 and up) only: W = 61 epochs (30 min), G = 60 epochs (30 min), which cut
the development error from 176.4 to 90.6 min. Then it was applied once to the
20 evaluation people:

| Model | Kappa without → with | Sleep-time error without → with (min) |
| --- | --- | --- |
| v2 | 0.579 → 0.625 | 235.9 → 158.1 |
| cnn | 0.086 → 0.103 | 803.2 → 768.8 |
| yasa | 0.675 → 0.785 | 110.3 → 41.1 |

It removes a third of v2's error (236 to 158 min) and two thirds of YASA's (110 to 41 min, kappa
0.675 to 0.785). It can't help the CNN, which calls almost everything sleep, so its "longest run"
is most of the recording.

On the night-only datasets the same frozen detector makes things slightly worse, because it
throws away real sleep before or after a long awakening:

| Dataset and model | Sleep-time error without → with (min) |
| --- | --- |
| DOD-H v2 | 57.5 → 59.1 |
| DOD-H cnn | 26.8 → 37.8 |
| DOD-H yasa | 23.4 → 36.5 |
| HMC v2 | 48.8 → 75.2 |
| HMC cnn | 21.7 → 41.3 |
| HMC yasa | 37.1 → 55.3 |

So it belongs in front of long recordings only. A rule like "apply when the recording is longer
than about 14 hours" is the obvious next step; it hasn't been tested.

## Prereg-v2: does the CNN lean on 16–30 Hz power? (`results/prereg_v2/prereg_v2_results.json`)

The addendum at the end of `PREREGISTRATION.md` was written and committed before either model
was trained. Both are the frozen CNN architecture, seed 20261108, 10 passes over the BOAS
training nights, nothing else changed:

- `cnn_lowpass16`: every FFT bin above 16 Hz is set to zero, in training and in evaluation.
- `cnn_hfdrop`: during training only, the 16–30 Hz bins of each epoch are zeroed with
  probability 0.5. Evaluation inputs are untouched.

Each was evaluated once. A caveat on blindness: DOD-H and HMC had already been scored for the
frozen models, so I knew the plain CNN's wake recall there when I wrote the predictions. The
predictions were fixed before training, but this is not a blind test in the way the first
external test was.

BOAS test set (20 people), kappa:

| Model | Kappa | Drop vs cnn | Prediction (drop ≤ 0.03) |
| --- | --- | --- | --- |
| cnn | 0.728 | – | – |
| cnn_lowpass16 | 0.726 | 0.003 | holds |
| cnn_hfdrop | 0.724 | 0.004 | holds |

Wake recall, change against the plain CNN, paired person-level bootstrap (10,000 resamples):

| Dataset | cnn | cnn_lowpass16 | change [95% CI] | cnn_hfdrop | change [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Sleep-EDF | 0.130 | 0.126 | −0.004 [−0.028, 0.019] | 0.178 | +0.049 [0.027, 0.073] |
| DOD-H | 0.503 | 0.478 | −0.025 [−0.079, 0.025] | 0.555 | +0.052 [0.015, 0.090] |
| HMC | 0.638 | 0.607 | −0.031 [−0.097, 0.035] | 0.709 | +0.071 [0.034, 0.108] |

The pre-registered wake prediction is supported for `cnn_hfdrop` on all three datasets and not
supported for `cnn_lowpass16` on any of them. Kappa, the secondary measure, barely moves:
`cnn_hfdrop` gains 0.013 [0.002, 0.024] on Sleep-EDF, 0.010 on DOD-H and 0.004 on HMC (the last
two with intervals that include zero); `cnn_lowpass16` changes kappa by less than 0.01 everywhere.

What this means, on this seed alone (the replicate seeds in the next section weaken it a lot).
Making the network cope with missing beta power during training lifts wake recall by 5 to 7 points
on every external dataset at a cost of 0.004 kappa on BOAS. That fits
the idea that the plain CNN uses 16–30 Hz power as a wake cue, and that this cue is weak on the
other recorders. Cutting beta out entirely does not help: the network then has no beta cue to
lose, but it also has nothing that replaces it. The effect is real but small. On Sleep-EDF the
CNN still recognises fewer than one wake epoch in five, so beta reliance is one part of the
collapse, not the whole of it; the 23-hour recordings and the gain mismatch measured earlier
are still the larger problems.

## Prereg-v2.1: checks added after the prereg-v2 result

Written into `PREREGISTRATION.md` (addendum prereg-v2.1) before any of these was run. The replicate
seeds go against the prereg-v2 sentence that ruled out other seeds, and are logged there as a
deviation. They don't change the prereg-v2 verdicts above. They change how those verdicts should
be read.

### Replicate seeds (`results/prereg_v2_1/seeds.json`)

All three recipes (`cnn`, `cnn_lowpass16`, `cnn_hfdrop`) were retrained with seeds 20261109 and
20261110, everything else unchanged, and scored with the prereg-v2 code.

Mean wake recall:

| Seed | Model | BOAS kappa | Sleep-EDF | DOD-H | HMC |
| --- | --- | --- | --- | --- | --- |
| 20261108 (original) | cnn | 0.728 | 0.130 | 0.503 | 0.638 |
| | cnn_lowpass16 | 0.726 | 0.126 | 0.478 | 0.607 |
| | cnn_hfdrop | 0.724 | 0.178 | 0.555 | 0.709 |
| 20261109 | cnn | 0.722 | 0.304 | 0.578 | 0.762 |
| | cnn_lowpass16 | 0.714 | 0.169 | 0.470 | 0.556 |
| | cnn_hfdrop | 0.720 | 0.287 | 0.549 | 0.655 |
| 20261110 | cnn | 0.724 | 0.315 | 0.479 | 0.611 |
| | cnn_lowpass16 | 0.708 | 0.375 | 0.463 | 0.511 |
| | cnn_hfdrop | 0.715 | 0.355 | 0.515 | 0.538 |

Change in wake recall for `cnn_hfdrop` against the plain CNN of the same seed [paired 95% CI]:

| Seed | Sleep-EDF | DOD-H | HMC |
| --- | --- | --- | --- |
| 20261108 | +0.049 [0.027, 0.073] | +0.052 [0.016, 0.090] | +0.071 [0.034, 0.107] |
| 20261109 | −0.017 [−0.039, 0.006] | −0.029 [−0.061, 0.001] | −0.107 [−0.153, −0.067] |
| 20261110 | +0.040 [0.019, 0.063] | +0.036 [0.003, 0.065] | −0.073 [−0.125, −0.027] |
| Pooled over seeds | +0.024 [0.007, 0.043] | +0.019 [−0.008, 0.046] | −0.036 [−0.066, −0.009] |

**By the rule fixed in advance, the `cnn_hfdrop` effect is not robust:** four of the nine
matched-seed changes are at or below zero. Pooled over seeds, it helps a little on Sleep-EDF,
does nothing measurable on DOD-H and hurts on HMC. `cnn_lowpass16` is worse: pooled changes of
−0.026, −0.049 and −0.112.

The larger lesson is about the plain CNN itself. Its wake recall on Sleep-EDF is 0.13, 0.30 and
0.31 across three seeds with the same data and recipe, and on HMC 0.64, 0.76 and 0.61. BOAS kappa
moves by only 0.006 across the same seeds. The seed chosen for the frozen model happens to sit at
the bottom of that range on Sleep-EDF. So a single-seed difference of 5 to 7 points in external
wake recall is inside what the seed alone does, and the prereg-v2 result was most likely that.
Every external CNN number in this repository comes from one seed and carries this uncertainty.
The conclusions that depend only on the direction and size of the collapse (wake recall far below
the headband's 0.87 on every PSG dataset, on every seed) still hold. A claim that hinges on a
difference of a few points between two single-seed CNNs does not.

What survives of the 16-30 Hz story is what the band occlusion below shows on the frozen model: the
CNN does use beta power as a wake cue. Training it to do without that cue does not reliably help it
on other devices.

### Correction, 28 November: matched training length (`results/prereg_v2_1/seeds_matched_passes.json`)

While writing the model cards I found that the comparison above is not like for like. The frozen
plain CNN in `results/cnn/` was trained for 25 passes (pass 21 selected). `cnn_lowpass16`,
`cnn_hfdrop` and all six replicate-seed models were trained for 10, despite the prereg-v2 text
saying the recipe matched. So the seed 20261108 row above compares a 25-pass plain CNN with 10-pass
variants. The correction was written into the deviations log and pushed before its result: train a
10-pass plain CNN at seed 20261108 (`results/cnn_seed20261108/`) and use it as that seed's baseline.
The rule for "robust" is unchanged. The table above is left as it was.

Plain CNN, 10 passes at every seed, mean wake recall:

| Seed | BOAS kappa | Sleep-EDF | DOD-H | HMC |
| --- | --- | --- | --- | --- |
| 20261108 | 0.719 | 0.100 | 0.452 | 0.649 |
| 20261109 | 0.722 | 0.304 | 0.578 | 0.762 |
| 20261110 | 0.724 | 0.315 | 0.479 | 0.611 |
| frozen model (25 passes, seed 20261108) | 0.728 | 0.130 | 0.503 | 0.638 |

Training length is not what moves the number: at the same seed, 10 and 25 passes give 0.10 and 0.13 on
Sleep-EDF. The seed does: 0.10 to 0.31 with everything else fixed.

`cnn_hfdrop` against the 10-pass plain CNN of the same seed, change in wake recall [95% CI]:

| Seed | Sleep-EDF | DOD-H | HMC |
| --- | --- | --- | --- |
| 20261108 | +0.079 [0.051, 0.108] | +0.103 [0.063, 0.144] | +0.059 [0.032, 0.084] |
| 20261109 | −0.017 [−0.039, 0.006] | −0.029 [−0.061, 0.001] | −0.107 [−0.153, −0.067] |
| 20261110 | +0.040 [0.019, 0.063] | +0.036 [0.003, 0.065] | −0.073 [−0.125, −0.027] |
| Pooled over seeds | +0.034 [0.017, 0.053] | +0.036 [0.008, 0.064] | −0.040 [−0.069, −0.014] |

The conclusion does not change: four of nine matched comparisons are at or below zero, so the effect
is not robust. Pooled, the dropout helps a little on Sleep-EDF and DOD-H and hurts on HMC.
`cnn_lowpass16` pooled: −0.016, −0.032 and −0.116.

### Multiple testing (`results/prereg_v2_1/holm.json`)

The prereg-v2 verdicts used uncorrected intervals. With two-sided bootstrap p-values and a Holm
correction across the six wake tests, all three `cnn_hfdrop` results stay significant (Holm-adjusted
p = 0.0012 on Sleep-EDF, 0.0192 on DOD-H, 0.0020 on HMC). None of the `cnn_lowpass16` tests
comes close (adjusted p = 1.0 for all three). This corrects for the number of tests, not for
seed-to-seed variation. The replicate seeds above show that the latter is the bigger problem.

### Band occlusion on the frozen CNN (`results/prereg_v2_1/band_occlusion.json`)

One band at a time was removed from the input at test time, and the change in wake probability
on reference-wake epochs was measured. The prediction was that the three 16-30 Hz bands together
would matter more on BOAS than on any external dataset. **It failed.** The summed drop in wake
probability is 0.048 on BOAS, against 0.059 on Sleep-EDF, 0.066 on DOD-H and 0.088 on HMC.

| Band (Hz) | BOAS | Sleep-EDF | DOD-H | HMC |
| --- | --- | --- | --- | --- |
| 0.5-4 | +0.081 | +0.668 | +0.387 | +0.266 |
| 4-8 | −0.001 | +0.057 | −0.013 | +0.036 |
| 8-12 | −0.062 | −0.018 | −0.066 | −0.063 |
| 12-16 | −0.016 | −0.019 | −0.031 | −0.040 |
| 16-20 | −0.035 | −0.027 | −0.043 | −0.055 |
| 20-25 | −0.013 | −0.019 | −0.017 | −0.026 |
| 25-30 | −0.000 | −0.013 | −0.006 | −0.007 |

*Change in mean wake probability on reference-wake epochs when the band is removed.*

So the CNN treats 16-30 Hz power as wake evidence on every dataset, not only on the headband. What
the external datasets lack is the power, not the reliance. DOD-H carries about half the 16-30 Hz
share of BOAS (section 8 of the pre-registration). It does not fit the narrower story that beta
is a headband-specific cue. And because the replicate seeds above show that training without beta
doesn't reliably help, the reliance is real but removing it is not a fix.

All occlusion numbers come from the frozen model (seed 20261108) only.

The 0.5-4 Hz row looks dramatic: removing it lifts Sleep-EDF wake recall from 0.13 to 0.98. A
check added after the first run (labelled post hoc in the file) shows the effect isn't specific to
wake. Removing 0.5-4 Hz also raises the wake probability of reference-sleep epochs, by 0.50 to 0.61
on every dataset, and kappa falls by 0.68 on BOAS, 0.27 on DOD-H and 0.34 on HMC. Low-frequency
power is the network's general sleep signal; take it away and everything looks like wake. The
16-30 Hz bands are the opposite case. Removing them lowers wake probability on wake epochs about
two to four times as much as on sleep epochs (16-20 Hz on BOAS: −0.035 against −0.015; on HMC: −0.055 against −0.015), which is what a
specific wake cue looks like.

### A length rule for the detector (`results/prereg_v2_1/length_rule.json`)

T was fixed from lengths outside the test sets: the longest BOAS fitting night is 9.2 h and the
shortest Sleep-EDF development recording 19.5 h, so T = 14.4 h. Every test recording falls clearly
on one side: BOAS 7.0-8.8 h, DOD-H 5.2-9.9 h, HMC 3.9-10.3 h, Sleep-EDF 21.4-23.5 h. So the rule
applies the detector to all 20 Sleep-EDF recordings and to none of the others. Each dataset gets
the better of the two columns in the earlier tables: v2's Sleep-EDF error falls from 236 to 158 min,
and YASA's from 110 to 41 min. The night-only datasets stay exactly as they were without the
detector (for example HMC v2 48.8 min, where always applying the detector gives 75.2). The rule is
crude, and its result here was predictable once T was fixed. It is worth stating because it makes
the detector safe to ship in front of any recording, but it is not evidence about anything new.

### Error analysis (`results/prereg_v2_1/error_analysis.json`)

**Why H1 failed on HMC.** H1 compares v2 with and without q75 gain calibration. On DOD-H, where
the estimated gain is 4.06, calibration helps where it should: wake F1 0.17 → 0.40, N3 0.09 → 0.52,
N2 0.66 → 0.71. On HMC the gain is only 1.38, and calibration pushes the signal past the BOAS
level. Predicted wake rises from 22% to 42% of epochs against a reference of 13%. REM takes most of
the damage: the share of reference REM called wake goes from 11% to 43%, and REM F1 falls from 0.43
to 0.11. REM is low-amplitude, mixed-frequency EEG, the stage closest to wake, so it is the first
to cross the line when amplitudes are scaled up. Gain calibration helps a device that is far off,
and harms one that was nearly right.

**Where the plain CNN goes wrong.** On Sleep-EDF 67% of reference wake is called N2. On DOD-H
the figure is 37% and on HMC 14%. N3 is under-called everywhere outside BOAS: the CNN finds 7-22% of
reference N3 on the external sets, and its predicted N3 share is 1-3% against a reference share of
5-21%. That fits A5, where BOAS has 3.6% N3 against 14-21% elsewhere.

**Calibration.** Expected calibration error of the plain CNN is 0.053 on BOAS, 0.134 on DOD-H,
0.187 on HMC and 0.419 on Sleep-EDF. On Sleep-EDF its mean top probability is 0.69 while it is right
on 27% of epochs. `cnn_hfdrop` barely changes this (0.046, 0.134, 0.190, 0.396). So its
probabilities can't be trusted to flag the epochs it gets wrong on a new device, which agrees with
the claim-gating result.

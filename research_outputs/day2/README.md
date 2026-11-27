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

## A4, channel and montage

Pending: needs the 20 original Sleep-EDF PSG files, being downloaded.

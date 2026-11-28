# Model card: the CNN family

**Files:** `results/cnn/` (the plain CNN used in the pre-registered test), `results/cnn_balanced/`,
`results/cnn_sqrtbalanced/`, `results/cnn_augmented/`, `results/cnn_context*/`, and the retrained
models `results/cnn_lowpass16*/`, `results/cnn_hfdrop*/`, `results/cnn_seed*/`. Code in
`src/cnn_challenger.py` and `src/cnn_context.py`. Weight hashes are in each `cnn_freeze_record.json`.
**Status:** research models. The plain CNN and its variants were frozen on 7 November 2026. Not a
medical device and not for clinical use.

## What they do

A small one-dimensional convolutional network (about 44k parameters, four convolution layers) reads
each 30-second epoch of the 64 Hz signal, after robust scaling per recording. Instance normalisation
follows every convolution, which makes the network close to insensitive to signal amplitude. Each
channel is classified with shared weights and the probabilities are averaged, then smoothed with the
same hidden-Markov step as v2. The variants differ only in class weighting (balanced, square-root
balanced), augmentation, or an added temporal head over neighbouring epochs (context models).

## Training data

BOAS headband recordings, 59 fitting nights, AASM consensus labels. The training pass, smoother weight
and temperature were chosen on separate BOAS people. The 20 test people were not used for any choice.

## Performance

Mean over people, whole recording. Full table with intervals: `results/paper/benchmark.json`.

| Dataset | Plain CNN kappa | Plain CNN wake recall | Balanced CNN kappa | Plain CNN sleep-time error (min) |
| --- | --- | --- | --- | --- |
| BOAS test (20) | 0.728 | 0.868 | 0.479 | 16.5 |
| Sleep-EDF (20, inspected during development) | 0.086 | 0.130 | 0.181 | 803.2 |
| DOD-H (25, pre-registered test) | 0.297 | 0.503 | 0.445 | 26.8 |
| HMC (25, pre-registered test) | 0.372 | 0.638 | 0.430 | 21.7 |

The balanced CNN has the lowest BOAS kappa of the family and the highest on DOD-H and HMC. Choosing a
variant by its BOAS score would have picked the wrong one.

## Known failure modes

- **Wake on clinical recordings.** On Sleep-EDF the plain CNN finds 13% of wake and calls 67% of it N2,
  so on 23-hour recordings it reports about 13 extra hours of sleep. On night-only PSG it finds half
  (DOD-H) to two thirds (HMC) of wake. It uses 16-30 Hz power as a wake cue, which forehead headband
  wake carries about twice as much of as Fpz-Cz wake.
- **Seed sensitivity.** Five copies of the plain CNN, trained on the same data with the same recipe
  and different seeds, score between 0.719 and 0.727 kappa on the headband. Off it they disagree:
  wake recall runs from 0.10 to 0.37 on Sleep-EDF, 0.45 to 0.65 on DOD-H and 0.61 to 0.76 on HMC.
  Treat a difference between two single-seed CNNs on external data as noise unless it is well
  outside that spread.
- **Retraining without beta does not fix it.** `cnn_hfdrop` (16-30 Hz dropped at random during
  training) looked like a fix on one seed and was not robust across three. `cnn_lowpass16` was worse.
- **Confidence.** On Sleep-EDF its most confident epochs are mostly wrong (risk-coverage AURC 0.64),
  so its probabilities cannot flag the epochs to distrust on a new device.
- **N3.** Like v2 it under-calls N3, a consequence of the little N3 in BOAS.

## Training-length note

The frozen plain CNN was trained for 25 passes. The retrained models and the replicate seeds were
trained for 10. Comparisons in the seed check use a 10-pass plain CNN at every seed
(`results/prereg_v2_1/seeds_matched_passes.json`); see the 28 November entry of the deviations log in
`PREREGISTRATION.md`.

## Intended use

Research on headband sleep staging and on how models fail under device change. Do not use for
diagnosis, treatment decisions or any individual's care.

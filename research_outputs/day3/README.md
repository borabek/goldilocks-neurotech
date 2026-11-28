# Day 3 (28 November 2026): the paper, and the checks it needed

Today was for turning two days of results into a paper, `paper/main.pdf`. Writing it exposed a few
things that had to be measured properly first, and one mistake. Everything below is exploratory
unless it says otherwise. The plans for the two new analyses were written into `PREREGISTRATION.md`
and pushed before they ran.

## One table for everything (`analysis/benchmark.py`, `results/paper/benchmark.json`)

The numbers in the paper's main table had come from four different scripts written on different
days. `analysis/benchmark.py` recomputes every model's per-person metrics on all four datasets from the
saved predictions alone, then checks its means against the metric files written at evaluation time.
All 108 checks match to within 1e-9, so the table and the earlier write-ups agree.

## Does the headband ranking predict anything? (`results/paper/ranking_extended.json`)

With the eight frozen headband models, Kendall's tau between BOAS kappa and external kappa was -0.43
on Sleep-EDF, -0.14 on DOD-H and 0.07 on HMC. Eight points is not much, so I widened the test to every
headband-trained model with predictions on all four datasets: 19, counting the retrained CNNs and the
extra seeds. People were resampled within each dataset to get an interval for tau.

| Dataset | Kendall tau (19 models) | 95% interval | Rank of the best BOAS model |
| --- | --- | --- | --- |
| Sleep-EDF | -0.27 | -0.47 to 0.02 | 13 of 19 |
| DOD-H | 0.27 | -0.19 to 0.33 | 10 of 19 |
| HMC | 0.24 | -0.23 to 0.36 | 3 of 19 |

Every interval includes zero. The headband score is not a usable guide to the ranking on a new device,
in either direction.

YASA against the best headband model on each dataset (chosen after the fact, which favours the
headband side), paired difference in kappa: +0.096 (0.021 to 0.170) on Sleep-EDF, +0.210 (0.149 to
0.273) on DOD-H, +0.184 (0.136 to 0.233) on HMC.

## The seed spread, with five seeds

Two more copies of the 10-pass plain CNN were trained (seeds 20261111 and 20261112). The rule written
beforehand was that the paper's seed claim stands if the five-seed range of Sleep-EDF wake recall is at
least 0.10 while the BOAS kappa range stays under 0.02. It holds:

| | Minimum | Maximum |
| --- | --- | --- |
| BOAS kappa | 0.719 | 0.727 |
| Sleep-EDF wake recall | 0.10 | 0.37 |
| DOD-H wake recall | 0.45 | 0.65 |
| HMC wake recall | 0.61 | 0.76 |

## A mistake, and its correction

Writing the model cards, I found that the frozen plain CNN had been trained for 25 passes while every
retrained CNN and replicate seed had 10. The prereg-v2 addendum had said the recipes matched. I logged
this in the deviations section and pushed it before training a 10-pass plain CNN at the original seed,
then re-ran the seed check against it (`results/prereg_v2_1/seeds_matched_passes.json`; details in
`research_outputs/day2/README.md`). The conclusion did not change, and training length turned out not to
matter much: 10 and 25 passes at the same seed give Sleep-EDF wake recall of 0.10 and 0.13.

A smaller one: I had written one seed's Sleep-EDF wake recall as 0.32 in two places. It is 0.3147, so
0.31. `scripts/verify_claims.py` did not catch it because 0.32 happens to match some other stored
value; it is fixed.

## The vendor gap (`analysis/bitbrain_attempt.py`)

One attempt, decided on the six selection people only: average the probabilities of v2 and the
seven-CNN ensemble, with v2's weight chosen from 0 to 1 in steps of 0.1. The selection people chose 0.3.
On the 20 test people, on the epochs the vendor's scorer labels, kappa went from 0.746 for the ensemble
to 0.767, against 0.787 for the vendor. The remaining gap, -0.021 (-0.044 to 0.003), is no longer
distinguishable from zero. The test people have been looked at many times, so this is a hint, not a
result.

## The drug-delivery track, rerun from the data

The exposure model and its leakage audit were rerun from the original workbook, downloaded again from
the publisher (same checksum as the file used at the event). Every one of the stored values in both
output files came out the same (`neurorelease/exposure_model/out/rerun_check.json`). The workbook is
licensed CC BY-NC 4.0 and stays out of the repository.

## Also done

- `paper/`: LaTeX source, `make_numbers.py` and `make_figures.py`, which write every number and figure
  from the result files, and the built PDF.
- `docs/model_cards/`: v2 and the CNN family, including how each fails.
- `scripts/verify_claims.py` now checks the paper and the model cards as well as the README and the
  hackathon slides; 548 figures, none unsupported. Putting made-up numbers into the paper makes it fail,
  which I checked.
- `.github/workflows/checks.yml`: on every push, run the tests and `verify_claims.py`, regenerate the
  benchmark, the paper's numbers, the external-test scores and the Holm correction and compare them with
  the committed files, and confirm the pre-registered files still match their frozen hashes. It runs on
  `requirements.txt` alone; I checked every step in a fresh environment first.

## Re-run

```
python analysis/benchmark.py
python analysis/ranking_extended.py --boas-cache <boas cache> --ext-cache <sleep-edf cache> --new-cache <dod-h/hmc cache>
python analysis/bitbrain_attempt.py --boas-cache <boas cache>
python paper/make_numbers.py && python paper/make_figures.py
```

# Goldilocks: forehead-EEG sleep staging, tested on devices it never saw

Bora Bayrakci, University of Essex.

This started as a project at the Oxford NeuroTech Hackathon (7-8 November 2026) and is now an
individual research output. The question is narrow: a sleep stager trained on a two-channel
forehead headband looks good on held-out people from the same headband. How much of that
survives on other recorders, and why does it break where it does?

Two model families are trained on the BOAS headband recordings (OpenNeuro ds005555) and frozen:

- **v2**, gradient boosting on 14 spectral features per channel, with a sequence smoother;
- **a CNN family**, a small 1-D network on the raw 64 Hz signal, with several variants.

They are then tested on Sleep-EDF and, in a test written down before any label was seen
(`PREREGISTRATION.md`, tag `prereg-v1`), on DOD-H and a 25-recording subset of HMC. YASA is the
off-the-shelf baseline. DOD-H's five human scorers give a ceiling.

## What came out

- On the headband both families do well (BOAS test kappa 0.661 for v2, 0.728 for the CNN). On
  other recorders neither comes close to YASA (kappa 0.655 on DOD-H, 0.614 on HMC) or to the human
  scorers (0.690 to 0.789 on DOD-H).
- A label-free gain calibration, fixed in advance, helps v2 where the amplitude mismatch is large
  (DOD-H, estimated gain 4.06) and hurts it where the mismatch is small (HMC, 1.38). On HMC it
  pushes REM over the wake line.
- The CNN's wake recognition collapses on PSG recordings: 0.87 wake recall on the headband against
  0.13 on Sleep-EDF for the frozen model. Band occlusion shows it uses 16-30 Hz power as a wake
  cue on every dataset.
- Retraining the CNN without that cue looked like a fix on one seed. Across three seeds it isn't.
  The plain CNN's own Sleep-EDF wake recall ranges from 0.13 to 0.32 with the seed alone, so
  single-seed comparisons between CNN variants of a few points carry no weight.
- A label-free sleep-window detector, applied only to recordings longer than 14.4 hours, cuts
  YASA's sleep-time error on Sleep-EDF from 110 to 41 min and v2's from 236 to 158 min, and leaves
  night recordings untouched.

The full write-ups, with every number traced to a result file, are in
[`research_outputs/day1/`](research_outputs/day1/README.md) (exploratory checks) and
[`research_outputs/day2/`](research_outputs/day2/README.md) (the pre-registered test and the
checks that followed it).

## Layout

```
PREREGISTRATION.md        hypotheses, thresholds, deviations log and addenda
pipeline.py               shared core: features, metrics, protocol (see note below)
v2_model.py               v2 model
yasa_evaluate.py          masks, windows and scoring used by every evaluation
protocol.json             the BOAS person-level split
v2_protocol.json          v2 design, selection and calibration people
preregistration_hashes.json, SHA256SUMS.txt

src/                      model training, prediction and evaluation
analysis/                 ablations and follow-up analyses (one script per question)
scripts/                  data download and compaction, verify_claims.py
tests/                    unit tests
data/                     derived features, selections and manifests (raw recordings are downloaded)
results/                  every saved model, prediction and metric
research_outputs/         dated write-ups of the research rounds
docs/                     data sources and licences, where files moved
hackathon/                the hackathon submission, kept as it was
neurorelease/             the drug-delivery research track from the hackathon
```

`pipeline.py`, `v2_model.py` and `yasa_evaluate.py` stay at the top level on purpose. The
pre-registered evaluation code imports them from there, and its SHA-256 hashes are frozen in
`preregistration_hashes.json`, so moving or editing them would break the record of what was fixed
before the test. Files moved on 27 November 2026; [`docs/LAYOUT.md`](docs/LAYOUT.md) lists every
old and new path, which matters when reading `PREREGISTRATION.md` and the hackathon documents.

## Running it

Install the pinned `requirements.txt`, plus `requirements_live.txt` for the CNN scripts (JAX).
Run everything from the repository root.

```
python -m pytest -q                      # unit tests
python scripts/verify_claims.py          # every figure on the slides and handout against the
                                         # saved results, and every file against SHA256SUMS.txt
python src/reproduce_v2.py               # v2 tests and evaluation from the saved weights
python src/external_eval.py score        # re-score the pre-registered external test
```

The raw recordings are not in the repository. `scripts/fetch_external_data.py`,
`scripts/fetch_dodh_compact.py` and `node scripts/acquire.mjs` download them;
`src/build_epoch_cache.py` builds the 64 Hz caches the CNN scripts read. Each script's docstring
gives its exact command line. Datasets and licences: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## How the evidence is kept honest

- Hypotheses, thresholds and the analysis were committed before the external labels were
  downloaded (tag `prereg-v1`), and the frozen models were hashed. Anything added later is written
  as a dated addendum, and every departure from the plan is logged in its deviations section.
- Evaluation is person-level throughout, with paired bootstrap intervals (10,000 resamples) and
  a Holm correction over the confirmatory tests.
- Results that went against the hypotheses are reported the same way as those that held.

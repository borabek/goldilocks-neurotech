# Goldilocks AI — start here

**A wearable can stage sleep well and still get the night wrong. We measured where ours stops
being trustworthy.**

One sleep-stage model for a forehead EEG headband. Frozen before testing, evaluated
person-held-out on 20 people, then run unchanged on a device it had never seen.

## The three numbers that matter

| | kappa | sleep-time error |
|---|---|---|
| BOAS headband — ours | **0.661** | **31.8 min** |
| BOAS headband — YASA 0.7.0 baseline | 0.532 | 57.6 min |
| Sleep-EDF, unseen device — ours | 0.579 | **235.9 min** |

Per-epoch agreement barely moved. The night summary broke. On one night the reference holds
473 minutes of sleep and we reported 1383.

**And the model that scored best at home was the worst one away.** The CNN we trained overnight
reached kappa 0.728 on the headband and 0.086 on Sleep-EDF. Choosing a model by its in-domain
score would have picked the one that fails hardest.

## Check it yourself, in about thirty seconds

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce_v2.py
```

Expect 7 tests passed, kappa 0.6612737502932162, sleep-time error 31.8 min. No network is used.
This has reproduced to the last digit on three machines and two operating systems.

And to check that the presentation itself did not drift from the evidence:

```
.venv/bin/python verify_claims.py --verbose
```

It pulls every figure out of the handout and the slides, looks for each one among the metrics
stored under `results/`, and verifies `SHA256SUMS.txt`. Expect 102 figures checked, none
unsupported, no hash problems. Its docstring states what each half of that check can and cannot
catch.

One thing to know about the order: `reproduce_v2.py` rewrites six files under `results/v2/` as
it runs, so their hashes change even though every number in them stays identical. Check the
hashes before reproducing, or regenerate `SHA256SUMS.txt` afterwards with

```
find . -type f -not -path "./.venv/*" -not -path "*/__pycache__/*" \
  -not -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
```

For the CNN and the live path, also install `requirements_live.txt`, then:

```
.venv/bin/python live_inference.py --replay data/live_demo/sub-4.npz | head -40
./demo.sh                      # the whole story, paced, about 1 min 45
```

## What to open

| file | what it is |
|---|---|
| `handout/goldilocks_handout.pdf` | every number and baseline on one page |
| `slides/goldilocks_support_slides.pdf` | 10 slides of technical backup, including what we withdrew |
| `results/cnn/Goldilocks_Validation_Explorer_v3.html` | the validation explorer: eight lanes, one night at a time |
| `plan/GOLDILOCKS_Plan_Rev19.md` | the full record, including the work we did not do and why |
| `QA_CARD.md` | the nine hardest questions, with the answers and the numbers behind them |
| `SHA256SUMS.txt` | every file in the package, hashed |

## What we are not claiming

The BOAS cohort was inspected at v1, so results on it are a development comparison, not
independent confirmation. Prediction intervals missed their 90% target — 60% on BOAS, 20%
externally — so calibrated-range claims are switched off in the release gate and exported
intervals are null. No clinical, ADHD, medication or paediatric claim is made anywhere.

Three claims were withdrawn during the build because they fell inside the 0.015 macro-F1 spread
we measured across random seeds. A leakage audit removed a fourth by merging two duplicated
studies. Those removals are in the record on purpose.

## Data

BOAS via OpenNeuro (CC0, `doi:10.18112/openneuro.ds005555.v1.1.3`) and Sleep-EDF Expanded 1.0.0
via PhysioNet (ODC-By 1.0, `10.13026/C2X676`). Public, de-identified, no teammate recording.

Ceren — biomaterials · Bora — coding · Jacob — neurotechnology

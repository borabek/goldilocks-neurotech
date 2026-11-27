# Goldilocks v2: development and independent transfer evaluation

Open `results/v2/Goldilocks_Validation_Explorer_v2.html` offline. All outputs are precomputed retrospective analysis. V1 artefacts remain available unchanged.

## What changed

The classifier shares features across EEG channels, uses centered temporal context, and smooths stage probabilities using transitions estimated on fitting people. This uses future epochs and is not a live decoder. Sequence strength was selected on six development people; temperature was fitted on six different people. Twenty original BOAS people calibrated the two fixed sleep-time interval methods. No historical-test or external labels selected those choices.

`v2_protocol.json` records the design; `results/v2/freeze_record.json` records the weights/protocol hashes and pre-reference freeze. `v2_model.py` refuses to overwrite an existing protocol. Future experiments need a new version, development split and untouched evaluation set; do not delete the protocol to silently reuse this test.

## Measured results

| Cohort | Mean person kappa | Sleep-time MAE | Global / scaled interval coverage |
|---|---:|---:|---:|
| Historical BOAS, v1 | 0.584 | 27.6 min | 80% / not tested |
| Historical BOAS, v2 development comparison | 0.661 | 31.8 min | 60% / 60% |
| New Sleep-EDF, full scored recording (primary) | 0.579 | 235.9 min | 20% / 25% |
| New Sleep-EDF, reference-defined sleep window (secondary) | 0.515 | 36.8 min | 30% / 30% |

There are 20 people in each evaluated cohort. Historical BOAS people were inspected in v1, so their v2 result is development evidence. Macro-F1 improved from 0.471 to 0.561 there, but the released Bitbrain scorer still leads. On the same vendor-covered mask v2 sleep-time MAE is 31.3 versus 14.4 minutes for Bitbrain.

On historical BOAS, temperature calibration improved log loss from 0.739 to 0.545, Brier score from 0.320 to 0.281 and confidence ECE from 0.128 to 0.087. Probability calibration and total-sleep prediction-interval coverage are separate properties. Both interval methods missed their nominal 90% target. `release_gate.json` disables calibrated-range claims; numerical intervals remain in research CSVs, while UI exports contain `calibrated_interval: null`.

Twenty independent Sleep-EDF signal/reference pairs were downloaded and matched against PhysioNet SHA-256 checksums. Selection and preprocessing were fixed before reference access; the model was saved before reference downloads. The single Fpz-Cz channel is resampled 100 to 256 Hz. R&K stages 3/4 are mapped to N3, movement/unscored excluded. Primary evaluation uses all valid scored epochs; the secondary window uses the first-to-last reference sleep span plus 30-minute margins. This reference-defined secondary window is an evaluation convention, not a deployable lights-off detector.

Sleep-EDF changes device, montage, scoring and recording duration, including long daytime wake. This is a domain-shift test. The poor full-recording sleep-time result stays primary and visible. These data do not validate ADHD diagnosis, treatment response, dosing, hydrogel compatibility or clinical sleep quantities.

## Reproduce the frozen evaluation without raw downloads

Create the environment using `requirements.txt`, then run:

```bash
.venv/bin/python reproduce_v2.py
```

The build includes all 128 BOAS and 20 external derived feature files. The command tests invariants, loads frozen weights, recomputes both evaluations and rebuilds the report. It never fits a model. Separate original external EDF ZIPs are for signal/annotation inspection. For acquisition, `node acquire_external.mjs` retrieves signals; `node acquire_external.mjs --labels` requires an existing model freeze.

Seven tests passed. Desktop/mobile controls, reference reveal, cohort switching, rejection messages and evidence download were checked in Chrome with no page errors. The app has no clinical decision function.

## Independent dataset attribution

Sleep-EDF Database Expanded, version 1.0.0, DOI `10.13026/C2X676`. Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on Biomedical Engineering 47(9):1185–1194, 2000. PhysioNet: https://physionet.org/content/sleep-edfx/1.0.0/ . Licence: Open Data Commons Attribution License v1.0. The original licence page, source page, published hash manifest and acquisition records accompany the files. Follow the dataset's citation instructions when reusing it.

## Next decision

Keep v1 as a reference, v2 as a stage-decoding development result, and the new external result as a documented failure boundary. Do not promote v2 as universally superior. A next model needs a task-aligned night-window definition, improved wake/artefact handling and separate development/calibration data, followed by another untouched evaluation. The present 20 external people are now inspected and cannot be called a fresh test again. No additional unrelated datasets or delivery modules are needed to continue building the research product.

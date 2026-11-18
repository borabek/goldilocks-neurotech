# Improvement backlog — one round at a time

Each overnight round takes the highest item that is **ready**, does it, records the result here and
in the plan, and marks it done. The ordering is by evidence value per unit of risk, not by novelty.

Rules that do not change, whatever a round finds:
- Never select anything on the 20 test people. Selection happens on the six selection people only.
- Never change the evaluation mask, the window definitions or `stage_metrics`: that is what makes
  v2, YASA and every CNN variant comparable.
- Report every variant tried, including the ones that lose. A round that produces a worse number
  still writes it down.
- Both cohorts are inspected. Nothing produced now is a fresh independent test, and every new
  result is labelled development.
- No metric substitution to make a number look better. Person-averaged macro-F1 stays the default;
  pooled macro-F1, weighted F1 and accuracy are reported beside it, always labelled.

## Ready now, no epoch cache needed (uses saved predictions only)

| # | Item | Why it is worth doing | Expected effect |
|---|---|---|---|
| 1 ✅ **DONE 00:52** | **Failure gallery for the best model.** Pick the three worst held-out nights for the ensemble, plot reference against prediction, write one cautious sentence each about what went wrong. | The plan lists it as delivery-pack priority 10 and it is the one piece of the three-act demo still missing for tonight's models. `results/failure_gallery.json` only covers v1. | No F1 change. Fills a named gap in the demo. |
| 2 ✅ **DONE 07:26** | **Per-stage table in the viewer.** Add a lane and a per-stage F1 row for the ensemble to `results/yasa/Goldilocks_Validation_Explorer_v2_1_yasa.html`, as a new copy, leaving the existing viewers untouched. | The demo still shows v2 and YASA only. The headline result of the night is not in the thing the judge clicks. | No F1 change. Large demo value. |
| 3 ✅ **DONE 00:52** | **Confusion matrices, pooled and per person.** For v2, YASA, the context model and the ensemble. | Tells the reader *where* N1 goes, which is the question every sleep person asks. Cheap and it constrains claims. | No F1 change. |
| 4 ✅ **DONE 00:52** | **Agreement against the vendor scorer on its covered mask**, for the new models. | We compare to the Bitbrain scorer for v2 only. The claim "we approach the vendor on its own hardware" needs the same mask for the new models. | No F1 change. Strengthens the strongest honest comparison. |

## Ready now, needs the epoch caches in ~/Downloads

| # | Item | Why | Expected effect |
|---|---|---|---|
| 5 ✅ **DONE 01:05** | **Seed variance of the context head.** Retrain the temporal head on the frozen trunk with five seeds; report the spread of macro-F1 and kappa. The head trains in about 30 seconds. | Every CNN number in the plan is a single seed. If the spread is ±0.02, then the gap between several variants is noise, and the plan should say so before a judge does. | No gain expected; it may *shrink* claims, which is the point. |
| 6 ✅ **DONE 07:38** | **Context window sweep.** The head uses ±7 epochs because published work plateaus there. Try ±3, ±5, ±7, ±11, selecting on the six selection people. | One cheap, principled hyperparameter with a literature prior. | +0.00 to +0.02 macro-F1. |
| 7 ✅ **DONE 07:38** | **Wider temporal head.** Two convolutional layers at 64 channels is small. Try 128 channels and a third layer, selected the same way. | The trunk is frozen, so the head is the only place left where capacity is cheap. | +0.00 to +0.02 macro-F1. |
| 8 🔄 **IN FLIGHT, resumable only by rerunning** | **Train the trunk and the head together, end to end, on multi-epoch input.** Code is written and tested: `cnn_endtoend.py`. It saves only at the end, so an interrupted run is lost and must be restarted. |~~ The present context model freezes a trunk that never saw context, which the plan records as a lower bound. | This is the single largest remaining modelling gain, and the only one with a clear mechanism: the encoder gets to reshape its features for the context task. | +0.02 to +0.05 macro-F1. Costs about an hour of CPU. |
| 9 | **Person-weighted variant, finished.** Its training was interrupted. | The one item from the user's original list that never completed. | Unknown; the balanced variants suggest small. |

## Needs a decision, not a machine

| # | Item | Why it is blocked |
|---|---|---|
| 10 | **A second wearable cohort (DOD-H, DOD-O).** 25 and 55 nights, five scorers each, publicly available. | Downloading and integrating a new dataset is outside the hackathon scope rules in section 6, and the network here refuses most hosts. It is the right next step *after* the event, and it targets transfer, which is the real failure. |
| 11 | **Recordings containing daytime wake.** | Same: the sleep-window rule failed because no training night contains daytime wake. No public substitute is in the build. |
| 12 | **Team items.** Rehearsals, two adversarial Q&A rounds, real names on the credits slide, the organiser message and the confirmed code freeze. | People, not compute. |

## What will not raise macro-F1, and why, so no round wastes time on it

- **Scalar temperature scaling.** It is a monotone transform of the logits; the winning class never
  changes, so macro-F1 cannot move. Confirmed, not assumed.
- **Prediction intervals.** They do not touch stage predictions.
- **More BOAS-like nights.** Published learning curves for this task plateau at roughly 12 to 25
  training recordings; this build fits on 59 nights from 48 people, well past that.
- **Chasing 0.80 person-averaged macro-F1.** Six of the 20 test nights contain no reference N3, so
  mean N3 F1 cannot exceed 0.70 under this convention. With the best published N1 ever reported for
  this task (0.598, on full polysomnography with EOG and EMG) and N3 at that cap, the ceiling is
  0.758. The honest target is 0.65 to 0.70.


## Completed, with measured outcomes

| # | Outcome |
|---|---|
| 1 | `results/vendor_and_failures/failure_gallery.{png,json}`. The three worst nights for the ensemble fail identically: N1 recovered at 20–45% and most of the rest called Wake. |
| 3 | `results/vendor_and_failures/confusion_matrices.json`, counts and row-normalised, for the frozen v2 model, YASA, the context model and the ensemble. |
| 2 | `results/cnn/Goldilocks_Validation_Explorer_v3.html`, built by `make_viewer_v3.py`. Three new lanes (CNN challenger, context model, ensemble) on both cohorts, six new rows in the results table, the existing v2 and v2.1 viewers untouched. Verified in a headless browser: eight lanes draw with distinct colours, thirteen table rows, twenty-two records selectable, no script error. |
| 6, 7 | No improvement found, and the evidence is in `results/context_sweep/`. Windows of 7, 11, 15 and 23 epochs and a wider 128×3 head were each selected on the six selection people. They chose the shipped configuration, 15 epochs with a 64×2 head. The best test macro-F1 across all five runs is 0.617 against the shipped 0.618, and the largest difference between any two runs is smaller than the 0.0147 head-seed spread. Two items closed with nothing gained: the hyperparameters were already at a sensible point, and the selection people independently landed on the same window the literature prior predicted. |
| 5 | Five head seeds, trunk frozen: macro-F1 0.601–0.616 (mean 0.607, spread 0.015), kappa spread 0.004, sleep-time error 9.6–20.3 min (sd 4.2). Forced three withdrawals, now in the plan: the two context variants are no longer ordered, the 8.7-minute figure is replaced by "roughly halves 31.8 minutes", and kappa is named as the stable metric. Also found that the same seed gives 0.618 on the Mac workspace and 0.616 in the cloud one, so training reproduces to about two decimals across environments while the frozen evaluation reproduces to fifteen significant figures. |
| 4 | On the mask the released Bitbrain scorer covers (98.2% of scored epochs): vendor kappa 0.787, ensemble 0.746 — the vendor keeps the lead on stage agreement. But the ensemble is ahead on macro-F1, 0.632 against 0.616, and halves the vendor's sleep-time error, 8.8 against 14.4 minutes. This is the like-for-like benchmark the pitch should use. |

## Done in the first session, 7–8 November

Submission package; proven one-command reproduction; exposure-model leakage audit; the CNN challenger and
six variants; three context models; the seven-model ensemble; every one scored on both cohorts; post-hoc
prior adjustment and N1/N3 decision offsets; smoothing and selection-metric ablations; four-metric summary;
paired person-level bootstrap intervals; vendor-mask comparison; failure gallery; confusion matrices; plan
Revision 19; README; seven support slides; one-page handout; file-hash manifest.

Leading result on the headband, 20 held-out people: **ensemble**, person-averaged macro-F1 0.629, pooled
macro-F1 0.702, weighted F1 0.852, accuracy 0.847, kappa 0.741, sleep-time error 9.3 minutes. Every member
of the CNN family, the ensemble included, collapses on Sleep-EDF; the frozen feature model is the only one
of ours that transfers at all, and YASA beats everything there.


## Added on request: the live EEG path

| Item | Outcome |
|---|---|
| Causal streaming inference (`live_inference.py`) | Done. Trailing-window normalisation, per-epoch CNN, forward-only HMM filtering, running sleep total. LSL input for a real headset, replay input for testing without one. Verified end to end on a replayed night. |
| Measured cost of going causal | kappa 0.728 → 0.690, macro-F1 0.595 → 0.556, sleep-time error 16.5 → 17.7 min. The causal path still beats the frozen v2 model's retrospective 0.661 kappa. `results/live/`. |
| Still needs the team | the device itself and its channel map; the three successful rehearsals Gate E requires; and a re-evaluation on that device, which needs referenced recordings from it. None of these can be produced here. |

## The one unfinished item, and what a proper run of it looks like

Item 8, the end-to-end sequence model, is the only avenue left with a clear mechanism for a real
gain. It has been written and smoke-tested twice; both attempts were stopped for reasons that had
nothing to do with the method. Do it once, properly, in a single uninterrupted block of about two
hours, rather than in snatched fragments.

**Where.** Not the Mac's Linux workspace. Two attempts died there: the workspace is suspended
whenever the Mac sleeps and the job does not resume, which also killed all six scheduled overnight
rounds. Run it in a cloud session that stays active for the whole block.

**What a quality run means here.**

1. **Overlapped sampling, stride 7.** Every epoch then appears in three sequence positions per
   pass, so no epoch is always stuck at a window edge. This is the version that matters; the
   stride-21 run is its control, not its replacement.
2. **Random window offsets per pass.** Shift the whole sequence grid by a random 0–20 epochs at the
   start of each pass. This costs nothing and removes the last bit of positional bias that even
   stride 7 leaves. Not yet implemented: it is a small change in the dataset-building loop.
3. **25 passes**, matching every other variant, with the pass chosen on the six selection people as
   always. At roughly 206 seconds a pass that is about 90 minutes.
4. **Both arms reported.** Run stride 7 and stride 21 and publish both, with the seed-variance
   spread of 0.015 macro-F1 printed beside them, so a difference smaller than the spread is not
   read as a result.
5. **Then the external cohort.** Add the winner to `external_all.py` and score it on Sleep-EDF. The
   expectation from every other member of the family is that it collapses there too; if it does
   not, that is the most interesting result of the whole project and needs saying loudly.
6. **Then the outputs**: `metrics_summary.py`, `make_presentation.py`, `update_readme.py`,
   `make_plan_rev19.py`, `make_viewer_v3.py`, and `SHA256SUMS.txt` last.

**Honest expectation.** Published work puts context worth a few points of macro-F1, and the frozen
trunk already captured part of that. A gain of 0.02 to 0.04 would be a real result; anything under
0.015 is inside the seed spread and must be reported as no finding.

## How to resume the one unfinished item

`cnn_endtoend.py` (backlog item 8) is written, syntax-checked and was running when the session
stopped. It checkpoints nothing, so it has to be run again from the start. One command, from inside
the submission folder, with the epoch cache as its argument:

```bash
python cnn_endtoend.py <epoch-cache> 25 7     # the overlapped run that matters
python cnn_endtoend.py <epoch-cache> 25 21    # the no-overlap control
```

About 45 minutes: 12 passes at roughly 206 seconds, then the sequence-weight selection, the
temperature fit and the 20 test people. It writes `results/cnn_endtoend/` and touches nothing else.
Afterwards run `metrics_summary.py`, then `make_presentation.py`, `update_readme.py` and
`make_plan_rev19.py`, then regenerate `SHA256SUMS.txt`.

Why it is worth the 45 minutes: it is the only remaining item with a clear mechanism for a real
gain. Every other avenue tried last night — class weighting, augmentation, two channels, person
weighting, prior adjustment, decision offsets, wider heads, other context windows — either lost or
moved less than the measured seed spread of 0.015 macro-F1.

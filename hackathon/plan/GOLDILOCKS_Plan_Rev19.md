# GOLDILOCKS AI
## Uncertainty-aware wearable sleep decoding, with a retained NEURORELEASE research roadmap

**Revision 19.** Adds the submission package, a proven one-command reproduction, the exposure-model leakage audit, the CNN challenger family and its external collapse, two ablations, post-hoc adjustments, and the support slides and handout. Nothing from Revision 18 is removed or shortened.

**Competition and research specification, Revision 18 | 7 November 2026**  
**Entry: Stream 2, AI / Neural Decoding, sponsored by Tobii, as registered by the team.**  
**Status: v2 built and independently transfer-tested; stage decoding largely transferred, sleep totals did not, and calibrated-range claims are withheld. Revision 17 records the executed section 15 actions: number check passed; YASA run on both cohorts (better than v2 on Sleep-EDF, worse on the headband); the sleep-window rule did not fix the totals failure. Revision 18 adds the coating/core evidence update (C01–C10) to the research backlog as Appendix K; it changes nothing in the scored core.**

> Decode sleep from wearable EEG, then test how uncertainty and missing data affect the sleep quantities we report.

This document merges NEURORELEASE Revision 6 with **Goldilocks AI — Battle Plan v3 (AI stream)** [45]. It supersedes conflicting task priorities, lock timestamps and readiness statements in either input. Official event instructions remain authoritative. The source PDF's statements that models are tested, code is complete or data checks have passed are not adopted without the corresponding evidence. Revision 12 incorporates the AI requirements, hydrogel catalogue, matched pairs and coating/core evidence supplied afterwards; section 12 records their actual contents and support limits.

**Implementation update:** section 14 is the current execution record; section 13 retains v1 history. Revision 15 integrates the transfer-failure narrative, protected scope, provenance requirements, current Q&A and presentation acceptance checks. Revision 16 (section 15) reframes the opening as finding-first, adds a YASA baseline on both cohorts and an optional time-boxed sleep-window rule, and closes planning. Readiness and claims follow the measured endpoint-specific gates, including failed interval coverage.

## Competition brief: one challenge, a clear evidence boundary

### TL;DR and 30-second opening

**Core challenge:** estimate sleep stages from wearable EEG and test how prediction uncertainty and missing data affect recording-window sleep summaries. ADHD medication timing is a motivating future use case, not the validated task.

**TL;DR:** Goldilocks trains/adapts a sleep decoder and evaluates it against reference labels and baselines on held-out people. Its sleep-evidence viewer makes errors, missingness, uncertainty and summary definitions inspectable. The question is which quantities remain reliable under the tested conditions, not whether a child's medicine or dose is suitable.

**30-second pitch (Revision 17, after the YASA run):** “Sleep-stage decoding transfers to a new device reasonably well: ours went from kappa 0.66 on the BOAS headband to 0.58 on 20 new Sleep-EDF people, and YASA, the standard open-source stager, reached 0.68 there. Sleep totals don't transfer: on complete recordings with daytime wake, our totals were off by 236 minutes and even YASA's by 110. So our viewer shows where totals break and refuses to give ranges it can't back. On the forehead headband itself, our model beats YASA. ADHD medication timing motivates future research; today we evaluate adult EEG, not treatment response.”

Show the full comparison on one slide (section 15, Revision 17 execution record): v2 and YASA, both cohorts, both windows. The 235.9 is never hidden, and YASA's better external result is never hidden either. Use this opening before medication context. Display the full results table and cohort/window definitions beside the headline. Retain stage metrics and baseline comparisons: historical BOAS improvement is development evidence, and the independent failure is not superiority evidence. Presentation emphasis can change under section 11; the frozen analysis and population claims cannot.

**The sleep-evidence viewer is the product. The evaluated neural model is its core.** It compares neural estimates with reference and baseline, exposes unsupported summary interpretations and exports a traceable result. An optional timing/formulation overlay marks evidence boundaries; it does not demonstrate a link between unpaired EEG, drug and gel studies.

Keep **“one chain, three links: decode; interrogate; specify the next experiment”** as a workflow explanation, not a three-model promise. The earlier “ADHD medicine is a chain” hook can appear later as motivation, explicitly showing which links have not been measured together. It is no longer the default first sentence.

**Build promise:** one trained sleep model, one held-out evaluation and one working sleep-evidence view. Daytime prediction, PK prediction, medicine scenarios and the four NEURORELEASE modules are optional/context or later research; none is needed for the core demonstration to make sense. The detailed formulation, host-state and standards material is retained below.

**Current readiness:** the EEG loader, feature model, held-out evaluation, interval diagnostics and offline viewer are implemented. Original release datasets and a bounded PLGA experiment are available. Current results and remaining scientific limitations are recorded in section 14. Presentation choices cannot turn planning or a vendor-label audit into a completed model result.

## 1. One project and a clear build hierarchy

Goldilocks supplies the required neural-data experiment and sleep-evidence viewer. NEURORELEASE supplies a retained drug-evidence, compatibility, hydrogel-release and exposure-evidence research backlog. Showing these sources in an optional view does not establish a shared subject, paired study or causal treatment-response model.

| Tier | Deliverable | Requirement |
|---|---|---|
| Required AI core | BOAS forehead-EEG sleep staging and recording-window sleep summaries | Train or meaningfully adapt a model; compare with baselines on held-out people |
| Required integration | Aligned model/reference/baseline sleep results, uncertainty/missingness and recording-window summaries | The sleep task stands alone; sources, masks, definitions and actual errors are visible |
| Gated extension | Pre-stimulus prediction of task errors/slower responses from consenting teammates | Independent person-level evaluation and causal input windows; live only after technical checks |
| Retained research context | Four NEURORELEASE modules with route-specific requirements, compatibility rules, property-model gates and coating/core evidence | Section 12 defines decisions and data readiness; optional implementation/later research, independent of the core demo |
| Later research | Paired-PK exposure prediction, neuronal nanoparticle effects, individual proteome models | Separate datasets, baselines and validation; not additional compulsory hackathon models |

The former nanoparticle PK predictor is no longer the scored AI core. Consequently the entry no longer depends on interpreting PK-only tables as “neural or physiological data” under the brief: actual EEG is central [1]. The PK specification remains intact as a conditional research appendix. This change is deliberate, not a claim that the old proposal was already an accepted AI entry.

The brief requires meaningful model training/adaptation, core neural or physiological input, held-out evaluation, a baseline and at least one metric [1,44]. Hardware is optional. Disclosed simulations are allowed but do not validate biology. Teams have five or six people, enter one stream and submit one project; follow the announced pitch/submission rules. Attribute public data, pretrained models and open-source assets and check sponsor-tool terms.

The hackathon user is a researcher inspecting decoded signals, evidence and illustrative scenarios. ADHD medication review is a future clinical application hypothesis. BOAS adults and teammate task recordings do not validate home monitoring in children, medication titration, clinical sleep assessment or personalised release design. The plan does not diagnose ADHD or infer health conditions from people at the event.

The final scope contains one required trained model, not three mandatory models. The day model cannot delay the night-model floor result. The minimum product remains complete without a drug/gel view; when that view is present it reports its own evidence limits and cannot inherit validity from sleep-model performance.

## 2. BOAS source check and what is actually known

The public repository identifies BOAS as simultaneous PSG/headband recordings, with expert-consensus PSG stages and released automatic scores [46]. We independently inspected participants.tsv, dataset_description.json, channel/event definitions, one headband acquisition sidecar, and all 256 PSG/headband event TSVs. The repository tree observed was `face09565c4337ddbce7a048c9b02060ad20df45`; the dataset description identifies DOI `10.18112/openneuro.ds005555.v1.1.3` and CC0. Pin the actual snapshot used by the build.

| Independently checked item | Result | Consequence |
|---|---|---|
| Recording/person identities | 128 recording rows, 100 distinct pid values; 20 repeat participants account for 48 nights | Split by pid, never by recording name |
| Annotation timing | All 128 paired event files have matching epoch counts, onsets and durations | Annotation alignment checked; EDF sample/time alignment still needs loader checks |
| Epoch/reference labels | 30-second annotations; human stages 0-4 and disconnection code 8 | Use the human reference, not stage_ai, as training labels |
| Human disconnections | 336 epochs across 37 nights | Exclude unknown reference labels from stage scoring; report lost time |
| Released headband abstentions | 4,869 predictions are -2 among 119,759 valid-reference epochs | Do not hide these by reporting conditional accuracy alone |
| Released-score benchmark | Coverage 95.93%; conditional accuracy 86.74%, kappa 0.769 on 114,890 valid predictions | This is a reproduced full-release reference statistic, not our model's held-out result |
| Example headband metadata | HB_1/HB_2, 256 Hz, microvolt channel units, left-mastoid reference | Verify each actual EDF and preserve units/reference |

The repository describes an adult general-population cohort with exclusions, not a paediatric ADHD medication-response cohort. Do not simplify this to proof that every participant is disease-free. The global annotation audit has already been performed; we must not claim that test labels were never inspected at all. The held-out promise is that designated participants are not used for fitting, tuning or performance-driven decisions.

Human-valid class counts in the inspected release are Wake 19,137; N1 4,462; N2 72,181; N3 5,225; REM 18,754. These are annotation counts, not independent sample counts. At the original annotation audit, raw EDF signals were not downloaded or checked. Section 13 records the subsequent completed acquisition and signal checks. Optional pulse/motion availability, total signal-download size, usable signal coverage and any lights-off marker in additional material still require acquisition checks. No clinical lights-off marker is established by the inspected event schema.

Acquire one paired record first and verify signal length, channel names, calibration/units, resampling and annotation boundaries. Then use a fixed participant manifest and an explicitly described acquisition subset. Selecting a manageable initial subset must not use favourable model scores. A small technical run is not the full study result. Treat the PDF's suggested 40-person start as a development budget, not a validated sample-size threshold.

## 3. Required night model: precise experiment

### Task and comparisons

Predict Wake, N1, N2, N3 and REM from headband EEG using human consensus in the PSG event file as reference. No PSG signal, human stage, released stage_ai or target-derived summary enters the headband predictors. Use available demographics only if prespecified and compared fairly; do not add them merely to improve a test result.

The mandatory trained floor model is a compact EEG-feature classifier with a prespecified feature set. A small CNN with limited temporal context is the primary challenger if time and compute permit. This ensures an evaluated trained model is feasible without relying on foundation-model fine-tuning. CBraMod/LaBraM or another public model is a credited stretch, subject to actual licensing, channel compatibility and compute.

Compare against a training-derived majority-stage baseline, the released Bitbrain headband scores, and YASA where its inputs are supported. YASA's documented preference is a central derivation; forehead use is a domain limitation, not a perfectly matched reference [47]. Do not choose its channel using held-out results. Run its documented preprocessing path separately: its guidance discourages pre-filtering or z-scoring before SleepStaging. The team's own model can have a different prespecified preprocessing pipeline.

Released Bitbrain scores have their original cross-validation/training provenance. They are a reference comparator, not a model retrained under our split. Record what is known about person-wise independence; do not silently equate the published procedure with ours. Their hard labels do not supply probabilities for temperature scaling.

### Splits, temporal context and missingness

Freeze a participant-wise train/calibration/test manifest before new-model training. Tune only inside training participants, with grouped validation. A practical target is approximately 60/20/20% by person when the available cohort supports it; report actual counts and all subset/exclusion rules. Every repeat night stays with its person.

For confirmatory sleep-summary uncertainty, use one prespecified night per calibration/test person or a justified participant-level aggregation. Report additional nights with clustered analysis. Never treat thousands of correlated epochs as thousands of independent calibration subjects. If calibration groups are too few for a finite interval at the desired nominal level, report that limitation rather than a nominal guarantee.

Validate EDF-to-event indexing after resampling, reject out-of-range epochs and reset temporal context across missing segments or recording boundaries. Keep signal-quality rules independent of model success. Reference code 8 is unknown; headband score -2 is an abstention. Neither becomes Wake. Report signal/reference/prediction availability separately.

The floor analysis is retrospective full-night scoring. If the model uses future neighbouring epochs, label playback retrospective; animating it does not make it real-time prediction. A causal variant requires past-only preprocessing/context and its own latency/accuracy evaluation. Do not transfer a night model to the day headset without a separate study.

### Outcomes and uncertainty

Primary stage results are held-out-person kappa and macro-F1, with per-stage F1, confusion matrix, participant counts and coverage. Compare paired participants, not only pooled epochs. Report both coverage-conditioned metrics and performance/coverage on the full valid-reference set; a scorer that abstains on difficult epochs must not appear superior solely because those epochs disappear.

Define sleep summaries before fitting:

- **Recording-start to first scored sleep:** time from the recording origin to the first N1/N2/N3/REM epoch under the fixed rule. This is not lights-off sleep-onset latency.
- **Valid-window total sleep time:** accumulated scored sleep duration over the declared observation window, with unknown time reported separately.
- **Post-onset wake in the recording window:** Wake after the defined first-sleep point, using a declared endpoint. Do not call it a full lights-off-to-lights-on clinical WASO measurement without those markers.

If missing epochs make first sleep or total sleep uncertain, show missing-data bounds or withhold the complete-window value. Use the same observation definition and declared mask for model/reference comparison. Distinguish these bounds from statistical model intervals.

Temperature scaling applies to model probabilities and is assessed on separate people. Epoch confidence does not automatically create valid uncertainty for a whole-night summary. For summary intervals, use a justified calibration method on independent participant-level errors. State nominal level, observed coverage, interval width and calibration size; temporal dependence and domain shift limit guarantees. A person's label sequence is not regenerated as independent samples to manufacture coverage.

Report summary MAE in minutes, signed bias and a clearly defined paired-error/Bland-Altman analysis with participant-aware uncertainty. Improvement in stage kappa is not automatically improvement in these summaries. If the challenger fails to beat the trained feature baseline, ship the reproducible baseline and report the failure.

### The central uncertainty experiment

Stage confidence and summary reliability are different hypotheses. On the same held-out people, assess (1) stage discrimination and probability calibration, (2) error in the prespecified recording-window summaries, and (3) empirical coverage and width of supported summary intervals. A lower stage error or calibrated epoch probabilities do not automatically establish reliable total-sleep-time intervals.

Keep calibration roles separate. Fit temperature scaling, if used, on grouped validation people inside the training allocation; freeze the point predictor and its probability transformation before using the dedicated participant-level summary-calibration group. Do not fit that predictor/temperature on the same people whose residuals are then treated as untouched calibration errors. Final test people fit neither component.

The default summary-interval candidate uses absolute summary residuals from independent calibration people, one prespecified night per person or a justified person-level aggregation. Evaluate actual interval coverage and width on separate test people; report calibration size and the limits of any exchangeability assumption. Where a nominal level cannot be supported with a finite useful interval, show the limitation. Compare point/summary errors alongside interval results so a very wide interval is not marketed as a useful improvement merely for covering the reference.

A stage-probability-weighted summary is at most a prespecified alternative point estimator; adding independent epoch samples does not create valid whole-night uncertainty. Keep unknown-time bounds separate from prediction-error intervals. Assess a baseline under the same observation rules and document training/probability limitations of released comparators.

One optional, prespecified sensitivity analysis can mask contiguous valid signal windows and rerun the declared pipeline to measure changes in summary error, availability and interval behaviour. Choose gap rules/seeds without final-test outcomes and use identical masks for comparable methods. Label this controlled missing-signal simulation. It is not validation under a different device, age group, ADHD population or natural distribution shift. Cut this extension before the main trained-model and baseline evaluation.

## 4. Optional day experiment and live gate

The day task predicts a future behavioural outcome during a short go/no-go task. “Attention slip” is a convenient task label, not a validated diagnosis or a direct measurement of a hidden clinical attention state.

Freeze the task schedule, event timestamps, response deadline and outcome definitions. Distinguish commission errors, omissions and slower responses. A personalised slow-response threshold may come from that person's calibration block only; never from their test outcomes. If the proposed 80th-percentile rule is used, record it as a task-design choice, not a clinical cutoff.

Features can include a fixed pre-stimulus EEG window, validated blink proxies, head motion and past response history. Nothing after the prediction timestamp, the current response or future target type enters prediction. Fast trials may make windows overlap previous stimuli/responses; measure that overlap and use adequate timing/chronological guards. Frontal EEG blink features are not a Tobii eye tracker or a verified eye-movement measurement.

Verify the actual borrowed device, montage, sample rate, units, SDK/LSL support and motion availability. Webcam acquisition is optional and separately consented, with no bystander recording. Headset hardware in the supplied PDF is a proposed resource, not a confirmed booking.

Use an interpretable feature model first. Compare response-history-only, EEG-added and eye/motion-added variants on the same held-out people. Keep all sessions from one teammate together. Distinguish zero-shot leave-person-out evaluation from adaptation using the held-out person's first calibration block; evaluate the remaining chronological block only. Calibration-time curves need the same held-out participants and test block.

Report per-person class counts, AUC where defined, precision/recall at an alert threshold set without test optimisation, alert rate and latency. Report undefined AUC when a person has only one class; do not silently drop them. Pooled AUC does not replace participant-wise results. Any null test must respect session/time dependence. A few teammates provide a proof of concept, not population or clinical validation.

Record only consenting teammates, make acquisition visible, keep recordings local under the agreed retention choice, and permit stopping. Do not infer ADHD or other health traits. Three successful live rehearsals test reliability, not predictive validity. If acquisition, evaluation or reliability fails, use a labelled recorded replay or omit the day act; never fabricate live EEG or performance.

## 5. The timing overlay and medicine layer

This optional overlay is an evidence-boundary/scenario interface, not the minimum product. It does not establish that a drug caused a measured sleep or performance pattern. BOAS, teammate recordings, product-label studies and hydrogel assays are unpaired sources. Do not combine them into the record of one real child or suggest that the same participant supplied them all.

Use separate provenance labels: **PUBLIC ADULT EEG**, **CONSENTED TEAM TASK**, **LITERATURE/ASSAY EVIDENCE**, **MODEL ESTIMATE**, **SIMULATED SCENARIO**. An entire scenario using assumed drug-response relationships remains simulated even if some illustrative parameter came from real data. Source, population, device, time origin, units and confidence/missingness status stay visible.

Drug what-ifs leave measured EEG, actual behaviour and reference labels unchanged. Only clearly simulated curves may move. Do not say “this child's dose does not fit” or “the mismatch closes” as an experimentally established result. A hypothetical scenario can illustrate a question to study, not recommend a dose or establish a better treatment.

Keep three layers separate: release from the dosage form, systemic pharmacokinetics, and pharmacodynamic/clinical outcomes. Half-life/onset/Tmax summaries do not identify a complete concentration-time or response curve. A simple first-order compartment curve is a labelled model with explicit parameters and assumptions, not a measured patient trajectory. Use normalised shapes when absolute parameters are unsupported.

The starting evidence catalogue can cover methylphenidate IR, a named methylphenidate ER product, lisdexamfetamine, atomoxetine and guanfacine ER. Every actual value needs the exact formulation, source label, population and date. Product-specific extended-release behaviour must not be replaced by a universal two-peak curve. Lisdexamfetamine needs its prodrug-conversion distinction. Atomoxetine and guanfacine still have time-varying PK; slower treatment-response timescales do not justify drawing their concentrations as flat [50,51]. No profile yields a clinical effect threshold by itself.

Treatment history is metadata and a future covariate. A reported DAT change does not provide a numerical individual sensitivity function. No literature-derived receptor multiplier is inserted into a simulated child and presented as fitted physiology.

For the hydrogel module, a target release profile is a declared research constraint or an explicit scenario assumption. It is not a prescription derived from BOAS or a three-minute game. Tablets, nasal gels, local depots and different drugs cannot be ranked as interchangeable treatments from in-vitro release alone. Compare only compatible assay profiles, or show model-dependent scenario comparisons with the assumptions visible. No optimiser chooses a patient dose, administration time or an invasive delivery route.

## 6. Build execution and retained formulation research

The required implementation is a trained sleep decoder, person-held-out comparison and sleep-evidence viewer. The formulation rationale and four modules in Appendix A remain a research backlog, with source audits and acceptance cases preserved. An atomoxetine NLC/in-situ-gel card is optional context; its dementia-like animal-model outcomes do not validate ADHD benefit or improve the EEG model's evidence.

The complete core path is: select a frozen held-out night; compare predictions/reference/baseline; inspect an uncertain or missing interval; inspect the resulting summary, error and supported uncertainty; export test conditions and the next measurement. A clinical-sounding summary unavailable from the source, such as lights-off latency without a lights-off marker, is a useful unsupported-case demonstration inside this same task.

| Gate | Required result | Scope or fallback |
|---|---|---|
| 0: task and claims | One EEG question; frozen analysis/claim contract, roles and source terms | Keep day/PK/formulation work outside the core dependency |
| A: data/loader | One EDF/reference pair aligns; units, quality, labels and person manifest verified | Resolve loading or use a justified manageable subset without performance-based selection |
| B: trained floor | Trained feature sleep model versus declared baselines on held-out people | Cut architecture stretch; retain/report the actual result, including failure |
| C: uncertainty/summaries | Prespecified summary definitions, missingness and independently evaluated uncertainty where supported | Show supported errors/bounds; remove unsupported interval or clinical-summary claims |
| D: sleep-evidence viewer | Aligned results, source/definition reveal and summary/error path work end to end | Use a simpler working sleep-only view; no medicine card is needed |
| E: optional extensions | Day acquisition, missing-signal sensitivity or formulation view only after their own checks and core capacity | Omit the extension without changing the core question |
| F: submission | Frozen result artefacts, credits, repeatable permitted demo and limitations | Use actual organiser timings and a truthful fallback |

Assign core responsibilities first: data/loading; model training; independent evaluation/calibration; summary/viewer integration; reproducibility/demo. With six people, add a quality/source-review role or share it across those tasks. Day acquisition and formulation implementation start only after core owners confirm capacity. Bora can lead integration and Jacob the retained biology/formulation material if agreed; actual names and contributions must be confirmed.

All project code must respect the event window [1]. Public libraries/data/models need attribution; planning sketches and pre-event work are not newly written event code. No model training, live predictor, medication-response result or working viewer is produced merely by this plan revision.

## 7. Definition of done, demonstration and remaining work

The required result is one implemented EEG model with a genuine person-held-out comparison and a working sleep-evidence viewer. Night stage/summary results must show actual participant counts, metrics, coverage and failures. Optional models have their own tables; their absence does not get replaced with invented values. Drug/gel acceptance rows below apply only if that optional view is included; they do not make it part of the minimum product.

| Check | Expected behaviour |
|---|---|
| Repeat nights from one pid | Always remain in one split |
| PSG consensus versus stage_ai | Human reference is the target; automatic labels never leak into predictors |
| Code 8 or scorer -2 | Unknown reference or abstention handled explicitly; neither is Wake |
| Vendor baseline coverage differs | Report coverage and comparable-mask/full-reference results |
| Lights-off unavailable | No clinical sleep-onset-latency claim |
| Noncausal night context | Playback labelled retrospective |
| Small calibration set | No unsupported finite interval or nominal-coverage guarantee |
| Day timestamps/labels unavailable | No pre-stimulus prediction claim |
| Too few day errors | Counts and undefined metrics visible |
| Drug what-if activated | Measured EEG/behaviour remains fixed; only scenario curves change |
| Unpaired source populations | No fictitious real patient joining them |
| Hydrogel curve without PK/PD data | Release-only output; exposure/benefit remain unknown |
| Source conflict or unsupported formulation | Evidence request rather than a confident ranking |
| Model loses to baseline | Report it; no AI-superiority claim |

Use the three-act choreography in section 9: decode the night, challenge the evidence, and inspect the resulting summary/report. The optional day recording is an insert after its own gate passes; the main story works without it. Say the population and simulation status aloud. The judge should be able to reveal reference labels, compare a baseline, inspect a source and trigger an unsupported case. A scrolling precomputed hypnogram is a replay, not proof of live inference. Keep a recorded/offline backup and return borrowed kit under organiser rules.

The scientific claim is about the implemented neural task and evaluated summaries. The optional medication/gel connection is research context and a hypothesis, not measured treatment benefit. Neither showing confidence intervals nor adopting a careful evaluation protocol establishes novelty alone: compare with relevant sleep systems and demonstrate a useful empirical finding or improvement in summary error, uncertainty or evidence inspection. The target is a strong, inspectable result rather than an unsupported claim of being first.

Future validation needs paired day/night signals, medication/formulation timing, adherence, relevant outcomes and confounders from the same participants. Clinical use would require a separately designed supervised study and qualified intended-use/regulatory assessment. Person-level proteome or genotype effects need paired delivery outcomes; the autism/OCD/ADHD maps do not supply those coefficients.

**Completion status:** the two plans are one specification, with a judging-criteria map, three-act demonstration, visible provenance, readiness board, Q&A and the evidence/rehearsal checklist in section 10. Handouts, video, rehearsals, outreach and measured headline results are planned work, not completed outputs of this update. The core question and default pitch are now both wearable sleep decoding and summary reliability; PK-only eligibility cannot determine the entry's core validity. Raw-signal loading, training, held-out evaluation and working integration still require execution. The older PK model, additional neural nanoparticle experiments and personalised delivery predictions remain gated research work, not promised hackathon results.

## 8. Merge decisions and source traceability

| Topic | Goldilocks contribution | NEURORELEASE contribution | Merged decision |
|---|---|---|---|
| AI core | EEG sleep model; optional task predictor | Conditional measured-PK model | EEG is required; PK modelling is later research |
| ADHD story | Day/night medication timing | Delivery rationale and payload evidence | Motivation after the sleep-core opening; no demonstrated child treatment response |
| Medicine layer | Label-based timing scenarios | Dated existing/emerging-drug evidence | Exact product/source/date and explicit PK/PD assumptions |
| Hydrogels | Release-profile extension | Loading, compatibility, materials and mass balance | Retained optional research backlog; not a minimum demo dependency |
| Validation | Person-wise splits and uncertainty | Source-group validation, leakage/domain checks | Apply each independence unit to its own task |
| Disease/proteome work | Background and roadmap | Compartment/model-specific evidence maps | Retained as typed evidence; no personal uptake multipliers |
| Readiness claims | Source PDF says locked/tested/data passed | Prior plan records unresolved empirical work | Only independently checked metadata/annotations are reported as completed here |
| Standards | Adds medical-software lifecycle and risk management | Existing ISO/MIRIBEL mapping | Conditional ISO 14971/IEC 62304 roadmap added; no certification claims |

Goldilocks v3 was read across all 20 pages [45]. All prior project-source dispositions remain in Appendix I, updated for this merge. In particular, the full formulation/safety/host-state material is retained below rather than disappearing into an unsupported “future work” sentence. Its presence does not mean every module has already been coded or validated.

## 9. Competition execution: judging evidence, demo and Q&A

### One user, one research question

Keep the primary user a neurotechnology researcher evaluating wearable sleep stages and their derived summaries. The task is to inspect a decoded night, compare stated uncertainty with observed error, and identify the measurement needed for a stronger conclusion. The initial output is a traceable sleep-analysis report and next-measurement brief, not a treatment decision.

The main screen is a sleep-evidence viewer: neural estimates, reference/baseline, uncertainty/missingness and recording-window summaries. An optional evidence drawer can expose the retained formulation research. That drawer marks absent cross-domain evidence rather than implying integration of unpaired biological studies.

### Impact and route to adoption

Initial adoption is a reproducible sleep-analysis and reliability-inspection workflow for neurotechnology research teams. A later ADHD application would start with a clinician-supervised paired-study pilot, collecting signals, medication timing and outcomes before any treatment-response claim. The initial product is the sleep-evidence viewer and report; a later clinical product could support medication titration and review only after the required validation and intended-use/regulatory work. A proposed business model is a software licence sold to clinics per patient per titration period, using compatible existing hardware rather than requiring the team to manufacture a headset. This is a commercial hypothesis, not established demand or an agreed price: validate workflow fit, willingness to pay, procurement and device compatibility with prospective users. Do not invent market size, revenue, savings, pricing or clinical-benefit numbers. Keep the living drug-evidence tracker and formulation research as later extensions of the same workflow.

### Evidence for the five official judging criteria

These are the exact criterion names in the supplied brief [1,44]. No weights or predicted scores are assumed.

| Criterion | Evidence to prepare | Failure to avoid |
|---|---|---|
| Scientific validity | Trained decoder, frozen person split, matched baselines, calibration/summary errors, coverage and missingness | Presenting model confidence as established reliability or a simulated medication effect as measured |
| Feasibility and implementation | One repeatable trained-model result and an end-to-end view; working offline/replay backup | Relying on three unfinished models or a hardware booking |
| Originality | A specific, evaluated improvement or useful evidence-inspection workflow compared with existing approaches | Calling a dashboard, two common tasks or uncertainty alone novel |
| Impact and marketability | A clear sleep-research user, traceable summaries and a realistic later paired-study route | Claiming clinical savings, ADHD benefit or willingness to pay without evidence |
| Presentation and demonstration | Three acts, judge-operated reveal, visible provenance and a controlled unsupported case | A demo that collapses when an optional model is absent |

For a small usability check, ask an available domain mentor to complete prespecified tasks: find the source for one value, identify which panel is simulated, and identify the missing evidence for a medication-response claim. Record the answers and confusion. This is formative workflow feedback, not clinical validation, a representative user study or proof of market demand.

### Three-act demo that survives scope cuts

Use a three-minute rehearsal cut only if compatible with the announced pitch format. The timings below are rehearsal allocations, not event rules. Freeze model outputs and selection rules before rehearsing.

| Act | Rehearsal allocation | Judge action and visible result |
|---|---|---|
| 1. Decode | 0:00-0:55 | Pick a case from the frozen held-out set; reveal the human reference, baseline and model errors. Show actual cohort size, stage/summary metrics and coverage. |
| 2. Challenge | 0:55-1:40 | Inspect an uncertain/missing segment, reference disagreement and coverage. Ask for lights-off latency; if no marker exists, the viewer gives only its supported recording-start measure. |
| 3. Summarise | 1:40-2:40 | Inspect total-sleep-time error, missing-time bounds and independently supported prediction interval where available. Export source, definitions, test conditions and next measurement. Optional medicine scenarios belong in spare time or Q&A. |
| Close | 2:40-3:00 | State what was measured, what was modelled, what was simulated and what the next paired study must establish. |

An actual measured gel curve is displayed as an assay observation under its assay conditions. An assumed kinetic curve is labelled simulation. Neither is called measured in-vivo exposure. Do not put real public EEG and a hypothetical medicine curve into a fictitious child's patient record.

For the reveal, the judge chooses among available held-out cases after outputs are frozen. A worked example can be preselected for explanation, but name that selection and keep aggregate results and failure cases accessible. Avoid using a particularly good case as a substitute for the held-out table.

The core unsupported case requests clinical lights-off sleep-onset latency without a lights-off marker. Retain the supported recording-start-to-first-sleep value and name the missing marker. An optional gel-to-ADHD-benefit question can demonstrate a second boundary, but is not required for the core story.

### Provenance is part of the visible product

Use two separate labels so source and runtime are not confused:

| Dimension | Allowed labels and meaning |
|---|---|
| Origin | PUBLIC ADULT EEG; CONSENTED TEAM TASK; LITERATURE/ASSAY; SIMULATED |
| Execution | LIVE ACQUISITION; LIVE INFERENCE; RETROSPECTIVE REPLAY; PRECOMPUTED RESULT |
| Derived output | MODEL ESTIMATE, with model version, task and relevant uncertainty/coverage |
| Scenario status | SIMULATED WHAT-IF, persistent on the panel and any exported image/report |

A live acquisition badge says only that samples are arriving. Live inference means the displayed estimate was computed from available inputs at that time; it is not proof of predictive validity. Retrospective night context must never receive a causal live-prediction label.

If an optional medication scenario is shown, say: **“This is a hypothetical timing scenario. We have not measured how this medicine changes this person's sleep or ADHD symptoms.”** Do not say “this child's dose does not fit,” even with a small simulation banner elsewhere. Measured EEG, reference stages and recorded task responses remain fixed when a scenario control changes.

### Sponsor alignment with an identifiable experiment

Tobii sponsors the AI stream, whose examples include EEG, eye movements and multimodal prediction [1]. Explain the relevance of the optional eye-signal experiment rather than implying use of Tobii equipment.

If the day experiment runs, prespecify response-history-only, EEG-only, blink-proxy-only and combined comparisons where data support them. Keep held-out people and prediction windows identical. The useful question is whether eye-related information adds predictive value, and whether apparent EEG performance is actually dominated by ocular artefacts. Keep proxy extraction and its limitations visible.

A verified eye tracker could add measured blink/gaze features after device access, SDK/licence, calibration, timestamps and consent are checked. Forehead-EEG blink proxies remain labelled proxies. No sponsor-hardware claim is made without actual use. Neither eye acquisition nor a day-model result is required for the main demo.

### Gate 0 and the readiness board

**Scope, ranked: cut from the bottom: day model, then extra modules; never the night model or its baselines.** Extra modules include additional drug/material catalogues, added scenarios and conditional exposure-model work. Keep the core evaluation and complete sleep-result/summary inspection path; no drug/gel card is compulsory. Protecting the night model means reducing architecture complexity before sacrificing its trained-model result, person-held-out evaluation or baseline comparisons; it never means inventing a successful result.

Before parallel feature work, confirm the single EEG-core entry, the dataset/source licences, the raw-loader plan, model ownership and scope cuts. Freeze the headline-result contract and analysis choices described in section 10. Brief an available mentor on the full task, evidence, boundaries and missing measurement, not only the optional PK-eligibility question. The brief's stated stream lock is 2pm; use the organisers' actual event-day instructions and avoid reusing an old 14:00/14:10 readiness timestamp.

If proposing the exposure model as part of the assessed AI contribution, ask the stream mentor: **“Would a model using measured systemic drug exposure as a core input count as physiological data under 'or similar' in this stream?”** Record the exact task description and answer. This question has not been answered by the present document.

A yes does not establish data compatibility, validation or build capacity. A no does not threaten the EEG entry. Neither answer expands the default plan into three compulsory models. In this specification the exposure model remains conditional research unless the team explicitly revises scope after satisfying its data/evaluation gates.

| Readiness item | Status at this revision | Evidence required to change status |
|---|---|---|
| Event brief and judging criteria | Reviewed | Source-linked brief and current organiser instructions |
| BOAS metadata/annotations | Audited | Completed source audit described in section 2 |
| Raw EDF loader and usable waveforms | Implemented and checked; basic quality mask only | All 128 BOAS acquisitions/loader checks and 20 independent external signal/reference pairs; sections 13–14 |
| Trained EEG model and evaluation | V1/V2 implemented; endpoint-specific failures retained | Saved weights/protocol, historical development comparison and independent transfer result; no Bitbrain-superiority or reliable-interval claim |
| Integrated sleep-evidence view | Offline sleep viewer implemented and browser-checked; drug/scenario integration not implemented | Two-cohort selection, BOAS comparator tracks, reference reveal, claim rejection and JSON export; no live-inference claim |
| Day acquisition/prediction | Optional, pending | Device/consent/timestamps, evaluation and rehearsals |
| YASA external baseline | Done (Revision 17): run on both frozen cohorts after v2 reproduction passed | `results/yasa/`; section 15 execution record |
| Label-free sleep-window rule | Done; did not fix the failure; moved to next work | `results/window_rule/`; exploratory label on Sleep-EDF |
| Number check (slides, plan, viewer) | Done: all quoted numbers match saved files | `results/yasa/v2_reproduction_check.json` |
| Viewer with YASA lane | Done: separate v2.1 copy, browser-checked | `results/yasa/Goldilocks_Validation_Explorer_v2_1_yasa.html` |
| Pitch card, Q&A bank, credits text | Drafted; updated with YASA | Team rehearsal evidence |
| Coating/core evidence update (C01–C10) | Recorded as research backlog (Appendix K); nothing fitted; described code not found on the Mac | Study-count gate: 10 studies (continuous) or 5 per group (yes/no) |
| Exposure prediction | Conditional research | Eligible source-group corpus, any relevant mentor decision, model evaluation |

Keep stage-specific statuses. “Annotation audit passed” does not become “dataset ready” or “model validated.” Implementation checks record actual outcomes and times once performed, not planned pass statements.

### Team handoffs and definition of done

The evaluation lead freezes the person manifest and scoring contract before fitting; the model lead delivers predictions with IDs, labels, scores and availability; integration consumes this stable result contract. Optional medicine/formulation work supplies a traced anchor card only after core capacity is protected. Day acquisition cannot borrow core model/evaluation time at the expense of Gates A-D.

The final evidence pack contains the model/task card, split manifest, baseline/result table, summary error and coverage definitions, source snapshot, scenario assumptions where applicable, acceptance log, reproducible run instructions and offline demo backup. Sections 13–14 identify the delivered technical artefacts. Video, public repository/QR, printed handout and rehearsals remain separate tasks until completion evidence is recorded.

Section 7's neural/integration checks form the core acceptance register. All **16 delivery-workbench cases in Appendix J** remain in a conditional register for any implemented formulation view/model. Mark each as passed, failed, pending or not applicable with a reason and actual evidence. Conditional-model cases are not marked passed merely because that model was omitted.

### Short answers for difficult questions

| Judge question | Prepared answer |
|---|---|
| “Isn't this two projects?” | “The required project is wearable sleep decoding and evaluation of its summary reliability. Drug/formulation material is optional research context; we do not claim a measured bridge between those studies.” |
| “What is actually new?” | “Sleep staging and calibration are established. We implemented a workflow that tests and reports stage confidence, missingness and sleep-summary errors together, including a frozen external transfer test. Its failed coverage changes what the viewer permits us to claim. We do not claim a first method or universal failure detector.” |
| “Why ADHD if BOAS contains adults from the general population?” | “ADHD medication timing motivates a future study. Today we evaluated an adult EEG task and measured important limits. We have not validated paediatric ADHD prediction, medication response or treatment timing.” |
| “What did you build rather than reuse?” | “We identify the event-written training/adaptation, evaluation and integration code, credit public data/libraries, and show the result artefacts. Planning notes and existing components are not represented as new event code.” |
| “What if the night challenger loses to the baseline?” | “We report the loss. We keep the trained feature-model result and whichever supported comparison is informative; we make no superiority claim. The interface does not conceal the failure.” |
| “Where are the nanoparticles and hydrogels?” | “They remain in a source-audited research backlog. If the optional view is implemented, a formulation card can show loading/release evidence and its gaps. The sleep task stands alone; new delivery-response predictions require matched data.” |
| “Is this using Tobii?” | “Tobii sponsors the stream. We name only hardware and software actually used. EEG-derived blink proxies are not an eye tracker; a measured eye-feature extension needs separate acquisition and evaluation.” |
| “Does changing this curve improve sleep?” | “No such effect has been measured here. The control changes a hypothetical formulation/timing assumption. The observed EEG stays fixed; benefit needs a paired, supervised study.” |
| “Is the exposure model eligible?” | “EEG makes our core eligibility clear. We would document a mentor's interpretation before treating PK-only input as an additional assessed AI task. Approval would not substitute for a valid dataset.” |
| “Are the data/model ready?” | “Raw loading, V1/V2 training, saved evaluation and an offline viewer are implemented. V2 improves historical stage metrics but still trails Bitbrain and fails sleep-time interval coverage. Twenty independent external participants expose transfer limits. Video, QR/handout and rehearsal have separate completion checks.” |
| “Does the AI know when it is wrong?” | “Not generally. Reference labels let us measure failures retrospectively. The viewer withholds calibrated ranges because evaluated coverage failed; that is an evidence-based release rule, not a validated per-recording error or out-of-distribution detector.” |
| “Did a different device cause the external error?” | “We cannot isolate that cause. Device, montage, scoring, population and recording-window composition differ. Primary complete-recording MAE is 235.9 minutes; the prespecified secondary sleep-window MAE is 36.8 minutes. Both remain visible.” |
| “Why show a failure?” | “It changes the product claim: a stage-decoding gain does not establish reliable sleep totals. The viewer makes this inspectable. Reporting failure does not make our model better than a stronger baseline.” |
| “Who would adopt this first?” | “A neurotech research team inspecting wearable sleep estimates and their limits. A later paired timing study is a separate step; clinical deployment and market demand are not yet established.” |


## 10. Evidence, rehearsal and submission checklist

This section turns the supplied brainstorming list into build/rehearsal tasks. It adds no compulsory prediction model and does not authorise outreach, public posting, participant recruitment or endorsements. The presentation/rehearsal tasks below remain planned unless a completion record says otherwise; section 14 records implemented technical checks. Evidence and a working EEG core come before presentation polish. A working viewer does not imply a recorded video, public repository, printed handout or finished rehearsal.

### One headline result, selected without cherry-picking

Use one memorable number, with the comparator and denominator beside it. The default presentation endpoint is **mean absolute error in recording-window total sleep time, in minutes, averaged by held-out person**. Freeze the window, masks, repeat-night aggregation, eligible participants and one compatible baseline before looking at the final test comparison. Keep kappa and macro-F1 as the scientific primary stage metrics already specified in section 3; the presentation endpoint does not replace them.

The eventual sentence reports the actual model error, actual baseline error and held-out person count. Add the paired difference and participant-level uncertainty where supported. Missing time and the sleep-window definition remain visible. It is not a clinical lights-off-to-lights-on sleep measure unless those markers exist. If this endpoint proves technically unavailable, document the reason and a replacement such as person-averaged kappa before final test evaluation, not after searching for a favourable result.

At Gate B, fill the headline from the actual frozen comparison and write the pitch around its observed result. Do not select N1 F1 merely because it is the only improved class, or compare a held-out team result with the full-release conditional N1 statistic as though the populations and scoring masks were identical. The audited released-score N1 F1 of approximately 0.34 is context, not a fair matched comparison by itself.

If there is no improvement, say so. A separately labelled coverage/error audit can be a useful secondary finding, but “we abstain, therefore we are honest/safe/better” is not an evaluation. Set rejection thresholds using training/calibration people, report retained coverage, conditional error and full-reference outcomes under a declared abstention rule, and show the unfiltered comparison. An audit of someone else's released labels cannot substitute for the required trained/adapted neural model.

### The same-night visual and failure gallery

The implemented BOAS diagnostic figure has four aligned lanes: expert reference, released Bitbrain, frozen v1 and frozen v2. The external figure has reference and v2; Bitbrain/YASA are unavailable there. YASA was not run and is not a curve to manufacture. Use the same participant/night, epoch boundaries, time axis, stage colours and declared scoring window. Show unknown reference and scorer abstention as distinct gaps/marks rather than Wake. If YASA cannot run appropriately, label that lane unavailable and explain why; do not manufacture a fourth curve.

The judge physically selects a night or opens an evidence card where the format allows. Predictions are frozen beforehand; the click changes the inspected case, not the model. An offline version must support the same interaction. If judges cannot operate the device, offer the choice verbally and let the operator perform the selected action.

Add up to three annotated error examples, selected by a stated rule such as the first eligible instance of each prespecified confusion type. Show reference, prediction, raw-signal window, confidence/availability and impact on the summary. Include the aggregate confusion matrix so the gallery is not the result. An explanation like “blink artefact caused this” requires supporting evidence; otherwise distinguish the observed disagreement from a candidate explanation. If fewer than three qualifying cases exist, show the available cases rather than inventing errors.

### Highest-priority delivery pack

Do these after protecting Gates A-D. Roles below are responsibilities within the existing five/six-person team, not extra team members.

| Priority | Task and owner | Completion evidence |
|---|---|---|
| 1 | Headline-result contract; evaluation lead | Endpoint/comparator/window frozen before final test; actual result, denominator and uncertainty available |
| 2 | Same-night comparison and judge click; model + integration leads | Aligned lanes, gaps/abstentions visible, working held-out case selector |
| 3 | Rehearse the unsupported case; formulation + pitch leads | A working evidence rule, a source and a concise spoken explanation of the exact missing measurement |
| 4 | Two adversarial mock Q&A rounds; evaluation + pitch leads | An outside volunteer challenges limitations; unanswered questions and corrections logged after each round |
| 5 | One-page printable handout and verified repository QR; integration lead | Final headline, demo summary, provenance, main limitation, source/repo link and revision/commit identifier; QR opened successfully from another device |
| 6 | Approximately two-minute edited backup video; demo operator | Actual working result capture, readable labels, source/runtime status retained, offline playback tested on the presentation device |
| 7 | Two timed 30-second pitches per teammate; pitch lead/timekeeper | Each person delivers the core claim, scope and simulation boundary from memory; record only with their consent |
| 8 | “What we cut and why” slide; technical lead | Actual scope decisions and reasons; night model, held-out evaluation and baselines protected |
| 9 | Full-plan technical mentor briefing; relevant leads | Core task, validation, boundaries and missing measurement explained; feedback and any eligibility interpretation recorded |
| 10 | Failure-case gallery; evaluation lead | Up to three genuine errors with cautious interpretation and their relation to aggregate performance |

For the core unsupported-case rehearsal, the judge asks for lights-off sleep-onset latency. If that marker is absent, say: **“This recording does not establish lights-off. We can report recording-start to first scored sleep, but clinical latency needs the missing marker.”** An optional formulation scenario may use the gel-to-benefit gap as an additional example; neither response establishes clinical safety.

The backup video target is approximately two minutes: 0:00–0:20 task/public-data status; 0:20–0:50 BOAS comparison and baseline; 0:50–1:20 external primary failure and labelled secondary result; 1:20–1:45 calibrated-range rejection and export; 1:45–2:00 scope/limits. This is a storyboard, not a video already recorded. The video can be edited for clarity but cannot hide failures, change model outputs, remove simulation labels or imply unmeasured real-time speed. Mark precomputed output and retrospective replay. A video does not eliminate the three-successful-rehearsals gate for any proposed live acquisition/inference path, and cannot guarantee the same assessment as a successful live demo.

The handout is printed only after genuine results and an accessible, shareable repository URL exist; until then it is a draft with results explicitly pending. Never generate a QR for an invented repository. Before printing, a second teammate scans the actual QR on another device, opens the repository logged out, follows the README and locates code, weights, results and the frozen-evaluation command. Include the revision/commit and model/protocol identifiers, data-access/licence instructions and links to raw archives; gigabytes of raw data need not sit inside Git. Record the destination and check date. A downloadable build exists, but this update does not assert a public repository/verified QR. Remove private participant recordings before public release. Printed pitch cards may reuse the same approved 30-second script.

### Supporting slides and proof points

These are concise backup/support material, not thirty extra main-pitch slides. Fit the live talk to the organiser's format.

| Item | What to include and what it establishes |
|---|---|
| What would change our mind | A failed matched baseline comparison, poor participant-level interval coverage, or failure on a new population changes the corresponding claim. More exposure without paired functional outcomes still does not establish benefit. |
| Competitive landscape | YASA: established automatic staging [47]; Bitbrain: released headband/reference comparison [46]; Yousfan: existing brain-delivery modelling [5]; Qiu: existing ML/optimisation and grouped procedures [29]. Compare each to the exact task it addresses; delivery models are adjacent work, not sleep-staging competitors. Add named consumer apps only after checking their actual claims. |
| Why this is practical now | BOAS is an accessible paired EEG/reference resource already audited at metadata/annotation level [46]. Explain the concrete development opportunity, not an unsupported 2026 market inflection or a universal claim about device readiness. Actual hardware/SDK availability remains a check. |
| Concrete user journey | A researcher opens a decoded night, inspects an uncertain segment against reference/baseline, checks its effect on the summary and exports test conditions/missing measurement. A parent/clinical treatment journey remains future work. |
| Privacy and ethics | Visible acquisition, teammate consent, local storage/access, purpose, retention/deletion choice, stopping procedure and no covert recording or attendee health inference. Consent to EEG collection does not automatically permit a public face/video recording or publication. |
| Team and contribution | Confirm actual names against the roles in section 6; show what each person implemented, evaluated or sourced. Separately label reused assets and planning work. Do not invent people, titles or completed contributions. |
| Mentor quotation | Include only an actual, accurately attributed comment with permission for that use. Distinguish technical feedback from endorsement; neither is clinical validation. No invented quote or promised advocacy in judging. |
| Learning and scope | Use genuine findings, for example the audited difference between 128 nights and 100 people and its effect on splitting. Describe actual model/build lessons only once observed. Explain prioritisation without calling other choices dishonest. |
| Funding/use of award | If asked for a one-month funding scenario, present a scoped research/protocol and data-access plan with costs supported by actual quotes. Do not promise that a hypothetical $10k buys an approved paediatric study or a completed child dataset in one month. |
| Credits | Name the organiser/sponsor accurately; credit Bitbrain/BOAS, OpenNeuro and software/data contributors used. Credit Tobii as stream sponsor; claim g.tec or other hardware use only if it occurred. Follow logo/asset terms. |

### The next 30 days: proposed work, not booked commitments

| Period after the event | Proposed deliverable |
|---|---|
| Days 1-7 | Reproduce the frozen result, organise attributed code and data-access instructions, resolve documented failures and review privacy before sharing |
| Days 8-14 | Seek workflow feedback from a relevant research/clinical partner; record actual responses rather than claiming an existing partnership |
| Days 15-21 | Draft a paired-study question, endpoints, recruitment/access requirements, confounders, device compatibility and required ethics/data-governance steps |
| Days 22-30 | Produce a protocol and costed feasibility decision; identify and verify relevant data-access/funding opportunities before naming an application as submitted |

This roadmap does not schedule an automation, contact anyone or claim an appointment, dataset application, grant application or agreement already exists. Paediatric recruitment and clinical work proceed only through the appropriate approved study process; the calendar is a planning target, not permission or guaranteed delivery.

### Reproducibility, analysis freeze and presentation operations

Freeze the split manifest, endpoint definitions, preprocessing/model choices, seeds, exclusion rules and calibration procedure before final fitting/evaluation. Save them with a timestamped version/commit and change log. The global annotation audit already occurred: do not describe it as prospectively preregistered. An optional OSF record would state exactly which analyses remain prospective and which changes are amendments. Registration does not remove bias or prove the analysis correct. Publishing to OSF or making a repository public is separate from adding this checklist and requires an actual team decision.

The reproducibility statement names the code commit, data snapshot, person-split manifest hash, package/environment file, seeds, hardware assumptions, model artefact and command used. Claim “one command” only after a fresh-environment run succeeds with documented data prerequisites. Keep consented/private recordings outside a public repository; a public reproduction can use permitted public data.

Assign a demo operator, timekeeper and judge greeter within the existing team. The greeter gives the same accurate explanation through permitted event interactions; the role is not lobbying for special treatment. Practise once with a person who does not know the project and, if access is allowed, once in the actual room. Check screen legibility, sound, click target sizes, cable/power and offline assets. Share water/food/break responsibility so one person does not carry the whole presentation.

Ask organisers to confirm code freeze, submission deadline, permitted post-freeze materials and pitch duration in an official written channel where possible; record date/time zone and source. Do not substitute an old draft timestamp for the actual rule or alter code after freeze without explicit event permission. This is a task for the team; no organiser message has been sent by this update.

### Written fallback for every gate

| Gate fails | Pre-agreed fallback and honest demonstration |
|---|---|
| 0: scope/eligibility interpretation | Keep the single EEG core; leave PK-only AI outside the assessed contribution. Resolve the actual event rules without promising mentor approval. |
| A: raw loading | Repair units/alignment or use a prespecified manageable valid subset without selecting on performance. If raw neural loading remains unavailable, show the audit as partial work; do not call it a trained decoding project. |
| B: trained result | Drop complex challengers and retain an evaluated trained feature model if available. If training/evaluation cannot be completed, report an incomplete result; a UI or vendor-score audit cannot satisfy the missing model work. |
| C: uncertainty/summaries | Show supported point metrics, missingness and explicit limitations; omit unsupported interval/clinical-summary claims. |
| D: integration | Reduce to the evaluated neural result and a traceable static sleep-result/summary view if it works; disclose missing interaction rather than stage a fake click. |
| E: day/live extension | Omit it or use a clearly labelled genuine replay. Keep the three-act core independent of it. |
| F: submission/presentation | Use the tested offline backup and only materials permitted by the official freeze/submission rules; do not promise an extension or covertly change the evaluated artefact. |

The final release check is the sleep headline result plus its denominator, the aligned visual, the narrow summary/evidence-gap response, a functioning sleep-only demo path and a reproducible result. Section 11 fixes which conclusions may change with results and which scope boundaries do not. No unsupported “first”, “top 20%”, numerical winning probability or guarantee is added. These tasks improve clarity and inspectability; competition outcomes remain unknown.


## 11. Core question, six design risks and result-dependent presentation

**The fixed scientific question is:** from wearable forehead EEG, how accurately can we estimate sleep stages and recording-window summaries, and how well do stated uncertainty and missingness limits describe their actual errors on held-out people?

The fixed population is the inspected BOAS adult cohort. The model/task/evaluation—not a simulated medication overlay—is the core contribution. Sleep staging and calibration are established methods: the project must demonstrate a useful empirical result or improvement under a clear protocol, rather than claim that a new architecture or a properly grouped split alone is novel.

### Six risks and their closure evidence

| Risk from the supplied critique | Design decision | Evidence or remaining limitation |
|---|---|---|
| C1: data/story mismatch | Lead with wearable sleep decoding; introduce ADHD later as motivation | BOAS supplies paired wearable EEG and an expert PSG reference for this task. This is a positive task-fit reason, not a claim it is the only usable public dataset. It does not validate children, ADHD or medication response. |
| C2: multiple unrelated projects | A complete sleep-only core and demo; drug/day/formulation work is optional research context | Viewer shows the boundary between observations, estimates and absent paired evidence. An aligned clock or shared screen is not a measured biological bridge. |
| C3: weak AI novelty | Test stage confidence, summary errors, missingness and interval behaviour in one declared evaluation | Report the actual comparison and failure modes. Proper protocol is necessary but not automatically original. Appendix J's 16 formulation checks do not validate EEG uncertainty; use sections 3 and 7's neural evaluation. |
| C4: marginal eye-tracker use | Keep EEG central; describe any actual eye features precisely | The official AI brief permits EEG and makes hardware optional [1]. Its five published criteria do not establish a separate sponsor-use score or an automatic penalty for no Tobii hardware. Do not promise a future integration is already working. |
| C5: PK-only eligibility | Exposure model stays conditional research, outside the core dependency | A mentor's answer affects only that optional assessed task. Neither an answer nor a sleep-model gain validates drug-response prediction. |
| C6: excessive build scope | One trained/adapted model plus one sleep-reliability evaluation/viewer | Extra architectures, day acquisition, drug scenarios and formulation code are cut before the required model, baselines, summaries and honest result. Retained appendices are not a promise to implement everything. |

### A, B and C are presentation choices, not different experiments

Use **A: uncertainty-aware sleep decoding** by default from Gate 0. The core question, data, split, metrics and claims stay fixed whether the model wins or loses.

| Evidence actually available | Permitted emphasis | Claim that remains unavailable |
|---|---|---|
| A: trained sleep result and an evaluated uncertainty/summary analysis | “We test wearable sleep estimates and the conditions under which their uncertainty is informative.” | Universal trust, clinical safety or reliable intervals not supported by actual coverage/width |
| B: model fails to improve, but a valid matched comparison reveals informative differences | “Which sleep estimates can we rely on under these tested conditions?” Show the trained result, all comparator errors, coverage and limitations. | That a comparison-only vendor audit satisfies the required training/adaptation, or that losing a benchmark is itself an AI improvement |
| Sleep model improves under the prespecified comparison | Strengthen the same A story with the actual effect, denominator and participant-level uncertainty | Automatic promotion to medication timing, paediatric ADHD or dose-fit claims |
| C: an optional timing/formulation evidence view also works | Show it as a bounded research application in spare time/Q&A, explicitly exposing absent links | Even successful independent night, day and PK models do not establish a jointly validated treatment-response chain |
| Raw training/evaluation remains incomplete | Demonstrate partial work and name the missing result | A completed, evaluated neural-decoding project |

Use grouped development results to choose implementation scope. At Gate B, and again before submission, choose only presentation emphasis from the available evidence. Freeze final model/analysis choices before the final held-out evaluation; do not retrain on its results, switch the primary endpoint, omit failed comparisons or present a selected positive class as the full result.

An observed improvement is reported at its measured size. If uncertainty includes no improvement, say so; do not convert a small point difference into a decisive superiority claim. A benchmark loss does not force a new project: the original reliability question still has an honest answer. Unsupported uncertainty cannot be rescued by a compelling narrative.

Use an event-relative Gate B checkpoint. The pasted 13:45 lock and 21:00 decision times are not adopted as verified rules or current readiness. Any actual checkpoint must leave time for evaluation, integration and rehearsal under the organisers' confirmed schedule.

### Research extensions, ranked within the same evidence boundaries

| Direction suggested in the critique | Place and prerequisite |
|---|---|
| Uncertainty in sleep summaries | Already part of the core question; distinguish recording-window quantities, missing-data bounds and independently evaluated prediction intervals |
| Robustness to another device or age group | Later external-validation study using a separately verified licensed dataset, compatible signal definitions/reference and no test-cohort tuning. Controlled gaps in BOAS are not evidence for this transfer. |
| Real eye-tracker/EEG fusion | Optional later paired acquisition study with verified device access, consent, synchronisation and person-held-out evaluation; no core dependency or assumed sponsor-score benefit |
| Formulation evidence engine | Retained NEURORELEASE research line. Any separate event entry would require its own task, validation and stream-fit review; this plan does not switch streams or establish drug-only eligibility. |

**Team explanation:** “Our project has one technical challenge: make wearable sleep estimates inspectable, and test whether their uncertainty matches the errors in the summaries. ADHD explains a potential future need. The drug and hydrogel research stays in the roadmap until paired data can test that connection.”


## 12. New AI-decision contract and four-file dataset audit

This revision incorporates **Add AI determine.docx** [52], **hydrogel_ai_datasets.xlsx** [53], **matched_pairs.xlsx** [54] and **coating_core_evidence.md** [55]. Both workbooks were opened locally; all six sheets, populated rows, formulas and cached formula results were inspected. The source workbooks are unchanged.

The formulation component now has a concrete specification: researcher-declared requirements, route-specific compatibility checks, source-linked candidate comparisons and property predictors only where independently evaluated data support them. It is not a learned disease-to-gel selector. The sleep-decoding core remains independent of these optional research functions.

### Actual contents and readiness

| Inspected location | Finding | Consequence |
|---|---|---|
| Hydrogel: Datasets, rows 4-22 | 19 catalogue entries: 18 “Excerpt only”, one “To build” | Acquisition register, not 19 downloaded/verified training datasets |
| Hydrogel: Datasets B24/B25 | Two formulas; cached results 19 and 18 match independent counts | Counts checked; no workbook edit or Excel recalculation claimed |
| Hydrogel: Core table template, rows 4-6 | Three labelled examples; C4:C6 all have incomplete DOI placeholders | Extraction leads excluded from accepted training |
| Hydrogel: example properties | One numeric modulus, no numeric degradation time, one 35-day entry in the t95 column | Not enough for a property model. “Release over five weeks” does not establish t95 = 35 days. Verify the fraction/time definition. |
| Hydrogel: System design and Sources | Four workflow stages; 23 source-link rows | Proposed design and leads, not verified raw datasets or commercial permissions |
| Matched pairs: rows 2-10 | Nine rows, eight distinct paper-label strings; classes A=3, C=3, D=2, E=1 | Resolve publications/groups; rows are not independent studies |
| Matched pairs: status | Seven A (abstract/excerpt), two R (earlier full-text status), both R rows from Johnsen | Inherited status, not proof all methods were re-audited here |
| Matched pairs: quantitative fields | No typed numeric dose or fold-change cells; folds include “<1” and “about 3” as text | Preserve approximate evidence. No regression-ready matched-pair corpus is established by this upload. |

Inspected SHA-256 fingerprints: hydrogel 829d4818c180841e4ea3eee714f5824387cd794799a008f4e74dfa280b7f5670; matched pairs c04e4278cbdb495cde45c3d87d7ce2fa1cbf28c4711c31d4ad7a279c22b72f73. Counts describe these files, not all literature or biological replicates.

### Which decisions belong to AI, rules and the researcher

| Decision | Mechanism | Support requirement/current status |
|---|---|---|
| Sleep stages and sleep summaries | Required trained/adapted EEG model | Section 3; loading/training/evaluation pending |
| Disease, target, payload, route and desired duration | Researcher-declared question; reviewed extraction/retrieval can assist | Disease associations inform candidates, not automatic therapy, route or dosing-window selection |
| Route/payload/gel incompatibility | Explicit, versioned rules tied to conditions and sources | Known conflicts exclude under the research policy; missing critical fields remain unresolved |
| Gelation, modulus, swelling or release | Separate family/assay-specific predictors | Defined targets, verified rows, independent-group baselines/tests; none fitted from these examples |
| Candidate shortlist | Transparent constraint/Pareto comparison or declared criterion-based ordering | Priorities and missingness visible; no unvalidated score advertised as efficacy probability or clinically best gel |
| Targeted/untargeted nanoparticle contrast | Verified matched comparison, then a conditional model if data permit | Same endpoint, analyte, dose basis, route, interval and comparator; no pooled uptake/transport/imaging/benefit target |
| Host-specific PEG/heparosan or iron/gold choice | Evidence retrieval and missing-experiment response today | No paired varying-host-property/outcome dataset supplied; no personalised coating/core recommendation |
| Next useful measurement | Identify the unresolved variable that could change comparison | “Cheapest” requires real cost/availability; no invented information-gain/cost optimum |
| New/old drugs and trials | Dated reviewed source tracking | Approval by indication/route/formulation differs from trial or mechanism evidence |
| Cure, human safe dose or disease diagnosis | Outside supported decisions | No automatic release-to-exposure-to-symptom conversion |

Label rules and retrieval as rules and retrieval. Automated extraction is not experimental evidence, a validated property predictor or the scored EEG contribution.

### All ten bioengineering gaps become explicit fields

| Gap in the AI document | Fields and corrected interpretation |
|---|---|
| 1. Route/barrier model | Route, deposition/site, nasal residence, mucus/epithelium, pathway evidence, systemic exposure, interstitial spread and sampled compartment. The gel does not itself establish BBB passage; released material may involve systemic and direct pathways. Local placement bypasses entry, not tissue distribution. |
| 2. Outcome label | Retention, defined released fraction/time, gelation, swelling, tissue reaction or named functional endpoint; retain units, censoring and definition. No universal “best gel” label or pooling graft survival with release. |
| 3. Release mechanics | Mesh/cargo dimensions, swelling, diffusion/erosion/binding context, degradation, burst, loading mechanism and kinetic-model domain; a fitted curve is not an independent experiment. |
| 4. Spatial coverage/window | Region, measured distribution volume/distance, release location and observation interval. “A few millimetres” is not a universal bound; local delivery does not automatically suit a distributed circuit. |
| 5. Species translation | Species/anatomy, device/deposition, head position where relevant, mucociliary clearance and enzyme stability; no unvalidated animal-to-human/nose-to-IV transfer coefficient. |
| 6. Tissue/CSF limits | Actual-medium dimensions, interstitial conditions, CSF compartment/turnover and measured clearance. No universal 40-100 nm acceptance band. |
| 7. Cargo stability | Retained activity over time, loading loss, pH, enzyme degradation, payload identity, anti-drug immunity and coating-antibody evidence as separate fields. |
| 8. Chronic/repeat use | Schedule, age/sex, follow-up, inflammation, foreign-body/glial response, clearance and repeat-dose observations. The note's 14-day statement is not a universal literature limit. |
| 9. Handling/manufacturing | Viscosity/shear-thinning conditions, needle/catheter geometry, gelation time/temperature, sterilisation, endotoxin, residual reagents and shelf life. |
| 10. Model/readout | Exact model (SHR, 6-OHDA, MPTP or alpha-synuclein as applicable), species/strain, sex, age, region and assay. PD outcomes are not ADHD efficacy labels. |

Do not make 1-4 kPa a universal gate. Record modulus type (Young's, storage or another quantity), method, frequency/rate, temperature, species and site. These measures are not interchangeable. Safety rules require concentration, route, exposure and endpoint context; absence of a listed toxic crosslinker does not prove safety.

### Matched-pair and coating/core rules

Keep A-G assay classes as task labels; suggested Yokel mappings are not verified universal ordinal scores. P1's TEER note uses another study's value: leave the actual experiment's TEER unextracted until checked. Split multi-arm P6 only after extracting each comparison, retaining the shared source/control.

P3a/P3b are the same Johnsen publication in different contexts. Retrieved primary results identify DOI **10.1038/s41598-017-11220-1** and increased capillary-depleted platinum in vivo, not proof of intact liposome passage [59]. Preserve different assay directions and keep both rows in one source group; precise effects/variance/matching still need extraction. P8 measures drug and changes surfactant context; do not call it a pure receptor-ligand effect.

For coating choice, record coating, chain mass/density/linker, core, size/shape, dose basis, medium/corona, cell identity and host-property assay. The heparosan primary result supports different uptake in certain immune cell lines for a studied gold-particle comparison, not longer circulation, brain transport or an individual donor rule [60]. Anti-PEG antibodies are a candidate measured property, not proof that heparosan is the right replacement. Heparosan and heparan-sulfate evidence are not interchangeable.

Iron/gold comparison needs genuinely comparable arms and an explicit particle-number/mass/surface-area dose basis. A simultaneous polymer/size/coat change cannot be credited to the core. Gas-core bubbles need separate gas and shell/component fields. FUS/tumour observations remain a distinct intervention/barrier context.

A future crossed coating-by-core experiment needs pairing, batch controls, replicates and adequate independent hosts. The note's “10 donors/studies”, “8 distinct values”, “5 per group” and “within 25% size” are design heuristics, not validated sufficiency/matching guarantees. Donors and studies are different independence levels; choose sample size from the estimand, variability and design, and group all donor/source data in evaluation.

The CSV and executable paths mentioned in the coating note were not included in these four uploads. Their implementation and refusal behaviour have not been checked or run.

### Acquisition and modelling priorities

- **Bannigan [56]:** the primary record lists 181 profiles, 43 drug-polymer combinations and 3,783 measurements, with two downloadable TSVs. These are polymeric long-acting injectables, not nasal-gel/ADHD-response data. Some early release inputs used imputation from curves extending to two days: audit that before an early-cutoff forecast. No TSV downloaded here.
- **PLGA microparticles [57]:** the primary description reports 321 experiments, 4,913 points and 113 publications; initial/processed files are on Mendeley. It targets small-molecule emulsion microparticles and excludes protein/peptide/nanoparticle search categories. It does not validate protein-gel or nanoparticle-gel transfer. Article and dataset permissions are separate checks; no dataset files downloaded here.
- **B3DB [58]:** small-molecule BBB annotation/benchmarking; a BBB label does not decide if a gel is needed, intact-particle transport or gel performance.
- **Other leads:** peptide-gel formation, PNIPAAm swelling, poloxamer rheology, tissue mechanics and safety sources remain separate acquisition tasks. Open Targets is target/disease evidence, not disease-to-gel labels. Search-excerpt sizes, permissions and performance are not independently verified.

Acquire one relevant permission-checked source and reconstruct DOI, assay, material, interval and units. Define one property endpoint and compare a simple baseline with one modest learner on independent groups. Hold out whole curves/formulations/sources appropriate to the claim. Prefix forecasts use only measurements genuinely available by the cutoff, including imputation. Deduplicate overlapping corpora before pooling/cross-dataset testing.

Enable a property model only after those checks, actual held-out evaluation and support/missingness behaviour. A PLGA release model cannot populate a hydrogel predictor merely because both release drugs.

### Updated project data path and acceptance

The formulation path is **source record → reviewed observation → researcher requirements → compatibility decision → comparable evidence or eligible property model → report and missing measurement**. Preserve raw text/value, typed value, units, source location, extraction status, context and study/host IDs. “Not extracted”, “not reported”, “not applicable”, censored and unresolved are separate; none becomes zero.

| New acceptance case | Required behaviour |
|---|---|
| Three EXAMPLE rows with DOI placeholders | Exclude from accepted training; identify extraction work |
| 35-day narrative entered as t95 | Withhold t95 until its released-fraction definition is verified |
| Borrowed TEER in P1 | Do not assign another experiment's value |
| P3a/P3b across assay classes | Preserve context/common study; no pooling or split leakage |
| Approximate fold without denominator/variance | Retain approximation/source; no exact fictional replicates |
| High anti-PEG without paired comparison | Request matched outcomes; no heparosan recommendation |
| BBB+ used to choose a gel | Reject the inference; retain the molecule's sourced annotation |
| Missing critical route/stability/chronic fields | Unresolved requirement; no safety/efficacy pass |
| Future-imputed early release input | Exclude/reconstruct for the prediction cutoff |
| Unavailable property/host-choice model | Evidence/measurement request, not a numerical invention |

**Update completed:** four sources accounted for, workbook contents audited, and the decision contract/data architecture specified. No application repository was supplied with these files. No disease-to-gel, coating/core-choice or release model was trained; measured evaluation is required before advertising those predictors as implemented.

## 13. Executed build and dataset acquisition: 7 November 2026

This section supersedes the earlier planning-stage “pending” status for the work actually executed below. Design constraints, unsupported clinical claims and unimplemented extensions remain in force.

### Acquired and verified

- All **128 BOAS headband EDFs / 100 people** acquired; 13.440 GB of original source bytes. SHA-256 checked against the pinned repository annex pointers. Onsets, durations, sample boundaries, units and two task channels checked for every recording.
- Lossless digital HB_1/HB_2 samples and EDF calibration retained for all recordings. Large extra-channel original EDFs were acquisition cache, replaced after verified extraction by task-channel archives. URLs/hashes permit original reacquisition; the first original EDF is also retained locally.
- Bannigan authors' 14/17-feature data acquired from the pinned authors' GitHub mirror. Both files match the primary Zenodo MD5 values. Their .tsv extension is misleading: the actual containers are Excel. Both variants share the same 3,783 outcomes and are not independent datasets.
- Four original PLGA workbooks acquired from Mendeley v2; SHA-256 checked against its file manifest; dataset licence CC BY 4.0 verified. 4,913 observations, 321 formulations, 113 source DOIs. No cross-corpus pooling.

### Frozen EEG experiment and actual outcomes

48 fitting people; 12 separate probability-validation people; 20 summary-calibration people; 20 test people. Repeated nights stay together. The first night per person was selected before fitting for calibration/test. Model: 28 EEG-only spectral/statistical features, fixed histogram gradient boosting; no target/reference/vendor labels in the features.

| Held-out result | Actual value and interpretation |
|---|---|
| Participant-average stage kappa | 0.584 feature model; 0.765 released Bitbrain; majority baseline 0.000 |
| Participant-average macro-F1 | 0.471 feature model versus 0.607 released Bitbrain |
| Usable-window sleep-time MAE | 27.6 minutes; this is not a whole-night clinical measure where signal/reference is missing |
| Identical vendor-covered mask | 27.1 model versus 14.4 vendor minutes; paired difference CI approximately -2.4 to 27.8 minutes |
| Nominal 90% summary interval | Observed 80% coverage (16/20), mean width 64.0 minutes; achieved 90% coverage is not claimed |
| Probability temperature scaling | Held-out log loss, Brier score and confidence ECE did not improve; the failure is retained |

The trained model exceeds a majority baseline but trails the released vendor comparator. It is an implemented initial baseline, not a demonstrated competitive advantage. Vendor training provenance differs from the team's split. Further model development must be labelled development on an already inspected cohort; a new independent confirmatory result requires an untouched evaluation design/data.

### Separate, bounded materials experiment

PLGA release model trained with whole-DOI separation: 90 training papers, 23 test papers, 98 test formulations. Study-averaged MAE is 0.144 release fraction versus 0.161 for a fixed time-bin median. This is a descriptive point difference, not established transferable superiority. There are 108 observations outside the nominal 0-1 range; original observations are retained and flagged, not silently clipped.

This is in-vitro small-molecule PLGA microparticle modelling. It does not establish hydrogel prediction, protein-payload transfer, brain exposure, personalised coating/core choice, ADHD treatment response or dosing. Missing matched host/clinical data have not been invented or replaced by unrelated rows.

### Working deliverables and verification

The Goldilocks source package includes protocols, pinned acquisition, extracted features, trained model artefacts, per-person metrics, full-reference availability diagnostics, probability-calibration diagnostics, three genuine failure examples, original release inputs and a self-contained offline evidence explorer. The separate EEG data archive contains the lossless task-channel recordings and provenance.

Meaningful tests passed for person-group separation/repeat nights, unknown-label handling and spectral feature behaviour. All 128 loader checks and digital reconstruction/MNE calibration spot checks completed. Browser checks verified 20 selectable test nights, reference reveal, aligned plots, unsupported-latency response and evidence JSON download. Desktop/mobile layouts were inspected; mobile plots scroll horizontally to retain readable labels.

YASA, CNN, live EEG, day-task predictor and clinical/host-specific formulation models have not been implemented in this build. The public-data report is precomputed retrospective analysis, not live inference. The next scientific need is improving and independently confirming the EEG model and interval reliability, not adding more unrelated modules.

## 14. V2 execution, independent test and revised claim gates

This section is the current implementation record. Section 13 preserves the original experiment. The new work improves stage decoding and empirical probability calibration, but does not establish reliable sleep totals or calibrated sleep-time ranges. Earlier aspirational acceptance cases remain requirements where not implemented; their presence in this plan is not evidence that they passed.

### Implemented model and frozen evaluation

V2 uses a channel-shared feature classifier with 5/15-epoch centered context and training-derived sequence smoothing. Context resets at signal/timestamp gaps. Its use of future epochs makes this retrospective inference. Forty-eight people fit the model. The original 12 probability-validation people are prospectively separated into six sequence-selection people and six probability-calibration people. The original 20 summary-calibration people determine fixed global and scaled 90%-target residual intervals. Repeated nights stay grouped. Three sequence weights were compared only on the six selection people; 0.5 was selected. Temperature is 2.4522. These choices did not use historical-test, summary-calibration or external outcomes to select the model.

`v2_protocol.json` and `results/v2/freeze_record.json` preserve the protocol, selection results and timestamps. Frozen model SHA-256: `121837c16b4ff3a045421355048b538c53158119b2afd52dcbd32a641021cb0b`. External reference acquisition occurred after that freeze. Frozen evaluation loads saved weights and never refits. The 20 BOAS test people were inspected at v1: their v2 results are development comparisons, not new independent confirmation.

### Additional dataset actually obtained

Sleep-EDF Expanded 1.0.0 [61], 20 different people (subject IDs 00–19, first night), selected before external reference access. All 20 original PSG EDFs and 20 reference EDFs were acquired and checked against the publisher's SHA-256 manifest. Original source/licence pages and acquisition records accompany the files. Licence: Open Data Commons Attribution License v1.0; DOI 10.13026/C2X676. The original external signals are supplied separately; all 20 derived feature files are in the build package.

The fixed preprocessing uses Fpz-Cz alone, 100→256 Hz resampling and R&K 3/4→N3 mapping. Movement/unscored reference is excluded rather than changed to Wake. Device, montage, reference convention and recording duration differ from BOAS. This is independent domain-shift evidence, not same-device headband confirmation or ADHD validation. Primary evaluation includes all valid scored epochs of the complete recording. A prespecified secondary evaluation uses the first-to-last reference sleep span plus 30-minute margins; this is not a model-generated deployable sleep window.

### Results that govern the product

| Evaluation, 20 people per cohort | Mean-person kappa | Sleep-time MAE | Global / scaled 90%-target interval coverage |
|---|---:|---:|---:|
| V1 historical BOAS | 0.584 | 27.6 min | 80% / not tested |
| V2 historical BOAS, development comparison | 0.661 | 31.8 min | 60% / 60% |
| V2 independent Sleep-EDF, complete scored recording, primary | 0.579 | 235.9 min | 20% / 25% |
| V2 independent Sleep-EDF, reference-defined sleep window, secondary | 0.515 | 36.8 min | 30% / 30% |

Historical BOAS macro-F1 increases 0.471→0.561. On that cohort, temperature calibration improves log loss 0.739→0.545, Brier score 0.320→0.281 and confidence ECE 0.128→0.087. These are empirical probability diagnostics; they do not imply valid uncertainty for total sleep. The released Bitbrain stage comparator still leads. Common vendor-covered sleep-time MAE is 31.3 minutes for v2 versus 14.4 for Bitbrain.

Do not hide the 235.9-minute primary external sleep-time error behind the more favourable secondary window. Long daytime wake differs from the BOAS night window; that is a plausible contributor to transfer failure, not a proven sole cause. Confidence in per-epoch probabilities and confidence in a sum over a recording are different properties. Twenty people also leave wide coverage uncertainty: the global interval's 95% Wilson range is approximately 8–42% externally, versus a 90% target. Neither fixed interval method is released as reliably calibrated.

### Updated interface, acceptance and presentation

The new self-contained explorer displays both cohorts, all failures, aligned reference/model tracks, and BOAS v1/Bitbrain tracks after reference reveal. YASA remains unavailable. A calibrated-range request is rejected with the measured failure reason. A clinical lights-off-latency request is rejected for absent validated markers. JSON export records the model hash and `calibrated_interval: null`. Raw diagnostic intervals stay in research outputs, never silently relabelled confidence bounds.

Seven invariant tests passed, including participant separation, channel-order invariance, context gap handling and small-sample interval behaviour. Browser checks covered desktop/mobile controls, cohort changes, reference reveal, both rejection cases and downloaded JSON, with no page errors. Reproduction uses `reproduce_v2.py`; v1 outputs are retained. Neither live acquisition nor a full formulation application is implied by these checks.

The current presentation frame is the benchmark/reliability frame from section 11. The headline may state the historical stage-decoding gain only when its development status appears alongside it. The independent failure is a substantive result, not evidence of superiority. Demonstrate one BOAS case, one external failure and the claim rejection. Do not switch to the three-link treatment story or imply a proven ADHD–drug–gel bridge.

### Current presentation contract and release checks (Revision 15)

Lead with the observed transfer boundary, then show the product response. Report **31.8 minutes of sleep-time MAE on historical BOAS development people versus 235.9 minutes on independent complete-recording Sleep-EDF**, 20 people per cohort. Different cohorts/windows make this a transfer comparison, not a matched causal estimate of device effects. Show 36.8 minutes for the prespecified secondary external sleep window alongside the primary failure. The 60% historical and 20%/25% external interval coverage explains withholding calibrated-range claims. Do not generalise this to all black-box neurotech or say Goldilocks automatically recognises every erroneous case.

**Scope freeze:** preserve the evaluated night model, baseline comparisons, window definitions, failure/coverage reporting and working offline sleep-only path. Cut the day model first, then extra drug/gel/PK modules and architectural complexity. The retained PLGA experiment is separate research evidence, not a required demo act or a measured ADHD link. This revision adds no new model.

**Provenance acceptance:** show origin and execution separately with persistent, readable labels. The current result uses `PUBLIC ADULT EEG` and `PRECOMPUTED RESULT / RETROSPECTIVE`; tracks/results distinguish `EXPERT REFERENCE` and `MODEL ESTIMATE`. Any future assumed drug curve carries `SIMULATED WHAT-IF` on its panel and export even when a parameter is literature-derived. A release experiment is `LITERATURE/ASSAY EVIDENCE`, not clinical benefit. A live badge, speed claim or child-treatment implication requires the corresponding measurement. Check label legibility at presentation distance. These are acceptance requirements, not a claim that future panels already exist.

**Three-act demo:** (1) select BOAS and reveal reference/vendor/v1/v2 tracks, noting historical development status; (2) switch to Sleep-EDF, show aggregate primary/secondary results and a genuine failure case without suggesting one selected case represents everyone; (3) request a calibrated range, read the measured rejection reason and export the evidence record. Say: “We stop this claim because the measured interval coverage failed.” The clinical lights-off rejection is a second missing-input example, not a safety certificate.

**Delivery gates:** the evaluation lead checks displayed values against saved results; the integration lead checks labels/offline controls; the operator records the storyboard video and tests local playback on the actual presentation device; the pitch lead runs two adversarial Q&A rounds and confirms each teammate can answer novelty, ADHD, causal-attribution and failure-detection questions. The handout owner completes the logged-out QR/destination check in section 10. Completion evidence changes status, not an intended deadline. Existing browser/reproduction checks passed; video, QR scan, printout and rehearsal are not newly marked complete here.

### Build decision and remaining work

Continue building the research product with the available data. More unrelated datasets are not required for this version. The core model is implemented; broad brainstorming and extra drug/delivery modules should not displace integration, reproducibility and rehearsal. V2 is not universally better: preserve v1 as a reference rather than replacing every endpoint with v2.

If a further scientific iteration is undertaken, first define a task-aligned night-window policy usable without reference labels, strengthen wake/artefact handling and obtain separate development and interval-calibration data under that policy. Freeze the resulting model, window rules and uncertainty method before another untouched test. The 20 external people used here are now inspected and cannot be reused as fresh confirmation. A larger cohort alone does not repair a mismatched task or establish exchangeability. ADHD medication response and hydrogel/coating personalisation still require the paired measurements listed in section 12; neither public EEG nor unpaired PLGA data supplies them.

## 15. Final review actions (Revision 16)

The science is complete for this event. This section adds exactly three changes and four smaller fixes, then closes planning. Nothing here adds a module, model family or dataset beyond YASA as a comparator.

### Keep exactly as it is

- The frozen v2 model (hash recorded), person-held-out evaluation with repeat nights grouped, and the independent Sleep-EDF test.
- Honest reporting: the model trails the released Bitbrain scores, interval coverage failed, and the viewer refuses unsupported claims.
- One clean core: the day model, PK prediction and formulation work stay outside it.

### Change 1: reframe the headline (about 30 minutes, highest value)

Lead with the finding, not the failure. The results support two statements, and both are shown:

| Finding | Numbers (20 people per cohort) | Meaning |
|---|---|---|
| Stage decoding largely transferred | Mean-person kappa 0.661 on historical BOAS (development) to 0.579 on Sleep-EDF (different device, single Fpz-Cz channel, different population, R&K scoring) | Per-epoch skill survives a substantial domain shift |
| Sleep totals did not transfer | Sleep-time MAE 235.9 min on complete Sleep-EDF recordings (primary) versus 36.8 min in the sleep window (secondary); BOAS 31.8 min | Small per-epoch errors accumulate over long daytime wake |
| The product's response | Global / scaled interval coverage 20% / 25% externally against a 90% target | The viewer withholds calibrated ranges and says why |

The secondary window (first to last reference sleep plus 30 minutes either side) resembles how Sleep-EDF is commonly evaluated, so it is a fair secondary result, not a cherry-pick. It is reference-defined, not deployable without labels: say so. The primary 235.9-minute result stays on the same slide.

Opening line: **“Our sleep decoder kept most of its skill on a new device and population. Its sleep totals didn't, so our viewer refuses to give ranges it can't back.”** The 30-second pitch at the top of this document is updated to match.

### Change 2: YASA baseline on both cohorts (1 to 2 hours)

On Sleep-EDF the only comparator is the majority baseline, the weakest point under scientific validity. Add YASA, the standard open-source sleep stager [47], as an external baseline.

- Follow YASA's documented SleepStaging path: no pre-filtering or z-scoring before it.
- Run it on the same frozen people, epochs, masks and windows as v1 and v2; never choose its channel from held-out results.
- Label the channel mismatch on every display: YASA prefers a central derivation; BOAS offers forehead AF7/AF8 and Sleep-EDF offers frontal Fpz-Cz.
- Report kappa, macro-F1, sleep-time MAE (primary and secondary windows) and coverage, whether YASA wins or loses.
- Add a YASA lane to the same-night figure and the viewer; replace “YASA unavailable” only once real outputs exist.
- Credit YASA (Vallat and Walker, 2021) and its licence.

### Change 3, optional and time-boxed: label-free main-sleep-period rule (3 to 4 hours, firm stop time)

The plan's own next step for the 235.9-minute failure is a night-window rule that needs no reference labels.

- Rule: the longest block of predicted sleep, tolerating interruptions up to a fixed gap length, with fixed margins.
- Set gap and margin parameters on BOAS training or calibration people only; freeze them with a timestamp before touching Sleep-EDF.
- Sleep-EDF has already been inspected, so its result under this rule is **exploratory development, not confirmation**. Put that label on the slide and in the export.
- Report sleep-time MAE under the rule beside the primary and reference-window results; keep coverage results unchanged unless recalibrated on separate people.
- Stop rule: if it is not working end to end by a time the team fixes tonight, drop it and present it as next work. It never displaces Change 2 or rehearsal.

If it works, the story becomes: “we found the failure mode, and here is the first fix, labelled exploratory.”

### Smaller fixes

1. **Sharper impact line:** “Anyone who reports total sleep from a wearable, such as researchers using home EEG as a study endpoint, needs to know when that number can't be trusted.” ADHD medication timing stays as one motivation slide.
2. **Name the model behind each number:** v2 is better on stages (kappa 0.661 against 0.584); v1 is better on BOAS sleep time (27.6 against 31.8 minutes). The viewer and slides show both and say which is which.
3. **One-page pitch card** for every teammate: the four headline numbers, the three demo acts, the five hardest Q&A answers (novelty, ADHD, cause of the external error, does the AI know when it is wrong, why show a failure) and the provenance labels.
4. **New Q&A answer:** “Why lead with transfer rather than the failure?” — “Both are measured. Stage skill transferred; totals didn't. Showing only one would misstate the result.”

### Do not do

No CNN challenger, day model, PK model, extra datasets or extra formulation modules. No further planning revisions: remaining gains come from rehearsal, two adversarial Q&A rounds, checking displayed numbers against saved results, the backup video and the QR check (section 10).

### Order of work

| Order | Task | Owner (proposed) | Done when |
|---|---|---|---|
| 1 | Replace the opening sentence and slide headline | Pitch lead | Pitch card and slide show all four numbers |
| 2 | YASA on both cohorts | Evaluation lead | Results table and viewer lane filled from real outputs |
| 3 | Main-sleep-period rule (optional) | Model lead | Frozen parameters, exploratory label, or dropped at the stop time |
| 4 | Pitch card, Q&A rounds, number check, video, QR | Integration lead and pitch lead | Completion evidence recorded per section 10 |

### Revision 17 execution record (7 November 2026, about 17:45–17:55 UTC)

All work below ran against the frozen v2 build in `Documents/Codex/2026-11-07/new-chat-2/goldilocks`. Nothing in v1 or v2 was refit or changed; new outputs are in `results/yasa/` and `results/window_rule/`.

**Number check: passed.** `yasa_evaluate.py` recomputed every v2 metric from the saved per-epoch predictions using the build's own masks, windows and `stage_metrics` code, and all match the saved results (`results/yasa/v2_reproduction_check.json`). Every number quoted in this plan and the pitch card matches the saved files (v1: kappa 0.584, macro-F1 0.471, sleep MAE 27.6 min, coverage 80%, width 64.0, vendor-mask 27.1 against 14.4; Bitbrain kappa 0.765, macro-F1 0.607; v2 calibration log loss 0.739→0.545, Brier 0.320→0.281, ECE 0.128→0.087; external Wilson 8–42%). The v2 viewer's embedded data is identical to the saved result files.

**Change 2, YASA: done.** YASA 0.7.0 pre-trained SleepStaging (0.7.0 because the Mac's Linux environment has Python 3.10; 0.8.1 needs a newer Python), documented path, no metadata, fixed channels (BOAS HB_1, Sleep-EDF Fpz-Cz, both mismatched: YASA prefers a central channel). Same 20 people per cohort, same epochs, masks and windows as v2. YASA produced no abstentions (coverage 100%).

| Cohort and window (20 people each) | v2 kappa | YASA kappa | v2 sleep-time MAE | YASA sleep-time MAE | YASA signed bias |
|---|---:|---:|---:|---:|---:|
| BOAS headband, development | 0.661 | 0.532 | 31.8 min | 57.6 min | −57.6 min |
| Sleep-EDF, complete recording (primary) | 0.579 | 0.675 | 235.9 min | 110.3 min | +101.6 min |
| Sleep-EDF, reference sleep window (secondary) | 0.515 | 0.628 | 36.8 min | 23.1 min | −3.2 min |

Macro-F1: v2 0.561 / 0.478 against YASA 0.518 / 0.624 (BOAS / Sleep-EDF primary).

Reading:

- On the forehead headband, v2 beats YASA on stages and totals. On Sleep-EDF's Fpz-Cz PSG channel, YASA beats v2 on both. YASA was trained on PSG nights, which plausibly suits Fpz-Cz better than a forehead band; that is a candidate explanation, not a tested cause.
- Totals fail to transfer for both systems on complete recordings: YASA's 110.3 min is better than v2's 235.9 but still far from its 23.1 min within the sleep window. This supports the core claim that sleep totals, not just one model, need reliability checks across recording windows.
- Pitch consequence: never claim v2 is the best stager. Claim the workflow, the headband result and the measured totals failure, with YASA's numbers shown.

**Change 3, sleep-window rule: done, did not fix the failure; moved to next work.** Tuned on the 20 BOAS summary-calibration people (frozen v2 inference only, verified to reproduce saved predictions on two test people). Chosen: 30-minute gap, 30-minute margin, frozen 2026-11-07T17:50:37Z (`results/window_rule/window_params.json`). On the calibration people it barely changed error (9.7 against 9.8 min) because BOAS nights contain little daytime wake.

| Applied with frozen parameters | v2 unwindowed → rule | YASA unwindowed → rule |
|---|---:|---:|
| BOAS development | 31.8 → 45.3 min (worse) | 57.6 → 72.5 min (worse) |
| Sleep-EDF, EXPLORATORY (already inspected) | 235.9 → 177.5 min | 110.3 → 73.9 min |

The rule trims spurious daytime sleep externally but cuts real sleep on BOAS nights, and v2's external error stays far above its within-window error. A night-only cohort cannot tune a rule for long recordings. Next work: development and calibration data that contain daytime wake, then a fresh untouched test.

**Viewer: updated copy added.** `results/yasa/Goldilocks_Validation_Explorer_v2_1_yasa.html` adds a YASA lane (after reference reveal, both cohorts) and three YASA results rows; the original v2 viewer is unchanged. Headless-browser check: 7 table rows, lanes correct in both cohorts, range and latency refusals work, JSON export keeps `calibrated_interval: null` and the model hash, no page errors.

**New Q&A answers.**

- “Why not just use YASA?” — “On Sleep-EDF's PSG channel YASA transfers better than our model; on the forehead headband ours beats it, kappa 0.66 against 0.53 and totals 32 against 58 minutes. Neither gives trustworthy totals on complete recordings: that is what our viewer exposes.”
- “Did your window rule fix the 236 minutes?” — “No. Tuned on night-only data it cut the external error to 178 minutes but made BOAS worse. Fixing it needs development data that contain daytime wake.”

**Still open (team tasks):** rehearsals and two adversarial Q&A rounds; backup video; public repository and QR check (remove private recordings first); organiser confirmation of code freeze, submission format and pitch length.

### Revision 19 execution record (7–8 November 2026, overnight)

All work below ran against the frozen v2 build in `Documents/Codex/2026-11-07/new-chat-2/goldilocks`, with a
tidied copy in `Documents/Codex/2026-11-07/new-chat-2/Goldilocks_Submission`. No v1 or v2 artefact was refitted,
reselected or overwritten. Every new output sits in its own folder. Built 08 November 2026, 11:47 UTC.

**1. Submission package assembled.** `Goldilocks_Submission/` holds the code, the derived features the frozen
evaluation actually reads, every result, the viewer, the backup video, the plan, a README, credits, licences and
a file-hash manifest, in 36 MB. The 3.4 GB of raw and compacted EEG, the 955 MB of original Sleep-EDF files and
the non-redistributable Yousfan workbook are deliberately excluded, with published checksums and an acquisition
script left in place so anyone can fetch and verify the originals. There is no public repository, by the team's
decision, so the build identifier is a content hash rather than a commit: frozen protocol
`ad3fb648f8eedcd429884abe8f018f24c5eb3e85b54a3f87ffc164183f29882c`, frozen model `121837c16b4ff3a045421355048b538c53158119b2afd52dcbd32a641021cb0b`, both equal to the values written into
`results/v2/freeze_record.json` before the external reference labels were opened. Appendix M has the manifest.

**2. One-command reproduction: proven, and the deviation measured.** A clean machine, a clean virtual
environment built from `requirements.txt` with every pin satisfied exactly, then `python reproduce_v2.py`:
7 tests passed, both evaluations recomputed and the report rebuilt, exit code 0, 31 seconds. Of 38 result files
31 came back byte-identical. The other 7 differ only in floating-point accumulation order: across 294,208
numeric fields, 52 differ at all, the largest relative difference is 2.6 × 10⁻¹⁵, and **nothing differs at four
decimal places**. A second run in the same environment reproduced all 38 files byte-for-byte. The honest claim
is therefore "reproducible to about 15 significant figures across environments, bitwise within one", not
"bitwise reproducible". Full report in `REPRODUCTION.md`.

**3. Leakage audit of the exposure model: a win removed.** `research/exposure_model/leakage_audit.py` tests the
assumption that a normalised study title identifies a source study. Four detectors, each proved able to fire
against an injected synthetic violation before the real data is touched (11 self-test checks, all passing). It
found 1 bit-identical and
2 four-significant-figure
duplicate outcome pairs sitting under two different study titles: the haloperidol solid-lipid-nanoparticle pair
and the lectin/UEA-I intranasal nanoparticle pair. Merging them with union-find takes
159 title groups to 157 source groups. Before
interpreting the rerun the audit reproduces the saved title-grouped numbers with its own code to within 1e-9.
One headline verdict changes: `beats_route_ratio_baseline` goes from true (interval −0.109 to −0.004) to
**false** (−0.097 to +0.015). The two findings that matter are unchanged: measured blood exposure carries the
signal, and formulation descriptors do not transfer across studies. `EXPOSURE_RESULT.md` was updated to say so.

**4. The CNN challenger family, and the result that sharpens the pitch.** Section 15 listed a CNN challenger as
an optional stretch item. It was built, and then five further variants, all on the identical frozen split, the
same six selection people, the same six calibration people, the same evaluation mask, the same `stage_metrics`
and the same sleep-time definition. Appendix L has the protocol in full.

| Model | BOAS kappa | BOAS macro-F1 | BOAS sleep-time error | Sleep-EDF kappa | Sleep-EDF sleep-time error |
|---|---:|---:|---:|---:|---:|
| Frozen v2 feature model | 0.661 | 0.561 | 31.8 min | 0.579 | 235.9 min |
| YASA 0.7.0 | 0.532 | 0.518 | 57.6 min | 0.675 | 110.3 min |
| Plain CNN | 0.728 | 0.595 | 16.5 min | 0.086 | 803.2 |
| Class-balanced CNN | 0.479 | 0.525 | 8.4 min | 0.181 | 716.7 |
| Square-root-balanced CNN | 0.662 | 0.604 | 9.8 min | 0.114 | 853.9 |
| Augmented CNN | 0.727 | 0.591 | 11.5 min | 0.075 | 855.7 |
| Two-channel CNN | 0.653 | 0.554 | 9.4 min | not run | not run |
| Context model | 0.734 | 0.618 | 8.7 min | not run | not run |
| Context model, sqrt-balanced trunk | 0.737 | 0.607 | 10.2 min | not run | not run |
| Context model, balanced trunk | 0.712 | 0.592 | 10.1 min | not run | not run |
| Ensemble of seven models | 0.741 | 0.629 | 9.3 min | not run | not run |

The finding that matters is not which model wins. On the forehead headband the CNN beats the frozen feature
model clearly — kappa 0.728 against 0.661, sleep-time
error 16.5 against 31.8 minutes,
winning on 14 of 20 people. On
Sleep-EDF the same frozen CNN collapses: kappa 0.086
against the feature model's 0.579 and YASA's 0.675, and a sleep-time error of
803.2 minutes against
235.9 and 110.3, because it calls daytime wake sleep.

**The best model at home is the worst model away.** Choosing a wearable sleep model by its in-domain score
would have selected exactly the model that fails hardest on a new device. That is the argument for the
evaluation this project is actually about, and it is now demonstrated rather than asserted.

**5. Post-hoc adjustments that need no retraining.**

| Model and setting | kappa | macro-F1 | sleep-time error |
|---|---:|---:|---:|
| cnn — no adjustment | 0.728 | 0.595 | 16.5 min |
| cnn — prior adjustment chosen for kappa | 0.728 | 0.595 | 16.5 min |
| cnn — prior adjustment chosen for macro-F1 | 0.709 | 0.608 | 13.2 min |
| cnn — N1/N3 decision offsets chosen for macro-F1 | 0.700 | 0.611 | 12.1 min |
| cnn_context — no adjustment | 0.734 | 0.618 | 8.7 min |
| cnn_context — prior adjustment chosen for kappa | 0.734 | 0.618 | 8.7 min |
| cnn_context — prior adjustment chosen for macro-F1 | 0.734 | 0.618 | 8.7 min |
| cnn_context — N1/N3 decision offsets chosen for macro-F1 | 0.724 | 0.618 | 8.5 min |
| cnn_context_sqrtbalanced — no adjustment | 0.737 | 0.607 | 10.2 min |
| cnn_context_sqrtbalanced — prior adjustment chosen for kappa | 0.737 | 0.607 | 10.2 min |
| cnn_context_sqrtbalanced — prior adjustment chosen for macro-F1 | 0.737 | 0.607 | 10.2 min |
| cnn_context_sqrtbalanced — N1/N3 decision offsets chosen for macro-F1 | 0.713 | 0.615 | 10.1 min |
| cnn_sqrtbalanced — no adjustment | 0.662 | 0.604 | 9.8 min |
| cnn_sqrtbalanced — prior adjustment chosen for kappa | 0.662 | 0.604 | 9.8 min |
| cnn_sqrtbalanced — prior adjustment chosen for macro-F1 | 0.662 | 0.604 | 9.8 min |
| cnn_sqrtbalanced — N1/N3 decision offsets chosen for macro-F1 | 0.609 | 0.588 | 9.8 min |
| cnn_balanced — no adjustment | 0.479 | 0.525 | 8.4 min |
| cnn_balanced — prior adjustment chosen for kappa | 0.479 | 0.525 | 8.4 min |
| cnn_balanced — prior adjustment chosen for macro-F1 | 0.479 | 0.525 | 8.4 min |
| cnn_balanced — N1/N3 decision offsets chosen for macro-F1 | 0.479 | 0.525 | 8.4 min |
| cnn_augmented — no adjustment | 0.727 | 0.591 | 11.5 min |
| cnn_augmented — prior adjustment chosen for kappa | 0.727 | 0.591 | 11.5 min |
| cnn_augmented — prior adjustment chosen for macro-F1 | 0.710 | 0.608 | 8.9 min |
| cnn_augmented — N1/N3 decision offsets chosen for macro-F1 | 0.698 | 0.609 | 8.5 min |
| cnn_twochannel — no adjustment | 0.653 | 0.554 | 9.4 min |
| cnn_twochannel — prior adjustment chosen for kappa | 0.653 | 0.554 | 9.4 min |
| cnn_twochannel — prior adjustment chosen for macro-F1 | 0.629 | 0.567 | 9.9 min |
| cnn_twochannel — N1/N3 decision offsets chosen for macro-F1 | 0.630 | 0.558 | 9.5 min |
| ensemble_of_cnn+cnn_context+cnn_context_sqrtbalanced+cnn_sqrtbalanced+cnn_balanced+cnn_augmented+cnn_twochannel — no adjustment | 0.741 | 0.629 | 9.3 min |
| ensemble_of_cnn+cnn_context+cnn_context_sqrtbalanced+cnn_sqrtbalanced+cnn_balanced+cnn_augmented+cnn_twochannel — prior adjustment chosen for kappa | 0.741 | 0.629 | 9.3 min |
| ensemble_of_cnn+cnn_context+cnn_context_sqrtbalanced+cnn_sqrtbalanced+cnn_balanced+cnn_augmented+cnn_twochannel — prior adjustment chosen for macro-F1 | 0.741 | 0.629 | 9.3 min |
| ensemble_of_cnn+cnn_context+cnn_context_sqrtbalanced+cnn_sqrtbalanced+cnn_balanced+cnn_augmented+cnn_twochannel — N1/N3 decision offsets chosen for macro-F1 | 0.659 | 0.610 | 9.3 min |

Alpha and the offsets were chosen on the six selection people only and then applied unchanged to the 20 test people. Scalar temperature cannot appear here: it preserves the winning class, so it cannot move macro-F1 at all.


**6. Two ablations.**

| Variant | sequence weight | test kappa | test macro-F1 | test N1 F1 | test N1 recall |
|---|---:|---:|---:|---:|---:|
| cnn | 0.0 | 0.593 | 0.472 | 0.058 | 0.038 |
| cnn | 0.5 | 0.728 | 0.595 | 0.263 | 0.252 |
| cnn | 1.0 | 0.679 | 0.602 | 0.289 | 0.347 |
| cnn_sqrtbalanced | 0.0 | 0.577 | 0.518 | 0.221 | 0.244 |
| cnn_sqrtbalanced | 0.5 | 0.662 | 0.604 | 0.346 | 0.575 |
| cnn_sqrtbalanced | 1.0 | 0.535 | 0.545 | 0.289 | 0.675 |
| cnn_balanced | 0.0 | 0.492 | 0.497 | 0.238 | 0.409 |
| cnn_balanced | 0.5 | 0.479 | 0.525 | 0.269 | 0.708 |
| cnn_balanced | 1.0 | 0.346 | 0.436 | 0.215 | 0.825 |
| cnn_augmented | 0.0 | 0.605 | 0.489 | 0.082 | 0.057 |
| cnn_augmented | 0.5 | 0.727 | 0.591 | 0.240 | 0.222 |
| cnn_augmented | 1.0 | 0.675 | 0.599 | 0.283 | 0.310 |
| cnn_twochannel | 0.0 | 0.506 | 0.431 | 0.036 | 0.022 |
| cnn_twochannel | 0.5 | 0.653 | 0.554 | 0.175 | 0.160 |
| cnn_twochannel | 1.0 | 0.583 | 0.553 | 0.228 | 0.300 |

- **cnn**: smoothing moves test kappa from 0.593 to 0.728 and N1 F1 from 0.058 to 0.263, so it does **not** cost N1. Kappa would choose weight 0.5, macro-F1 would choose 1.0 — the two metrics disagree.
- **cnn_sqrtbalanced**: smoothing moves test kappa from 0.577 to 0.662 and N1 F1 from 0.221 to 0.346, so it does **not** cost N1. Kappa would choose weight 0.5, macro-F1 would choose 0.5 — the two metrics agree.
- **cnn_balanced**: smoothing moves test kappa from 0.492 to 0.479 and N1 F1 from 0.238 to 0.269, so it does **not** cost N1. Kappa would choose weight 0.0, macro-F1 would choose 0.5 — the two metrics disagree.
- **cnn_augmented**: smoothing moves test kappa from 0.605 to 0.727 and N1 F1 from 0.082 to 0.240, so it does **not** cost N1. Kappa would choose weight 0.5, macro-F1 would choose 0.5 — the two metrics agree.
- **cnn_twochannel**: smoothing moves test kappa from 0.506 to 0.653 and N1 F1 from 0.036 to 0.175, so it does **not** cost N1. Kappa would choose weight 0.5, macro-F1 would choose 0.5 — the two metrics agree.


**7. Support slides and a one-page handout.** `make_presentation.py` builds seven support slides and a
single-page printable handout directly from the result files, so no number on a slide is typed by hand. Outputs
in `slides/` and `handout/`, as HTML and PDF.

**8. Deviations and environment notes, recorded rather than smoothed over.**

- The pinned environment cannot be built inside the Linux workspace on the team's Mac: its package mirror offers
  numpy only up to 2.2.6, so `numpy==2.3.5` will not install there. The fresh-environment proof ran in a separate
  cloud Linux workspace from the snapshot tarball, with every pin satisfied.
- The CNN work uses JAX, not PyTorch. The PyTorch CPU wheel index is blocked by the network, and the PyPI wheel
  refuses to import without CUDA libraries. Neither machine has a GPU, so CUDA would have bought nothing; JAX on
  CPU was the working path, not a preference.
- The external epoch cache was read with MNE 1.12.1 rather than the pinned 1.13.2, again a mirror limit. Only
  signal reading is affected; the epoch grid, reference labels and quality mask come from the frozen v2
  prediction file, not from the reader.
- The CNN variants spend the six selection people on two choices (number of passes and sequence weight) where v2
  spent them on one. That is a small advantage to the CNN, not to v2.
- Every CNN result is a single seed and a single architecture. No repeated-seed variance is reported, so a
  difference of a few hundredths between two variants should not be read as a real ordering.
- Both evaluation cohorts have now been inspected by every model here. Nothing in this revision is a fresh
  independent test, and the external CNN rows are labelled exploratory development for that reason.

**9. A reporting convention that must be explained, not quietly changed.** Six of the 20 BOAS test nights contain
no reference N3 at all. Under the person-average convention this project uses, a stage absent from a night scores
F1 = 0 for that person, which no model can fix. The mean N3 F1 is therefore capped at 14/20 = 0.70 whatever the
model does, and the macro-F1 ceiling with perfect N1 and the best attainable N3 is 0.833. Published figures near
0.90 come from full polysomnography with many derivations plus EOG and EMG, scored against a five-technologist
consensus; this project has two forehead channels and a three-scorer consensus. The two numbers are not
comparable, and the convention stays as it is rather than being swapped for one that flatters the score.

**10. Every model met the external cohort, and the whole family failed it.** The ensemble became the
best model on the headband, so claiming "best overall" from one cohort would have repeated exactly the
mistake this project exists to expose. `external_all.py` scored every context variant and the ensemble on
Sleep-EDF with `yasa_evaluate.py`'s windows, masks and metrics, refitting nothing.

| Model on Sleep-EDF, 20 people | kappa | macro-F1 | sleep-time error |
|---|---:|---:|---:|
| Frozen v2 feature model | 0.579 | 0.478 | 235.9 |
| YASA 0.7.0 baseline | 0.675 | 0.624 | 110.3 |
| cnn | 0.086 | 0.188 | 803.2 |
| cnn_balanced | 0.181 | 0.307 | 716.7 |
| cnn_sqrtbalanced | 0.114 | 0.246 | 853.9 |
| cnn_augmented | 0.075 | 0.187 | 855.7 |
| cnn_context | 0.103 | 0.205 | 849.4 |
| cnn_context_sqrtbalanced | 0.099 | 0.210 | 842.4 |
| cnn_context_balanced | 0.089 | 0.196 | 846.7 |
| ensemble_of_7 | 0.098 | 0.217 | 842.6 |

Not one member of the CNN family transfers. The frozen feature model, which loses at home, is the only
model of ours that survives the move at all, and YASA beats everything. The two-channel variant could not be
scored here, because a model fitted on two headband channels cannot read a single-channel recording; that is
recorded rather than worked around.

**11. Confidence intervals, because twenty people is a small cohort.** `confidence_intervals.py` resamples
the 20 people 10,000 times, paired, and reports the 95% interval of each model's difference from the frozen
v2 model. On the headband the gains are established: the ensemble is +0.079 kappa [+0.032, +0.132] and
+0.067 macro-F1 [+0.040, +0.097], and its sleep-time error is 22.5 minutes lower [-37.9,
-10.1]. Two variants are not established by this cohort at all: the square-root-balanced CNN's kappa
difference is +0.000 [-0.057, +0.062] and the two-channel variant's is -0.008 [-0.056, +0.040]. There is no
multiplicity correction across twelve comparisons, so an interval that only just excludes zero is weak
evidence, and the file says so.

**12. The vendor comparison, and a failure gallery for the model that now leads.**
`vendor_and_failures.py` scores every model on the mask the released Bitbrain scorer covers, which is 98.2%
of scored epochs and the only like-for-like benchmark available: same headband, same people, same epochs,
against the model the manufacturer ships.

| On the vendor-covered mask, 20 people | kappa | macro-F1 | sleep-time error |
|---|---:|---:|---:|
| Released Bitbrain scorer | 0.787 | 0.616 | 14.4 min |
| Ensemble of seven models | 0.746 | 0.632 | 8.8 min |
| Context model | 0.739 | 0.621 | 8.4 min |
| CNN challenger | 0.735 | 0.599 | 15.9 min |
| Frozen v2 feature model | 0.666 | 0.563 | 31.3 min |
| YASA 0.7.0 | 0.534 | 0.518 | 55.9 min |

The vendor scorer keeps the lead on stage agreement. The ensemble is ahead of it on macro-F1 and halves its
sleep-time error. This, not a published figure from full polysomnography, is the comparison the pitch should
make.

The failure gallery (`results/vendor_and_failures/failure_gallery.png` and `.json`) takes the three
held-out nights where the leading model's total-sleep estimate is worst. All three fail the same way: N1 is
recovered at between 20% and 45% and most of the rest is called Wake. Three nights out of twenty illustrate the error;
they do not explain the aggregate, and the file says so.

**13. Seed variance, and the claims it forces us to withdraw.** Every CNN number in this plan comes from
a single seed. `seed_variance.py` retrains only the temporal head, trunk frozen, with five seeds and the
same selection procedure each time.

| Across five head seeds | mean | spread |
|---|---:|---:|
| macro-F1 | 0.607 | 0.015 |
| kappa | 0.731 | 0.004 |
| sleep-time error | 13.0 min | 10.7 min |

Three things follow, and the plan adopts all three. The 0.011 macro-F1 gap between the context model and
the one on the square-root-balanced trunk is **smaller than the seed spread**, so those two are not ordered
anywhere in this document. The 8.7-minute sleep-time figure is the fragile one: across seeds it ranges
9.6 to 20.3 minutes, so the defensible claim is that the context model roughly halves the frozen
model's 31.8 minutes, not that it reaches 8.7. Kappa is stable to 0.004 and its ordering survives.

A fourth observation, recorded because it bears on the reproduction claim: the shipped context model was
trained on the Mac's workspace with JAX 0.6.2 and the five runs above in the cloud workspace with JAX
0.12.0.1. The same seed gives 0.618 there and 0.616 here. Training reproduces to about two decimal places
across environments; only the frozen evaluation reproduces to fifteen significant figures. This study
changed no published result: the shipped model was restored afterwards.

**14. Two hyperparameter sweeps that found nothing, reported anyway.** `context_sweep.py` varied the
temporal head's context window over 7, 11, 15 and 23 epochs and then tried a wider 128×3 head, each
configuration selected on the six selection people alone. The selection people chose the shipped
configuration, 15 epochs with a 64×2 head. Across all five runs the test macro-F1 spans 0.601 to 0.617
against the shipped 0.618, and every difference is smaller than the 0.015 head-seed spread measured above.
Nothing here is an improvement. Two observations are still worth keeping: the window the selection people
chose is the one the published plateau predicted, which is a small independent check on the procedure; and
the selection people did *not* pick the configuration that scored best on the test people, which is exactly
why selection never touches them.

**15. A causal, streaming inference path, and what going live costs.** Everything else in this project
reads the future: the v2 features use a centred window, the smoother runs forward and backward over the
whole night, and even the amplitude normalisation divides by the whole recording's scale. A live demo built
on those functions would be a replay in costume, which section 7 already forbids. `live_inference.py` adds a
path where nothing at epoch t depends on epoch t+1: trailing-window normalisation warmed up over the first
10 epochs, the per-epoch CNN (chosen because it needs no context by construction, so going causal changes
nothing about its input), forward-only HMM filtering at the frozen sequence weight, and a running sleep-time
total. It reads a Lab Streaming Layer stream or replays a cached recording epoch by epoch.

The cost was measured on the same 20 people with the same mask and metric:

| Same CNN, 20 held-out people | kappa | macro-F1 | sleep-time error |
|---|---:|---:|---:|
| Retrospective, as reported elsewhere | 0.728 | 0.595 | 16.5 min |
| Causal and streaming | 0.690 | 0.556 | 17.7 min |
| Cost of going live | -0.038 | -0.039 | +1.2 min |

Two things worth saying out loud at the demo. The causal path still scores above the frozen v2 feature
model's retrospective 0.661 kappa, so streaming is not what breaks this system. And the frozen v2 model is
deliberately *not* offered in live mode: its features are defined over a centred window, and feeding it
trailing windows would quietly change what its weights mean.

What this does not establish, and must not be claimed: that the model is accurate on a device it has never
seen. The transfer failure measured in item 10 dominates anything measured here, and a live path that runs
is not a live path that is right.

**16. New Q&A answers, for the question bank.**

- *"You trained a better model tonight — why isn't that your result?"* — "Because it is the clearest evidence
  for our actual point. The CNN beats our frozen model on the headband, kappa 0.728 against 0.661, and then
  collapses on Sleep-EDF at kappa 0.086 with 803.2 minutes of sleep-time error. The best model at home is
  the worst one away. If we had picked a model by in-domain score, we would have shipped that one."
- *"Your macro-F1 is only about 0.629. Published sleep stagers report 0.90."* — "Different setting. Those
  numbers come from full polysomnography, many derivations plus EOG and EMG, scored against a five-technologist
  consensus. We have two forehead channels and a three-scorer consensus. And six of our twenty test nights
  contain no N3 at all, which under our person-average convention caps mean N3 F1 at 0.70 whatever the model
  does. We kept the convention and explain it rather than switching to one that flatters us."
- *"Would more data fix it?"* — "Not for stage agreement: the published learning curves for this task saturate
  at roughly 12 to 25 training recordings and we fit on 59 nights from 48 people. What more data would fix is
  transfer, and the right data is nights from other devices and recordings that contain daytime wake, which is
  exactly where both our model and YASA fail."
- *"Did anything you checked tonight go against you?"* — "Three things, and they are all in the plan. The CNN
  beat our frozen model. Our leakage audit removed the exposure model's only win. And the reproduction is not
  bitwise, only identical to about fifteen significant figures."

**Still open after this revision (team tasks):** rehearsals and the two adversarial Q&A rounds; the team's real
names and contributions on the credits slide; posting the organiser message and recording the confirmed code
freeze, submission format and pitch length in writing.

## Appendix A. Formulation rationale and the four research modules

These detailed modules are retained optional research/context, not a minimum hackathon build requirement. Their scientific constraints apply whenever such a module is implemented; no required sleep result depends on them. Section 12 is the latest decision contract: reviewed requirements and route-specific rules precede comparisons; a learned property or host-choice result requires its own eligible data and evaluation.

### Formulation user and evidence rationale

The user is a formulation or neuropharmacology researcher considering a documented payload/carrier/gel system. They need two separate answers:

| Decision | Evidence required |
|---|---|
| Why use this carrier/gel instead of the appropriate existing formulation? | A specific delivery limitation, a plausible mechanism and a relevant control |
| Why use the learned model? | Better held-out prediction or a demonstrated decision benefit beyond simple baselines |

More brain exposure is not automatically better. Methylphenidate already crosses the BBB and modified-release products exist [2,3]. Nanocarrier arguments must therefore specify the proposed advantage, such as a measured release or exposure difference, without assuming symptom benefit or lower toxicity.

**Primary formulation evidence anchor:** atomoxetine NLC in a poloxamer/carbopol in-situ gel, DOI 10.3390/pharmaceutics15071985 [28]. Its formulation/release sections and PK Tables 5-6 were inspected. It directly matches the requested particle-plus-gel format. Its behavioural study concerns L-methionine-related dementia-like impairment, not ADHD. Use it as an ADHD-payload delivery example, not ADHD efficacy evidence. Its few group means cannot train and independently validate a general model.

The atomoxetine lyophilised nasal-insert study is a separate record with unresolved reporting conflicts [20]. Methylphenidate transfersomal in-situ gel is another literature lead [4]. An insert is not automatically a thermogelling hydrogel. Keep these systems, their controls and their source findings distinct.

For the NLC-gel anchor, a same-route free-drug comparison is useful, but differences in nanoparticles, gel matrix and other formulation properties still need checking. It does not automatically isolate one nanoparticle mechanism. No numerical prediction for a new ADHD payload is presented as supported merely because the molecule has an established therapeutic indication.

### Four-module specification

| Module | Required inputs | Output and acceptance condition |
|---|---|---|
| Drug evidence | Compound identity/synonyms, indication, jurisdiction, source record and retrieval dates | Existing/emerging agents with source-linked mechanism, trial status and approval evidence; a trial entry cannot become an approval claim |
| Compatibility | Drug form and activity, particle, polymer, loading, gelation, route, stability and assay context | Supported observations, contradictions and unmeasured properties; every positive statement has matching evidence |
| Release over time | Measured active-payload loading, time-stamped observations, sampling/recovery protocol and assay conditions | Observed release and mass accounting; a forecast only within the evaluated task and support range |
| Exposure/outcome evidence | Verified systemic measurements, brain endpoint, interval, cohort and units; separately, functional/symptom measurements | Supported exposure result or a precise missing-data explanation; no automatic release-to-symptom conversion |

Drug evidence can use ClinicalTrials.gov, official regulator records and ChEMBL indication/mechanism resources [25-27]. Store approval status by indication, route, formulation, population and jurisdiction. A drug approved for another indication or route is not an approved nanoparticle-gel therapy. A completed trial without reported results is not positive evidence. Claims of new approvals in the uploaded research notes remain verification items until their official records are checked.

Use a small dated source snapshot for the hackathon. A later refresh records additions, changed status, source dates and conflicts. Scheduled ingestion is a future program feature, not an automation already operating. An optional LLM can assist retrieval/extraction, but sources and fields require review; its generated text is not experimental evidence or the main evaluated AI contribution.

The research report exports the target question, eligible records, source links, measured/modelled/simulated status, predictions and baseline errors where available, support limits, conflicting findings and next measurement. The disease selector changes evidence requirements. It does not automatically create a disease-specific model.

## Appendix B. Conditional delivery-exposure model — later research

This preserves the earlier NEURORELEASE experiment as a separate research protocol. It is not the required hackathon AI model, not an EEG-to-drug-response bridge and not an implementation commitment. It starts only after eligible paired PK data, independent source groups and the stated units/compartments have been verified. Its inference requires measured systemic exposure and cannot be driven by a sleep-stage estimate instead.

### Task and inference time

Predict natural-log brain AUC from measured natural-log systemic AUC, formulation descriptors and declared context. State the numerical unit convention for both logs and the verified systemic compartment. Freeze one harmonised endpoint/basis before fitting. The decision occurs after the systemic observation interval; using full-interval AUC cannot be advertised as an early-time or pre-laboratory forecast.

Separate whole blood, plasma and serum unless a source-supported conversion exists. Separate brain-tissue mass, homogenate-volume and other concentration bases. Time units can be converted explicitly; tissue mass cannot be turned into volume using an invented density or dilution factor. Do not pool incompatible assay families simply by adding a categorical flag.

### Hypotheses and baselines

1. Formulation information improves prediction beyond systemic exposure and context.
2. Measured physiology improves prediction beyond descriptors/context alone.
3. Support restrictions reduce error at an explicitly reported loss of coverage.

Compare the full regularised linear model with two matched linear ablations: systemic/context only, and descriptors/context only. Also include a training-derived route-specific brain/systemic partition-ratio baseline where the compartments and units support it. One modest nonlinear model is a secondary comparison; do not attribute a feature gain to physiology when learner complexity also changed. Compare variants on identical eligible records.

For this conditional PK task, the primary metric is source-held-out MAE on log brain AUC. Also report original-unit errors after back-transformation, independent study count, per-study errors and the covered/rejected fraction. Report failed improvements honestly. A model that does not beat the baseline cannot carry an AI-superiority claim.

### Independence and leakage

- Resolve study-title strings to source identities; group linked publications, repeated formulations, cohorts and time points where they share experimental information.
- Fit imputation, encoding, scaling, feature selection and tuning on training groups only. Do not impute missing target values.
- Exclude brain AUC, brain Cmax, brain/systemic ratios, DTE/DTP and other target-derived quantities from inputs. Avoid mathematical coupling when a predictor also occurs in a ratio target.
- Do not delete held-out studies because the model predicts them poorly. Exclusion rules concern source validity and intended scope, are recorded before performance comparison, and apply consistently.
- Keep the ADHD/hydrogel application case outside training if it is used to assess payload transfer. A study split does not demonstrate support for an unseen payload or gel.
- Never create independent samples by slicing one curve into random rows, treating electrodes as independent cultures, or sampling group means and SDs into fictional subjects.

Use source-grouped validation appropriate to the actual independent groups. With sparse groups, report leave-study-out case results as exploratory and limit the claim. Do not invent a sample-size threshold that certifies generalisation. Freeze the test assignment and model choices before final reporting.

### Uncertainty, support and candidate comparison

Report empirical study-resampling/model stability. It is not automatically a calibrated prediction interval. Coverage claims require separate calibration/testing and enough independent groups; new chemistry/routes may invalidate them.

Support checks include payload class, material, route, gel context, assay, units, time, physiological range and required fields. Missing is not zero. Unsupported inputs produce a specific measurement request rather than a confident numerical ranking. Evaluate error versus retained coverage; rejection is not itself proof of safety.

The initial candidate set consists of documented formulations. Independently editing size, charge or PEG does not establish a manufacturable recipe or causal benefit. A measured plasma input from formulation A cannot be held fixed while changing formulation B and interpreted as B's exposure. Within-study ranking is secondary and requires compatible controls. A target-centred screener must not use the held-out study's outcome mean to make a deployable absolute prediction; retrospective centring for a ranking metric must be labelled accordingly.

## Appendix C. Delivery data inventory and completed source investigation

| Source | What was actually inspected | Use and unresolved issue |
|---|---|---|
| Yousfan supporting workbooks [5,18] | Both XLSX files opened; larger table has 403 identified records, positive numeric brain/blood AUC pairs and 160 distinct trimmed study-title strings | Candidate general corpus; these are raw counts, not accepted training counts |
| Smaller Yousfan workbook [18] | 133 rows after the header, 15 columns | Reduced table; do not append it as an independent corpus |
| Atomoxetine NLC-gel [28] | Primary formulation/release text and PK Tables 5-6 | Lead application evidence; summary groups and a non-ADHD behavioural model |
| Atomoxetine inserts [20] | Article, methods and Table 7 | Separate audit case; route/endpoint conflicts must be reconciled |
| IFE.PTML [7] | 431 rows, 16 columns; binary output with 243 zeroes and 188 ones | Processed classification data, not quantitative release trajectories or paired PK; definitions, provenance and reuse need checking |
| LipoDB [6] | Landing page; archive retrieval did not produce usable bytes | No claim that its actual curves or brain endpoints were audited |
| EPA neural nanoparticle archive [15,16] | 98 entries; all 17 workbooks contain statistical-model output fields; referenced master CSV absent | Cannot supply the linked early/late observations required for the neural forecasting alternative |
| PC12 nanoparticle paper [21] | Article and supplement descriptions | Described normal-culture traces do not establish a replicated exposed/control forecasting corpus; workbook not audited |
| ADHD methylphenidate EEG [8] | Published access statement | Restricted individual data; not a verified paired nanocarrier source |
| New hydrogel workbook [53] | All four sheets; 19 catalogue entries and three example records; two cached formula results checked | 18 excerpt-only leads and one table to build; not a downloaded hydrogel training corpus |
| New matched-pair workbook [54] | Both sheets; nine rows/eight paper labels; no typed numeric fold or dose cells | Curation register and interpretation cases; source grouping and quantitative extraction pending |
| New AI decision and coating/core notes [52,55] | Both texts read; fields, route distinctions and evidence gaps incorporated | Decision contract and future experiments; no verified host-specific coating/core predictor |

Yousfan route counts are IV 175, IN 157, PO 48 and IP 23. A keyword check did not find methylphenidate, amphetamine, atomoxetine, guanfacine, clonidine or ADHD in those records. This is a scoped audit finding, not proof that every possible ADHD-relevant mechanism is absent. Study titles are not verified DOIs or biological replicate counts; the Subject column is not established as a patient ID.

**Source discrepancy A:** Subject 0 / Excel row 2 cites the baicalin study [19]. Curated blood AUC AM2 is 267 under an hours-based header. Source Table 3 reports normal-rat liposome plasma AUC values 2266.38 and 2676.03 for different integration limits on a minutes-based scale. Cohort, units and interval have not been reconciled. Quarantine the record pending independent extraction; do not guess a replacement or declare the entire corpus invalid.

**Source discrepancy B:** the atomoxetine insert paper describes an intraperitoneal reference in methods but labels it IV in Table 7. S4 finite brain AUC differs between table and narrative, and its tabulated infinite AUC is below its finite value [20]. Preserve the conflict and withhold affected quantitative comparisons. This is not an allegation about the authors.

ACS Figshare lists the supplements as CC BY-NC 4.0 [18]; preserve attribution and the noncommercial restriction rather than claiming unrestricted commercial training rights. A code licence does not automatically license the underlying compiled data. The downloaded workbook MD5 values match publisher metadata. No accepted modelling corpus, fitted model, held-out performance or working application was produced by the planning audit.

## Appendix D. Biological evidence and comparability engine

Every observation has a DOI/source location, cohort, formulation, analyte, assay, units/denominator, route, interval, model and evidence status. Comparability rules are explicit research logic, evaluated separately from the learned predictor.

For each declared target, count unique included source studies N, studies with at least one eligible observation n, unresolved studies u with no eligible observation but at least one plausible incomplete record, and known mismatches. Deduplicate the union when a study occupies several route/readout cells. Show perfusion and functional-readout coverage with their denominators. These are corpus descriptions, not model accuracy or a systematic-review prevalence estimate. Later validation can test whether accepted comparisons agree more often across independent sources than rejected ones, for one prespecified common endpoint.

### Barrier and route table

| Context | Required distinctions |
|---|---|
| Healthy unopened BBB | Species, region, age, assay, perfusion and localisation evidence |
| Disease-modified BBB | Exact model, stage, age and measured barrier change; disease name alone is insufficient |
| Glioma/blood-tumour barrier | Tumour and region, host immune status, vascular versus interstitial signal; not a healthy-BBB surrogate |
| FUS-assisted delivery | Acoustic dose card and opening/injury evidence; distinct intervention strata |
| Peripheral nerve/BNB | Vessel bed, injury state and sampled compartment; do not transfer it into brain rows |
| Intranasal | Deposition, nasal residence, measured systemic exposure and evidence for each pathway; not direct proof of IV BBB transport |
| Direct brain/ICV/intrathecal | Local delivery; do not score as proof of crossing an unopened BBB |
| In-vitro barrier | Cell/co-culture model, flow, medium, TEER/tracer and measurement format; separate from in-vivo uptake |

Intranasal delivery may include systemic and direct routes; neither “all BBB crossing” nor “none involves systemic exposure” is assumed. Do not impose one universal leak or TEER threshold from a proposal note. Record measured values and a justified context-specific classification.

### Supported claims, with compartment

Association, demonstrated barrier passage, target localisation, intracellular function and therapeutic effect are separate supported-claim fields. Endothelial expression can demonstrate intracellular function in endothelium without demonstrating parenchymal passage. Do not automatically mark every lower “rung” true. Bulk brain label, intact carrier, released active payload and protein expression are different analytes/readouts.

Imaging signal alone does not automatically prove intact-particle parenchymal delivery. Perfusion and capillary depletion are different procedures; a hand-entered vascular-risk score cannot numerically subtract vascular binding without measurements. Reporter expression needs its cell compartment. An efficacy-only paper does not provide an uptake label.

For intracellular payloads, record endosomal escape/cytosolic availability separately from cellular internalisation. Evidence of passage through a barrier also does not establish subsequent movement through brain interstitium. Treat barrier passage and tissue penetration as two different measured questions, without a universal particle-size cutoff.

### Unit and control rules

| Comparison | Required action |
|---|---|
| %ID versus %ID/g | Require organ mass, same analyte/dose basis, time and sampling; preserve the conversion |
| Tissue amount versus %ID | Require injected amount on the same basis |
| Brain/systemic ratio | Require compatible units, compartment normalisation and matched interval; retain numerator and denominator |
| Share of total organ signal | Keep denominator/cohort and absolute signal; no ranking against %ID/g |
| Fold change | Require matching outcome, comparator type, dose basis, timing and normalisation |

Drug targeting efficiency is not percentage of injected dose. Administered cargo mass, particle mass, particle count and surface area are distinct dose bases. A label may detach or degrade; elemental or radiolabel measurement does not by itself identify intact active material.

For each pair list changed formulation, procedure and assay fields, plus unknowns. A count is not a causal-confounding score or proof of exchangeability. A single recorded difference does not exclude unmeasured confounding. Known context conflict produces a mismatch verdict; missing critical information produces an unresolved verdict and named evidence request.

Carry the original Ohta exposure, Han absolute-signal/share, masked-perfusion, endothelial-function and unit/control cases into verification [38,40]. They are stress tests of interpretation, not training labels borrowed from unrelated systems. A FUS record includes pressure value/type, frequency, burst/repetition, duration, microbubble type/dose, injection timing, opening confirmation, region, anaesthetic/carrier gas and observed injury. Dextran pressure/size findings are context flags, never universal nanoparticle pore cutoffs.

## Appendix E. Materials, payloads and host-state knowledge

### Material and payload ontology

Keep bulk hydrogels, nanogels, nanoparticles, protein/antibody shuttles and viral vectors distinct. A full record links payload, carrier, ligand and gel; none can be inferred from the other. Payload classes include small molecules, defined peptides, peptide mixtures, proteins/enzymes, antibodies/fragments and nucleic acids. A peptide mixture has no single justified molecular weight or charge. DNA encoding GDNF is not GDNF protein. A transferrin-binding fusion protein is not automatically a nanoparticle.

The supplied carrier landscape covers PLGA/PLA/PBCA polymers, liposomes, SLN/NLC, LNPs, micelles, dendrimers, nanogels, extracellular vesicles and inorganic materials [35-37]. They are an evidence catalogue, not interchangeable candidates or a licence to predict new material classes. Retain size method/medium, PDI, zeta medium, PEG presence/MW/density/role, ligand identity/amount/affinity/valency where measured, corona conditions, loading and retained activity. Co-varying ligand, PEG and size must stay visible.

### Hydrogel families

| Family in supplied research | Place in this project |
|---|---|
| Nasal poloxamer with mucoadhesive partner | Lead formulation family to characterise, anchored to the actual atomoxetine NLC-gel study |
| Chitosan/starch/CMCh and thermoresponsive protein/peptide systems | Separate literature candidates; require their own payload/route/assay evidence |
| HA/HAMC, collagen and self-assembling peptides | Local protein/graft/depot research context; direct CNS delivery cannot become an ADHD nasal-delivery validation |
| PLGA-PEG-PLGA peripheral/local depots | Distinguish release location from measured brain exposure |
| Silk fibroin, ROS-responsive and PNIPAM systems | Background candidates with unresolved clearance, trigger or safety questions; no generic “non-toxic” selection |

Record gelation trigger, sol-gel conditions/time, polymer/crosslinker composition, rheology/modulus, injectability, swelling/erosion, active recovery, degradation, storage and route-specific tolerability. Poloxamer 188 and 407 are not interchangeable identities. Drug dissolved in gel, encapsulated in particles, adsorbed onto particles and covalently attached are distinct loading mechanisms.

Batch reproducibility, sterilisation effects, residual reagents and payload activity after storage are translation questions. The software records available evidence and requests missing tests; it does not certify manufacturability or set a formulation recipe.

The Alzheimer hydrogel DOCX and Parkinson hydrogel notes remain source-specific research summaries, not clinical endorsements. Do not import their abstract-derived concentrations, release durations, stiffness ranges or trigger thresholds as universal formulation recipes. An adsorption hypothesis for oxytocin requires verified charge, loading/activity and combined-system assays.

### Disease and host-state coverage

| Disease/context | Retained role and boundary |
|---|---|
| ADHD | Initial drug-evidence context; target-side DAT/NET evidence is not endothelial transport evidence; hypertension confounds SHR barrier interpretations |
| Autism | Model-, age-, tissue- and method-specific receptor/barrier evidence; genotype or antibody data are hypotheses for particle uptake, not validated personalised coefficients |
| OCD | Supplied map reports sparse barrier evidence; missing host state stays missing rather than a neutral multiplier |
| Parkinson/Alzheimer | Separate exact models, immune status, region and compartment; uptake, graft survival, protein release and symptom effects are different outcomes |
| MPS and other genetic/metabolic conditions | Protein-delivery and ADHD-like-symptom context, not relabelling these conditions as ADHD |
| Glioma, stroke and peripheral nerve | Preserve the original mechanism comparisons; distinct evidence strata, not interchangeable training outcomes |

The supplied receptor maps [31-34] are incorporated as an evidence layer, with their limits. Blood junction-protein levels do not directly measure endothelial receptor density. Transcript, protein abundance, ligand binding and particle transport do not substitute for one another. Altered small-substrate transport does not establish intact transport of an amino-acid-ligand nanoparticle. A receptor increase is not a guaranteed uptake gain.

Group-level host values constant within study cannot identify individual host effects separately from study effects. Do not manufacture donor-level training pairs from separate proteomic and nanoparticle papers. Personal-proteome predictions require paired donor/particle outcomes and donor-held-out validation; that experiment is outside this hackathon. Conflicting measurements retain method and source rather than being averaged into a disease multiplier.

### Therapeutic landscape

Keep oxytocin, TAT-DATNT, Semax, Cortexin/Cerebrolysin mixtures, neurotrophins, Parkinson biologics, enzyme therapies and gene therapies as separately typed evidence leads [35-37]. Vendor claims and untraced percentages do not enter scored records. A preclinical peptide injected into ventricles is not a validated nasal therapy. A biomarker or genetic association is not an effective therapeutic agent. A failed or negative efficacy study remains visible even if delivery improves. Clinical mimic/comorbidity lists do not become an automated diagnostic or treatment-selection algorithm.

## Appendix F. Loading, release, onset and clearance

For an assay with measured initial active mass M0 and cumulative released fraction F(t), released mass is M0 × F(t). Remaining mass equals M0 × (1 − F(t)) only if degradation and other losses are negligible. Otherwise account for degradation, adsorption, sampling loss and recovery, or mark remaining active mass unknown. State the loading denominator, volume, free/encapsulated fraction, sink conditions, medium replacement and analytical recovery.

Preserve observations outside expected recovery/mass balance as flagged measurements; do not silently clip them. Plot measured, fitted, predicted and simulated segments with different labels. AUC/Cmax/Tmax summaries do not identify a full observed curve. The timeline does not actuate an administered gel.

If real profiles support an auxiliary forecast, fix an early observation window, predict the unseen portion and compare with a suitable kinetic/persistence baseline given the same information. Keep whole profiles, formulations and source studies grouped. One curve demonstrates a fit, not generalisation. In-vitro release cannot become in-vivo release or brain exposure without matched evidence.

Maintain separate times for gelation, first measured release, systemic/brain Tmax, observed functional response, toxicity onset and clearance/recovery. Release, dissolution, biodegradation and elimination are different processes. Short-term absence of tissue damage does not demonstrate chronic nasal tolerability or long-term brain clearance. No generic safe dose, human onset time or removal method is inferred from heterogeneous cell/animal values.

The safety evidence record includes material/payload dose basis, route, concentration, exposure/repeat schedule, endpoints, cell types, medium/corona, solvent/surfactant/crosslinker residues, recovery interval, inflammation and clearance measurements. Missing toxicity means unknown. Delayed injury, glial/endothelial effects, ion release and retained active species are explicit review questions. Literature thresholds remain scoped to their actual experiment; the prototype offers no human dose calculator.

## Appendix G. Delivery prior work and screener implementation status

Qiu et al. already describe ML and optimisation for brain delivery [29]. The released utility inspected here uses GroupKFold with `number`; its mapping to source studies requires provenance confirmation. The inspected main script also filters groups using out-of-fold rank error. Do not claim that Qiu used only random row splits or that grouped validation alone is our novelty. Do not copy performance-conditioned exclusion into our test protocol.

Yousfan already models brain-delivery data [5]. Release ML also has precedents [17]. Neither an uncertainty band nor a dashboard makes this the first formulation AI system. The conditional formulation-research contribution would be the evaluated incremental value of measured physiology inside a source-audited, route/compartment-aware workflow that exposes unsupported transfers and asks for a decision-changing measurement. It remains a proposed contribution until evaluated.

The screener README describes synthetic smoke testing, assay separation, matched-control contrasts and host-state fields [30]. The supplied ZIP is a research pack and contains no executable screener Python source. We have not rerun its claimed smoke test. Existing code or a six-record prototype mentioned in older plans must not be presented as inspected/current implementation without obtaining and checking it. Respect the event's code-timing rules.

## Appendix H. Standards carried forward with boundaries

| Reference | Concrete use | Claim boundary |
|---|---|---|
| ISO/IEC 17025:2017 [9] | Method/context, uncertainty, decision basis and review log | Process analogy; no laboratory accreditation or automatic metrological traceability |
| ISO 45001:2018, 2024 amendment [10] | Qualified material-form, handling and hazard-review prompts | No completed workplace risk assessment or handling clearance |
| ISO 9001:2026 [11] | Design inputs, verification records, consequences and change history | Edition status checked; no unverified clause mapping or certification |
| ISO 14001:2026 [12] | Material, waste and end-of-life questions | No life-cycle assessment or environmental safety/compliance score |
| ISO 13485:2016 [13] | Conditional future reference if intended use establishes a relevant device role | Not the default regulatory route for a nanoparticle medicine; no conformity claim |
| ISO 14971:2019 [48] | Conditional future medical-device risk-management reference: intended use, hazards, risk controls and verification | No completed risk-management file, acceptable-risk declaration or certification; classification depends on intended use |
| IEC 62304:2006 + AMD1:2015 [49] | Conditional medical-device-software lifecycle roadmap: requirements, architecture, verification, configuration and problem resolution | No lifecycle compliance claim; not an automatic certification requirement for a research hackathon prototype |
| MIRIBEL [14] | Material/biological characterisation and protocol fields | Reporting guidance, not ISO; do not claim a complete checklist without checking it |

Current ISO 9001/14001 edition status was rechecked against official ISO material. Preserve the older standards numbers in source history while using the current reference where appropriate. Source links and an evidence log do not constitute most of a quality system. A mentor's feedback is not clinical validation. Question-specific reporting coverage means verified required fields divided by required fields, not probability of efficacy or certification.

ISO 45001 does not establish human-participant consent or a clinical research ethics framework. Teammate consent, privacy, retention and stopping rules must be addressed directly. Adding ISO 14971 and IEC 62304 is a future intended-use roadmap, not a claim that the prototype is a cleared medical device or that its predictions are clinically safe.

## Appendix I. Attachment and conversation traceability audit

| Supplied material / requirement | Review and final disposition |
|---|---|
| Add AI determine.docx [52] | Both pages read; all ten gaps converted to fields/rules in section 12; universal route/size/stiffness/duration assumptions qualified |
| hydrogel_ai_datasets.xlsx [53] | All four sheets inspected, including formulas and cached counts; catalogue versus examples/training distinction recorded |
| matched_pairs.xlsx [54] | Both sheets inspected; assay/source grouping, missing quantitative fields and approximate effects retained without invented labels |
| coating_core_evidence.md [55] | Read in full; PEG/heparosan, core, gas-bubble and donor-study distinctions incorporated; unprovided CSV/code and threshold claims not treated as executed/validated |
| NEURORELEASE Revision 6 | Read in full; formulation/PK/evidence content incorporated here, with the former PK core moved to conditional Appendix B |
| Goldilocks AI v3 [45] | All 20 pages read; EEG night core, gated day task, timing overlay, roles and demo integrated; unsupported readiness/causal/clinical claims corrected |
| Team-supplied core/challenge critique (C1-C6; routes A-C) | Incorporated in sleep-first scope, the uncertainty experiment and section 11; performance-based promotion to treatment claims, assumed sponsor penalties and unverified lock/checkpoint times not adopted |
| Team-supplied competition brainstorming list | Useful items incorporated in section 10 with priorities, owners and completion evidence; unsupported winning probabilities, causal error explanations, readiness and endorsement claims not adopted |
| Team-supplied v4-versus-Revision-7 comparison | Pasted feedback incorporated in the competition brief and section 9; separate v4 PDF not located or claimed reviewed |
| BOAS public metadata and annotations [46] | Participants, acquisition/event definitions and all 256 paired event TSVs inspected; counts and released-score coverage reproduced; raw EDF loading/training still pending |
| Both participant-information PDFs [1,44] | All four pages read; duplicate rules reconciled; AI input/evaluation, code timing and one-project rule retained |
| Current 40-page NeuroBridge plan and earlier 36-page plan [40,41] | Text retrieved; updated biological method, evidence cases, standards and operations reconciled; old scope/track/superiority assertions superseded |
| VETO [42] and earlier THREAD discussion | Superseded alternative concepts; not merged into this entry |
| Dataset screenshot [43] | Visually read; all three linked sources accounted for; no unsupported drug/gel linkage created |
| Landscape notes, part 1 (PDF) [36] | All seven pages visually inspected; peptide/mixture candidates, cargo identity and evidence limits retained as landscape leads |
| Landscape notes, part 2 (PDF) [37] | All seven pages visually inspected; carrier examples, atomoxetine NLC-gel lead, route/label/control caveats incorporated |
| ZIP research pack [35] | All ten entries inventoried; six Markdown notes read, hydrogel DOCX text extracted/reviewed, duplicate PDF hashes checked; no executable screener source present |
| Six ZIP Markdown notes | Combined research; in-situ materials; PD hydrogels; ADHD-like conditions; nanoparticle platforms; imaging: covered in Appendices C-G and their source-specific limitations |
| Alzheimer hydrogel DOCX inside ZIP | Nasal/local/peripheral route separation, material families, release-versus-exposure and trigger/safety gaps retained; abstract-level claims not promoted |
| additional bioengineering parts.pdf [38] | All ten pages' text reviewed; 13 gap categories, typed units, exposure, functional compartment, FUS card and source corrections carried forward |
| nanoparticle_screener_readme.md [30] | Read; assay/grouping/contrast ideas retained; synthetic smoke-test and code-completion claims not represented as verified implementation |
| evidence_map_and_proteome_feasibility.md [31] | Read; host-state identifiability and uptake-versus-efficacy cohort separation incorporated |
| ADHD/OCD/autism receptor maps [32-34] | Read; tissue/method/model specificity, missingness and individual-versus-group boundaries incorporated; unsupported uptake multipliers corrected |
| 13024_2018_Article_252.pdf [39] | All three pages read; spectroscopy commentary, not a nanoparticle training dataset or evidence of drug efficacy; future readout context only |
| Therapeutic agents, nanoparticles and in-situ hydrogel discussion | Chemical/biological payload meaning preserved; four modules, typed cargo/carrier/gel system and timeline specified |
| Symptoms, diseases and new/old drug tracking | Research indication/outcome metadata and dated source tracking; no symptom-to-cure inference |
| ISO request | All carried references mapped in Appendix H with current-edition and conformity boundaries |
| Standout/top-university competition objective | Prior-work comparison, falsifiable model contribution, source challenge and judge-operated reveal; no unsupported winning probability |
| “Include everything” | Every accessible project source family has a disposition; relevant evidence retained, contradictions corrected, unsupported claims explicitly withheld rather than copied |

The combined research note references additional original notes and many primary papers that were not individually supplied. Its summaries do not make those originals independently reviewed. Untraced vendor figures, old clinical recommendations, claimed current approvals without official confirmation, universal safety/size/TEER cutoffs and receptor-to-particle transport assumptions are excluded from supported outputs. The final plan can be complete in its accounting while those scientific claims remain unverified.

## Appendix J. Delivery-workbench acceptance cases

These apply to the formulation workbench and, where relevant, the conditional PK model. They are required future checks; they have not been executed against a built application by this document audit.

| Case | Required behaviour |
|---|---|
| Supported measured record | Correct endpoint/basis, source and model/baseline output |
| Missing systemic exposure | Withhold physiology-conditioned prediction; request the measurement |
| Blood/plasma, tissue/volume or interval mismatch | Reject comparison or require a documented conversion; no silent pooling |
| Target-derived feature | Exclude it from fitting and inference |
| Duplicate source across train/test | Stop evaluation and repair grouping |
| New protein, drug, carrier, gel or route | Apply support rules; no confident transfer without evaluation |
| Missing host-state value | Keep missing; no neutral disease multiplier |
| Group proteome presented as individual | Reject personalised interpretation |
| Organ share compared with %ID/g | Reject numerical ranking and request the missing denominator |
| Endothelial expression used as brain entry | Restrict claim to the measured compartment |
| Loading/degradation/recovery mismatch | Flag mass accounting; do not invent a remaining active mass |
| Toxicity, onset or clearance absent | Show unknown, not zero or safe |
| Clinical trial or off-indication approval | Retain the exact source/status; no approval/efficacy promotion |
| Source values conflict | Preserve both locations; quarantine affected magnitude |
| Model does not beat baseline | Report failure; remove superiority claim |
| Retrieval fails or internet unavailable | Use the dated attributable snapshot and show its age |

## Appendix K. Coating and core evidence update (C01–C10), Revision 18

Research backlog only. Nothing here is part of the scored EEG core, the demo or the pitch; it extends section 12's coating/core rules. **Nothing can be fitted yet**: the data hold one computed feature, four pairs from a single study and weak priors. The coating/core choice model needs matched pairs from at least 10 studies (continuous host property) or 5 studies per group (yes/no property), and refuses to recommend below that.

### Computed from reported numbers (arithmetic re-checked 7 Nov 2026)

| Quantity | Value | Status and limits |
|---|---|---|
| Anti-PEG antibody occupancy = S / (S + Kd) | 10 kDa PEG: 0.22, 0.84, 0.98 at 0.5, 9.6, 110.5 µg/mL IgG; 2 kDa at 9.6 µg/mL: 0.045 | Hypothesis feature, not a validated predictor. S assumes two binding sites per 150 kDa IgG and one site per chain; multi-site binding on a dense coat is ignored, so binding is understated. Kd: 2 kDa 2690, 5 kDa 249, 10 kDa 24, 20 kDa 44 nM (C03); log-log interpolation only within 2–20 kDa |
| Heparosan minus PEG uptake, log10, macrophage-type cells | 2.36 (RAW264.7, 55 nm), 1.85 (J774A.1), 1.65 (DC2.4), 1.32 (RAW264.7, 15 nm) | All “up to” values from one study (C01); four non-phagocytic lines reported only as “similar”, so magnitude missing |
| Anti-PEG prevalence prior | 83% of 500 donors; other cohorts 24–72% (C03) | Record assay and positivity threshold with every value |
| Anti-heparosan prior | 0 of 3 rats positive; exact one-sided 95% upper bound 0.63 (C02) | No human data, so not wired in |

### References and their effect on the model

| ID | Source | Effect on the model | Built? |
|---|---|---|---|
| C01 | Yang et al. 2022, Nano Letters | Coating contrasts fitted against a host property, never as one global heparosan effect; per-cell-line effects load as seed pairs; cell uptake is its own model family; heparosan gets no assumed stealth advantage | Phagocytic index (or yes/no flag) column on every cell-uptake row: not built |
| C02 | Heparosan-coated liposomes (authors not shown in the page read) | No usable rows (peripheral tumour, no brain outcome); sets the no-head-start prior for heparosan circulation | Prior only |
| C03 | Deuker, Mailänder, Morsbach, Landfester 2023, Nanoscale Horizons | `peg_antibody_occupancy` from donor titre (`anti_peg_igg_ug_ml`) and PEG chain mass; prevalence priors | Described as `priors.py`; no outcome pairs, so no effect estimated |
| C04 | Loughborough University repository: stealth nanoparticles and brain cell populations | `cell_type` column so coating contrasts are fitted within cell type; control coat is dextran, so only PEG versus dextran; neuron values under 0.5% are censored, not entered | Column not built; per-cell values still need extracting from figures |
| C05 | Neubert et al. 2015, Int J Nanomedicine 10:2033–2049 | Excluded from contrasts (coating, size and charge all differ); outcome is toxicity | `culture_type` column (mono versus coculture) not built |
| C06 | Dasgupta et al. 2024, Beilstein J Nanotechnol | Method precedent: fit cell types separately, hold out whole studies; test AUCs 0.891, 0.861, 0.565 as predictability bounds | Not entered as rows |
| C07 | Bittner et al. 2019, Beilstein J Nanotechnol | Pair rejected (coating differs with core) | **Gap:** pairing rule must also check zeta potential (these differ by 10–20 mV) |
| C08 | Hu et al. 2024, Adv Compos Hybrid Mater 7:76 | Core class `gold_shell_on_iron_oxide`; route and magnetic field as context columns | Columns not built; abstract, text and table disagree, so any rows carry a flag |
| C09 | Fan et al. 2016, Theranostics 6:1542 | Dropped by existing rules (disrupted barrier; 3–5 µm outside the 3–1000 nm check); adds core class `gas_core_lipid_shell` | Class only |
| C10 | Int J Mol Sci 2024, 25(15):8404 | Candidate host feature: glycocalyx coverage (brain 40.1%, heart 15.1%, lung 3.7%) and thickness (301, 136, 65 nm); varies by organ, not person; enters through the host-state layer with species as covariate | Not wired in; no particle outcomes |

### Data file and open items

- All 61 extracted values are in `coating_core_numeric.csv` (C03 20, C01 15, C10 7, C02 4, C08 4, C06 3, C09 3, C04 2, C07 2, C05 1). Six rows have no value. Values come from page text, not figures; several papers contradict themselves (C08's magnetic-field result, C01's coating density). Check a row against its paper before relying on it.
- The code described for this update (`priors.py`, the pairing-rule changes) and the notes-column version of the table were **not found on the team's Mac** on 7 Nov; retrieve them from the chat that produced them before claiming they exist.
- To build, in order: zeta-potential check in the pairing rule; `cell_type`, `culture_type`, phagocytic-index, route and field columns; figure extraction for C04; then re-run the study-count gate. Until that gate passes, the module outputs evidence requests, never a coating or core recommendation.

## Appendix L. CNN challenger family: protocol, variants and results

This appendix exists so the CNN results can be checked rather than believed. Code: `cnn_challenger.py`,
`cnn_external.py`, `cnn_context.py`, `cnn_posthoc.py`, `cnn_ablation.py`. Outputs: `results/cnn*/`.

### What is held identical to the frozen v2 experiment

- The person split exactly as recorded in `protocol.json` and `v2_protocol.json`: 48 fitting people over 59
  nights, six selection people (pids 10, 15, 18, 36, 38, 84), six probability-calibration people (32, 33, 62,
  80, 96, 98) and the 20 held-out test people, confirmatory nights.
- Channel sharing: one model over single-channel epochs with the two channels' probabilities averaged, which is
  what `v2_model.probabilities` does. The two-channel variant is the deliberate exception and is labelled.
- The retrospective sequence smoother from `v2_model.smooth`, its candidate weights [0, 0.5, 1.0], and selection
  by mean person score on the six selection people.
- Temperature fitted on the six calibration people with v2's objective.
- Evaluation mask `good & (reference < 5)`, `stage_metrics` from `pipeline.py`, and total sleep time as
  0.5 minutes per non-wake epoch inside the mask. The test reference, quality mask and epoch onsets are asserted
  equal to the frozen `results/v2/historical_boas_predictions.json` before any score is computed.

### What differs, and why

- Input is the signal: 30-second epochs decimated 256 → 64 Hz with an anti-aliasing filter (1920 samples),
  divided by a per-recording robust scale computed from the signal alone, clipped at 20. On trainable epochs the
  clip touches about 0.5% of samples.
- The model is a four-layer 1-D convolutional network, 44,581 parameters, instance normalisation after each
  convolution, mean-and-max pooling into a five-way output.
- Training: AdamW, learning rate 3e-4, weight decay 1e-4, batch 128, seed 20261108, 25 passes over
  109,436 epoch-channel samples, with the pass chosen by the selection people.
- No class weighting in the plain variant, matching v2's plain fit.

### The variants

| Variant | Change | Selected by |
|---|---|---|
| Plain CNN | unweighted loss, selection by kappa | kappa |
| Class-balanced CNN | sklearn "balanced" class weights, selection by macro-F1 | macro_f1 |
| Square-root-balanced CNN | inverse square-root class weights, selection by macro-F1 | macro_f1 |
| Augmented CNN | amplitude scaling 0.8–1.25 and Gaussian noise, selection by kappa | kappa |
| Two-channel CNN | both headband channels as one input, no probability averaging | kappa |
| Context model | frozen plain-CNN trunk plus a temporal head over 15 epochs | kappa |
| Context model, sqrt-balanced trunk | frozen sqrtbalanced trunk plus the same temporal head | kappa |
| Context model, balanced trunk | frozen balanced trunk plus the same temporal head | kappa |
| Ensemble of seven models | equal-weight average of the seven variants' frozen probabilities, no retraining and no adjustment needed | none |

### Results

| Model | BOAS kappa | BOAS macro-F1 | BOAS sleep-time error | Sleep-EDF kappa | Sleep-EDF sleep-time error |
|---|---:|---:|---:|---:|---:|
| Frozen v2 feature model | 0.661 | 0.561 | 31.8 min | 0.579 | 235.9 min |
| YASA 0.7.0 | 0.532 | 0.518 | 57.6 min | 0.675 | 110.3 min |
| Plain CNN | 0.728 | 0.595 | 16.5 min | 0.086 | 803.2 |
| Class-balanced CNN | 0.479 | 0.525 | 8.4 min | 0.181 | 716.7 |
| Square-root-balanced CNN | 0.662 | 0.604 | 9.8 min | 0.114 | 853.9 |
| Augmented CNN | 0.727 | 0.591 | 11.5 min | 0.075 | 855.7 |
| Two-channel CNN | 0.653 | 0.554 | 9.4 min | not run | not run |
| Context model | 0.734 | 0.618 | 8.7 min | not run | not run |
| Context model, sqrt-balanced trunk | 0.737 | 0.607 | 10.2 min | not run | not run |
| Context model, balanced trunk | 0.712 | 0.592 | 10.1 min | not run | not run |
| Ensemble of seven models | 0.741 | 0.629 | 9.3 min | not run | not run |

### The external run

`cnn_external.py` applies the frozen CNN unchanged to the 20 Sleep-EDF records: 100 Hz Fpz-Cz resampled to
64 Hz, the frozen epoch grid, reference and quality mask taken from `results/v2/external_predictions.json`, and
`yasa_evaluate.py`'s windows, masks and metrics so all three systems are scored identically. Nothing is
refitted. The CNN was fitted on two forehead headband channels and is applied to a single frontal PSG channel:
a device, montage and scorer shift, and the collapse should be read in that light rather than as a claim about
convolutional networks in general.

### Known limitations of this appendix

- Single seed, single architecture, no variance estimate.
- The context model freezes the plain CNN trunk as an epoch encoder and trains only a temporal head over it, so
  it is a lower bound on what end-to-end multi-epoch training would give.
- The selection people were used for more decisions here than in v2, as recorded above.
- Both cohorts are now inspected; these are development results.

## Appendix M. Submission package, reproduction and verification

### What a judge receives

`Goldilocks_Submission/`, 36 MB, no public repository by team decision.

| Path | Contents |
|---|---|
| `README.md` | Headline result, the comparison including the baselines, the main limitation, how to run it |
| `REPRODUCTION.md` | The reproduction report summarised in item 2 of the Revision 19 record |
| `CREDITS.md`, `LICENSES.md` | Attribution and licence terms for every dataset and library used |
| `SHA256SUMS.txt` | Hash of every file in the folder |
| `*.py`, `*.mjs`, `requirements.txt` | All code at the folder root so its relative paths work |
| `v2_protocol.json`, `protocol.json`, `release_protocol.json` | The frozen protocols |
| `source_sha256.json`, `source_sha256_v2.json` | Code hash manifests; 33 of 33 files verify |
| `data/features/`, `data/external_sleepedf/features/` | The derived features the frozen evaluation reads |
| `data/boas/`, `data/release/`, `data/loader_audits/` | Dataset metadata, formulation source data, per-person loader audits |
| `results/` | Every metric, per-person table, prediction file, frozen model, viewer, video, YASA baseline, window rule, CNN family |
| `research/` | Formulation evidence module with 18 tests, exposure model, leakage audit |
| `plan/` | This plan and the coating/core evidence table |
| `slides/`, `handout/` | Support slides and the printable one-page handout, HTML and PDF |

### Identity

| Item | SHA-256 |
|---|---|
| Frozen protocol `v2_protocol.json` | `ad3fb648f8eedcd429884abe8f018f24c5eb3e85b54a3f87ffc164183f29882c` |
| Frozen model `results/v2/frozen_model.joblib` | `121837c16b4ff3a045421355048b538c53158119b2afd52dcbd32a641021cb0b` |

Seeds: 20261108 for the v2 and CNN experiments, 20261107 for the v1 pipeline and the exposure model.

### How to verify in three steps

1. Build the environment from `requirements.txt` and run `python reproduce_v2.py`. Expect 7 passing tests and
   about 31 seconds.
2. Check the headline numbers in `results/v2/historical_boas_metrics.json` and
   `results/v2/external_metrics.json`: kappa 0.661 and 31.8 minutes on
   BOAS, kappa 0.579 and 235.9 minutes on Sleep-EDF.
3. Recompute the file hashes against `SHA256SUMS.txt`.

### What the one command does not rebuild

The YASA baseline and the window rule need the full 9.3 GB build, including the compacted BOAS signals and the
original Sleep-EDF files; their outputs ship as JSON and CSV. The CNN family needs the 64 Hz epoch caches built
by `build_cache.py` from the same raw data. The exposure model needs the Yousfan workbook, which is CC BY-NC
and not redistributed here.

## Appendix N. Clinical measurement requirements for the retained delivery track

Status: **not implemented, and no data for any measure below exists in this project.** This appendix
records what would have to be measured before any clinical or delivery claim could be made. It is a
statement of the evidence burden, not of evidence held. Nothing here is wired into code, no model in
this submission consumes any of it, and the scored sleep-staging core is unaffected. It is written down
for the same reason the "what would change our mind" slide exists: the boundary of a claim is easier to
audit when the missing evidence is named.

### N.1 Entry and outcome measures, by indication

| Indication | Entry / confirmation | Outcome measures |
|---|---|---|
| Parkinson's | Dopamine transporter or FDOPA PET for dopamine terminals; alpha-synuclein seed amplification assay to confirm biology at entry | Motor scales; wearable sensors; levodopa-equivalent dose |
| ADHD | Symptom scales; attention tasks, and fMRI only if the study is research-grade | Symptom scales; attention tasks |
| ADHD-like disease (MPS IIIA, fragile X) | Disease-specific biomarkers, e.g. CSF heparan sulfate for MPS IIIA | Developmental scales |
| Drug levels (all) | — | Plasma, and CSF where available; PET with a labelled drug where brain exposure is the question |

Two limits are part of the record, not footnotes to it. **ADHD has no validated imaging biomarker**, which
is one reason it is a weak first demonstration: without an objective entry measure the endpoint is a
symptom scale, and a symptom scale cannot separate a delivery effect from a placebo or an expectation
effect. **Brain exposure in humans is hard to measure directly**, so PET with a labelled drug is the best
available option rather than a convenient one; a plasma level is not a brain level, and the exposure
model elsewhere in this plan refuses to fit precisely because matched pairs linking the two do not exist
in the collected sources.

### N.2 Safety monitoring, by route

| Route | Monitoring |
|---|---|
| Nasal gels | Repeat smell testing, smell loss being the key risk; endoscopy; nasal lavage cytokines (IL-1-beta, IL-6, TNF-alpha) |
| Injectable gels | Serial MRI for oedema, haemorrhage and gel position; blood or CSF markers of injury (GFAP, neurofilament light) |
| Protein cargo | Anti-drug antibodies |
| All routes | Adverse event logs |

The longest repeated-dose nasal evidence found in the sources reviewed for this plan is about **14 days**.
A first pilot should therefore stay short: the duration of a study is bounded by the duration over which
safety has actually been observed, and extending past that boundary would be an assumption presented as a
protocol.

### N.3 Why this is an appendix and not a feature

A decision component that consumed PET, CT or any measure in the tables above would need matched outcomes,
a frozen protocol, a person-held-out test split and an independent evaluation — the same machinery the
sleep-staging core required, built again for imaging, with none of the data in hand. Writing that code
against absent data would produce a system that returns a number for every input and has been shown to be
right about none of them. The submission's one durable claim is that its numbers are checkable; a
component with no validation path would cost that claim more than the component could add.

## Sources


[1] Supplied event brief, all four pages  
[2] Ritalin Summary of Product Characteristics, pharmacokinetics: https://www.medicines.org.uk/emc/product/15987/smpc  
[3] NHS Specialist Pharmacy Service, modified-release methylphenidate: https://sps.nhs.uk/articles/prescribing-and-switching-between-modified-release-methylphenidate/  
[4] Khalil et al., intranasal methylphenidate transfersomal gel. DOI: 10.1016/j.jddst.2024.106125. https://www.sciencedirect.com/science/article/pii/S1773224724007949  
[5] Yousfan et al., nanoparticle brain delivery. DOI: 10.1021/acs.molpharmaceut.3c00880. https://pmc.ncbi.nlm.nih.gov/articles/PMC10762658/ ; code/README: https://github.com/Introvertuoso/BrainTargeting  
[6] LipoDB-BBB landing page: https://zenodo.org/records/19677556  
[7] IFE.PTML repository: https://github.com/Aruize/IFE.PTML_for_NP-Systems_in_NDDs  
[8] Controlled methylphenidate EEG study and restricted data-access statement: https://pmc.ncbi.nlm.nih.gov/articles/PMC6544236/  
[9] ISO/IEC 17025:2017: https://www.iso.org/standard/66912.html  
[10] ISO 45001:2018: https://www.iso.org/standard/63787.html  
[11] ISO 9001:2026: https://www.iso.org/quality-management/iso-9001-2026  
[12] ISO 14001:2026: https://www.iso.org/standard/14001  
[13] ISO 13485:2016: https://www.iso.org/standard/59752.html  
[14] Faria et al., MIRIBEL: https://www.nature.com/articles/s41565-018-0246-4  
[15] EPA nanoparticle/MEA dataset catalogue and downloaded archive, DOI 10.23719/1413280; inventory and all 17 workbook schemas inspected: https://catalog.data.gov/dataset/screening-silver-nanoparticles-for-potential-neurotoxicity-using-cortical-neurons-grown-on ; https://pasteur.epa.gov/uploads/10.23719/1413280/MEA_NanoAg.zip  
[16] Strickland et al. (2016), DOI 10.1016/j.tox.2016.05.009; study abstract inspected through EPA HERO and PubMed, full methods not audited: https://hero.epa.gov/reference/4126859/ ; https://pubmed.ncbi.nlm.nih.gov/27179409/  
[17] He et al. (2025), Drug Release Nanoparticle System Design: Data Set Compilation and Machine Learning Modeling, DOI 10.1021/acsami.4c16800; publication record checked, dataset not inspected: https://pubmed.ncbi.nlm.nih.gov/39800937/  
[18] Publisher supporting datasets, retrieved and inspected; metadata lists CC BY-NC 4.0: https://acs.figshare.com/articles/dataset/A_Comprehensive_Study_on_Nanoparticle_Drug_Delivery_to_the_Brain_Application_of_Machine_Learning_Techniques/24768612 ; https://acs.figshare.com/articles/dataset/A_Comprehensive_Study_on_Nanoparticle_Drug_Delivery_to_the_Brain_Application_of_Machine_Learning_Techniques/24768615 . Verified MD5 values: `mp3c00880_si_002.xlsx` = `7fc79f136584a0b068164918c5c841d0`; `mp3c00880_si_003.xlsx` = `fdd98a50447d9fd21288b0c2ca653601`. Public API metadata inspected at https://api.figshare.com/v2/articles/24768612 and https://api.figshare.com/v2/articles/24768615 .  
[19] Baicalin source study; Table 3 and relevant methods/context inspected, DOI 10.3390/molecules23071747: https://pmc.ncbi.nlm.nih.gov/articles/PMC6100585/  
[20] Atomoxetine nasal-insert rat study; article, methods and Table 7 inspected, DOI 10.3390/ph16020326: https://pmc.ncbi.nlm.nih.gov/articles/PMC9958713/  
[21] Zhang et al. (2022), PC12 nanoparticle electrophysiology; article and supplement descriptions inspected, supplemental workbook not downloaded, DOI 10.1371/journal.pone.0277942: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277942  
[22] Gopal et al. (2007), methylphenidate functional neurotoxicity in cultured networks; abstract identified, open observation data not verified: https://pubmed.ncbi.nlm.nih.gov/17576514/  
[23] Albrecht et al. (2015), dexamphetamine EEG deposit; landing page inspected, data not audited for our task: https://zenodo.org/records/33432  
[24] EPA data-use terms, inspected: https://pasteur.epa.gov/license/sciencehub-license.html  
[25] ClinicalTrials.gov public study-record access: https://clinicaltrials.gov/data-api/how-download-study-records  
[26] Official Drugs@FDA data description: https://open.fda.gov/data/drugsfda/ ; FDA distinction between approval information and trial records: https://www.fda.gov/about-fda/cder-offices-and-divisions/drug-application-and-approval-process-questions-and-answers  
[27] ChEMBL indication, mechanism and related public web services: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
[28] Primary atomoxetine NLC/in-situ-gel study, DOI 10.3390/pharmaceutics15071985; formulation/release sections and PK Tables 5-6 inspected: https://pmc.ncbi.nlm.nih.gov/articles/PMC10386213/
[29] Qiu et al., DOI 10.1016/j.celbio.2025.100130: https://www.sciencedirect.com/science/article/pii/S3050562325001217 . Released code inspected without execution: https://github.com/jxzly/NanoMed/blob/master/main.py and https://github.com/jxzly/NanoMed/blob/master/utils.py . Repository tree SHA observed: f51c218d40774517c046e3d68de31b6589e3a630. Group provenance and reuse terms remain verification items.
[30] Supplied nanoparticle screener README
[31] Supplied evidence map and proteome feasibility review
[32] Supplied ADHD receptor evidence map
[33] Supplied OCD receptor evidence map
[34] Supplied autism receptor evidence map
[35] Supplied research ZIP, including six Markdown notes, Alzheimer hydrogel DOCX, additional bioengineering PDF and two duplicate copies of the carrier screenshot PDF. The archive contains research materials, not executable screener Python source.
[36] Supplied landscape notes, part 1 (PDF), seven pages visually inspected
[37] Supplied landscape notes, part 2 (PDF), seven pages visually inspected; byte-identical to both copies of the notes inside the ZIP
[38] Supplied additional bioengineering parts.pdf, ten-page text reviewed
[39] Supplied Paraskevaidi et al. spectroscopy commentary, DOI 10.1186/s13024-018-0252-x, three pages read
[40] Current NeuroBridge Battle Plan, 40 pages, version 6
[41] Earlier Oxford NeuroTech Hackathon Battle Plan, 36 pages
[42] Superseded VETO plan
[43] Dataset screenshot, visually inspected. It lists https://openneuro.org/datasets/ds005555/ ; https://github.com/unicorn-bi/Hackathon ; https://neurallatents.github.io/datasets.html . The NLB dataset overview and repository landing information were checked; recordings were not audited as drug-delivery data.
[44] Duplicate supplied participant information sheet, all four pages read
[45] Supplied Goldilocks AI — Battle Plan v3 (AI stream), all 20 pages read
[46] BOAS official OpenNeuro repository, README, participants.tsv, dataset_description.json, acquisition/channel/event definitions and 256 event TSVs inspected: https://github.com/OpenNeuroDatasets/ds005555 . Dataset DOI in inspected metadata: https://doi.org/10.18112/openneuro.ds005555.v1.1.3 . Observed repository tree SHA: `face09565c4337ddbce7a048c9b02060ad20df45`. Counts and released-score statistics in section 2 were independently recomputed on 7 November 2026; they are not new-model performance. That sentence described the earlier metadata audit; section 13 records the subsequent complete raw-signal acquisition/loader checks.
[47] YASA official SleepStaging documentation, input/preprocessing and channel guidance inspected: https://yasa-sleep.org/generated/yasa.SleepStaging.html
[48] ISO 14971:2019 official catalogue, current status inspected: https://www.iso.org/standard/72704.html
[49] IEC 62304:2006 + AMD1:2015 consolidated edition 1.1, official catalogue inspected: https://webstore.iec.ch/en/publication/22794
[50] Atomoxetine Summary of Product Characteristics; pharmacokinetics and clinical-response distinction checked: https://www.medicines.org.uk/emc/product/100476/smpc
[51] Intuniv (guanfacine prolonged-release) Summary of Product Characteristics; time-varying pharmacokinetics checked: https://www.medicines.org.uk/emc/product/5099/smpc

[52] Supplied Add AI determine.docx, both pages read
[53] Supplied hydrogel_ai_datasets.xlsx, all four sheets/rows/formulas inspected
[54] Supplied matched_pairs.xlsx, both sheets inspected
[55] Supplied coating_core_evidence.md, full text read; mentioned CSV and code not included
[56] Bannigan long-acting injectable dataset primary record, counts/file listing and early-feature imputation description inspected, initial metadata-stage review; the original files were subsequently acquired and verified in section 13: https://zenodo.org/records/7309021 ; associated study DOI 10.1038/s41467-022-35343-w: https://www.nature.com/articles/s41467-022-35343-w
[57] Bao et al., PLGA microparticle dataset, methods/data-record sections inspected, initial publication-stage review; the original workbooks were subsequently acquired and verified in section 13; DOI 10.1038/s41597-025-04621-9: https://pmc.ncbi.nlm.nih.gov/articles/PMC11873201/ ; data record: https://data.mendeley.com/datasets/zzvtdrcy76/2
[58] B3DB primary description and domain inspected, DOI 10.1038/s41597-021-01069-5: https://pmc.ncbi.nlm.nih.gov/articles/PMC8556334/ ; source repository: https://github.com/theochem/B3DB
[59] Johnsen et al. 2017, primary results retrieved for the workbook's P3 case; complete methods not re-audited in this update. DOI 10.1038/s41598-017-11220-1: https://www.nature.com/articles/s41598-017-11220-1 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC5583399/
[60] Heparosan surface-engineering primary abstract/results retrieved; complete methods/supplements not re-audited: https://pmc.ncbi.nlm.nih.gov/articles/PMC9540343/

[61] Sleep-EDF Expanded 1.0.0, official source, licence and published hashes checked; 20 PSG/reference pairs acquired and verified: https://physionet.org/content/sleep-edfx/1.0.0/ ; https://physionet.org/content/sleep-edfx/view-license/1.0.0/ ; https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt . DOI 10.13026/C2X676.

Primary-paper review is limited to the sources and sections explicitly described. A supplied literature summary does not establish independent verification of its underlying primary studies. Current-status claims require dated official evidence at implementation time.

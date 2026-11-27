# Goldilocks first build report


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

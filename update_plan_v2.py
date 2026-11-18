"""Record executed v2 work without rewriting earlier experimental history."""
from pathlib import Path

root=Path(__file__).resolve().parent
plan=root.parent/'outputs/NEURORELEASE_Stream2_ADHD_Nanoparticle_Plan.md'
section='''## 14. V2 execution, independent test and revised claim gates

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

### Build decision and remaining work

Continue building the research product with the available data. More unrelated datasets are not required for this version. The core model is implemented; broad brainstorming and extra drug/delivery modules should not displace integration, reproducibility and rehearsal. V2 is not universally better: preserve v1 as a reference rather than replacing every endpoint with v2.

If a further scientific iteration is undertaken, first define a task-aligned night-window policy usable without reference labels, strengthen wake/artefact handling and obtain separate development and interval-calibration data under that policy. Freeze the resulting model, window rules and uncertainty method before another untouched test. The 20 external people used here are now inspected and cannot be reused as fresh confirmation. A larger cohort alone does not repair a mismatched task or establish exchangeability. ADHD medication response and hydrogel/coating personalisation still require the paired measurements listed in section 12; neither public EEG nor unpaired PLGA data supplies them.

'''
s=plan.read_text()
if '## 14. V2 execution' not in s:
    s=s.replace('specification, Revision 13 |','specification, Revision 14 |',1)
    start=s.index('**Status:');end=s.index('\n',start)
    s=s[:start]+'**Status: v2 built and independently transfer-tested; stage decoding improved on historical BOAS, sleep-time calibration failed, and calibrated-range claims are withheld. See section 14 for current decisions and results.**'+s[end:]
    s=s.replace('**Implementation update:** section 13 supersedes earlier audit-stage pending statements. It records the executed first baseline, including failed vendor and calibration comparisons.','**Implementation update:** section 14 is the current execution record; section 13 retains v1 history. Readiness and claims follow the measured endpoint-specific gates, including failed interval coverage.')
    s=s.replace('Current results and remaining scientific limitations are recorded in section 13.','Current results and remaining scientific limitations are recorded in section 14.')
    s=s.replace('## Appendix A. Formulation rationale and the four research modules',section+'## Appendix A. Formulation rationale and the four research modules',1)
    s=s.replace('Raw EDF signals remain uninspected.','That sentence described the earlier metadata audit; section 13 records the subsequent complete raw-signal acquisition/loader checks.')
    s=s.replace('underlying TSVs not downloaded:','initial metadata-stage review; the original files were subsequently acquired and verified in section 13:')
    s=s.replace('underlying workbooks not downloaded;','initial publication-stage review; the original workbooks were subsequently acquired and verified in section 13;')
    s=s.replace('\nPrimary-paper review is limited', '\n[61] Sleep-EDF Expanded 1.0.0, official source, licence and published hashes checked; 20 PSG/reference pairs acquired and verified: https://physionet.org/content/sleep-edfx/1.0.0/ ; https://physionet.org/content/sleep-edfx/view-license/1.0.0/ ; https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt . DOI 10.13026/C2X676.\n\nPrimary-paper review is limited')
    plan.write_text(s)
(root/'BUILD_REPORT_V2.md').write_text('# Goldilocks v2 execution report\n\n'+section.split('\n',1)[1])

from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parent;output=ROOT.parent/'outputs'
m=json.loads((ROOT/'results/metrics.json').read_text());r=json.loads((ROOT/'results/release/metrics.json').read_text());a=json.loads((ROOT/'data/boas_acquisition.json').read_text())
section=f'''## 13. Executed build and dataset acquisition: 7 November 2026

This section supersedes the earlier planning-stage “pending” status for the work actually executed below. Design constraints, unsupported clinical claims and unimplemented extensions remain in force.

### Acquired and verified

- All **128 BOAS headband EDFs / 100 people** acquired; {sum(x['bytes'] for x in a)/1e9:.3f} GB of original source bytes. SHA-256 checked against the pinned repository annex pointers. Onsets, durations, sample boundaries, units and two task channels checked for every recording.
- Lossless digital HB_1/HB_2 samples and EDF calibration retained for all recordings. Large extra-channel original EDFs were acquisition cache, replaced after verified extraction by task-channel archives. URLs/hashes permit original reacquisition; the first original EDF is also retained locally.
- Bannigan authors' 14/17-feature data acquired from the pinned authors' GitHub mirror. Both files match the primary Zenodo MD5 values. Their .tsv extension is misleading: the actual containers are Excel. Both variants share the same 3,783 outcomes and are not independent datasets.
- Four original PLGA workbooks acquired from Mendeley v2; SHA-256 checked against its file manifest; dataset licence CC BY 4.0 verified. 4,913 observations, 321 formulations, 113 source DOIs. No cross-corpus pooling.

### Frozen EEG experiment and actual outcomes

48 fitting people; 12 separate probability-validation people; 20 summary-calibration people; 20 test people. Repeated nights stay together. The first night per person was selected before fitting for calibration/test. Model: 28 EEG-only spectral/statistical features, fixed histogram gradient boosting; no target/reference/vendor labels in the features.

| Held-out result | Actual value and interpretation |
|---|---|
| Participant-average stage kappa | {m['methods']['feature_model']['kappa']:.3f} feature model; {m['methods']['released_bitbrain']['kappa']:.3f} released Bitbrain; majority baseline 0.000 |
| Participant-average macro-F1 | {m['methods']['feature_model']['macro_f1']:.3f} feature model versus {m['methods']['released_bitbrain']['macro_f1']:.3f} released Bitbrain |
| Usable-window sleep-time MAE | {m['model_tst_mae_minutes']:.1f} minutes; this is not a whole-night clinical measure where signal/reference is missing |
| Identical vendor-covered mask | {m['common_mask_tst']['model_mae_minutes']:.1f} model versus {m['common_mask_tst']['vendor_mae_minutes']:.1f} vendor minutes; paired difference CI approximately -2.4 to 27.8 minutes |
| Nominal 90% summary interval | Observed {m['observed_summary_coverage']*100:.0f}% coverage (16/20), mean width {m['mean_interval_width_minutes']:.1f} minutes; achieved 90% coverage is not claimed |
| Probability temperature scaling | Held-out log loss, Brier score and confidence ECE did not improve; the failure is retained |

The trained model exceeds a majority baseline but trails the released vendor comparator. It is an implemented initial baseline, not a demonstrated competitive advantage. Vendor training provenance differs from the team's split. Further model development must be labelled development on an already inspected cohort; a new independent confirmatory result requires an untouched evaluation design/data.

### Separate, bounded materials experiment

PLGA release model trained with whole-DOI separation: {r['train_sources']} training papers, {r['test_sources']} test papers, {r['test_formulations']} test formulations. Study-averaged MAE is {r['study_averaged_model_mae_fraction']:.3f} release fraction versus {r['study_averaged_baseline_mae_fraction']:.3f} for a fixed time-bin median. This is a descriptive point difference, not established transferable superiority. There are {r['outside_0_1_observations']} observations outside the nominal 0-1 range; original observations are retained and flagged, not silently clipped.

This is in-vitro small-molecule PLGA microparticle modelling. It does not establish hydrogel prediction, protein-payload transfer, brain exposure, personalised coating/core choice, ADHD treatment response or dosing. Missing matched host/clinical data have not been invented or replaced by unrelated rows.

### Working deliverables and verification

The Goldilocks source package includes protocols, pinned acquisition, extracted features, trained model artefacts, per-person metrics, full-reference availability diagnostics, probability-calibration diagnostics, three genuine failure examples, original release inputs and a self-contained offline evidence explorer. The separate EEG data archive contains the lossless task-channel recordings and provenance.

Meaningful tests passed for person-group separation/repeat nights, unknown-label handling and spectral feature behaviour. All 128 loader checks and digital reconstruction/MNE calibration spot checks completed. Browser checks verified 20 selectable test nights, reference reveal, aligned plots, unsupported-latency response and evidence JSON download. Desktop/mobile layouts were inspected; mobile plots scroll horizontally to retain readable labels.

YASA, CNN, live EEG, day-task predictor and clinical/host-specific formulation models have not been implemented in this build. The public-data report is precomputed retrospective analysis, not live inference. The next scientific need is improving and independently confirming the EEG model and interval reliability, not adding more unrelated modules.
'''
(ROOT/'BUILD_REPORT.md').write_text('# Goldilocks first build report\n\n'+section.split('\n',1)[1])
plan=output/'NEURORELEASE_Stream2_ADHD_Nanoparticle_Plan.md';s=plan.read_text()
if 'specification, Revision 12 |' in s:
    s=s.replace('specification, Revision 12 |','specification, Revision 13 |')
    start=s.index('**Status:');end=s.index('\n',start)
    s=s[:start]+'**Status: first build executed; 128 raw EEG recordings and original release datasets acquired/verified; trained models, held-out results and offline explorer available. See section 13 for current outcomes and limitations.**'+s[end:]
    s=s.replace('## Competition brief:','**Implementation update:** section 13 supersedes earlier audit-stage pending statements. It records the executed first baseline, including failed vendor and calibration comparisons.\n\n## Competition brief:',1)
    s=s.replace('## Appendix A. Formulation rationale and the four research modules',section+'\n## Appendix A. Formulation rationale and the four research modules',1)
    plan.write_text(s)
source_manifest={}
for p in ROOT.glob('*.py'):source_manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['acquire.mjs','requirements.txt','protocol.json','release_protocol.json']:
    source_manifest[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(ROOT/'source_sha256.json').write_text(json.dumps(source_manifest,indent=2))
output.mkdir(exist_ok=True)
with zipfile.ZipFile(output/'Goldilocks_Build_v1.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part.startswith('.') or part in ['__pycache__','eeg_digital'] for part in rel.parts):continue
        if p.suffix in ['.edf','.log','.exit','.pyc'] or p.name.startswith('explorer_'):continue
        if p.name in ['mendeley_page.html','mendeley_metadata_response.json']:continue
        z.write(p,'goldilocks/'+str(rel))
with zipfile.ZipFile(output/'Goldilocks_BOAS_EEG_Data.zip','w',zipfile.ZIP_STORED,allowZip64=True) as z:
    for p in (ROOT/'data/eeg_digital').glob('*.npz'):z.write(p,'goldilocks/data/eeg_digital/'+p.name)
    for name in ['boas_acquisition.json','source_tree.json']:z.write(ROOT/'data'/name,'goldilocks/data/'+name)
    z.writestr('README.txt','BOAS ds005555, CC0. Lossless original int16 HB_1/HB_2 digital samples plus EDF physical/digital calibration and sample rate. Convert: uV=(digital-digital_min)*(physical_max-physical_min)/(digital_max-digital_min)+physical_min, broadcasting calibration per channel. See the build package for annotations, protocols, checks and original file URLs/SHA256. Original full EDFs are reproducibly obtainable from those URLs. Not clinical/ADHD validation.')
print(json.dumps({p.name:p.stat().st_size for p in [output/'Goldilocks_Build_v1.zip',output/'Goldilocks_BOAS_EEG_Data.zip']},indent=2))


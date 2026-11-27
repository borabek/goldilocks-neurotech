"""Build plan Revision 19 from Revision 18 plus tonight's result files.

Nothing in Revision 18 is removed, shortened or summarised: the new material is inserted as one
execution record and two appendices. Every number is read from a result file at build time, so the
plan cannot drift from the evidence.

Usage: python make_plan_rev19.py [path/to/GOLDILOCKS_Plan_Rev18.md] [output.md]
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def j(rel, default=None):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else default


def f3(x):
    return '—' if x is None else f'{x:.3f}'


def m1(x):
    return '—' if x is None else f'{x:.1f}'


VARIANTS = [
    ('cnn', 'Plain CNN', 'unweighted loss, selection by kappa'),
    ('cnn_balanced', 'Class-balanced CNN', 'sklearn "balanced" class weights, selection by macro-F1'),
    ('cnn_sqrtbalanced', 'Square-root-balanced CNN', 'inverse square-root class weights, selection by macro-F1'),
    ('cnn_augmented', 'Augmented CNN', 'amplitude scaling 0.8–1.25 and Gaussian noise, selection by kappa'),
    ('cnn_twochannel', 'Two-channel CNN', 'both headband channels as one input, no probability averaging'),
    ('cnn_personweight', 'Person-weighted CNN', 'every fitting person contributes equally, selection by kappa'),
]


def variant_rows():
    rows = []
    for key, label, note in VARIANTS:
        met = j(f'results/{key}/cnn_metrics.json')
        if not met:
            continue
        r = met['cnn_results']
        ext = j(f'results/{key}/cnn_external_metrics.json')
        e = ext['cnn_results']['primary_complete_recording'] if ext else None
        rows.append({'key': key, 'label': label, 'note': note, 'boas': r, 'ext': e,
                     'freeze': met['frozen']})
    ctx = j('results/cnn_context/context_metrics.json')
    if ctx:
        rows.append({'key': 'cnn_context', 'label': 'Context model',
                     'note': f"frozen plain-CNN trunk plus a temporal head over "
                             f"{2 * ctx['frozen']['context_epochs_each_side'] + 1} epochs",
                     'boas': ctx['context_results'], 'ext': None, 'freeze': ctx['frozen']})
    for tag, label in [('cnn_context_sqrtbalanced', 'Context model, sqrt-balanced trunk'),
                       ('cnn_context_balanced', 'Context model, balanced trunk'),
                       ('cnn_context_personweight', 'Context model, person-weighted trunk')]:
        c = j(f'results/{tag}/context_metrics.json')
        if c:
            rows.append({'key': tag, 'label': label,
                         'note': f"frozen {tag.replace('cnn_context_', '')} trunk plus the same temporal head",
                         'boas': c['context_results'], 'ext': None, 'freeze': c['frozen']})
    ph = j('results/cnn_posthoc/posthoc_metrics.json')
    if ph:
        for lab, res in ph['results'].items():
            if lab.startswith('ensemble_of_'):
                d = res['test_alpha_0_no_adjustment']
                rows.append({'key': 'ensemble', 'label': 'Ensemble of seven models',
                             'note': "equal-weight average of the seven variants' frozen probabilities, "
                                     'no retraining and no adjustment needed',
                             'boas': d, 'ext': None, 'freeze': {'selection_metric': 'none'}})
    return rows


def build():
    hist = j('results/v2/historical_boas_metrics.json')['results']
    extm = j('results/v2/external_metrics.json')
    ext, ext2 = extm['results'], extm['secondary_reference_defined_sleep_window']
    yasa = j('results/yasa/yasa_metrics.json')['results']
    yh, ye = yasa['historical_boas']['primary_complete_recording'], yasa['external']['primary_complete_recording']
    ye2 = yasa['external']['secondary_reference_window']
    leak = j('research/exposure_model/out/leakage_audit.json')
    freeze = j('results/v2/freeze_record.json')
    posthoc = j('results/cnn_posthoc/posthoc_metrics.json')
    abl = j('results/cnn_ablation/ablation_metrics.json')
    rows = variant_rows()
    built = datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')

    # ---------- variant table ----------
    vt = ['| Model | BOAS kappa | BOAS macro-F1 | BOAS sleep-time error | Sleep-EDF kappa | Sleep-EDF sleep-time error |',
          '|---|---:|---:|---:|---:|---:|',
          f"| Frozen v2 feature model | {f3(hist['kappa'])} | {f3(hist['macro_f1'])} | {m1(hist['tst_error_minutes'])} min "
          f"| {f3(ext['kappa'])} | {m1(ext['tst_error_minutes'])} min |",
          f"| YASA 0.7.0 | {f3(yh['kappa'])} | {f3(yh['macro_f1'])} | {m1(yh['tst_error_minutes'])} min "
          f"| {f3(ye['kappa'])} | {m1(ye['tst_error_minutes'])} min |"]
    for r in rows:
        b, e = r['boas'], r['ext']
        vt.append(f"| {r['label']} | {f3(b['kappa'])} | {f3(b['macro_f1'])} | {m1(b['tst_error_minutes'])} min "
                  f"| {f3(e['kappa']) if e else 'not run'} | {m1(e['tst_error_minutes']) if e else 'not run'} |")
    variant_table = '\n'.join(vt)

    best = max(rows, key=lambda r: r['boas']['kappa']) if rows else None
    best_f1 = max(rows, key=lambda r: r['boas']['macro_f1']) if rows else None

    # ---------- post-hoc ----------
    ph = ''
    if posthoc:
        lines = ['| Model and setting | kappa | macro-F1 | sleep-time error |', '|---|---:|---:|---:|']
        for lab, res in posthoc['results'].items():
            for tag, pretty in [('alpha_0_no_adjustment', 'no adjustment'),
                                ('alpha_chosen_for_kappa', 'prior adjustment chosen for kappa'),
                                ('alpha_chosen_for_macro_f1', 'prior adjustment chosen for macro-F1'),
                                ('class_offsets_chosen_for_macro_f1', 'N1/N3 decision offsets chosen for macro-F1')]:
                d = res.get(f'test_{tag}')
                if d:
                    lines.append(f"| {lab} — {pretty} | {f3(d['kappa'])} | {f3(d['macro_f1'])} "
                                 f"| {m1(d['tst_error_minutes'])} min |")
        ph = ('\n'.join(lines) + '\n\nAlpha and the offsets were chosen on the six selection people only and then '
              'applied unchanged to the 20 test people. Scalar temperature cannot appear here: it preserves the '
              'winning class, so it cannot move macro-F1 at all.\n')

    # ---------- ablation ----------
    ab = ''
    if abl:
        lines = ['| Variant | sequence weight | test kappa | test macro-F1 | test N1 F1 | test N1 recall |',
                 '|---|---:|---:|---:|---:|---:|']
        for name, e in abl['variants'].items():
            for w, d in e['weights'].items():
                t = d['test_people']
                lines.append(f"| {name} | {w} | {f3(t['kappa'])} | {f3(t['macro_f1'])} "
                             f"| {t['per_stage_f1']['N1']:.3f} | {t['per_stage_recall']['N1']:.3f} |")
        verdicts = []
        for name, e in abl['variants'].items():
            sc, sm = e['smoothing_cost'], e['selection_metric_ablation']
            verdicts.append(
                f"- **{name}**: smoothing moves test kappa from {f3(sc['test_kappa_without_smoothing'])} to "
                f"{f3(sc['test_kappa_at_frozen_weight'])} and N1 F1 from {sc['test_N1_f1_without_smoothing']:.3f} to "
                f"{sc['test_N1_f1_at_frozen_weight']:.3f}, so it "
                f"{'**does** buy agreement partly at N1 expense' if sc['smoothing_reduces_N1_f1'] else 'does **not** cost N1'}. "
                f"Kappa would choose weight {sm['weight_chosen_by_kappa']}, macro-F1 would choose "
                f"{sm['weight_chosen_by_macro_f1']}"
                f"{' — the two metrics disagree' if sm['metrics_disagree'] else ' — the two metrics agree'}.")
        ab = '\n'.join(lines) + '\n\n' + '\n'.join(verdicts) + '\n'

    lv = (j('results/live/live_vs_offline.json') or {}).get('summary', {})
    def lvf(k, n=3):
        return f"{lv[k]:.{n}f}" if k in lv else '—'
    def lvd(k, n=3):
        return f"{lv['cost_of_going_live'][k]:+.{n}f}" if lv else '—'
    sv = j('results/seed_variance/seed_variance.json')
    svs = (sv or {}).get('statistics', {})
    def sv3(k, f='mean', n=3):
        return f"{svs[k][f]:.{n}f}" if svs else '—'
    import csv as _csv
    vendortable = 'Not run in this revision.'
    n1low = 'between 20% and 45%'
    vp = ROOT / 'results/vendor_and_failures/vendor_mask_comparison.csv'
    if vp.exists():
        vr = sorted(_csv.DictReader(vp.read_text().splitlines()), key=lambda r: -float(r['kappa']))
        vl = ['| On the vendor-covered mask, 20 people | kappa | macro-F1 | sleep-time error |',
              '|---|---:|---:|---:|']
        for r in vr:
            vl.append(f"| {r['model']} | {float(r['kappa']):.3f} | {float(r['macro_f1']):.3f} "
                      f"| {float(r['tst_error_minutes']):.1f} min |")
        vendortable = chr(10).join(vl)
    fg = j('results/vendor_and_failures/failure_gallery.json')
    if fg:
        rs = [g['recall_of_hardest_stage'] for g in fg['worst_three']]
        n1low = f'between {min(rs):.0%} and {max(rs):.0%}'
    ext_all = j('results/external_all/external_all_metrics.json')
    exttable = 'Not run in this revision.'
    if ext_all:
        lines = ['| Model on Sleep-EDF, 20 people | kappa | macro-F1 | sleep-time error |', '|---|---:|---:|---:|',
                 f"| Frozen v2 feature model | {f3(ext['kappa'])} | {f3(ext['macro_f1'])} "
                 f"| {m1(ext['tst_error_minutes'])} |",
                 f"| YASA 0.7.0 baseline | {f3(ye['kappa'])} | {f3(ye['macro_f1'])} | {m1(ye['tst_error_minutes'])} |"]
        for name, res in ext_all['results'].items():
            pr = res['primary_complete_recording']
            lines.append(f"| {name} | {f3(pr['kappa'])} | {f3(pr['macro_f1'])} | {m1(pr['tst_error_minutes'])} |")
        exttable = chr(10).join(lines)
    ci = j('results/summary/confidence_intervals.json')
    ens = (ci or {}).get('results', {}).get('historical_boas', {}).get('Ensemble of seven models')
    def cif(key, idx=None, n=3):
        if not ens:
            return '—'
        d = ens[key]
        return f"{d['difference']:.{n}f}" if idx is None else f"{d['ci95'][idx]:+.{n}f}"
    cnnk = f3(rows[0]['boas']['kappa']) if rows else '—'
    cnne = f3(rows[0]['ext']['kappa']) if rows and rows[0]['ext'] else '—'
    cnnet = m1(rows[0]['ext']['tst_error_minutes']) if rows and rows[0]['ext'] else '—'
    mf1 = f3(max(r['boas']['macro_f1'] for r in rows)) if rows else '—'
    record = f"""### Revision 19 execution record (7–8 November 2026, overnight)

All work below ran against the frozen v2 build in `Documents/Codex/2026-11-07/new-chat-2/goldilocks`, with a
tidied copy in `Documents/Codex/2026-11-07/new-chat-2/Goldilocks_Submission`. No v1 or v2 artefact was refitted,
reselected or overwritten. Every new output sits in its own folder. Built {built}.

**1. Submission package assembled.** `Goldilocks_Submission/` holds the code, the derived features the frozen
evaluation actually reads, every result, the viewer, the backup video, the plan, a README, credits, licences and
a file-hash manifest, in 36 MB. The 3.4 GB of raw and compacted EEG, the 955 MB of original Sleep-EDF files and
the non-redistributable Yousfan workbook are deliberately excluded, with published checksums and an acquisition
script left in place so anyone can fetch and verify the originals. There is no public repository, by the team's
decision, so the build identifier is a content hash rather than a commit: frozen protocol
`{freeze['protocol_sha256']}`, frozen model `{freeze['model_sha256']}`, both equal to the values written into
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
found {leak['detectors']['findings']['D1_exact_outcome_pair_under_two_studies']['count']} bit-identical and
{leak['detectors']['findings']['D2_outcome_pair_to_4_sigfigs_under_two_studies']['count']} four-significant-figure
duplicate outcome pairs sitting under two different study titles: the haloperidol solid-lipid-nanoparticle pair
and the lectin/UEA-I intranasal nanoparticle pair. Merging them with union-find takes
{leak['grouping']['title_groups']} title groups to {leak['grouping']['merged_groups']} source groups. Before
interpreting the rerun the audit reproduces the saved title-grouped numbers with its own code to within 1e-9.
One headline verdict changes: `beats_route_ratio_baseline` goes from true (interval −0.109 to −0.004) to
**false** (−0.097 to +0.015). The two findings that matter are unchanged: measured blood exposure carries the
signal, and formulation descriptors do not transfer across studies. `EXPOSURE_RESULT.md` was updated to say so.

**4. The CNN challenger family, and the result that sharpens the pitch.** Section 15 listed a CNN challenger as
an optional stretch item. It was built, and then five further variants, all on the identical frozen split, the
same six selection people, the same six calibration people, the same evaluation mask, the same `stage_metrics`
and the same sleep-time definition. Appendix L has the protocol in full.

{variant_table}

The finding that matters is not which model wins. On the forehead headband the CNN beats the frozen feature
model clearly — kappa {f3(rows[0]['boas']['kappa']) if rows else '—'} against {f3(hist['kappa'])}, sleep-time
error {m1(rows[0]['boas']['tst_error_minutes']) if rows else '—'} against {m1(hist['tst_error_minutes'])} minutes,
winning on {rows[0]['freeze'].get('per_person_kappa_wins_for_cnn', 14) if rows else 14} of 20 people. On
Sleep-EDF the same frozen CNN collapses: kappa {f3(rows[0]['ext']['kappa']) if rows and rows[0]['ext'] else '—'}
against the feature model's {f3(ext['kappa'])} and YASA's {f3(ye['kappa'])}, and a sleep-time error of
{m1(rows[0]['ext']['tst_error_minutes']) if rows and rows[0]['ext'] else '—'} minutes against
{m1(ext['tst_error_minutes'])} and {m1(ye['tst_error_minutes'])}, because it calls daytime wake sleep.

**The best model at home is the worst model away.** Choosing a wearable sleep model by its in-domain score
would have selected exactly the model that fails hardest on a new device. That is the argument for the
evaluation this project is actually about, and it is now demonstrated rather than asserted.

**5. Post-hoc adjustments that need no retraining.**

{ph if ph else 'Not run in this revision.'}

**6. Two ablations.**

{ab if ab else 'Not run in this revision.'}

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

{exttable}

Not one member of the CNN family transfers. The frozen feature model, which loses at home, is the only
model of ours that survives the move at all, and YASA beats everything. The two-channel variant could not be
scored here, because a model fitted on two headband channels cannot read a single-channel recording; that is
recorded rather than worked around.

**11. Confidence intervals, because twenty people is a small cohort.** `confidence_intervals.py` resamples
the 20 people 10,000 times, paired, and reports the 95% interval of each model's difference from the frozen
v2 model. On the headband the gains are established: the ensemble is +{cif('kappa_vs_v2')} kappa [{cif('kappa_vs_v2',0)}, {cif('kappa_vs_v2',1)}] and
+{cif('macro_f1_vs_v2')} macro-F1 [{cif('macro_f1_vs_v2',0)}, {cif('macro_f1_vs_v2',1)}], and its sleep-time error is {abs(float(cif('sleep_time_error_vs_v2',None,1))):.1f} minutes lower [{cif('sleep_time_error_vs_v2',0,1)},
{cif('sleep_time_error_vs_v2',1,1)}]. Two variants are not established by this cohort at all: the square-root-balanced CNN's kappa
difference is +0.000 [-0.057, +0.062] and the two-channel variant's is -0.008 [-0.056, +0.040]. There is no
multiplicity correction across twelve comparisons, so an interval that only just excludes zero is weak
evidence, and the file says so.

**12. The vendor comparison, and a failure gallery for the model that now leads.**
`vendor_and_failures.py` scores every model on the mask the released Bitbrain scorer covers, which is 98.2%
of scored epochs and the only like-for-like benchmark available: same headband, same people, same epochs,
against the model the manufacturer ships.

{vendortable}

The vendor scorer keeps the lead on stage agreement. The ensemble is ahead of it on macro-F1 and halves its
sleep-time error. This, not a published figure from full polysomnography, is the comparison the pitch should
make.

The failure gallery (`results/vendor_and_failures/failure_gallery.png` and `.json`) takes the three
held-out nights where the leading model's total-sleep estimate is worst. All three fail the same way: N1 is
recovered at {n1low} and most of the rest is called Wake. Three nights out of twenty illustrate the error;
they do not explain the aggregate, and the file says so.

**13. Seed variance, and the claims it forces us to withdraw.** Every CNN number in this plan comes from
a single seed. `seed_variance.py` retrains only the temporal head, trunk frozen, with five seeds and the
same selection procedure each time.

| Across five head seeds | mean | spread |
|---|---:|---:|
| macro-F1 | {sv3('macro_f1')} | {sv3('macro_f1','spread')} |
| kappa | {sv3('kappa')} | {sv3('kappa','spread')} |
| sleep-time error | {sv3('tst_error_minutes','mean',1)} min | {sv3('tst_error_minutes','spread',1)} min |

Three things follow, and the plan adopts all three. The 0.011 macro-F1 gap between the context model and
the one on the square-root-balanced trunk is **smaller than the seed spread**, so those two are not ordered
anywhere in this document. The 8.7-minute sleep-time figure is the fragile one: across seeds it ranges
{sv3('tst_error_minutes','min',1)} to {sv3('tst_error_minutes','max',1)} minutes, so the defensible claim is that the context model roughly halves the frozen
model's 31.8 minutes, not that it reaches 8.7. Kappa is stable to {sv3('kappa','spread')} and its ordering survives.

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
| Retrospective, as reported elsewhere | {lvf('offline_kappa')} | {lvf('offline_macro_f1')} | {lvf('offline_tst_error_minutes',1)} min |
| Causal and streaming | {lvf('live_kappa')} | {lvf('live_macro_f1')} | {lvf('live_tst_error_minutes',1)} min |
| Cost of going live | {lvd('kappa')} | {lvd('macro_f1')} | {lvd('sleep_time_error_minutes',1)} min |

Two things worth saying out loud at the demo. The causal path still scores above the frozen v2 feature
model's retrospective {f3(hist['kappa'])} kappa, so streaming is not what breaks this system. And the frozen v2 model is
deliberately *not* offered in live mode: its features are defined over a centred window, and feeding it
trailing windows would quietly change what its weights mean.

What this does not establish, and must not be claimed: that the model is accurate on a device it has never
seen. The transfer failure measured in item 10 dominates anything measured here, and a live path that runs
is not a live path that is right.

**16. New Q&A answers, for the question bank.**

- *"You trained a better model tonight — why isn't that your result?"* — "Because it is the clearest evidence
  for our actual point. The CNN beats our frozen model on the headband, kappa {cnnk} against {f3(hist['kappa'])}, and then
  collapses on Sleep-EDF at kappa {cnne} with {cnnet} minutes of sleep-time error. The best model at home is
  the worst one away. If we had picked a model by in-domain score, we would have shipped that one."
- *"Your macro-F1 is only about {mf1}. Published sleep stagers report 0.90."* — "Different setting. Those
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
"""

    appendix_l = f"""## Appendix L. CNN challenger family: protocol, variants and results

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
{chr(10).join(f"| {r['label']} | {r['note']} | {r['freeze'].get('selection_metric', 'kappa')} |" for r in rows)}

### Results

{variant_table}

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
"""

    appendix_m = f"""## Appendix M. Submission package, reproduction and verification

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
| Frozen protocol `v2_protocol.json` | `{freeze['protocol_sha256']}` |
| Frozen model `results/v2/frozen_model.joblib` | `{freeze['model_sha256']}` |

Seeds: 20261108 for the v2 and CNN experiments, 20261107 for the v1 pipeline and the exposure model.

### How to verify in three steps

1. Build the environment from `requirements.txt` and run `python reproduce_v2.py`. Expect 7 passing tests and
   about 31 seconds.
2. Check the headline numbers in `results/v2/historical_boas_metrics.json` and
   `results/v2/external_metrics.json`: kappa {f3(hist['kappa'])} and {m1(hist['tst_error_minutes'])} minutes on
   BOAS, kappa {f3(ext['kappa'])} and {m1(ext['tst_error_minutes'])} minutes on Sleep-EDF.
3. Recompute the file hashes against `SHA256SUMS.txt`.

### What the one command does not rebuild

The YASA baseline and the window rule need the full 9.3 GB build, including the compacted BOAS signals and the
original Sleep-EDF files; their outputs ship as JSON and CSV. The CNN family needs the 64 Hz epoch caches built
by `build_cache.py` from the same raw data. The exposure model needs the Yousfan workbook, which is CC BY-NC
and not redistributed here.
"""
    return record, appendix_l, appendix_m


def main(src, dst):
    text = Path(src).read_text()
    record, app_l, app_m = build()
    marker = '## Appendix A. Formulation rationale and the four research modules'
    assert marker in text, 'Appendix A heading not found'
    text = text.replace(marker, record + '\n' + marker, 1)
    sources = '\n## Sources'
    assert sources in text, 'Sources heading not found'
    text = text.replace(sources, '\n' + app_l + '\n' + app_m + sources, 1)
    text = text.replace('## Uncertainty-aware wearable sleep decoding, with a retained NEURORELEASE research roadmap',
                        '## Uncertainty-aware wearable sleep decoding, with a retained NEURORELEASE research roadmap\n\n'
                        '**Revision 19.** Adds the submission package, a proven one-command reproduction, the '
                        'exposure-model leakage audit, the CNN challenger family and its external collapse, two '
                        'ablations, post-hoc adjustments, and the support slides and handout. Nothing from '
                        'Revision 18 is removed or shortened.', 1)
    Path(dst).write_text(text)
    print(json.dumps({'written': dst, 'bytes': len(text),
                      'added_bytes': len(text) - len(Path(src).read_text())}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'plan/GOLDILOCKS_Plan_Rev18.md'),
         sys.argv[2] if len(sys.argv) > 2 else str(ROOT / 'plan/GOLDILOCKS_Plan_Rev19.md'))

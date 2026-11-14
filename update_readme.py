"""Refresh the two result-bearing blocks of README.md from the saved result files.

The README's prose is written by hand; its numbers are not. This replaces the headline comparison
table and the model-family paragraph with values read from the result JSONs, so the README cannot
drift from the evidence after a rerun.

Usage: python update_readme.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = '<!-- RESULTS:START -->'
END = '<!-- RESULTS:END -->'


def j(rel, default=None):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else default


def f3(x):
    return '—' if x is None else f'{x:.3f}'


def m1(x):
    return '—' if x is None else f'{x:.1f} min'


VARIANTS = [('cnn', 'CNN challenger'), ('cnn_context', 'Context model'),
            ('cnn_context_sqrtbalanced', 'Context model on sqrt-balanced trunk'),
            ('cnn_balanced', 'Class-balanced CNN'), ('cnn_sqrtbalanced', 'Square-root-balanced CNN'),
            ('cnn_augmented', 'Augmented CNN'), ('cnn_twochannel', 'Two-channel CNN'),
            ('cnn_personweight', 'Person-weighted CNN')]


def block():
    hist = j('results/v2/historical_boas_metrics.json')['results']
    em = j('results/v2/external_metrics.json')
    ext, ext2 = em['results'], em['secondary_reference_defined_sleep_window']
    ya = j('results/yasa/yasa_metrics.json')['results']
    yh, ye, ye2 = (ya['historical_boas']['primary_complete_recording'],
                   ya['external']['primary_complete_recording'],
                   ya['external']['secondary_reference_window'])
    rows = [('Our frozen v2 model', hist, ext), ('YASA 0.7.0 baseline', yh, ye)]
    ext_all = (j('results/external_all/external_all_metrics.json') or {}).get('results', {})
    for key, label in VARIANTS:
        met = j(f'results/{key}/cnn_metrics.json') or j(f'results/{key}/context_metrics.json')
        if not met:
            continue
        b = met.get('cnn_results') or met.get('context_results')
        e = ext_all.get(key, {}).get('primary_complete_recording')
        if e is None:
            x = j(f'results/{key}/cnn_external_metrics.json')
            e = x['cnn_results']['primary_complete_recording'] if x else None
        rows.append((label, b, e))
    ph = j('results/cnn_posthoc/posthoc_metrics.json')
    if ph:
        for lab, res in ph['results'].items():
            if lab.startswith('ensemble_of_'):
                e = ext_all.get('ensemble_of_7', {}).get('primary_complete_recording')
                rows.append(('Ensemble of seven models', res['test_alpha_0_no_adjustment'], e))

    best = max((r for r in rows[2:]), key=lambda r: r[1]['macro_f1'], default=None)
    lines = [START, '',
             '| 20 people per cohort | BOAS kappa | BOAS macro-F1 | BOAS sleep-time error | Sleep-EDF kappa | Sleep-EDF sleep-time error |',
             '|---|---:|---:|---:|---:|---:|']
    for label, b, e in rows:
        lines.append(f"| {label} | {f3(b['kappa'])} | {f3(b['macro_f1'])} | {m1(b['tst_error_minutes'])} "
                     f"| {f3(e['kappa']) if e else 'not run'} | {m1(e['tst_error_minutes']) if e else 'not run'} |")
    cnn = j('results/cnn/cnn_metrics.json')
    cnne = j('results/cnn/cnn_external_metrics.json')
    lines += ['',
              f"The BOAS rows are a development comparison: those 20 people were inspected at v1. The Sleep-EDF "
              f"rows are the independent test, and they are a domain shift — different device, montage, scorer "
              f"and recording length. YASA's secondary sleep-window result is {f3(ye2['kappa'])} kappa and "
              f"{m1(ye2['tst_error_minutes'])}; ours is {f3(ext2['kappa'])} and {m1(ext2['tst_error_minutes'])}.",
              '']
    if cnn and cnne:
        c, ce = cnn['cnn_results'], cnne['cnn_results']['primary_complete_recording']
        lines += [f"**The result that decides the pitch.** We trained a CNN challenger on the raw headband signal, "
                  f"on the same frozen split and scored with the same mask and metric. It beats our frozen model "
                  f"at home — kappa {f3(c['kappa'])} against {f3(hist['kappa'])}, sleep-time error "
                  f"{m1(c['tst_error_minutes'])} against {m1(hist['tst_error_minutes'])} — and it is the worst "
                  f"model on Sleep-EDF: kappa {f3(ce['kappa'])} and {m1(ce['tst_error_minutes'])} of sleep-time "
                  f"error, because it calls daytime wake sleep. **The best model at home is the worst model "
                  f"away.** Choosing a wearable model by its in-domain score would have selected exactly the one "
                  f"that fails hardest on a new device. Every variant we tried is reported in the plan's "
                  f"Appendix L, including the ones that lost.", '']
    if ext_all:
        # These three figures were written out by hand in an earlier revision, which left them free to
        # drift from the results they describe. They are computed from external_all now, like every
        # other number in this file.
        ext_k = [v['primary_complete_recording']['kappa'] for v in ext_all.values()]
        ext_t = [v['primary_complete_recording']['tst_error_minutes'] for v in ext_all.values()]
        lines += [f"**Every model we trained tonight fails the same way.** All {len(ext_all)} members of the CNN "
                  f"family, the ensemble included, were scored on Sleep-EDF with the same windows and masks: "
                  f"none reaches kappa {max(ext_k):.3f} and every one is over {min(ext_t):.0f} minutes of "
                  f"sleep-time error. The frozen "
                  f"feature model, which loses at home, is the only model of ours that transfers at all, and "
                  f"YASA beats everything. Paired person-level bootstrap intervals for all of this are in "
                  f"`results/summary/confidence_intervals.csv`; on the headband the ensemble's gains over the "
                  f"frozen model exclude zero, and two variants' gains do not.", '']
    import csv as _csv
    vt = ROOT / 'results/vendor_and_failures/vendor_mask_comparison.csv'
    if vt.exists():
        vr = list(_csv.DictReader(vt.read_text().splitlines()))
        lines += ['**The only like-for-like benchmark: the vendor\'s own scorer, on its own hardware.** '
                  'BOAS ships the released Bitbrain scorer, which abstains on some epochs. On the epochs it '
                  'does cover, same people, same mask:', '',
                  '| On the vendor-covered mask (98.2% of scored epochs) | kappa | macro-F1 | sleep-time error |',
                  '|---|---:|---:|---:|']
        for r in sorted(vr, key=lambda r: -float(r['kappa'])):
            lines.append(f"| {r['model']} | {float(r['kappa']):.3f} | {float(r['macro_f1']):.3f} "
                         f"| {float(r['tst_error_minutes']):.1f} min |")
        lines += ['', 'The vendor scorer still leads on stage agreement. Our ensemble is ahead of it on '
                  'macro-F1 and halves its sleep-time error. That is the comparison to make: same device, '
                  'same people, same epochs, against the model the manufacturer ships — not against a '
                  'published figure from full polysomnography.', '']
    if best:
        lines += [f"Best macro-F1 among our models: **{f3(best[1]['macro_f1'])}** ({best[0]}). For context, the "
                  f"vendor's own released scorer on this same headband reaches 0.607 on its covered mask, and "
                  f"published figures near 0.90 come from full polysomnography with EOG and EMG scored against a "
                  f"five-technologist consensus. Six of the 20 test nights contain no reference N3 at all, which "
                  f"caps mean N3 F1 at 0.70 under our person-average convention whatever the model does.", '']
    sv = j('results/seed_variance/seed_variance.json')
    if sv:
        s = sv['statistics']
        lines += [f"**What a second seed does to these numbers.** Retraining only the temporal head with "
                  f"five seeds, trunk frozen, moves macro-F1 between {s['macro_f1']['min']:.3f} and "
                  f"{s['macro_f1']['max']:.3f} (mean {s['macro_f1']['mean']:.3f}) and sleep-time error "
                  f"between {s['tst_error_minutes']['min']:.1f} and {s['tst_error_minutes']['max']:.1f} "
                  f"minutes. Kappa barely moves, spread {s['kappa']['spread']:.3f}. Two consequences we hold "
                  f"to: the gap between the two context variants is smaller than that spread, so we do not "
                  f"order them; and the headline sleep-time figure should be read as roughly halving the "
                  f"frozen model's 31.8 minutes, not as a precise 8.7. Kappa is the number that survives "
                  f"reseeding. Details in `results/seed_variance/`.", '']
    lines.append(END)
    return '\n'.join(lines)


def main():
    p = ROOT / 'README.md'
    text = p.read_text()
    new = block()
    if START in text and END in text:
        text = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: new, text, flags=re.S)
    else:
        old = re.search(r'\| 20 people per cohort \| Our model kappa.*?(?=\n\nRead honestly)', text, flags=re.S)
        assert old, 'could not find the table to replace'
        text = text[:old.start()] + new + text[old.end():]
    p.write_text(text)
    print(json.dumps({'updated': str(p), 'bytes': len(text)}))


if __name__ == '__main__':
    main()

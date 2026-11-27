"""Paired person-level bootstrap intervals for every model against the frozen v2 model.

Twenty people is a small cohort. A table of point estimates invites the reading that 0.629 beats
0.561 and that is that. This resamples the 20 people 10,000 times, paired (both models scored on
the same resampled people), and reports the 95% interval of the difference. Where an interval
crosses zero, the difference is not established by this cohort, and the plan says so rather than
reporting the point estimate alone.

Deliberately not done: no multiplicity correction across the many variants, which means an
interval that just excludes zero among a dozen comparisons is weak evidence. That caveat is
printed with the results rather than buried.

Usage: python src/confidence_intervals.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/summary'
BOOT = 10000
SEED = 20261108

SOURCES = [
    ('CNN challenger', 'results/cnn/cnn_predictions.json', 'results/external_all/cnn_external_predictions.json'),
    ('Context model', 'results/cnn_context/context_predictions.json',
     'results/external_all/cnn_context_external_predictions.json'),
    ('Context model, sqrt-balanced trunk', 'results/cnn_context_sqrtbalanced/context_predictions.json',
     'results/external_all/cnn_context_sqrtbalanced_external_predictions.json'),
    ('Square-root-balanced CNN', 'results/cnn_sqrtbalanced/cnn_predictions.json',
     'results/external_all/cnn_sqrtbalanced_external_predictions.json'),
    ('Class-balanced CNN', 'results/cnn_balanced/cnn_predictions.json',
     'results/external_all/cnn_balanced_external_predictions.json'),
    ('Augmented CNN', 'results/cnn_augmented/cnn_predictions.json',
     'results/external_all/cnn_augmented_external_predictions.json'),
    ('Two-channel CNN', 'results/cnn_twochannel/cnn_predictions.json', None),
    ('Ensemble of seven models', 'results/cnn_posthoc/ensemble_predictions.json',
     'results/external_all/ensemble_of_7_external_predictions.json'),
    ('YASA 0.7.0 baseline', 'results/yasa/yasa_predictions.json:historical_boas',
     'results/yasa/yasa_predictions.json:external'),
]


def load(spec):
    if spec is None:
        return None
    path, _, key = spec.partition(':')
    p = ROOT / path
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get(key) if key else d


def per_person(pred, cohort):
    ref = json.loads((ROOT / f'results/v2/{cohort}_predictions.json').read_text())
    k, f, t, base_k, base_f, base_t = [], [], [], [], [], []
    for sub, fz in ref.items():
        if pred is not None and sub not in pred:
            return None
        y = np.array(fz['reference']); good = np.array(fz['good']).astype(bool)
        m = good & (y < 5)
        g = np.array(pred[sub]) if pred is not None else np.array(fz['model'])
        b = np.array(fz['model'])
        for arr, kk, ff, tt in [(g, k, f, t), (b, base_k, base_f, base_t)]:
            kk.append(cohen_kappa_score(y[m], arr[m]))
            ff.append(f1_score(y[m], arr[m], labels=list(range(5)), average='macro', zero_division=0))
            mt = float(np.sum((arr > 0) & m) * .5); rt = float(np.sum((y > 0) & m) * .5)
            tt.append(abs(mt - rt))
    return (np.array(k), np.array(f), np.array(t)), (np.array(base_k), np.array(base_f), np.array(base_t))


def interval(a, b, rng, lower_is_better=False):
    d = a - b
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    excludes = (lo > 0) or (hi < 0)
    better = (d.mean() < 0) if lower_is_better else (d.mean() > 0)
    return {'difference': float(d.mean()), 'ci95': [float(lo), float(hi)],
            'excludes_zero': bool(excludes),
            'verdict': ('better, interval excludes zero' if excludes and better else
                        'worse, interval excludes zero' if excludes else
                        'not established by this cohort')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {'baseline': 'frozen v2 feature model, same people, same masks',
              'people_per_cohort': 20, 'bootstrap_resamples': BOOT, 'paired': True,
              'caveat': 'No multiplicity correction across the twelve comparisons below. With this many '
                        'models, an interval that only just excludes zero is weak evidence. Twenty people '
                        'is a small cohort and these intervals are wide on purpose.',
              'results': {}}
    rows = []
    for label, boas_spec, ext_spec in SOURCES:
        for cohort, spec in [('historical_boas', boas_spec), ('external', ext_spec)]:
            pred = load(spec)
            if pred is None:
                continue
            got = per_person(pred, cohort)
            if got is None:
                continue
            (k, f, t), (bk, bf, bt) = got
            entry = {'kappa_vs_v2': interval(k, bk, rng),
                     'macro_f1_vs_v2': interval(f, bf, rng),
                     'sleep_time_error_vs_v2': interval(t, bt, rng, lower_is_better=True),
                     'model_means': {'kappa': float(k.mean()), 'macro_f1': float(f.mean()),
                                     'sleep_time_error_minutes': float(t.mean())},
                     'v2_means': {'kappa': float(bk.mean()), 'macro_f1': float(bf.mean()),
                                  'sleep_time_error_minutes': float(bt.mean())}}
            report['results'].setdefault(cohort, {})[label] = entry
            for metric in ['kappa_vs_v2', 'macro_f1_vs_v2', 'sleep_time_error_vs_v2']:
                e = entry[metric]
                rows.append({'cohort': cohort, 'model': label, 'metric': metric,
                             'difference': round(e['difference'], 4),
                             'ci_low': round(e['ci95'][0], 4), 'ci_high': round(e['ci95'][1], 4),
                             'verdict': e['verdict']})
    (OUT / 'confidence_intervals.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(OUT / 'confidence_intervals.csv', index=False)
    for cohort, models in report['results'].items():
        print(f'\n=== {cohort}: difference from the frozen v2 model, 95% paired bootstrap ===')
        print(f"{'model':36s}{'kappa':>22s}{'macro-F1':>22s}{'sleep-time (min)':>24s}")
        for label, e in models.items():
            def cell(m):
                d = e[m]
                star = '*' if d['excludes_zero'] else ' '
                return f"{d['difference']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]{star}"
            print(f"{label:36s}{cell('kappa_vs_v2'):>22s}{cell('macro_f1_vs_v2'):>22s}"
                  f"{cell('sleep_time_error_vs_v2'):>24s}")
    print('\n* interval excludes zero. No multiplicity correction; see the caveat in the JSON.')


if __name__ == '__main__':
    main()

"""One table of every model on the BOAS and Sleep-EDF cohorts, under four agreement metrics.

Why four. Published sleep-staging headline figures are not person-averaged macro-F1. The Dreem Open
Dataset paper reports per-stage F1 of 86.1 / 59.8 / 92.4 / 82.5 / 91.4 on DOD-H, whose average is
82.4, alongside a headline figure of 89.9 — the two only reconcile if the headline is
support-weighted rather than macro. So a comparison against published work has to name its metric.
This writes all four, side by side, for every model, and never substitutes one for another.

  person-averaged macro-F1  the strictest, and the one this project quotes by default. A stage
                            absent from a night scores 0 for that person, which caps mean N3 F1 at
                            14/20 on this cohort whatever a model does.
  pooled macro-F1           all epochs pooled, then macro-averaged over the five stages.
  weighted F1               pooled, averaged over stages weighted by how often each occurs. This is
                            the one comparable to published headline numbers.
  accuracy                  pooled; equals pooled micro-F1 for single-label classification.

Usage: python src/metrics_summary.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/summary'
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']

EXTERNAL_OVERRIDE = {
    'CNN challenger': 'results/external_all/cnn_external_predictions.json',
    'Context model': 'results/external_all/cnn_context_external_predictions.json',
    'Context model on sqrt-balanced trunk': 'results/external_all/cnn_context_sqrtbalanced_external_predictions.json',
    'Context model on balanced trunk': 'results/external_all/cnn_context_balanced_external_predictions.json',
    'Ensemble of seven models': 'results/external_all/ensemble_of_7_external_predictions.json',
    'Class-balanced CNN': 'results/external_all/cnn_balanced_external_predictions.json',
    'Square-root-balanced CNN': 'results/external_all/cnn_sqrtbalanced_external_predictions.json',
    'Augmented CNN': 'results/external_all/cnn_augmented_external_predictions.json',
}

MODELS = [
    ('Frozen v2 feature model', 'results/v2/{c}_predictions.json', 'model'),
    ('YASA 0.7.0 baseline', 'results/yasa/yasa_predictions.json', 'cohort'),
    ('CNN challenger', 'results/cnn/cnn_{e}predictions.json', None),
    ('Context model', 'results/cnn_context/context_predictions.json', None),
    ('Context model on sqrt-balanced trunk', 'results/cnn_context_sqrtbalanced/context_predictions.json', None),
    ('Context model on balanced trunk', 'results/cnn_context_balanced/context_predictions.json', None),
    ('Ensemble of seven models', 'results/cnn_posthoc/ensemble_predictions.json', None),
    ('Class-balanced CNN', 'results/cnn_balanced/cnn_{e}predictions.json', None),
    ('Square-root-balanced CNN', 'results/cnn_sqrtbalanced/cnn_{e}predictions.json', None),
    ('Augmented CNN', 'results/cnn_augmented/cnn_{e}predictions.json', None),
    ('Two-channel CNN', 'results/cnn_twochannel/cnn_{e}predictions.json', None),
    ('Person-weighted CNN', 'results/cnn_personweight/cnn_{e}predictions.json', None),
]


def reference(cohort):
    return json.loads((ROOT / f'results/v2/{cohort}_predictions.json').read_text())


def predictions(spec, kind, cohort):
    path = ROOT / spec.format(c=cohort, e='external_' if cohort == 'external' else '')
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if kind == 'cohort':
        d = d.get(cohort)
        if d is None:
            return None
    return d


def row(label, spec, kind, cohort):
    ref = reference(cohort)
    if cohort == 'external' and label in EXTERNAL_OVERRIDE:
        p = ROOT / EXTERNAL_OVERRIDE[label]
        pred = json.loads(p.read_text()) if p.exists() else None
    else:
        pred = predictions(spec, kind, cohort)
    if pred is None:
        return None
    per, Y, G = [], [], []
    for sub, fz in ref.items():
        if sub not in pred and kind != 'model':
            return None
        y = np.array(fz['reference']); good = np.array(fz['good']).astype(bool)
        m = good & (y < 5)
        g = np.array(fz['model']) if kind == 'model' else np.array(pred[sub])
        per.append(f1_score(y[m], g[m], labels=list(range(5)), average=None, zero_division=0))
        Y.append(y[m]); G.append(g[m])
    Y, G = np.concatenate(Y), np.concatenate(G)
    ps = np.mean(per, axis=0)
    return {'model': label, 'cohort': cohort, 'people': len(ref),
            'macro_f1_person_averaged': float(ps.mean()),
            'macro_f1_pooled': float(f1_score(Y, G, labels=list(range(5)), average='macro', zero_division=0)),
            'weighted_f1_pooled': float(f1_score(Y, G, labels=list(range(5)), average='weighted', zero_division=0)),
            'accuracy_pooled': float(accuracy_score(Y, G)),
            'kappa_pooled': float(cohen_kappa_score(Y, G, labels=list(range(5)))),
            **{f'f1_{s}_person_averaged': float(v) for s, v in zip(STAGES, ps)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for label, spec, kind in MODELS for cohort in ['historical_boas', 'external']
            for r in [row(label, spec, kind, cohort)] if r]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'metrics_summary.csv', index=False)
    note = ('Published headline sleep-staging figures near 0.90 are support-weighted, not person-averaged '
            'macro-F1: the Dreem Open Dataset paper\'s own per-stage F1 values average to 82.4 while its '
            'headline figure is 89.9. Compare like with like, and state which metric is meant. Six of the 20 '
            'BOAS test nights contain no reference N3, so under the person-averaged convention mean N3 F1 '
            'cannot exceed 0.70 for any model.')
    (OUT / 'metrics_summary.json').write_text(json.dumps({'note': note, 'rows': rows}, indent=2))
    for c in ['historical_boas', 'external']:
        sub = df[df.cohort == c]
        if sub.empty:
            continue
        print(f'\n=== {c} ===')
        print(f"{'model':28s}{'macro(person)':>14s}{'macro(pooled)':>14s}{'weighted':>10s}{'accuracy':>10s}")
        for _, r in sub.iterrows():
            print(f"{r['model']:28s}{r['macro_f1_person_averaged']:14.3f}{r['macro_f1_pooled']:14.3f}"
                  f"{r['weighted_f1_pooled']:10.3f}{r['accuracy_pooled']:10.3f}")
    print(json.dumps({'written': str(OUT), 'rows': len(rows)}))


if __name__ == '__main__':
    main()

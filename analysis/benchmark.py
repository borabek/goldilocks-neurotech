"""Main benchmark table and in-domain-to-external ranking, from saved predictions only.

Recomputes per-person metrics for every model on BOAS test, Sleep-EDF evaluation, DOD-H and HMC
(complete recording, scored epochs), checks the means against the metric files already in the
repository, and adds 95% person-level bootstrap intervals. Then asks whether the BOAS ranking of
the headband-trained models predicts their ranking on each external dataset (Kendall tau).

Exploratory: written for the paper on 28 November 2026, after every result it summarises.

Usage: python analysis/benchmark.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yasa_evaluate import score, windows  # noqa: E402

RESAMPLES, SEED = 10_000, 20261126
CNNS = ['cnn', 'cnn_balanced', 'cnn_sqrtbalanced', 'cnn_augmented']
CONTEXT = ['cnn_context', 'cnn_context_balanced', 'cnn_context_sqrtbalanced']
HEADBAND = ['v2'] + CNNS + CONTEXT          # every model trained on BOAS with predictions everywhere
DATASETS = ['boas', 'sleep_edf', 'dodh', 'hmc']
METRICS = ['kappa', 'macro_f1', 'wake_f1', 'wake_recall', 'tst_error_minutes']
TABLE_MODELS = ['yasa', 'v2', 'cnn', 'cnn_balanced']


def load(path):
    return json.loads((ROOT / path).read_text())


def cohorts():
    """{dataset: {recording: (reference, good, onset)}} and {dataset: {model: {recording: prediction}}}."""
    refs, preds = {}, {}
    hist, ext = load('results/v2/historical_boas_predictions.json'), load('results/v2/external_predictions.json')
    yasa = load('results/yasa/yasa_predictions.json')
    refs['boas'] = {s: (r['reference'], r['good'], r['onset']) for s, r in hist.items()}
    refs['sleep_edf'] = {s: (r['reference'], r['good'], r['onset']) for s, r in ext.items()}
    preds['boas'] = {'v2': {s: r['model'] for s, r in hist.items()}, 'yasa': yasa['historical_boas']}
    preds['sleep_edf'] = {'v2': {s: r['model'] for s, r in ext.items()}, 'yasa': yasa['external']}
    for m in CNNS:
        preds['boas'][m] = load(f'results/{m}/cnn_predictions.json')
    for m in CONTEXT:
        preds['boas'][m] = load(f'results/{m}/context_predictions.json')
    for m in CNNS + CONTEXT + ['ensemble_of_7']:
        preds['sleep_edf'][m] = load(f'results/external_all/{m}_external_predictions.json')
    for ds in ['dodh', 'hmc']:
        folder = f'results/external_{ds}/predictions'
        lab = load(f'{folder}/labels_and_grid.json')
        refs[ds] = {r: (v['reference'], v['good'], v['onset']) for r, v in lab.items()}
        preds[ds] = load(f'{folder}/predictions.json')
    return refs, preds


def per_person(ref, pred):
    rows = []
    for rec, (y, good, onset) in sorted(ref.items()):
        y, good, onset, p = np.array(y), np.array(good).astype(bool), np.array(onset, float), np.array(pred[rec])
        w = windows(y, onset)['primary_complete_recording']
        s = score(y, good, onset, p, w)
        m = good & w & (y == 0)
        rows.append({'recording': rec, 'kappa': s['kappa'], 'macro_f1': s['macro_f1'],
                     'wake_f1': s['per_stage_f1'][0], 'tst_error_minutes': s['tst_error_minutes'],
                     'wake_recall': float(np.mean(p[m] == 0)) if m.any() else float('nan')})
    return rows


def mean_ci(values, rng):
    v = np.array(values, float)
    v = v[np.isfinite(v)]
    boot = v[rng.integers(0, len(v), size=(RESAMPLES, len(v)))].mean(axis=1)
    return {'mean': float(v.mean()), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'people': int(len(v))}


def consistency_checks(table):
    """Means must equal the figures already published in the repository's metric files."""
    checks = []

    def add(name, got, want):
        checks.append({'check': name, 'recomputed': got, 'published': want, 'match': abs(got - want) < 1e-9})

    add('boas cnn kappa', table['boas']['cnn']['kappa']['mean'], load('results/cnn/cnn_metrics.json')['cnn_results']['kappa'])
    add('boas v2 kappa', table['boas']['v2']['kappa']['mean'],
        load('results/v2/historical_boas_metrics.json')['results']['kappa'])
    sedf = load('results/external_all/external_all_metrics.json')
    for m in CNNS + CONTEXT + ['ensemble_of_7']:
        add(f'sleep_edf {m} kappa', table['sleep_edf'][m]['kappa']['mean'],
            sedf['results'][m]['primary_complete_recording']['kappa'])
    add('sleep_edf v2 kappa', table['sleep_edf']['v2']['kappa']['mean'], sedf['frozen_v2_for_comparison']['kappa'])
    add('sleep_edf yasa kappa', table['sleep_edf']['yasa']['kappa']['mean'], sedf['yasa_for_comparison']['kappa'])
    for ds in ['dodh', 'hmc']:
        summ = load(f'results/external_{ds}/metrics.json')['summary']
        for m in summ:
            for k in ['kappa', 'macro_f1', 'wake_recall', 'tst_error_minutes']:
                add(f'{ds} {m} {k}', table[ds][m][k]['mean'], summ[m]['primary_complete_recording'][k])
    return checks


def main():
    refs, preds = cohorts()
    rng = np.random.default_rng(SEED)
    table, rows_all = {}, {}
    for ds in DATASETS:
        table[ds], rows_all[ds] = {}, {}
        for m in sorted(preds[ds]):
            rows = per_person(refs[ds], preds[ds][m])
            rows_all[ds][m] = rows
            table[ds][m] = {k: mean_ci([r[k] for r in rows], rng) for k in METRICS}
    checks = consistency_checks(table)
    bad = [c for c in checks if not c['match']]
    assert not bad, bad

    ranking = {'models': HEADBAND, 'note': 'headband-trained models with predictions on all four datasets; '
               'YASA is left out because it was not trained on BOAS', 'datasets': {}}
    boas = [table['boas'][m]['kappa']['mean'] for m in HEADBAND]
    best = HEADBAND[int(np.argmax(boas))]
    for ds in ['sleep_edf', 'dodh', 'hmc']:
        ext = [table[ds][m]['kappa']['mean'] for m in HEADBAND]
        tau, p = kendalltau(boas, ext)
        order = sorted(HEADBAND, key=lambda m: -table[ds][m]['kappa']['mean'])
        ranking['datasets'][ds] = {
            'kendall_tau_kappa': float(tau), 'p_value': float(p),
            'boas_best_model': best, 'boas_best_rank_here': order.index(best) + 1, 'of': len(HEADBAND),
            'best_here': order[0], 'yasa_kappa': table[ds]['yasa']['kappa']['mean'],
            'yasa_beats_every_headband_model': bool(table[ds]['yasa']['kappa']['mean'] > max(ext))}
    out = {'status': 'EXPLORATORY summary of saved predictions, written for the paper after all results were known',
           'window': 'primary_complete_recording', 'resamples': RESAMPLES, 'seed': SEED,
           'table': table, 'ranking': ranking, 'consistency_checks_passed': len(checks),
           'main_table_models': TABLE_MODELS}
    (ROOT / 'results/paper').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/paper/benchmark.json').write_text(json.dumps(out, indent=2))
    print(len(checks), 'consistency checks passed')
    for ds in DATASETS:
        print(ds, {m: round(table[ds][m]['kappa']['mean'], 3) for m in table[ds]})
    print(json.dumps(ranking['datasets'], indent=1))


if __name__ == '__main__':
    main()

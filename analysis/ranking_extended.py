"""Wider ranking test and five-seed spread (PREREGISTRATION.md, addendum of 28 November 2026). Exploratory.

1. Kendall's tau between BOAS kappa and external kappa over every headband-trained model with
   predictions on all four datasets: v2, the seven CNN variants, the 10-pass plain CNNs at five
   seeds, and the lowpass and dropout CNNs at three seeds each. A 95% interval comes from resampling
   people within each dataset and recomputing every model's mean kappa and the tau.
2. Paired kappa difference between YASA and the best headband model on each external dataset.
3. The five 10-pass plain CNN seeds: BOAS kappa and external wake recall, with the ranges, and the
   rule fixed in the addendum (Sleep-EDF wake recall range >= 0.10 while the BOAS kappa range < 0.02).

External rows for the retrained models come from the same per-person scoring as
analysis/prereg_v2_seeds.py (its row cache is reused when present).

Usage: python analysis/ranking_extended.py --boas-cache B --ext-cache E --new-cache N
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'analysis'))
import benchmark as bm  # noqa: E402
import prereg_v2_eval as pe  # noqa: E402
import prereg_v2_seeds as ps  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402

RESAMPLES, SEED = 10_000, 20261126
PLAIN_SEEDS = [20261108, 20261109, 20261110, 20261111, 20261112]
RETRAINED = (['cnn_lowpass16', 'cnn_lowpass16_seed20261109', 'cnn_lowpass16_seed20261110',
              'cnn_hfdrop', 'cnn_hfdrop_seed20261109', 'cnn_hfdrop_seed20261110'])
EXTERNAL = ['sleep_edf', 'dodh', 'hmc']


def recipe_of(name):
    return 'cnn_lowpass16' if name.startswith('cnn_lowpass16') else 'cnn'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    ap.add_argument('--new-cache', required=True)
    args = ap.parse_args()
    pe.cc.TWOCHANNEL = False
    refs, preds = bm.cohorts()
    trans, prior = fit_trans_prior(args.boas_cache)
    items = pe.cohort_items(args)

    # kappa[model][dataset] -> {recording: kappa}; wake[...] likewise
    kappa, wake = {}, {}
    for m in bm.HEADBAND + ['yasa']:
        kappa[m], wake[m] = {}, {}
        for ds in bm.DATASETS:
            rows = bm.per_person(refs[ds], preds[ds][m])
            kappa[m][ds] = {r['recording']: r['kappa'] for r in rows}
            wake[m][ds] = {r['recording']: r['wake_recall'] for r in rows}
    extra = [f'cnn_seed{s}' for s in PLAIN_SEEDS] + RETRAINED
    for m in extra:
        kappa[m], wake[m] = {}, {}
        boas_pred = json.loads((ROOT / 'results' / m / 'cnn_predictions.json').read_text())
        rows = bm.per_person(refs['boas'], boas_pred)
        kappa[m]['boas'] = {r['recording']: r['kappa'] for r in rows}
        wake[m]['boas'] = {r['recording']: r['wake_recall'] for r in rows}
        for ds in EXTERNAL:
            rows = ps.per_person(m, recipe_of(m), items[ds], trans, prior)
            kappa[m][ds] = {r['recording']: r['kappa'] for r in rows}
            wake[m][ds] = {r['recording']: r['wake_recall'] for r in rows}
        print('scored', m, flush=True)

    models = bm.HEADBAND + extra
    recs = {ds: sorted(refs[ds]) for ds in bm.DATASETS}
    for m in models + ['yasa']:
        for ds in bm.DATASETS:
            assert sorted(kappa[m][ds]) == recs[ds], (m, ds)
    K = {ds: np.array([[kappa[m][ds][r] for r in recs[ds]] for m in models]) for ds in bm.DATASETS}

    rng = np.random.default_rng(SEED)
    out = {'status': 'EXPLORATORY: addendum of 28 November 2026, written before this ran',
           'models': models, 'resamples': RESAMPLES, 'seed': SEED, 'ranking': {}}
    boas_mean = K['boas'].mean(axis=1)
    for ds in EXTERNAL:
        ext_mean = K[ds].mean(axis=1)
        tau, p = kendalltau(boas_mean, ext_mean)
        taus = np.empty(RESAMPLES)
        nb, ne = K['boas'].shape[1], K[ds].shape[1]
        for i in range(RESAMPLES):
            b = K['boas'][:, rng.integers(0, nb, nb)].mean(axis=1)
            e = K[ds][:, rng.integers(0, ne, ne)].mean(axis=1)
            taus[i] = kendalltau(b, e)[0]
        best = models[int(np.argmax(ext_mean))]
        d = np.array([kappa['yasa'][ds][r] - kappa[best][ds][r] for r in recs[ds]])
        db = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
        out['ranking'][ds] = {
            'kendall_tau_kappa': float(tau), 'p_value': float(p), 'models': len(models),
            'tau_ci95': [float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))],
            'best_headband_model_here': best, 'best_headband_kappa': float(ext_mean.max()),
            'boas_best_model': models[int(np.argmax(boas_mean))],
            'boas_best_rank_here': int(sorted(models, key=lambda m: -K[ds][models.index(m)].mean()).index(
                models[int(np.argmax(boas_mean))]) + 1),
            'yasa_minus_best_headband_kappa': {'mean_difference': float(d.mean()),
                                               'ci95': [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}}
        print(ds, json.dumps(out['ranking'][ds]), flush=True)

    plain = [f'cnn_seed{s}' for s in PLAIN_SEEDS]
    spread = {m: {'boas_kappa': float(np.mean(list(kappa[m]['boas'].values()))),
                  **{f'{ds}_wake_recall': float(np.nanmean(list(wake[m][ds].values()))) for ds in EXTERNAL}}
              for m in plain}

    def rng_of(key):
        v = [spread[m][key] for m in plain]
        return {'min': min(v), 'max': max(v), 'range': max(v) - min(v)}

    out['five_seeds'] = {'models': spread, 'ranges': {k: rng_of(k) for k in
                                                      ['boas_kappa'] + [f'{ds}_wake_recall' for ds in EXTERNAL]}}
    r = out['five_seeds']['ranges']
    out['five_seeds']['rule'] = {'statement': 'Sleep-EDF wake recall range >= 0.10 and BOAS kappa range < 0.02',
                                 'holds': bool(r['sleep_edf_wake_recall']['range'] >= 0.10 and r['boas_kappa']['range'] < 0.02)}
    (ROOT / 'results/paper').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/paper/ranking_extended.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out['five_seeds'], indent=1))


if __name__ == '__main__':
    main()

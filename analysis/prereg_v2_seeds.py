"""Replicate seeds for the prereg-v2 CNNs (PREREGISTRATION.md, addendum prereg-v2.1, item 1).

Post-result robustness check. Seed 20261108 is the original run (results/cnn, results/cnn_lowpass16,
results/cnn_hfdrop); seeds 20261109 and 20261110 are the same recipes retrained. The evaluation
code is the prereg-v2 code (analysis/prereg_v2_eval.py). For each seed, the wake recall change is
taken against the plain CNN of the same seed and against the frozen plain CNN. The cnn_hfdrop
effect is called robust only if the matched-seed mean change is above 0 for every seed on every
external dataset. A pooled estimate averages each person's wake recall over the three seeds
before the paired bootstrap.

Usage: python analysis/prereg_v2_seeds.py --boas-cache B --ext-cache E --new-cache /tmp/goldilocks_external_cache
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'analysis'))
import prereg_v2_eval as pe  # noqa: E402
from cnn_challenger import lowpass16, night_probabilities, rebuild_free, robust_scale, CLIP  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402
from yasa_evaluate import score, windows  # noqa: E402

SEEDS = [20261108, 20261109, 20261110]
RECIPES = ['cnn', 'cnn_lowpass16', 'cnn_hfdrop']
DATASETS = ['sleep_edf', 'dodh', 'hmc']


MATCHED = False   # --matched-passes: use the 10-pass plain CNN at seed 20261108 (deviations log, 28 Nov)


def folder(recipe, seed):
    if MATCHED and recipe == 'cnn' and seed == SEEDS[0]:
        return f'cnn_seed{seed}'
    return recipe if seed == SEEDS[0] else f'{recipe}_seed{seed}'


ROWS_CACHE = Path('/tmp/goldilocks_seed_rows')


def per_person(name, recipe, items, trans, prior):
    cached = ROWS_CACHE / f'{name}_{items[0][1].parent.name}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    out = _per_person(name, recipe, items, trans, prior)
    ROWS_CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out))
    return out


def _per_person(name, recipe, items, trans, prior):
    src = ROOT / 'results' / name
    freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
    params = rebuild_free(src / 'cnn_weights.json')
    out = []
    for rec, path, y in items:
        z = np.load(path)
        x = z['X'].astype('float32')
        if recipe == 'cnn_lowpass16':
            x = lowpass16(x)
        good, onset = z['good'].astype(bool), z['onset'].astype(float)
        med, mad = robust_scale(x, good)
        p = night_probabilities(params, np.clip((x - med) / mad, -CLIP, CLIP), good, onset, trans, prior,
                                freeze['selected_sequence_weight'], freeze['temperature'])
        g = p.argmax(axis=1)
        s = score(y, good, onset, g, windows(y, onset)['primary_complete_recording'])
        m = good & (y == 0)
        out.append({'recording': rec, 'kappa': s['kappa'],
                    'wake_recall': float(np.mean(g[m] == 0)) if m.any() else float('nan')})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    ap.add_argument('--new-cache', required=True)
    ap.add_argument('--only-seed', type=int, help='score one seed and stop (warms the row cache)')
    ap.add_argument('--matched-passes', action='store_true',
                    help='compare 10-pass models with the 10-pass plain CNN at every seed (writes seeds_matched_passes.json)')
    args = ap.parse_args()
    global MATCHED
    MATCHED = args.matched_passes
    pe.cc.TWOCHANNEL = False
    rng = np.random.default_rng(pe.SEED)
    trans, prior = fit_trans_prior(args.boas_cache)
    items = pe.cohort_items(args)
    first = json.loads((ROOT / 'results/prereg_v2/prereg_v2_results.json').read_text())['per_person']
    rows, boas = {}, {}
    if args.only_seed:
        for recipe in RECIPES:
            for ds in DATASETS:
                per_person(folder(recipe, args.only_seed), recipe, items[ds], trans, prior)
        return
    for seed in SEEDS:
        for recipe in RECIPES:
            name = folder(recipe, seed)
            boas[name] = json.loads((ROOT / 'results' / name / 'cnn_metrics.json').read_text())['cnn_results']['kappa']
            rows[name] = {}
            for ds in DATASETS:
                if name == recipe:            # the original runs, already scored by prereg_v2_eval.py
                    rows[name][ds] = [{k: r[k] for k in ['recording', 'kappa', 'wake_recall']}
                                      for r in first[ds][recipe]]
                else:
                    rows[name][ds] = per_person(name, recipe, items[ds], trans, prior)
            print(name, round(boas[name], 3),
                  {ds: round(float(np.nanmean([r['wake_recall'] for r in rows[name][ds]])), 3) for ds in DATASETS},
                  flush=True)
    frozen = {ds: [{k: r[k] for k in ['recording', 'kappa', 'wake_recall']} for r in first[ds]['cnn']]
              for ds in DATASETS}
    if MATCHED:
        rows['cnn_frozen_25_passes'] = frozen
        boas['cnn_frozen_25_passes'] = json.loads((ROOT / 'results/cnn/cnn_metrics.json').read_text())['cnn_results']['kappa']
    report = {'addendum': 'PREREGISTRATION.md, prereg-v2.1 item 1 (post-result robustness check)',
              'seeds': SEEDS, 'boas_kappa': boas, 'mean_wake_recall': {}, 'matched_seed_change': {},
              'change_vs_frozen_cnn': {}, 'pooled_over_seeds': {}, 'robust': {}}
    if MATCHED:
        report['matched_passes'] = ('every seed compared with the 10-pass plain CNN of the same seed; '
                                    'see the 28 November entry of the deviations log')
    for name in rows:
        report['mean_wake_recall'][name] = {ds: float(np.nanmean([r['wake_recall'] for r in rows[name][ds]]))
                                            for ds in DATASETS}
    for recipe in ['cnn_lowpass16', 'cnn_hfdrop']:
        report['matched_seed_change'][recipe], report['change_vs_frozen_cnn'][recipe] = {}, {}
        for seed in SEEDS:
            new, base = folder(recipe, seed), folder('cnn', seed)
            report['matched_seed_change'][recipe][str(seed)] = {
                ds: {'wake_recall': pe.paired(rows[base][ds], rows[new][ds], 'wake_recall', rng),
                     'kappa': pe.paired(rows[base][ds], rows[new][ds], 'kappa', rng),
                     'boas_kappa_drop': boas[base] - boas[new]} for ds in DATASETS}
            report['change_vs_frozen_cnn'][recipe][str(seed)] = {
                ds: pe.paired(frozen[ds], rows[new][ds], 'wake_recall', rng) for ds in DATASETS}
        pooled = {}
        for ds in DATASETS:
            avg = lambda r: [{'wake_recall': float(np.nanmean([rows[folder(r, s)][ds][i]['wake_recall'] for s in SEEDS]))}
                             for i in range(len(frozen[ds]))]
            pooled[ds] = pe.paired(avg('cnn'), avg(recipe), 'wake_recall', rng)
        report['pooled_over_seeds'][recipe] = pooled
        report['robust'][recipe] = all(report['matched_seed_change'][recipe][str(s)][ds]['wake_recall']['mean_change'] > 0
                                       for s in SEEDS for ds in DATASETS)
    (ROOT / 'results/prereg_v2_1').mkdir(parents=True, exist_ok=True)
    out_name = 'seeds_matched_passes.json' if MATCHED else 'seeds.json'
    (ROOT / 'results/prereg_v2_1' / out_name).write_text(json.dumps({**report, 'per_person': rows}, indent=2))
    for recipe in ['cnn_lowpass16', 'cnn_hfdrop']:
        print(recipe, 'robust', report['robust'][recipe])
        for s in SEEDS:
            m = report['matched_seed_change'][recipe][str(s)]
            print('  ', s, {ds: (round(m[ds]['wake_recall']['mean_change'], 3),
                                 [round(x, 3) for x in m[ds]['wake_recall']['ci95']],
                                 'boasdrop', round(m[ds]['boas_kappa_drop'], 3)) for ds in DATASETS})
        print('   pooled', {ds: (round(v['mean_change'], 3), [round(x, 3) for x in v['ci95']])
                            for ds, v in report['pooled_over_seeds'][recipe].items()})


if __name__ == '__main__':
    main()

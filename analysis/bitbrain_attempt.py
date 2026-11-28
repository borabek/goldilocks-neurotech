"""One exploratory attempt at the gap to the released Bitbrain scorer on BOAS.

The attempt: average the probabilities of the frozen v2 model and the frozen ensemble of seven CNN
variants, p = w * p_v2 + (1 - w) * p_cnn. Nothing is retrained. w is chosen from 0, 0.1, ..., 1 by
mean person kappa on the six selection people only (the v2 model-selection people, none of whom
any model was trained on), then applied once to the 20 BOAS test people. The test people have
been looked at many times before, so this is exploratory and says nothing about other devices.

Kappa is reported on two masks: every scored epoch with usable signal, and the subset the
released Bitbrain scorer also labels, which is the only mask where the vendor can be compared.

Usage: python analysis/bitbrain_attempt.py --boas-cache <64 Hz BOAS cache>
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from cnn_posthoc import variant_probabilities  # noqa: E402
from pipeline import stage_metrics  # noqa: E402
from v2_model import probabilities  # noqa: E402

MEMBERS = ['cnn', 'cnn_context', 'cnn_context_sqrtbalanced', 'cnn_sqrtbalanced', 'cnn_balanced',
           'cnn_augmented', 'cnn_twochannel']           # the seven of results/cnn_posthoc
WEIGHTS = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
RESAMPLES, SEED = 10_000, 20261126


def people():
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    sel_pids = set(json.loads((ROOT / 'v2_protocol.json').read_text())['model_selection_pids'])
    sel = [r['subject'] for r in rows if r['pid'] in sel_pids and r['confirmatory_night']]
    test = sorted(json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text()),
                  key=lambda s: int(s.split('-')[1]))
    return sel, test


def v2_probs(subs):
    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    out = {}
    for s in subs:
        z = dict(np.load(ROOT / 'data/features' / f'{s}.npz'))
        out[s] = (probabilities(frozen['model'], z['X'], z['good'], frozen['transition'], frozen['prior'],
                                frozen['weight'], frozen['temperature'], z['onset']), z)
    return out


def kappas(prob, z, vendor_mask):
    y, good = z['y'], z['good'].astype(bool)
    m = good & (y < 5)
    if vendor_mask:
        m &= (z['vendor'] >= 0) & (z['vendor'] < 5)
    return stage_metrics(y[m], prob[m].argmax(axis=1))['kappa']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    args = ap.parse_args()
    sel, test = people()
    v2 = v2_probs(sel + test)
    members = {}
    for name in MEMBERS:
        members[name], _ = variant_probabilities(name, args.boas_cache, sel + test)
        print('loaded', name, flush=True)
    cnn = {}
    for s in sel + test:
        p = np.mean([members[n][s][0] for n in MEMBERS], axis=0)
        cnn[s] = p / p.sum(axis=1, keepdims=True)
        assert len(cnn[s]) == len(v2[s][0]), s
        assert np.array_equal(members[MEMBERS[0]][s][1], v2[s][1]['y']), s

    def fused(s, w):
        return w * v2[s][0] + (1 - w) * cnn[s]

    grid = [{'w_v2': w, 'selection_mean_kappa': float(np.mean([kappas(fused(s, w), v2[s][1], False) for s in sel]))}
            for w in WEIGHTS]
    w = max(grid, key=lambda r: r['selection_mean_kappa'])['w_v2']

    rng = np.random.default_rng(SEED)
    test_rows = []
    for s in test:
        z = v2[s][1]
        vend = z['vendor'].astype(int)
        row = {'subject': s}
        for mask_name, vm in [('all_scored', False), ('vendor_covered', True)]:
            row[mask_name] = {'fusion': kappas(fused(s, w), z, vm), 'v2': kappas(v2[s][0], z, vm),
                              'cnn_ensemble': kappas(cnn[s], z, vm)}
        m = z['good'].astype(bool) & (z['y'] < 5) & (vend >= 0) & (vend < 5)
        row['vendor_covered']['bitbrain'] = stage_metrics(z['y'][m], vend[m])['kappa']
        test_rows.append(row)

    def summ(mask, model):
        v = np.array([r[mask][model] for r in test_rows])
        b = v[rng.integers(0, len(v), size=(RESAMPLES, len(v)))].mean(axis=1)
        return {'mean': float(v.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    def paired(mask, a, b):
        d = np.array([r[mask][a] - r[mask][b] for r in test_rows])
        bb = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
        return {'mean_difference': float(d.mean()), 'ci95': [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}

    out = {'status': 'EXPLORATORY: one attempt, chosen on the six selection people, applied once to the 20 BOAS test '
                     'people, who have been inspected before',
           'members': MEMBERS, 'selection_people': sel, 'test_people': test, 'selection_grid': grid,
           'chosen_w_v2': w,
           'test': {'all_scored': {m: summ('all_scored', m) for m in ['fusion', 'v2', 'cnn_ensemble']},
                    'vendor_covered': {m: summ('vendor_covered', m) for m in ['fusion', 'v2', 'cnn_ensemble', 'bitbrain']}},
           'paired': {'fusion_minus_cnn_ensemble_all_scored': paired('all_scored', 'fusion', 'cnn_ensemble'),
                      'fusion_minus_bitbrain_vendor_covered': paired('vendor_covered', 'fusion', 'bitbrain'),
                      'cnn_ensemble_minus_bitbrain_vendor_covered': paired('vendor_covered', 'cnn_ensemble', 'bitbrain')},
           'per_person': test_rows}
    (ROOT / 'results/paper').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/paper/bitbrain_attempt.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ['selection_grid', 'chosen_w_v2', 'test', 'paired']}, indent=1))


if __name__ == '__main__':
    main()

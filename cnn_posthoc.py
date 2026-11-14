"""Two post-hoc improvements that need no retraining: prior adjustment and an ensemble.

Prior adjustment. A classifier trained on a corpus that is 59% N2 predicts N2 too readily. Dividing
each probability by the training frequency raised to a power alpha shifts the decision towards the
rare stages. alpha is chosen on the six selection people only, from a fixed grid, and the same
chosen value is then applied to the 20 test people.

Ensemble. The average of the variants' probabilities, then the same alpha selection.

Nothing is retrained and nothing is selected on the test people. Every variant's own frozen
sequence weight and temperature are used, so each contributes the probabilities it was frozen to
produce. Both adjustments are reported for kappa and macro-F1 at once, because an alpha chosen for
one is usually worse for the other, and hiding that would be the dishonest part.

Usage: python cnn_posthoc.py <boas_cache> [variant,variant,...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

import cnn_challenger as cc
from cnn_challenger import N_CLASS, load_night, night_probabilities, rebuild_free
from pipeline import ROOT, stage_metrics

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OFFSETS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]      # additive log-odds boosts for N1 and N3 only
OUT = ROOT / 'results/cnn_posthoc'


def fit_trans_prior(cache):
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for r in rows:
        if r['split'] != 'fit':
            continue
        z = np.load(Path(cache) / f"{r['subject']}.npz")
        y, good = z['y'].astype('int8'), z['good'].astype(bool)
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
    prior /= prior.sum()
    return trans / trans.sum(axis=1, keepdims=True), prior


def context_probabilities(name, cache, subs):
    """Frozen probabilities for a context model: frozen trunk as encoder, frozen temporal head."""
    import cnn_context as cx
    src = ROOT / 'results' / name
    freeze = json.loads((src / 'context_freeze_record.json').read_text())
    trunk = rebuild_free(ROOT / freeze['trunk_source'] / 'cnn_weights.json')
    head = rebuild_free(src / 'context_head_weights.json')
    trans, prior = fit_trans_prior(cache)
    out = {}
    for sub in subs:
        e, y, good, onset = cx.night_embeddings(trunk, cache, sub, load_night)
        out[sub] = (cx.head_night_probabilities(head, e, good, onset, trans, prior,
                                                freeze['selected_sequence_weight'], freeze['temperature']),
                    y, good)
    return out, prior


def variant_probabilities(name, cache, subs):
    """Frozen probabilities for one variant, per night, exactly as that variant was frozen."""
    if name.startswith('cnn_context'):
        return context_probabilities(name, cache, subs)
    src = ROOT / 'results' / name
    freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
    params = rebuild_free(src / 'cnn_weights.json')
    cc.TWOCHANNEL = bool(freeze.get('two_channel_input', False))
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for r in rows:
        if r['split'] != 'fit':
            continue
        z = np.load(Path(cache) / f"{r['subject']}.npz")
        y, good = z['y'].astype('int8'), z['good'].astype(bool)
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
    prior /= prior.sum(); trans /= trans.sum(axis=1, keepdims=True)
    out = {}
    for sub in subs:
        x, y, good, onset = load_night(cache, sub)
        out[sub] = (night_probabilities(params, x, good, onset, trans, prior,
                                        freeze['selected_sequence_weight'], freeze['temperature']),
                    y, good)
    cc.TWOCHANNEL = False
    return out, prior


def adjust(p, prior, alpha):
    if alpha == 0:
        return p
    q = p / prior[None, :] ** alpha
    return q / q.sum(axis=1, keepdims=True)


def offset(p, b1, b3):
    """Additive log-odds boost for N1 and N3 only. Unlike temperature, this can change the
    predicted class, which is the point; the search is deliberately two-dimensional and coarse so
    six selection people cannot be fitted in detail."""
    if b1 == 0 and b3 == 0:
        return p
    q = p * np.array([1.0, np.exp(b1), 1.0, np.exp(b3), 1.0])[None, :]
    return q / q.sum(axis=1, keepdims=True)


def score_offsets(probs, b1, b3):
    ks, fs = [], []
    for sub, (p, y, good) in probs.items():
        m = good & (y < 5)
        g = offset(p, b1, b3)[m].argmax(axis=1)
        ks.append(cohen_kappa_score(y[m], g))
        fs.append(f1_score(y[m], g, labels=list(range(N_CLASS)), average='macro', zero_division=0))
    return float(np.mean(ks)), float(np.mean(fs))


def score_set(probs, prior, alpha):
    ks, fs = [], []
    for sub, (p, y, good) in probs.items():
        m = good & (y < 5)
        g = adjust(p, prior, alpha)[m].argmax(axis=1)
        ks.append(cohen_kappa_score(y[m], g))
        fs.append(f1_score(y[m], g, labels=list(range(N_CLASS)), average='macro', zero_division=0))
    return float(np.mean(ks)), float(np.mean(fs))


def main(cache, variants):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    v2p = json.loads((ROOT / 'v2_protocol.json').read_text())
    sel_pids = set(v2p['model_selection_pids'])
    sel = [r['subject'] for r in rows if r['pid'] in sel_pids and r['confirmatory_night']]
    frozen_test = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    test = sorted(frozen_test, key=lambda s: int(s.split('-')[1]))

    per_variant, prior = {}, None
    for name in variants:
        per_variant[name], prior = variant_probabilities(name, cache, sel + test)
        print(json.dumps({'loaded': name, 'nights': len(per_variant[name])}), flush=True)

    report = {'variants': variants, 'alpha_grid': ALPHAS, 'prior': prior.tolist(),
              'selection_people': sel, 'test_people': test, 'results': {}}

    def evaluate(label, probs):
        grid = [{'alpha': a, **dict(zip(['selection_kappa', 'selection_macro_f1'],
                                        score_set({s: probs[s] for s in sel}, prior, a)))} for a in ALPHAS]
        best_k = max(grid, key=lambda r: r['selection_kappa'])['alpha']
        best_f = max(grid, key=lambda r: r['selection_macro_f1'])['alpha']
        out = {'selection_grid': grid, 'alpha_chosen_for_kappa': best_k, 'alpha_chosen_for_macro_f1': best_f}
        grid2 = [{'n1_boost': b1, 'n3_boost': b3,
                  **dict(zip(['selection_kappa', 'selection_macro_f1'],
                             score_offsets({s: probs[s] for s in sel}, b1, b3)))}
                 for b1 in OFFSETS for b3 in OFFSETS]
        bo = max(grid2, key=lambda r: r['selection_macro_f1'])
        out['offset_grid_best'] = bo
        out['offset_grid_size'] = len(grid2)
        rows_o = []
        for sub in test:
            p, y, good = probs[sub]
            m = good & (y < 5)
            adj = offset(p, bo['n1_boost'], bo['n3_boost'])
            g = adj[m].argmax(axis=1)
            st = stage_metrics(y[m], g)
            mt = float(np.sum((adj.argmax(axis=1) > 0) & m) * .5); rt = float(np.sum((y > 0) & m) * .5)
            rows_o.append({'subject': sub, **st, 'tst_error_minutes': abs(mt - rt)})
        out['test_class_offsets_chosen_for_macro_f1'] = {
            'n1_boost': bo['n1_boost'], 'n3_boost': bo['n3_boost'], 'people': len(rows_o),
            **{k: float(np.mean([r[k] for r in rows_o]))
               for k in ['accuracy', 'kappa', 'macro_f1', 'tst_error_minutes']}}
        out.setdefault('per_person', {})['class_offsets'] = rows_o
        for tag, a in [('alpha_0_no_adjustment', 0.0), ('alpha_chosen_for_kappa', best_k),
                       ('alpha_chosen_for_macro_f1', best_f)]:
            rows_ = []
            for sub in test:
                p, y, good = probs[sub]
                m = good & (y < 5)
                g = adjust(p, prior, a)[m].argmax(axis=1)
                st = stage_metrics(y[m], g)
                mt = float(np.sum((adjust(p, prior, a).argmax(axis=1) > 0) & m) * .5)
                rt = float(np.sum((y > 0) & m) * .5)
                rows_.append({'subject': sub, **st, 'tst_error_minutes': abs(mt - rt)})
            out[f'test_{tag}'] = {'alpha': a, 'people': len(rows_),
                                  **{k: float(np.mean([r[k] for r in rows_]))
                                     for k in ['accuracy', 'kappa', 'macro_f1', 'tst_error_minutes']}}
            out.setdefault('per_person', {})[tag] = rows_
        safe = label if len(label) < 40 else 'ensemble'
        preds = {}
        for sub in test:
            pr, y, good = probs[sub]
            preds[sub] = adjust(pr, prior, 0.0).argmax(axis=1).tolist()
        (OUT / f'{safe}_predictions.json').write_text(json.dumps(preds))
        out['prediction_file'] = f'{safe}_predictions.json'
        out['members'] = variants if safe == 'ensemble' else [label]
        report['results'][label] = out
        print(json.dumps({label: {k: v for k, v in out.items() if k.startswith('test_')}}, indent=1), flush=True)
        return out

    for name in variants:
        evaluate(name, per_variant[name])

    if len(variants) > 1:
        ens = {}
        for sub in sel + test:
            ps = np.mean([per_variant[n][sub][0] for n in variants], axis=0)
            _, y, good = per_variant[variants[0]][sub]
            ens[sub] = (ps / ps.sum(axis=1, keepdims=True), y, good)
        evaluate('ensemble_of_' + '+'.join(variants), ens)

    saved = json.loads((ROOT / 'results/v2/historical_boas_metrics.json').read_text())['results']
    report['v2_results_for_the_same_people'] = {k: saved[k] for k in
                                                ['people', 'accuracy', 'kappa', 'macro_f1', 'tst_error_minutes']}
    report['created_unix'] = time.time(); report['seconds'] = round(time.time() - t0, 1)
    (OUT / 'posthoc_metrics.json').write_text(json.dumps(report, indent=2))
    tags = ['alpha_0_no_adjustment', 'alpha_chosen_for_kappa', 'alpha_chosen_for_macro_f1',
            'class_offsets_chosen_for_macro_f1']
    table = [{'model': lab, 'setting': tag, **{k: v for k, v in r[f'test_{tag}'].items()}}
             for lab, r in report['results'].items() for tag in tags if f'test_{tag}' in r]
    pd.DataFrame(table).to_csv(OUT / 'posthoc_table.csv', index=False)
    print(json.dumps({'written': str(OUT), 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    vs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['cnn', 'cnn_balanced']
    main(sys.argv[1], vs)

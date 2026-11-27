"""Two ablations the evidence asked for: what the sequence smoother costs, and what the selection
metric costs.

Question 1. The smoother raised mean person kappa on the selection people from 0.39 to 0.63. An
operation that large is suspicious for the rare stages: a transition model trained on a corpus
that is 61% N2 can erase short N1 segments, which raises kappa while destroying N1 recall. This
measures that per stage instead of assuming it either way.

Question 2. v2 chose its sequence weight by kappa. If macro-F1 would have chosen a different
weight, then the model was optimised for a different objective than the one being reported.

Both are reported on the six selection people (where choices are allowed) and on the 20 test
people (reported, never used to choose). Nothing is selected here.

Usage: python src/cnn_ablation.py <boas_cache> [variant,variant,...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, precision_score

import cnn_challenger as cc
from cnn_challenger import N_CLASS, load_night, night_probabilities, rebuild_free
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared core modules sit at the top level
from pipeline import ROOT

OUT = ROOT / 'results/cnn_ablation'
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def fit_stats(cache, rows):
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


def score_people(params, cache, subs, trans, prior, weight, temperature):
    ks, fs, per_stage, rec, prec, tst = [], [], [], [], [], []
    for sub in subs:
        x, y, good, onset = load_night(cache, sub)
        m = good & (y < 5)
        p = night_probabilities(params, x, good, onset, trans, prior, weight, temperature)
        g = p.argmax(axis=1)
        ks.append(cohen_kappa_score(y[m], g[m]))
        fs.append(f1_score(y[m], g[m], labels=list(range(N_CLASS)), average='macro', zero_division=0))
        per_stage.append(f1_score(y[m], g[m], labels=list(range(N_CLASS)), average=None, zero_division=0))
        rec.append(recall_score(y[m], g[m], labels=list(range(N_CLASS)), average=None, zero_division=0))
        prec.append(precision_score(y[m], g[m], labels=list(range(N_CLASS)), average=None, zero_division=0))
        mt = float(np.sum((g > 0) & m) * .5); rt = float(np.sum((y > 0) & m) * .5)
        tst.append(abs(mt - rt))
    ps = np.mean(per_stage, axis=0)
    return {'people': len(subs), 'kappa': float(np.mean(ks)), 'macro_f1': float(np.mean(fs)),
            'tst_error_minutes': float(np.mean(tst)),
            'per_stage_f1': {s: round(float(v), 4) for s, v in zip(STAGES, ps)},
            'per_stage_recall': {s: round(float(v), 4) for s, v in zip(STAGES, np.mean(rec, axis=0))},
            'per_stage_precision': {s: round(float(v), 4) for s, v in zip(STAGES, np.mean(prec, axis=0))}}


def main(cache, variants):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    v2p = json.loads((ROOT / 'v2_protocol.json').read_text())
    sel_pids = set(v2p['model_selection_pids'])
    sel = [r['subject'] for r in rows if r['pid'] in sel_pids and r['confirmatory_night']]
    test = sorted(json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text()),
                  key=lambda s: int(s.split('-')[1]))
    trans, prior = fit_stats(cache, rows)

    report = {'question_1': 'does the sequence smoother buy kappa by destroying the rare stages?',
              'question_2': 'would macro-F1 have chosen a different sequence weight than kappa?',
              'note': 'the test rows are reported, never used to choose anything',
              'variants': {}}
    for name in variants:
        src = ROOT / 'results' / name
        if not (src / 'cnn_weights.json').exists():
            print(json.dumps({'skipped': name}), flush=True); continue
        freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
        params = rebuild_free(src / 'cnn_weights.json')
        cc.TWOCHANNEL = bool(freeze.get('two_channel_input', False))
        temp = freeze['temperature']
        entry = {'frozen_sequence_weight': freeze['selected_sequence_weight'], 'temperature': temp,
                 'selection_metric_used': freeze.get('selection_metric', 'kappa'), 'weights': {}}
        for w in cc.CANDIDATE_WEIGHTS:
            entry['weights'][str(w)] = {'selection_people': score_people(params, cache, sel, trans, prior, w, temp),
                                        'test_people': score_people(params, cache, test, trans, prior, w, temp)}
            print(json.dumps({'variant': name, 'weight': w,
                              'sel_kappa': round(entry['weights'][str(w)]['selection_people']['kappa'], 3),
                              'sel_macro_f1': round(entry['weights'][str(w)]['selection_people']['macro_f1'], 3),
                              'sel_N1_f1': entry['weights'][str(w)]['selection_people']['per_stage_f1']['N1'],
                              'test_N1_f1': entry['weights'][str(w)]['test_people']['per_stage_f1']['N1']}),
                  flush=True)
        by_k = max(cc.CANDIDATE_WEIGHTS, key=lambda w: entry['weights'][str(w)]['selection_people']['kappa'])
        by_f = max(cc.CANDIDATE_WEIGHTS, key=lambda w: entry['weights'][str(w)]['selection_people']['macro_f1'])
        n1_0 = entry['weights']['0.0']['test_people']['per_stage_f1']['N1']
        n1_frozen = entry['weights'][str(freeze['selected_sequence_weight'])]['test_people']['per_stage_f1']['N1']
        entry['selection_metric_ablation'] = {
            'weight_chosen_by_kappa': by_k, 'weight_chosen_by_macro_f1': by_f,
            'metrics_disagree': bool(by_k != by_f)}
        entry['smoothing_cost'] = {
            'test_N1_f1_without_smoothing': n1_0, 'test_N1_f1_at_frozen_weight': n1_frozen,
            'smoothing_reduces_N1_f1': bool(n1_frozen < n1_0),
            'test_kappa_without_smoothing': entry['weights']['0.0']['test_people']['kappa'],
            'test_kappa_at_frozen_weight':
                entry['weights'][str(freeze['selected_sequence_weight'])]['test_people']['kappa']}
        report['variants'][name] = entry
        cc.TWOCHANNEL = False
        print(json.dumps({name: {'selection_metric_ablation': entry['selection_metric_ablation'],
                                 'smoothing_cost': entry['smoothing_cost']}}, indent=1), flush=True)

    report['created_unix'] = time.time(); report['seconds'] = round(time.time() - t0, 1)
    (OUT / 'ablation_metrics.json').write_text(json.dumps(report, indent=2))
    flat = [{'variant': n, 'sequence_weight': w, 'cohort': c,
             'kappa': d[c]['kappa'], 'macro_f1': d[c]['macro_f1'],
             'tst_error_minutes': d[c]['tst_error_minutes'],
             **{f'f1_{s}': d[c]['per_stage_f1'][s] for s in STAGES},
             **{f'recall_{s}': d[c]['per_stage_recall'][s] for s in STAGES}}
            for n, e in report['variants'].items() for w, d in e['weights'].items()
            for c in ['selection_people', 'test_people']]
    pd.DataFrame(flat).to_csv(OUT / 'ablation_table.csv', index=False)
    print(json.dumps({'written': str(OUT), 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    vs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['cnn']
    main(sys.argv[1], vs)

"""Score every remaining model on Sleep-EDF, including the ensemble.

Why this file exists. The ensemble became the best model on the headband, and the context models
beat the frozen v2 there too. None of them had met the external cohort. Claiming "best overall"
from one cohort is exactly the mistake this project exists to expose, so they all get the same
domain-shift test, with `yasa_evaluate.py`'s windows, masks and metrics — the same code that scored
v2, YASA and the plain CNN.

Status: EXPLORATORY DEVELOPMENT. These 20 external people were inspected when v2 was evaluated, so
nothing here is a fresh independent test for any model. Nothing is refitted: every model uses the
sequence weight and temperature it was frozen with.

Usage: python src/external_all.py <boas_cache> <external_cache>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cnn_challenger as cc
import cnn_context as cx
from cnn_challenger import N_CLASS, night_probabilities, rebuild_free
from cnn_external import load_external
from cnn_posthoc import fit_trans_prior
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared core modules sit at the top level
from yasa_evaluate import score, summarize, windows
from pipeline import ROOT

OUT = ROOT / 'results/external_all'
CNN_VARIANTS = ['cnn', 'cnn_balanced', 'cnn_sqrtbalanced', 'cnn_augmented', 'cnn_twochannel']
CTX_VARIANTS = ['cnn_context', 'cnn_context_sqrtbalanced', 'cnn_context_balanced']


def cnn_external_probabilities(name, boas_cache, ext_cache, subs, trans, prior):
    src = ROOT / 'results' / name
    freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
    if freeze.get('two_channel_input'):
        return None          # a two-channel model cannot read a single-channel recording
    params = rebuild_free(src / 'cnn_weights.json')
    cc.TWOCHANNEL = False
    out = {}
    for s in subs:
        x, _, good, onset = load_external(ext_cache, s)
        out[s] = night_probabilities(params, x, good, onset, trans, prior,
                                     freeze['selected_sequence_weight'], freeze['temperature'])
    return out


def ctx_external_probabilities(name, ext_cache, subs, trans, prior):
    src = ROOT / 'results' / name
    freeze = json.loads((src / 'context_freeze_record.json').read_text())
    trunk = rebuild_free(ROOT / freeze['trunk_source'] / 'cnn_weights.json')
    head = rebuild_free(src / 'context_head_weights.json')
    out = {}
    for s in subs:
        e, y, good, onset = cx.night_embeddings(trunk, ext_cache, s, lambda c, u: load_external(c, u))
        out[s] = cx.head_night_probabilities(head, e, good, onset, trans, prior,
                                             freeze['selected_sequence_weight'], freeze['temperature'])
    return out


def main(boas_cache, ext_cache):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frozen = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    subs = sorted(frozen)
    trans, prior = fit_trans_prior(boas_cache)

    probs = {}
    for name in CNN_VARIANTS:
        if not (ROOT / 'results' / name / 'cnn_weights.json').exists():
            continue
        p = cnn_external_probabilities(name, boas_cache, ext_cache, subs, trans, prior)
        if p is None:
            print(json.dumps({'skipped': name, 'why': 'two-channel model, single-channel recording'}), flush=True)
            continue
        probs[name] = p
        print(json.dumps({'scored': name}), flush=True)
    for name in CTX_VARIANTS:
        if not (ROOT / 'results' / name / 'context_head_weights.json').exists():
            continue
        probs[name] = ctx_external_probabilities(name, ext_cache, subs, trans, prior)
        print(json.dumps({'scored': name}), flush=True)

    members = [n for n in probs]
    probs['ensemble_of_' + str(len(members))] = {
        s: (lambda a: a / a.sum(axis=1, keepdims=True))(np.mean([probs[n][s] for n in members], axis=0))
        for s in subs}

    report = {'status': 'EXPLORATORY DEVELOPMENT: these external people were inspected at the v2 evaluation, '
                        'so this is not a fresh independent test for any model',
              'scoring': "yasa_evaluate.py windows, masks and metrics, identical to v2, YASA and the plain CNN",
              'refitted_anything': False, 'ensemble_members': members, 'results': {}}
    rows = []
    for name, per_sub in probs.items():
        windowed = {}
        for s in subs:
            y = np.array(frozen[s]['reference']); good = np.array(frozen[s]['good']).astype(bool)
            onset = np.array(frozen[s]['onset'], dtype=float)
            g = per_sub[s].argmax(axis=1)
            for w_name, w in windows(y, onset).items():
                windowed.setdefault(w_name, []).append({'subject': s, **score(y, good, onset, g, w)})
        report['results'][name] = {w: summarize(r) for w, r in windowed.items()}
        for w, rs in windowed.items():
            rows += [{'model': name, 'window': w, **r} for r in rs]
        pr = report['results'][name]['primary_complete_recording']
        print(json.dumps({'model': name, 'kappa': round(pr['kappa'], 3),
                          'macro_f1': round(pr['macro_f1'], 3),
                          'tst_error_minutes': pr['tst_error_minutes']}), flush=True)
        (OUT / f'{name}_external_predictions.json').write_text(
            json.dumps({s: per_sub[s].argmax(axis=1).tolist() for s in subs}))

    v2 = json.loads((ROOT / 'results/v2/external_metrics.json').read_text())
    ya = json.loads((ROOT / 'results/yasa/yasa_metrics.json').read_text())['results']['external']
    report['frozen_v2_for_comparison'] = v2['results']
    report['yasa_for_comparison'] = ya['primary_complete_recording']
    report['created_unix'] = time.time(); report['seconds'] = round(time.time() - t0, 1)
    (OUT / 'external_all_metrics.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).drop(columns='per_stage_f1').to_csv(OUT / 'external_all_people.csv', index=False)
    print(json.dumps({'written': str(OUT), 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

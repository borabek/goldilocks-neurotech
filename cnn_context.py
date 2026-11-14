"""Context model: give the CNN the same temporal context the frozen feature model already has.

Why this exists. `v2_model.temporal` gives the gradient-boosted model a centred 5- and 15-epoch
window of every feature. The CNN challenger sees one 30-second epoch and gets context only from
the sequence smoother afterwards, so the first comparison was not symmetric in what each model
could see. This closes that gap.

How, and its limitation. The CNN trunk trained by `cnn_challenger.py` is frozen and used as an
epoch encoder: each epoch becomes the 128-number pooled vector the trunk produces before its
classifier. A small temporal network is then trained over a centred window of CONTEXT*2+1 epoch
vectors. This is the same two-stage shape as v2 (per-epoch description, then context), and it
costs minutes instead of hours. It is **not** the same as training a CNN end to end on a
multi-epoch input: the encoder never got to reshape its own features for the context task, so
treat this as a lower bound on what context buys.

Protocol, unchanged from cnn_challenger.py: the trunk's fitting people train the head, the same
six selection people choose the number of passes and the sequence weight, the same six
calibration people fit the temperature, and the 20 test people are read only after freezing.

Usage: python cnn_context.py <boas_cache> [source_results_dir] [passes]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

import cnn_challenger as cc
from cnn_challenger import N_CLASS, SEED, BATCH, LR, arch, instance_norm, load_night, rebuild_free
from pipeline import ROOT, stage_metrics
from v2_model import segments, smooth

CONTEXT = 7                      # epochs each side; 15-epoch centred window
HEAD = [(128, 64, 5), (64, 64, 5)]


def encode(params, x):
    """Everything in cnn_challenger.forward except the final dense: the pooled epoch vector."""
    h = x
    for i, (_, _, _, stride) in enumerate(arch()):
        p = params[i]
        h = jax.lax.conv_general_dilated(h, p['w'], window_strides=(stride,), padding='SAME',
                                         dimension_numbers=('NCH', 'OIH', 'NCH'))
        h = h + p['b'][None, :, None]
        h = jax.nn.relu(instance_norm(h))
        if i in cc.POOL_AFTER:
            f = cc.POOL_AFTER[i]
            h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 1, f), (1, 1, f), 'SAME')
    return jnp.concatenate([h.mean(axis=-1), h.max(axis=-1)], axis=-1)


encode_jit = jax.jit(encode)


def night_embeddings(trunk, cache, sub, loader):
    x, y, good, onset = loader(cache, sub)
    n, c, _ = x.shape
    flat = x.reshape(n * c, 1, x.shape[-1])
    out = []
    for i in range(0, len(flat), 1024):
        out.append(np.asarray(encode_jit(trunk, jnp.asarray(flat[i:i + 1024]))))
    e = np.concatenate(out).reshape(n, c, -1).mean(axis=1)        # channel-shared, as in v2
    return e.astype('float32'), y, good, onset


def windows_of(e):
    """Centred CONTEXT-wide window per epoch, edge-padded, as (n, features, 2*CONTEXT+1)."""
    pad = np.pad(e, ((CONTEXT, CONTEXT), (0, 0)), mode='edge')
    idx = np.arange(len(e))[:, None] + np.arange(2 * CONTEXT + 1)[None, :]
    return np.transpose(pad[idx], (0, 2, 1))


def init_head(key, fin):
    keys = jax.random.split(key, len(HEAD) + 1)
    ps = []
    for i, (cin, cout, k) in enumerate(HEAD):
        cin = fin if i == 0 else cin
        ps.append({'w': jax.random.normal(keys[i], (cout, cin, k)) * np.sqrt(2.0 / (cin * k)),
                   'b': jnp.zeros((cout,))})
    ps.append({'w': jax.random.normal(keys[-1], (HEAD[-1][1] * 2, N_CLASS)) * np.sqrt(1.0 / (HEAD[-1][1] * 2)),
               'b': jnp.zeros((N_CLASS,))})
    return ps


def head_forward(ps, x):
    h = x
    for i in range(len(HEAD)):
        h = jax.lax.conv_general_dilated(h, ps[i]['w'], window_strides=(1,), padding='SAME',
                                         dimension_numbers=('NCH', 'OIH', 'NCH')) + ps[i]['b'][None, :, None]
        h = jax.nn.relu(instance_norm(h))
    pooled = jnp.concatenate([h.mean(axis=-1), h.max(axis=-1)], axis=-1)
    return pooled @ ps[-1]['w'] + ps[-1]['b']


head_proba = jax.jit(lambda ps, x: jax.nn.softmax(head_forward(ps, x), axis=-1))


def head_night_probabilities(ps, e, good, onset, trans, prior, weight, temperature=1.0):
    w = windows_of(e)
    out = []
    for i in range(0, len(w), 1024):
        out.append(np.asarray(head_proba(ps, jnp.asarray(w[i:i + 1024]))))
    p = np.concatenate(out)
    p = smooth(p, good, trans, prior, weight, onset)
    p = np.exp(np.log(p + 1e-12) / temperature)
    return p / p.sum(axis=1, keepdims=True)


def main(cache, source='results/cnn', passes=20):
    t0 = time.time()
    src = ROOT / source
    tag = Path(source).name
    OUT = ROOT / ('results/cnn_context' if tag == 'cnn' else f'results/cnn_context_{tag.replace("cnn_", "")}')
    OUT.mkdir(parents=True, exist_ok=True)
    trunk = rebuild_free(src / 'cnn_weights.json')
    src_freeze = json.loads((src / 'cnn_freeze_record.json').read_text())

    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    v2p = json.loads((ROOT / 'v2_protocol.json').read_text())
    sel_pids, cal_pids = set(v2p['model_selection_pids']), set(v2p['probability_calibration_pids'])
    fit = [r['subject'] for r in rows if r['split'] == 'fit']
    sel = [r['subject'] for r in rows if r['pid'] in sel_pids and r['confirmatory_night']]
    cal = [r['subject'] for r in rows if r['pid'] in cal_pids and r['confirmatory_night']]
    frozen_test = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    test = sorted(frozen_test, key=lambda s: int(s.split('-')[1]))

    emb = {}
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for sub in fit + sel + cal + test:
        emb[sub] = night_embeddings(trunk, cache, sub, load_night)
    for sub in fit:
        _, y, good, _ = emb[sub]
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
    prior /= prior.sum(); trans /= trans.sum(axis=1, keepdims=True)
    print(json.dumps({'embedded_nights': len(emb), 'seconds': round(time.time() - t0, 1)}), flush=True)

    X, Y = [], []
    for sub in fit:
        e, y, good, _ = emb[sub]
        m = good & (y < 5)
        X.append(windows_of(e)[m]); Y.append(y[m])
    X = np.concatenate(X); Y = np.concatenate(Y).astype('int32')
    print(json.dumps({'training_windows': int(len(X)), 'window_shape': list(X.shape[1:])}), flush=True)

    key = jax.random.PRNGKey(SEED)
    ps = init_head(key, X.shape[1])
    tx = optax.adamw(LR, weight_decay=1e-4)
    opt = tx.init(ps)

    @jax.jit
    def step(ps, opt, x, y):
        def loss(p):
            return optax.softmax_cross_entropy_with_integer_labels(head_forward(p, x), y).mean()
        l, g = jax.value_and_grad(loss)(ps)
        u, opt = tx.update(g, opt, ps)
        return optax.apply_updates(ps, u), opt, l

    def sel_score(ps, weight=0.0):
        from sklearn.metrics import cohen_kappa_score
        out = []
        for sub in sel:
            e, y, good, onset = emb[sub]
            m = good & (y < 5)
            p = head_night_probabilities(ps, e, good, onset, trans, prior, weight)
            out.append(float(cohen_kappa_score(y[m], p[m].argmax(axis=1))))
        return float(np.mean(out))

    rng = np.random.default_rng(SEED)
    history, best = [], None
    for ep in range(passes):
        t = time.time(); order = rng.permutation(len(X)); losses = []
        for i in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            ps, opt, l = step(ps, opt, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
            losses.append(float(l))
        k = sel_score(ps)
        history.append({'pass': ep + 1, 'train_loss': float(np.mean(losses)),
                        'selection_mean_person_kappa_unsmoothed': k, 'seconds': round(time.time() - t, 1)})
        print(json.dumps(history[-1]), flush=True)
        if best is None or k > best['k']:
            best = {'k': k, 'pass': ep + 1, 'ps': jax.tree.map(lambda a: a.copy(), ps)}
    ps = best['ps']

    candidates = [{'sequence_weight': w, 'mean_person_kappa': sel_score(ps, w)} for w in cc.CANDIDATE_WEIGHTS]
    weight = max(candidates, key=lambda r: r['mean_person_kappa'])['sequence_weight']
    print(json.dumps({'selected_pass': best['pass'], 'candidates': candidates, 'weight': weight}), flush=True)

    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss
    probs, targets = [], []
    for sub in cal:
        e, y, good, onset = emb[sub]
        m = good & (y < 5)
        probs.append(head_night_probabilities(ps, e, good, onset, trans, prior, weight)[m]); targets.append(y[m])
    def obj(t):
        return np.mean([log_loss(y, np.exp(np.log(p + 1e-12) / t) / np.exp(np.log(p + 1e-12) / t).sum(axis=1, keepdims=True),
                                 labels=list(range(N_CLASS))) for p, y in zip(probs, targets)]) + .02 * np.log(t) ** 2
    temperature = float(minimize_scalar(obj, bounds=(.5, 3), method='bounded').x)

    flat = {f'{i}/{k}': np.asarray(v).tolist() for i, p in enumerate(ps) for k, v in p.items()}
    (OUT / 'context_head_weights.json').write_text(json.dumps(flat))
    freeze = {'frozen_unix': time.time(), 'context_epochs_each_side': CONTEXT, 'head': HEAD,
              'trunk_source': source, 'trunk_weights_sha256': src_freeze['weights_sha256'],
              'trunk_frozen': True, 'selected_pass': best['pass'], 'passes_run': passes,
              'selection_candidates': candidates, 'selected_sequence_weight': weight,
              'temperature': temperature, 'history': history, 'seed': SEED,
              'head_weights_sha256': hashlib.sha256((OUT / 'context_head_weights.json').read_bytes()).hexdigest()}
    (OUT / 'context_freeze_record.json').write_text(json.dumps(freeze, indent=2))

    per_person, preds = [], {}
    for sub in test:
        e, y, good, onset = emb[sub]
        fz = frozen_test[sub]
        assert np.array_equal(np.array(fz['reference']), y) and np.array_equal(np.array(fz['good']).astype(bool), good)
        m = good & (y < 5)
        p = head_night_probabilities(ps, e, good, onset, trans, prior, weight, temperature)
        g = p.argmax(axis=1)
        stat = stage_metrics(y[m], g[m])
        mt = float(np.sum((g > 0) & m) * .5); rt = float(np.sum((y > 0) & m) * .5)
        per_person.append({'subject': sub, **stat, 'reference_tst_minutes': rt, 'model_tst_minutes': mt,
                           'tst_error_minutes': abs(mt - rt), 'available_epochs': int(m.sum())})
        preds[sub] = g.tolist()
        print(json.dumps({'evaluated': sub, 'kappa': round(stat['kappa'], 3),
                          'macro_f1': round(stat['macro_f1'], 3)}), flush=True)

    def mean(f): return float(np.mean([r[f] for r in per_person]))
    res = {'people': len(per_person), 'accuracy': mean('accuracy'), 'kappa': mean('kappa'),
           'macro_f1': mean('macro_f1'), 'tst_error_minutes': mean('tst_error_minutes')}
    saved = json.loads((ROOT / 'results/v2/historical_boas_metrics.json').read_text())['results']
    base = json.loads((src / 'cnn_metrics.json').read_text())['cnn_results']
    report = {'method': f'frozen CNN trunk from {source} as an epoch encoder, plus a temporal head over a '
                        f'{2 * CONTEXT + 1}-epoch centred window; v2 sequence smoother and temperature',
              'cohort': 'historical_boas, 20 held-out people, confirmatory nights',
              'limitation': 'the encoder was trained without context and is frozen here, so this is a lower '
                            'bound on what end-to-end multi-epoch training would give',
              'frozen': freeze, 'context_results': res, 'trunk_results_without_context': base,
              'v2_results_for_the_same_people': saved,
              'verdict': {'context_beats_trunk_on_kappa': bool(res['kappa'] > base['kappa']),
                          'context_beats_trunk_on_macro_f1': bool(res['macro_f1'] > base['macro_f1']),
                          'context_beats_v2_on_kappa': bool(res['kappa'] > saved['kappa']),
                          'context_beats_v2_on_macro_f1': bool(res['macro_f1'] > saved['macro_f1'])},
              'created_unix': time.time(), 'seconds': round(time.time() - t0, 1)}
    (OUT / 'context_metrics.json').write_text(json.dumps(report, indent=2))
    (OUT / 'context_predictions.json').write_text(json.dumps(preds))
    pd.DataFrame(per_person).drop(columns='per_stage_f1').to_csv(OUT / 'context_people.csv', index=False)
    print(json.dumps({'context': res, 'trunk': base, 'v2': {k: saved[k] for k in res},
                      'verdict': report['verdict']}, indent=2), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'results/cnn',
         int(sys.argv[3]) if len(sys.argv) > 3 else 20)

"""Backlog item 8: train the encoder and the temporal model together, end to end.

Compute budget, stated up front: a pass here costs about 206 seconds because the stride-7 sequence
sampling shows every epoch three times, so this run uses 12 passes rather than the 25 the other
variants used. In single-epoch exposure that is roughly 36 passes, more than any other variant got.
The budget is a constraint of the night, not a tuned choice, and it is recorded so the comparison is
not read as like-for-like on wall-clock training.

The context model in this build freezes a trunk that was trained one epoch at a time and bolts a
temporal head on top. The plan records that as a lower bound, because the encoder never got to
reshape its features for the context task. This removes that limitation.

How it stays affordable. Feeding each epoch its own 15-epoch window would encode every epoch
fifteen times. Instead a training sample is a contiguous run of SEQ epochs: the shared encoder runs
once per epoch, temporal convolutions run over the resulting sequence, and the loss is taken on
every valid epoch in the run. One pass therefore costs about what the plain CNN's pass costs, and
every epoch still sees context on both sides except at the two ends of a run.

Everything else is the frozen v2 protocol, unchanged: the same 48 fitting people over 59 nights,
the same six selection people choosing the number of passes and the sequence weight from
[0, 0.5, 1.0], the same six calibration people fitting the temperature, the same evaluation mask,
`stage_metrics` and sleep-time definition, and the 20 test people read only after freezing.

Usage: python cnn_endtoend.py <boas_cache> [passes]
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
from cnn_challenger import N_CLASS, N_IN, SEED, LR, arch, instance_norm, load_night
from pipeline import ROOT, stage_metrics
from v2_model import smooth

OUT = ROOT / 'results/cnn_endtoend'
SEQ = 21          # epochs per training sample
STRIDE = 7        # grid step; STRIDE == SEQ means no overlap. Ignored when SAMPLING == 'random'
SAMPLING = 'random'   # 'random': a fresh random window position every time an epoch is used, which
                      # removes the positional bias a fixed grid leaves. 'grid': the fixed-stride
                      # version, kept so the two can be compared.
EXPOSURE = 3      # how many times, on average, each epoch is used per pass under random sampling
BATCH = 6         # sequences per step: 126 epoch encodings, close to the plain CNN's 128
HEAD = [(128, 64, 5), (64, 64, 5)]
IGNORE = -1


def init(key):
    layers = arch()
    ks = jax.random.split(key, len(layers) + len(HEAD) + 1)
    trunk = []
    for i, (cin, cout, k, _) in enumerate(layers):
        trunk.append({'w': jax.random.normal(ks[i], (cout, cin, k)) * np.sqrt(2.0 / (cin * k)),
                      'b': jnp.zeros((cout,))})
    head = []
    for j, (cin, cout, k) in enumerate(HEAD):
        head.append({'w': jax.random.normal(ks[len(layers) + j], (cout, cin, k)) * np.sqrt(2.0 / (cin * k)),
                     'b': jnp.zeros((cout,))})
    out = {'w': jax.random.normal(ks[-1], (HEAD[-1][1], N_CLASS)) * np.sqrt(1.0 / HEAD[-1][1]),
           'b': jnp.zeros((N_CLASS,))}
    return {'trunk': trunk, 'head': head, 'out': out}


def encode(trunk, x):
    """x: (n, 1, N_IN) -> (n, features). The same body as the plain CNN, minus its classifier."""
    h = x
    for i, (_, _, _, stride) in enumerate(arch()):
        p = trunk[i]
        h = jax.lax.conv_general_dilated(h, p['w'], window_strides=(stride,), padding='SAME',
                                         dimension_numbers=('NCH', 'OIH', 'NCH')) + p['b'][None, :, None]
        h = jax.nn.relu(instance_norm(h))
        if i in cc.POOL_AFTER:
            f = cc.POOL_AFTER[i]
            h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 1, f), (1, 1, f), 'SAME')
    return jnp.concatenate([h.mean(axis=-1), h.max(axis=-1)], axis=-1)


def forward(params, seq):
    """seq: (B, L, 1, N_IN) -> logits (B, L, 5). The encoder is shared across positions."""
    b, l = seq.shape[0], seq.shape[1]
    e = encode(params['trunk'], seq.reshape(b * l, 1, N_IN)).reshape(b, l, -1)
    h = jnp.transpose(e, (0, 2, 1))                      # (B, features, L)
    for p in params['head']:
        h = jax.lax.conv_general_dilated(h, p['w'], window_strides=(1,), padding='SAME',
                                         dimension_numbers=('NCH', 'OIH', 'NCH')) + p['b'][None, :, None]
        h = jax.nn.relu(instance_norm(h))
    h = jnp.transpose(h, (0, 2, 1))                      # (B, L, features)
    return h @ params['out']['w'] + params['out']['b']


def loss_fn(params, seq, y):
    logits = forward(params, seq)
    valid = (y != IGNORE)
    per = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.where(valid, y, 0))
    return (per * valid).sum() / jnp.maximum(valid.sum(), 1)


@jax.jit
def night_logits(params, seq):
    return forward(params, seq)


def night_probabilities(params, x, good, onset, trans, prior, weight, temperature=1.0):
    """Score a whole night: each channel as its own sequence, probabilities averaged, then the v2
    sequence smoother and temperature, exactly as every other variant here."""
    n, c, _ = x.shape
    probs = []
    for ch in range(c):
        seq = jnp.asarray(x[:, ch][None, :, None, :])     # (1, n, 1, N_IN)
        out = []
        for i in range(0, n, 600):                        # chunk long nights to bound memory
            out.append(np.asarray(jax.nn.softmax(night_logits(params, seq[:, i:i + 600]), axis=-1))[0])
        probs.append(np.concatenate(out))
    p = np.mean(probs, axis=0)
    p = smooth(p, good, trans, prior, weight, onset)
    p = np.exp(np.log(p + 1e-12) / temperature)
    return p / p.sum(axis=1, keepdims=True)


def person_score(params, cache, subs, trans, prior, weight, temperature=1.0):
    from sklearn.metrics import cohen_kappa_score
    out = []
    for sub in subs:
        x, y, good, onset = load_night(cache, sub)
        m = good & (y < 5)
        p = night_probabilities(params, x, good, onset, trans, prior, weight, temperature)
        out.append(float(cohen_kappa_score(y[m], p[m].argmax(axis=1))))
    return float(np.mean(out))


def main(cache, passes=25):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    v2p = json.loads((ROOT / 'v2_protocol.json').read_text())
    sel_pids, cal_pids = set(v2p['model_selection_pids']), set(v2p['probability_calibration_pids'])
    fit = [r['subject'] for r in rows if r['split'] == 'fit']
    sel = [r['subject'] for r in rows if r['pid'] in sel_pids and r['confirmatory_night']]
    cal = [r['subject'] for r in rows if r['pid'] in cal_pids and r['confirmatory_night']]
    frozen_test = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    test = sorted(frozen_test, key=lambda s: int(s.split('-')[1]))
    assert len(fit) == 59 and len(sel) == 6 and len(cal) == 6 and len(test) == 20

    nights, X, Y = [], [], []
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for sub in fit:
        x, y, good, _ = load_night(cache, sub)
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
        lab = np.where(m, y, IGNORE).astype('int32')
        if SAMPLING == 'random':
            if len(y) >= SEQ:
                nights.append((x.astype('float16'), lab))
            continue
        for ch in range(x.shape[1]):
            for s in range(0, max(len(y) - SEQ + 1, 1), STRIDE):
                seg = slice(s, s + SEQ)
                if len(y[seg]) < SEQ or (lab[seg] != IGNORE).sum() == 0:
                    continue
                X.append(x[seg, ch][:, None, :].astype('float16')); Y.append(lab[seg])
    if SAMPLING == 'grid':
        X = np.stack(X); Y = np.stack(Y)    # float16; cast per batch below
        steps_per_pass = (len(X) - BATCH + 1) // BATCH
        positions = None
    else:
        positions = [(i, ch, len(nights[i][1]) - SEQ)
                     for i in range(len(nights)) for ch in range(nights[i][0].shape[1])]
        epoch_channels = sum(len(l) * x.shape[1] for x, l in nights)
        steps_per_pass = max(1, int(EXPOSURE * epoch_channels / (BATCH * SEQ)))
    prior /= prior.sum(); trans /= trans.sum(axis=1, keepdims=True)
    print(json.dumps({'sampling': SAMPLING, 'epochs_per_sequence': SEQ,
                      'stride': STRIDE if SAMPLING == 'grid' else None,
                      'target_exposure_per_epoch_per_pass': EXPOSURE if SAMPLING == 'random' else
                      round(SEQ / STRIDE, 2),
                      'steps_per_pass': steps_per_pass,
                      'fit_nights_held_in_memory': len(nights) if SAMPLING == 'random' else None,
                      'megabytes': round((sum(x.nbytes for x, _ in nights) if SAMPLING == 'random'
                                          else X.nbytes) / 1e6, 1),
                      'load_seconds': round(time.time() - t0, 1)}), flush=True)

    key = jax.random.PRNGKey(SEED)
    params = init(key)
    tx = optax.adamw(LR, weight_decay=1e-4)
    opt = tx.init(params)

    @jax.jit
    def step(params, opt, seq, y):
        l, g = jax.value_and_grad(loss_fn)(params, seq, y)
        u, opt = tx.update(g, opt, params)
        return optax.apply_updates(params, u), opt, l

    rng = np.random.default_rng(SEED)
    history, best = [], None

    def batch(rng):
        """A batch of sequences. Under random sampling every window position is drawn afresh, so no
        epoch is permanently stuck at a window edge the way a fixed grid leaves it."""
        if SAMPLING == 'grid':
            raise RuntimeError
        xs, ys = [], []
        for _ in range(BATCH):
            i, ch, hi = positions[rng.integers(len(positions))]
            s = int(rng.integers(0, hi + 1))
            x, lab = nights[i]
            xs.append(x[s:s + SEQ, ch][:, None, :].astype('float32')); ys.append(lab[s:s + SEQ])
        return np.stack(xs), np.stack(ys)

    for ep in range(passes):
        t = time.time(); losses = []
        if SAMPLING == 'grid':
            order = rng.permutation(len(X))
            for i in range(0, len(order) - BATCH + 1, BATCH):
                idx = order[i:i + BATCH]
                params, opt, l = step(params, opt, jnp.asarray(X[idx].astype('float32')),
                                      jnp.asarray(Y[idx]))
                losses.append(float(l))
        else:
            for _ in range(steps_per_pass):
                bx, by = batch(rng)
                params, opt, l = step(params, opt, jnp.asarray(bx), jnp.asarray(by))
                losses.append(float(l))
        k = person_score(params, cache, sel, trans, prior, 0.0)
        history.append({'pass': ep + 1, 'train_loss': float(np.mean(losses)),
                        'selection_mean_person_kappa_unsmoothed': k, 'seconds': round(time.time() - t, 1)})
        print(json.dumps(history[-1]), flush=True)
        if best is None or k > best['k']:
            best = {'k': k, 'pass': ep + 1, 'p': jax.tree.map(lambda a: a.copy(), params)}
    params = best['p']

    candidates = [{'sequence_weight': w, 'mean_person_kappa': person_score(params, cache, sel, trans, prior, w)}
                  for w in cc.CANDIDATE_WEIGHTS]
    weight = max(candidates, key=lambda r: r['mean_person_kappa'])['sequence_weight']
    print(json.dumps({'selected_pass': best['pass'], 'candidates': candidates, 'weight': weight}), flush=True)

    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss
    probs, targets = [], []
    for sub in cal:
        x, y, good, onset = load_night(cache, sub)
        m = good & (y < 5)
        probs.append(night_probabilities(params, x, good, onset, trans, prior, weight)[m]); targets.append(y[m])
    def obj(t):
        return np.mean([log_loss(yy, np.exp(np.log(pp + 1e-12) / t) /
                                 np.exp(np.log(pp + 1e-12) / t).sum(axis=1, keepdims=True),
                                 labels=list(range(N_CLASS))) for pp, yy in zip(probs, targets)]) \
            + .02 * np.log(t) ** 2
    temperature = float(minimize_scalar(obj, bounds=(.5, 3), method='bounded').x)

    flat = {f"trunk/{i}/{k}": np.asarray(v).tolist() for i, p in enumerate(params['trunk']) for k, v in p.items()}
    flat.update({f"head/{i}/{k}": np.asarray(v).tolist() for i, p in enumerate(params['head']) for k, v in p.items()})
    flat.update({f"out/{k}": np.asarray(v).tolist() for k, v in params['out'].items()})
    (OUT / 'endtoend_weights.json').write_text(json.dumps(flat))
    freeze = {'frozen_unix': time.time(), 'seq': SEQ, 'sampling': SAMPLING,
              'stride': STRIDE if SAMPLING == 'grid' else None,
              'target_exposure_per_epoch_per_pass': EXPOSURE if SAMPLING == 'random' else None,
              'steps_per_pass': steps_per_pass, 'batch_sequences': BATCH,
              'head': HEAD, 'seed': SEED, 'learning_rate': LR, 'passes_run': passes,
              'selected_pass': best['pass'], 'selection_candidates': candidates,
              'selected_sequence_weight': weight, 'temperature': temperature, 'history': history,
              'weights_sha256': hashlib.sha256((OUT / 'endtoend_weights.json').read_bytes()).hexdigest(),
              'test_labels_read_before_freeze': False}
    (OUT / 'endtoend_freeze_record.json').write_text(json.dumps(freeze, indent=2))

    per_person, preds = [], {}
    for sub in test:
        x, y, good, onset = load_night(cache, sub)
        fz = frozen_test[sub]
        assert np.array_equal(np.array(fz['reference']), y) and np.array_equal(
            np.array(fz['good']).astype(bool), good)
        m = good & (y < 5)
        p = night_probabilities(params, x, good, onset, trans, prior, weight, temperature)
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
    ctx = json.loads((ROOT / 'results/cnn_context/context_metrics.json').read_text())['context_results']
    v2 = json.loads((ROOT / 'results/v2/historical_boas_metrics.json').read_text())['results']
    sv = ROOT / 'results/seed_variance/seed_variance.json'
    spread = json.loads(sv.read_text())['statistics']['macro_f1']['spread'] if sv.exists() else None
    gain = res['macro_f1'] - ctx['macro_f1']
    report = {'method': f'end-to-end sequence model: shared encoder over {SEQ}-epoch runs with temporal '
                        f'convolutions, trained jointly; v2 sequence smoother and temperature on top',
              'cohort': 'historical_boas, 20 held-out people, confirmatory nights',
              'scope': 'Development comparison on people already inspected at v1; not fresh confirmation.',
              'removes_limitation': 'the frozen-trunk context model could not reshape its features for the '
                                    'context task; this one can',
              'frozen': freeze, 'endtoend_results': res,
              'frozen_trunk_context_model': ctx, 'frozen_v2': v2,
              'gain_over_frozen_trunk_context_macro_f1': gain, 'head_seed_spread_macro_f1': spread,
              'verdict': ('the gain is smaller than the head-seed spread, so it is not a finding'
                          if spread is not None and abs(gain) < spread
                          else 'the gain exceeds the head-seed spread'),
              'created_unix': time.time(), 'seconds': round(time.time() - t0, 1)}
    # Each arm also writes under its own name. The first run of this script wrote only the
    # canonical names, so the control arm overwrote the first arm's metrics in place and the
    # comparison had to be recovered from the run log. Per-arm names stop that recurring.
    tag = SAMPLING if SAMPLING == 'random' else f'grid{STRIDE}'
    (OUT / 'endtoend_metrics.json').write_text(json.dumps(report, indent=2))
    (OUT / f'endtoend_metrics_{tag}.json').write_text(json.dumps(report, indent=2))
    (OUT / 'endtoend_predictions.json').write_text(json.dumps(preds))
    (OUT / f'endtoend_predictions_{tag}.json').write_text(json.dumps(preds))
    pd.DataFrame(per_person).drop(columns='per_stage_f1').to_csv(OUT / 'endtoend_people.csv', index=False)
    pd.DataFrame(per_person).drop(columns='per_stage_f1').to_csv(OUT / f'endtoend_people_{tag}.csv', index=False)
    print(json.dumps({'endtoend': res, 'frozen_trunk_context': ctx, 'v2': {k: v2[k] for k in res},
                      'gain': gain, 'verdict': report['verdict']}, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        if sys.argv[3] == 'random':
            SAMPLING = 'random'
        else:
            SAMPLING, STRIDE = 'grid', int(sys.argv[3])
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25)

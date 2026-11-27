"""CNN challenger for the BOAS forehead headband, on the frozen v2 split.

Why: the plan lists a CNN challenger as an optional extension, and says to report the actual
result including a loss. This trains one from the raw headband signal and scores it with the
*same* people, masks, windows and metric function as the frozen feature model, so the comparison
is a comparison and not two different experiments.

What is held identical to v2 (`v2_protocol.json`, `v2_model.py`):
  - the person split: 48 fitting people (59 nights), 6 selection people, 6 probability-calibration
    people, 20 held-out test people, exactly the pids recorded in `protocol.json`;
  - channel sharing: one model over single-channel epochs, channel probabilities averaged;
  - the retrospective sequence smoother, its candidate weights [0, 0.5, 1.0], and selection by
    mean person kappa on the same six selection people;
  - temperature fitted on the same six probability-calibration people with the same objective;
  - evaluation mask `good & (reference < 5)`, `stage_metrics`, and total sleep time as
    0.5 minutes per non-wake epoch inside the mask.

What differs, stated rather than hidden:
  - input is the signal itself: 30-second epochs decimated 256 -> 64 Hz (1920 samples), divided by
    a per-recording robust scale from the signal only, clipped at 20;
  - the model is a small 1-D convolutional network (~44k parameters), not gradient-boosted
    spectral features;
  - the CNN spends the six selection people on two choices (how many passes, and the sequence
    weight); v2 spent them on one (the sequence weight). That is a small extra advantage to the
    CNN, not to v2;
  - no class weighting, matching v2's plain fit.

Nothing here refits, reselects or overwrites any v2 artefact. Test labels are read only in the
final evaluation, after the CNN weights and every choice are frozen and hashed.

Usage: python src/cnn_challenger.py <epoch_cache_dir> [passes]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared core modules sit at the top level
from pipeline import ROOT, stage_metrics
from v2_model import segments, smooth

SEED = 20261108                      # same seed the v2 development experiment used
FS, N_IN, N_CLASS = 64, 1920, 5
CANDIDATE_WEIGHTS = [0.0, 0.5, 1.0]
BATCH = 128
LR = 3e-4
CLIP = 20.0
OUT = ROOT / 'results/cnn'
BALANCED = False        # --balanced:   class-weighted loss, selection by macro-F1
AUGMENT = False         # --augment:    amplitude scaling and additive noise during training
TWOCHANNEL = False      # --twochannel: both headband channels as one input, no channel averaging
SQRTBALANCED = False    # --sqrtbalanced: inverse square-root class weights, a gentler balance
PERSONWEIGHT = False    # --personweight: every fitting person contributes equally, not every epoch
LOWPASS16 = False       # --lowpass16: every epoch low-passed at 16 Hz, in training and evaluation (prereg-v2)
HFDROP = False          # --hfdrop:    16-30 Hz zeroed in half the training epochs, evaluation untouched (prereg-v2)
ARCH = [(1, 16, 15, 4), (16, 32, 9, 1), (32, 64, 9, 1), (64, 64, 5, 1)]   # in, out, kernel, stride
POOL_AFTER = {0: 2, 1: 2, 2: 2}                                            # max-pool factor by layer


# ---------------------------------------------------------------- data

def robust_scale(x, good):
    """Per-recording, per-channel robust scale from the signal only. No labels, no other person."""
    g = x[good] if good.any() else x
    med = np.median(g, axis=(0, 2), keepdims=True)                 # (1, channels, 1)
    mad = np.median(np.abs(g - med), axis=(0, 2), keepdims=True) * 1.4826
    return med, np.maximum(mad, 1e-3)


def lowpass16(x):
    """Zero every FFT bin above 16 Hz (prereg-v2, model A)."""
    spec = np.fft.rfft(x, axis=-1)
    spec[..., np.fft.rfftfreq(x.shape[-1], 1 / FS) > 16] = 0
    return np.fft.irfft(spec, n=x.shape[-1], axis=-1).astype('float32')


def load_night(cache, sub):
    z = np.load(Path(cache) / f'{sub}.npz')
    x = z['X'].astype('float32')                     # (n, 2, 1920) microvolts
    if LOWPASS16:
        x = lowpass16(x)
    good, y, onset = z['good'].astype(bool), z['y'].astype('int8'), z['onset'].astype(float)
    med, mad = robust_scale(x, good)
    x = np.clip((x - med) / mad, -CLIP, CLIP)
    return x, y, good, onset


# ---------------------------------------------------------------- model

def arch():
    a = [list(x) for x in ARCH]
    if TWOCHANNEL:
        a[0][0] = 2
    return [tuple(x) for x in a]


def init_params(key):
    layers = arch()
    params, keys = [], jax.random.split(key, len(layers) + 1)
    for i, (cin, cout, k, _) in enumerate(layers):
        w = jax.random.normal(keys[i], (cout, cin, k)) * np.sqrt(2.0 / (cin * k))
        params.append({'w': w, 'b': jnp.zeros((cout,))})
    feat = layers[-1][1] * 2
    params.append({'w': jax.random.normal(keys[-1], (feat, N_CLASS)) * np.sqrt(1.0 / feat),
                   'b': jnp.zeros((N_CLASS,))})
    return params


def rebuild_free(path):
    """Load saved weights back into the parameter list shape."""
    flat = json.loads(Path(path).read_text())
    n = max(int(k.split('/')[0]) for k in flat) + 1
    return [{k.split('/')[1]: jnp.asarray(np.array(v, dtype='float32'))
             for k, v in flat.items() if int(k.split('/')[0]) == i} for i in range(n)]


def instance_norm(h):
    m = h.mean(axis=-1, keepdims=True)
    v = h.var(axis=-1, keepdims=True)
    return (h - m) / jnp.sqrt(v + 1e-5)


def forward(params, x):
    """x: (n, 1, 1920) single-channel epochs -> logits (n, 5)."""
    h = x
    for i, (_, _, _, stride) in enumerate(arch()):
        p = params[i]
        h = jax.lax.conv_general_dilated(h, p['w'], window_strides=(stride,), padding='SAME',
                                         dimension_numbers=('NCH', 'OIH', 'NCH'))
        h = h + p['b'][None, :, None]
        h = jax.nn.relu(instance_norm(h))
        if i in POOL_AFTER:
            f = POOL_AFTER[i]
            h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 1, f), (1, 1, f), 'SAME')
    pooled = jnp.concatenate([h.mean(axis=-1), h.max(axis=-1)], axis=-1)
    return pooled @ params[-1]['w'] + params[-1]['b']


def augment(key, x):
    """Amplitude scaling and additive noise only: no time reversal or circular shift, which would
    destroy the waveform asymmetries sleep stages are scored on."""
    k1, k2 = jax.random.split(key)
    scale = jax.random.uniform(k1, (x.shape[0], 1, 1), minval=0.8, maxval=1.25)
    noise = jax.random.normal(k2, x.shape) * 0.05
    return jnp.clip(x * scale + noise, -CLIP, CLIP)


def hf_drop(key, x):
    """Zero the 16-30 Hz FFT bins of each epoch with probability 0.5 (prereg-v2, model B)."""
    freq = jnp.fft.rfftfreq(x.shape[-1], 1 / FS)
    band = (freq >= 16) & (freq <= 30)
    drop = jax.random.bernoulli(key, 0.5, (x.shape[0], 1, 1))
    spec = jnp.fft.rfft(x, axis=-1)
    spec = jnp.where(drop & band, 0, spec)
    return jnp.fft.irfft(spec, n=x.shape[-1], axis=-1)


def loss_fn(params, x, y, sw=None):
    """sw is a per-sample weight, already carrying any class and person weighting."""
    logits = forward(params, x)
    per = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    if sw is None:
        return per.mean()
    return (per * sw).sum() / sw.sum()


@jax.jit
def predict_proba_batch(params, x):
    return jax.nn.softmax(forward(params, x), axis=-1)


def night_probabilities(params, x, good, onset, trans, prior, weight, temperature=1.0):
    """Channel-shared: score each channel, average probabilities, then the v2 sequence smoother.
    With TWOCHANNEL the epoch goes in whole and there is nothing to average."""
    n, c, _ = x.shape
    flat = x if TWOCHANNEL else x.reshape(n * c, 1, x.shape[-1])
    out = []
    for i in range(0, len(flat), 1024):
        out.append(np.asarray(predict_proba_batch(params, jnp.asarray(flat[i:i + 1024]))))
    p = np.concatenate(out)
    p = p.reshape(n, N_CLASS) if TWOCHANNEL else p.reshape(n, c, N_CLASS).mean(axis=1)
    p = smooth(p, good, trans, prior, weight, onset)
    p = np.exp(np.log(p + 1e-12) / temperature)
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- protocol

def person_kappa(params, cache, subs, trans, prior, weight, temperature=1.0, metric='kappa'):
    """Mean over people of the selection metric. 'kappa' for the plain variant, 'macro_f1' for the
    class-balanced one, which targets the rare stages and must be selected on what it targets."""
    from sklearn.metrics import cohen_kappa_score, f1_score
    out = []
    for sub in subs:
        x, y, good, onset = load_night(cache, sub)
        m = good & (y < 5)
        p = night_probabilities(params, x, good, onset, trans, prior, weight, temperature)
        g = p[m].argmax(axis=1)
        out.append(float(cohen_kappa_score(y[m], g)) if metric == 'kappa'
                   else float(f1_score(y[m], g, labels=list(range(N_CLASS)), average='macro', zero_division=0)))
    return float(np.mean(out)), out


def main(cache, passes):
    metric = 'macro_f1' if (BALANCED or SQRTBALANCED) else 'kappa'
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    v1 = json.loads((ROOT / 'v2_protocol.json').read_text())
    selection_pids, probcal_pids = set(v1['model_selection_pids']), set(v1['probability_calibration_pids'])
    fit = [r['subject'] for r in rows if r['split'] == 'fit']
    sel = [r['subject'] for r in rows if r['pid'] in selection_pids and r['confirmatory_night']]
    cal = [r['subject'] for r in rows if r['pid'] in probcal_pids and r['confirmatory_night']]
    frozen_test = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    test = sorted(frozen_test, key=lambda s: int(s.split('-')[1]))
    assert len(fit) == 59 and len(sel) == 6 and len(cal) == 6 and len(test) == 20, (len(fit), len(sel), len(cal), len(test))
    assert not ({*sel, *cal, *test} & set(fit)), 'a fitting night leaked into selection, calibration or test'

    # Training tensors from fitting nights only, with v2's transition matrix and prior.
    X, Y, P = [], [], []
    pid_of = {r['subject']: r['pid'] for r in rows}
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for sub in fit:
        x, y, good, _ = load_night(cache, sub)
        m = good & (y < 5)
        if TWOCHANNEL:
            X.append(x[m]); Y.append(y[m]); P.append(np.full(int(m.sum()), pid_of[sub]))
        else:
            X.append(x[m].reshape(-1, 1, N_IN)); Y.append(np.repeat(y[m], x.shape[1]))
            P.append(np.full(int(m.sum()) * x.shape[1], pid_of[sub]))
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
    X = np.concatenate(X); Y = np.concatenate(Y).astype('int32'); P = np.concatenate(P)
    prior /= prior.sum(); trans /= trans.sum(axis=1, keepdims=True)
    counts = np.bincount(Y, minlength=N_CLASS)
    class_w = None
    if BALANCED:                                   # sklearn 'balanced': n / (k * n_c)
        class_w = len(Y) / (N_CLASS * np.maximum(counts, 1))
    elif SQRTBALANCED:                             # gentler: the square root of the same ratio
        class_w = np.sqrt(len(Y) / (N_CLASS * np.maximum(counts, 1)))
        class_w = class_w / class_w.mean()
    sw = np.ones(len(Y))
    if class_w is not None:
        sw = sw * class_w[Y]
    person_w = None
    if PERSONWEIGHT:                               # one person, one vote, whatever the night length
        _, inv, counts = np.unique(P, return_inverse=True, return_counts=True)
        per_person = counts[inv].astype('float64')
        person_w = (1.0 / per_person); person_w = person_w / person_w.mean()
        sw = sw * person_w
    sw_jax = jnp.asarray(sw.astype('float32')) if (class_w is not None or PERSONWEIGHT) else None
    print(json.dumps({'training_epochs_times_channels': int(len(X)), 'fit_nights': len(fit),
                      'class_counts': np.bincount(Y, minlength=N_CLASS).tolist(),
                      'load_seconds': round(time.time() - t0, 1)}), flush=True)

    key = jax.random.PRNGKey(SEED)
    params = init_params(key)
    tx = optax.adamw(LR, weight_decay=1e-4)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y, swb, key):
        if AUGMENT:
            x = augment(key, x)
        if HFDROP:
            x = hf_drop(jax.random.fold_in(key, 1), x)
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, swb)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(SEED)

    history, best = [], None
    for ep in range(passes):
        t = time.time()
        order = rng.permutation(len(X))
        losses = []
        for i in range(0, len(order) - BATCH + 1, BATCH):
            idx = order[i:i + BATCH]
            key, sub_key = jax.random.split(key)
            swb = jnp.asarray(sw[idx].astype('float32')) if sw_jax is not None else None
            params, opt_state, l = train_step(params, opt_state, jnp.asarray(X[idx]), jnp.asarray(Y[idx]),
                                              swb, sub_key)
            losses.append(float(l))
        k_sel, per = person_kappa(params, cache, sel, trans, prior, 0.0, metric=metric)
        history.append({'pass': ep + 1, 'train_loss': float(np.mean(losses)),
                        f'selection_mean_person_{metric}_unsmoothed': k_sel,
                        'seconds': round(time.time() - t, 1)})
        print(json.dumps(history[-1]), flush=True)
        if best is None or k_sel > best['kappa']:
            best = {'kappa': k_sel, 'pass': ep + 1, 'params': jax.tree.map(lambda a: a.copy(), params)}
    params = best['params']
    print(json.dumps({'selected_pass': best['pass'], 'selection_kappa': best['kappa']}), flush=True)

    # Sequence weight on the same six selection people, exactly as v2 chose its own.
    candidates = []
    for w in CANDIDATE_WEIGHTS:
        k, _ = person_kappa(params, cache, sel, trans, prior, w, metric=metric)
        candidates.append({'sequence_weight': w, f'mean_person_{metric}': k})
        print(json.dumps(candidates[-1]), flush=True)
    weight = max(candidates, key=lambda r: r[f'mean_person_{metric}'])['sequence_weight']

    # Temperature on the six probability-calibration people, same objective as v2.
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss
    probs, targets = [], []
    for sub in cal:
        x, y, good, onset = load_night(cache, sub)
        m = good & (y < 5)
        probs.append(night_probabilities(params, x, good, onset, trans, prior, weight)[m]); targets.append(y[m])
    def objective(t):
        return np.mean([log_loss(y, np.exp(np.log(p + 1e-12) / t) / np.exp(np.log(p + 1e-12) / t).sum(axis=1, keepdims=True),
                                 labels=list(range(N_CLASS))) for p, y in zip(probs, targets)]) + .02 * np.log(t) ** 2
    temperature = float(minimize_scalar(objective, bounds=(.5, 3), method='bounded').x)

    flat = {f'{i}/{k}': np.asarray(v).tolist() for i, p in enumerate(params) for k, v in p.items()}
    (OUT / 'cnn_weights.json').write_text(json.dumps(flat))
    weight_sha = hashlib.sha256((OUT / 'cnn_weights.json').read_bytes()).hexdigest()
    extra = {'class_weighted_loss': bool(BALANCED or SQRTBALANCED), 'selection_metric': metric,
             'class_weight_scheme': ('balanced' if BALANCED else 'inverse_sqrt' if SQRTBALANCED else None),
             'person_normalised_sample_weights': bool(PERSONWEIGHT),
             'class_weights': (class_w.tolist() if class_w is not None else None),
             'augmentation': ('amplitude scale 0.8-1.25 and Gaussian noise 0.05' if AUGMENT else None),
             'input_filter': ('FFT low-pass at 16 Hz, training and evaluation' if LOWPASS16 else
                              '16-30 Hz zeroed in half the training epochs' if HFDROP else None),
             'two_channel_input': bool(TWOCHANNEL),
             'training_class_counts': counts.tolist()}
    freeze = {'frozen_unix': time.time(), 'weights_sha256': weight_sha, 'selected_pass': best['pass'],
              'passes_run': passes, 'selection_candidates': candidates, 'selected_sequence_weight': weight,
              'temperature': temperature, 'history': history, 'seed': SEED, 'learning_rate': LR,
              'batch': BATCH, 'architecture': ARCH, 'pool_after': POOL_AFTER, 'input_hz': FS,
              'samples_per_epoch': N_IN, 'clip': CLIP, 'test_labels_read_before_freeze': False, **extra}
    (OUT / 'cnn_freeze_record.json').write_text(json.dumps(freeze, indent=2))
    print(json.dumps({'frozen': True, 'weights_sha256': weight_sha, 'weight': weight,
                      'temperature': temperature}), flush=True)

    # ---- only now the test people ----
    per_person, preds = [], {}
    for sub in test:
        x, y, good, onset = load_night(cache, sub)
        fz = frozen_test[sub]
        assert np.array_equal(np.array(fz['reference']), y), f'{sub}: reference differs from the frozen file'
        assert np.array_equal(np.array(fz['good']).astype(bool), good), f'{sub}: quality mask differs'
        assert np.allclose(np.array(fz['onset'], dtype=float), onset), f'{sub}: onsets differ'
        m = good & (y < 5)
        p = night_probabilities(params, x, good, onset, trans, prior, weight, temperature)
        guess = p.argmax(axis=1)
        stat = stage_metrics(y[m], guess[m])
        model_tst = float(np.sum((guess > 0) & m) * .5); ref_tst = float(np.sum((y > 0) & m) * .5)
        v2_guess = np.array(fz['model'])
        per_person.append({'subject': sub, **stat, 'reference_tst_minutes': ref_tst,
                           'model_tst_minutes': model_tst, 'tst_error_minutes': abs(model_tst - ref_tst),
                           'available_epochs': int(m.sum()),
                           'v2_kappa': float(stage_metrics(y[m], v2_guess[m])['kappa'])})
        preds[sub] = guess.tolist()
        print(json.dumps({'evaluated': sub, 'cnn_kappa': round(stat['kappa'], 3),
                          'v2_kappa': round(per_person[-1]['v2_kappa'], 3),
                          'cnn_tst_error': per_person[-1]['tst_error_minutes']}), flush=True)

    def mean(field): return float(np.mean([r[field] for r in per_person]))
    saved = json.loads((ROOT / 'results/v2/historical_boas_metrics.json').read_text())['results']
    cnn = {'people': len(per_person), 'accuracy': mean('accuracy'), 'kappa': mean('kappa'),
           'macro_f1': mean('macro_f1'), 'tst_error_minutes': mean('tst_error_minutes')}
    verdict = {'cnn_kappa': cnn['kappa'], 'v2_kappa': saved['kappa'],
               'cnn_beats_v2_on_kappa': bool(cnn['kappa'] > saved['kappa']),
               'cnn_tst_error_minutes': cnn['tst_error_minutes'], 'v2_tst_error_minutes': saved['tst_error_minutes'],
               'cnn_beats_v2_on_sleep_time': bool(cnn['tst_error_minutes'] < saved['tst_error_minutes']),
               'per_person_kappa_wins_for_cnn': int(sum(r['kappa'] > r['v2_kappa'] for r in per_person))}
    report = {'method': ('single-channel 1-D CNN on 64 Hz headband epochs, v2 sequence smoother and '
                         'temperature' + (', class-weighted loss (sklearn "balanced"), selected by '
                         'macro-F1' if BALANCED else '')),
              'cohort': 'historical_boas, 20 held-out people, confirmatory nights',
              'scope': 'Development comparison on people already inspected at v1; not new independent '
                       'confirmation. BOAS forehead headband only; no external cohort, no ADHD claim.',
              'frozen': freeze, 'identical_to_v2': ['person split', 'channel sharing', 'sequence smoother and '
              'candidate weights', 'selection people', 'calibration people', 'evaluation mask', 'stage_metrics',
              'total sleep time definition'],
              'differences_from_v2': ['raw 64 Hz signal input instead of spectral features',
                                      '1-D CNN instead of gradient boosting',
                                      'selection people also chose the number of training passes',
                                      'no class weighting'],
              'cnn_results': cnn, 'v2_results_for_the_same_people': saved, 'verdict': verdict,
              'created_unix': time.time(), 'total_seconds': round(time.time() - t0, 1)}
    (OUT / 'cnn_metrics.json').write_text(json.dumps(report, indent=2))
    (OUT / 'cnn_predictions.json').write_text(json.dumps(preds))
    import pandas as pd
    pd.DataFrame(per_person).drop(columns='per_stage_f1').to_csv(OUT / 'cnn_people.csv', index=False)
    print(json.dumps({'cnn': cnn, 'v2': {k: saved[k] for k in cnn}, 'verdict': verdict}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    name = 'cnn'
    if '--balanced' in sys.argv:
        BALANCED = True; name = 'cnn_balanced'
    if '--augment' in sys.argv:
        AUGMENT = True; name = 'cnn_augmented'
    if '--twochannel' in sys.argv:
        TWOCHANNEL = True; name = 'cnn_twochannel'
    if '--sqrtbalanced' in sys.argv:
        SQRTBALANCED = True; name = 'cnn_sqrtbalanced'
    if '--personweight' in sys.argv:
        PERSONWEIGHT = True; name = 'cnn_personweight'
    if '--lowpass16' in sys.argv:
        LOWPASS16 = True; name = 'cnn_lowpass16'
    if '--hfdrop' in sys.argv:
        HFDROP = True; name = 'cnn_hfdrop'
    for a in sys.argv[1:]:
        if a.startswith('--seed='):          # robustness replicates: same recipe, another seed
            SEED = int(a.split('=', 1)[1]); name = f'{name}_seed{SEED}'
    OUT = ROOT / 'results' / name
    main(args[0], int(args[1]) if len(args) > 1 else 10)

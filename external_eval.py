"""Pre-registered external test on DOD-H and HMC (PREREGISTRATION.md, tag prereg-v1, amendment v1.1).

Two stages, run in this order and once:

  python external_eval.py predict dodh <compact dir> <scorers dir> <boas cache>   # every model
  python external_eval.py predict hmc  <compact dir> <boas cache>
  python external_eval.py score                                                   # scores once

The BOAS cache is only used for the CNN smoother's transition and prior counts (fitting nights),
exactly as cnn_external.py and external_all.py do.

`predict` builds the 256 Hz v2 features and the 64 Hz CNN epochs from the compact files, fixes
the dataset-level gains from the signals alone, and writes every model's stage predictions.
Labels are carried through to the score stage but nothing in `predict` reads them: the models,
the quality mask and the gains use the signal only.

`score` applies yasa_evaluate.py's masks, windows and metrics (the code that scored v2, YASA and
the CNNs on BOAS and Sleep-EDF), then the hypothesis tests exactly as written.

DOD-H reference: the DOD consensus, rebuilt from the five scorers' files in
Dreem-Organization/dreem-learning-evaluation with that repository's rule (majority vote, ties
broken by the scorer with the highest soft agreement). Each scorer is also scored against the
consensus of the other four, as the human ceiling.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'analysis'))
OUT = {'dodh': ROOT / 'results/external_dodh', 'hmc': ROOT / 'results/external_hmc'}
CACHE = Path('/tmp/goldilocks_external_cache')
UNSCORED = 8


# ---------------------------------------------------------------- DOD consensus (reference) --
def soft_agreement(hyp, others):
    prob = np.zeros((6, len(hyp)))
    for h in others:
        prob[np.array(h) + 1, np.arange(len(hyp))] += 1
    prob = prob / np.maximum(prob.max(0), 1)
    return float(prob[np.array(hyp) + 1, np.arange(len(hyp))].mean())


def consensus(ranked):
    """dreem-learning-evaluation build_consensus_hypnogram: majority vote, ties go to the
    best-ranked scorer. Input and output use -1 for not scored."""
    arr = np.array(ranked) + 1
    n = arr.shape[1]
    prob = np.zeros((6, n))
    for h in arr:
        prob[h, np.arange(n)] += 1
    cons = np.argmax(prob, 0)
    ties = (prob == prob[cons, np.arange(n)]).sum(0) > 1
    cons[ties] = arr[0][ties]
    return cons - 1


def rank_scorers(hyps):
    """Order scorers by soft agreement with the other scorers, best first."""
    names = sorted(hyps)
    sa = {s: soft_agreement(hyps[s], [hyps[o] for o in names if o != s]) for s in names}
    return sorted(names, key=lambda s: -sa[s]), sa


# The evaluation repository trims every scorer hypnogram to the span all five scored, and further
# to lights-off / lights-on for three DOD-H records. Its README lists those under older record IDs
# (lights-off at epoch 60 for two records, lights-on at epoch 620 for one); these are the three
# records whose distributed signals are exactly that much shorter.
DOD_LIGHTS_OFF = {'5bf0f969-304c-581e-949c-50c108f62846': 60, 'b5d5785d-87ee-5078-b9b9-aac6abd4d8de': 60}
DOD_LIGHTS_ON = {'3e842aa8-bcd9-521e-93a2-72124233fe2c': 620}


def dod_reference(scorers_dir, record):
    hyps = {d.name: np.array(json.loads((d / f'{record}.json').read_text()))
            for d in sorted(Path(scorers_dir).iterdir()) if (d / f'{record}.json').exists()}
    assert len(hyps) == 5, (record, len(hyps))
    assert len({len(h) for h in hyps.values()}) == 1, record
    lo = max(int(np.where(h >= 0)[0][0]) for h in hyps.values())
    hi = len(next(iter(hyps.values()))) - max(int(np.where(h[::-1] >= 0)[0][0]) for h in hyps.values())
    lo, hi = max(lo, DOD_LIGHTS_OFF.get(record, 0)), min(hi, DOD_LIGHTS_ON.get(record, hi))
    hyps = {s: h[lo:hi].tolist() for s, h in hyps.items()}
    order, _ = rank_scorers(hyps)
    cons = consensus([hyps[s] for s in order])
    to_code = lambda h: np.where((np.array(h) >= 0) & (np.array(h) <= 4), np.array(h), UNSCORED).astype('int8')
    ceiling = {}
    for s in sorted(hyps):
        others = {o: hyps[o] for o in hyps if o != s}
        o_order, _ = rank_scorers(others)
        ceiling[s] = {'scorer': to_code(hyps[s]), 'reference': to_code(consensus([others[o] for o in o_order]))}
    return to_code(cons), ceiling


# ----------------------------------------------------------------------------- epoching --
def quality(ep):
    return np.isfinite(ep).all(axis=(1, 2)) & (np.std(ep, axis=-1).min(axis=1) > 1e-6)


def epochs_dodh(path):
    z = np.load(path)
    sig, fs = z['signals'].astype('float64'), float(z['fs'])
    assert fs == 250
    n = len(z['hypnogram'])
    assert sig.shape[1] >= n * 7500
    ep = sig[:, :n * 7500].reshape(sig.shape[0], n, 7500).transpose(1, 0, 2)
    onset = np.arange(n) * 30.0
    v2_in = resample_poly(np.nan_to_num(ep), up=128, down=125, axis=-1)      # 250 -> 256 Hz
    cnn_in = resample_poly(np.nan_to_num(ep), up=32, down=125, axis=-1)      # 250 -> 64 Hz
    return {'v2': v2_in, 'cnn': cnn_in, 'good': quality(ep), 'onset': onset,
            'yasa': (sig[0, :n * 7500], fs, str(z['channels'][0])), 'channels': [str(c) for c in z['channels']]}


def epochs_hmc(path):
    z = np.load(path)
    sig, fs = z['signal'].astype('float64'), float(z['fs'])
    assert fs == 256
    onset = z['onset'].astype(float)
    assert np.allclose(np.diff(onset), 30), path
    start = np.rint(onset * fs).astype(int)
    ep = np.stack([sig[a:a + 7680] for a in start])[:, None, :]
    cnn_in = resample_poly(np.nan_to_num(ep), up=1, down=4, axis=-1)          # 256 -> 64 Hz
    first, last = start[0], start[-1] + 7680
    return {'v2': np.nan_to_num(ep), 'cnn': cnn_in, 'good': quality(ep), 'onset': onset,
            'yasa': (sig[first:last], fs, str(z['channel'])), 'channels': [str(z['channel'])],
            'lights': (float(z['lights_off']), float(z['lights_on']))}


# ------------------------------------------------------------------------------ models --
def yasa_stages(signal, fs, ch, n):
    import mne
    from yasa_evaluate import yasa_predict
    info = mne.create_info([ch], fs, ch_types=['eeg'])
    raw = mne.io.RawArray(signal[None, :] * 1e-6, info, verbose='ERROR')
    pred = yasa_predict(raw, ch)
    out = np.full(n, -2, dtype='int8')
    out[:min(n, len(pred))] = pred[:n]
    return out


def predict(dataset, compact_dir, scorers_dir=None):
    import joblib
    from pipeline import epoch_features
    from ablation_shift import percentile_log_sd, scale_features, v2_predict
    import cnn_challenger as cc
    import cnn_context as cx
    from cnn_challenger import night_probabilities, rebuild_free, robust_scale, CLIP
    from cnn_posthoc import fit_trans_prior

    out = OUT[dataset]
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    cache = CACHE / dataset
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(compact_dir).glob('*.npz'))
    if dataset == 'hmc':
        files = [f for f in files if f.stem.startswith('SN')]
    assert len(files) == 25, len(files)

    records, labels, extra = [], {}, {}
    for f in files:
        rec = f.stem
        e = epochs_dodh(f) if dataset == 'dodh' else epochs_hmc(f)
        X = epoch_features(e['v2'], 256)
        if dataset == 'dodh':
            y, ceiling = dod_reference(scorers_dir, rec)
            assert len(y) == len(e['onset']), rec
            extra[rec] = {'ceiling': {s: {k: v.tolist() for k, v in d.items()} for s, d in ceiling.items()}}
        else:
            y = np.load(f)['y'].astype('int8')
            extra[rec] = {'lights_off': e['lights'][0], 'lights_on': e['lights'][1]}
        np.savez_compressed(cache / f'{rec}.npz', X=e['cnn'].astype('float16'), y=y, good=e['good'], onset=e['onset'])
        np.savez(cache / f'{rec}_v2.npz', X=X, good=e['good'], onset=e['onset'])
        labels[rec] = {'reference': y.tolist(), 'good': e['good'].astype(int).tolist(), 'onset': e['onset'].tolist()}
        sig, fs, ch = e['yasa']
        np.savez(cache / f'{rec}_yasa.npz', signal=sig.astype('float32'), fs=fs, channel=ch)
        records.append((rec, {'onset': e['onset']}))
        print(json.dumps({'prepared': rec, 'epochs': len(e['onset']), 'valid_signal_epochs': int(e['good'].sum()),
                          'channels': e['channels']}), flush=True)
        del e

    # Dataset-level gains, from the signals only (PREREGISTRATION.md, H1 and H1b).
    proto = json.loads((ROOT / 'protocol.json').read_text())['records']
    fit = [dict(np.load(ROOT / 'data/features' / f"{r['subject']}.npz")) for r in proto if r['split'] == 'fit']
    target = [dict(np.load(cache / f'{rec}_v2.npz')) for rec, _ in records]
    gains = {}
    for q in [75, 90]:
        ref = float(np.median([percentile_log_sd(z, q) for z in fit]))
        gains[f'q{q}'] = float(np.exp(ref - np.median([percentile_log_sd(z, q) for z in target])))

    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    preds = {m: {} for m in ['v2', 'v2_gain_q75', 'v2_gain_q90', 'yasa']}
    for (rec, e), z in zip(records, target):
        preds['v2'][rec] = v2_predict(frozen, z).tolist()
        for q in ['q75', 'q90']:
            zz = dict(z)
            zz['X'] = scale_features(z['X'], gains[q])
            preds[f'v2_gain_{q}'][rec] = v2_predict(frozen, zz).tolist()
        ya = np.load(cache / f'{rec}_yasa.npz')
        preds['yasa'][rec] = yasa_stages(ya['signal'].astype('float64'), float(ya['fs']), str(ya['channel']),
                                         len(e['onset'])).tolist()
        print(json.dumps({'v2_and_yasa': rec}), flush=True)

    trans, prior = fit_trans_prior(BOAS_CACHE)

    def load(c, s):
        zz = np.load(Path(c) / f'{s}.npz')
        x = zz['X'].astype('float32')
        good = zz['good'].astype(bool)
        med, mad = robust_scale(x, good)
        return np.clip((x - med) / mad, -CLIP, CLIP), zz['y'].astype('int8'), good, zz['onset'].astype(float)

    probs = {}
    cc.TWOCHANNEL = False
    for name in ['cnn', 'cnn_balanced', 'cnn_sqrtbalanced', 'cnn_augmented']:
        src = ROOT / 'results' / name
        freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
        params = rebuild_free(src / 'cnn_weights.json')
        probs[name] = {}
        for rec, _ in records:
            x, _, good, onset = load(cache, rec)
            probs[name][rec] = night_probabilities(params, x, good, onset, trans, prior,
                                                   freeze['selected_sequence_weight'], freeze['temperature'])
        print(json.dumps({'scored_model': name}), flush=True)
    for name in ['cnn_context', 'cnn_context_sqrtbalanced', 'cnn_context_balanced']:
        src = ROOT / 'results' / name
        freeze = json.loads((src / 'context_freeze_record.json').read_text())
        trunk = rebuild_free(ROOT / freeze['trunk_source'] / 'cnn_weights.json')
        head = rebuild_free(src / 'context_head_weights.json')
        probs[name] = {}
        for rec, _ in records:
            emb, _, good, onset = cx.night_embeddings(trunk, cache, rec, load)
            probs[name][rec] = cx.head_night_probabilities(head, emb, good, onset, trans, prior,
                                                           freeze['selected_sequence_weight'], freeze['temperature'])
        print(json.dumps({'scored_model': name}), flush=True)
    members = list(probs)
    probs['ensemble_of_7'] = {rec: (lambda a: a / a.sum(axis=1, keepdims=True))(
        np.mean([probs[m][rec] for m in members], axis=0)) for rec, _ in records}
    for name, per in probs.items():
        preds[name] = {rec: p.argmax(axis=1).tolist() for rec, p in per.items()}

    (out / 'predictions' / 'predictions.json').write_text(json.dumps(preds))
    (out / 'predictions' / 'labels_and_grid.json').write_text(json.dumps(labels))
    (out / 'predictions' / 'extra.json').write_text(json.dumps(extra))
    (out / 'predictions' / 'gains.json').write_text(json.dumps(
        {'gains': gains, 'reference': 'median over BOAS fitting nights of the per-night percentile of epoch log-SD',
         'from_signals_only': True}, indent=2))
    print(json.dumps({'written': str(out / 'predictions'), 'gains': gains}), flush=True)


BOAS_CACHE = None

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'predict':
        BOAS_CACHE = sys.argv[-1]
        dataset = sys.argv[2]
        predict(dataset, sys.argv[3], sys.argv[4] if dataset == 'dodh' else None)
    elif stage == 'score':
        from external_score import main as score_main
        score_main()

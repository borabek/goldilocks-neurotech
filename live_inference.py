"""A causal, streaming inference path, added beside the offline work rather than replacing it.

Why this file is needed before any EEG device is used. Everything else in this project is
retrospective: `v2_model.temporal` gives the classifier a *centred* 5- and 15-epoch window, and
`v2_model.smooth` runs a forward-backward pass over the whole night. Both read the future. Even the
amplitude normalisation uses the whole recording's median absolute deviation. None of that can run
live, so a live demo built on those functions would be a replay wearing a costume.

What this does instead, all of it causal:

  * **Epoching.** 30-second epochs are emitted as they complete, from a live stream or a replayed
    recording. Nothing is buffered beyond the current epoch and the running statistics below.
  * **Normalisation.** A running robust scale over the trailing SCALE_EPOCHS epochs, warmed up on
    the first WARMUP_EPOCHS. The offline path divides by the whole night's scale; this cannot, and
    the difference is part of the measured cost below.
  * **Model.** The per-epoch CNN. It is used because it needs no context by construction, so going
    causal changes nothing about its input. The frozen v2 feature model is deliberately *not*
    offered here: its features are defined over a centred window, and feeding it trailing windows
    would silently change what its weights mean.
  * **Smoothing.** Forward-only HMM filtering with the transition matrix and prior estimated on the
    fitting people, at the same sequence weight the model was frozen with. The offline path uses
    forward-backward; posterior at epoch t here depends only on epochs 1..t.
  * **Running summary.** Cumulative sleep time as 0.5 minutes per non-wake epoch, reported as it
    accrues.

What it does not do, and must not be presented as doing: it does not make the model accurate on a
device it has never seen. This project's own evidence is that transfer is where these models fail —
every member of the CNN family drops from kappa 0.74 on the headband to below 0.22 on Sleep-EDF.
A live path that runs is not a live path that is right, and the two claims must be kept apart.

Usage:
  python live_inference.py --evaluate <boas_cache>        measure what going causal costs
  python live_inference.py --replay <night.npz>           stream a cached recording, epoch by epoch
  python live_inference.py --lsl --channels HB_1,HB_2     read a live Lab Streaming Layer stream
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import cnn_challenger as cc
from cnn_challenger import CLIP, N_CLASS, N_IN, rebuild_free
from pipeline import ROOT, stage_metrics

FS = 64                 # the sampling rate the CNN was trained at
EPOCH_S = 30
WARMUP_EPOCHS = 10      # epochs collected before the first score is emitted
SCALE_EPOCHS = 120      # trailing window for the running robust scale, one hour
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
OUT = ROOT / 'results/live'


class CausalScorer:
    """Scores epochs one at a time. Nothing it returns for epoch t depends on epoch t+1."""

    def __init__(self, weights='results/cnn/cnn_weights.json', freeze='results/cnn/cnn_freeze_record.json',
                 trans=None, prior=None):
        self.params = rebuild_free(ROOT / weights)
        fz = json.loads((ROOT / freeze).read_text())
        self.weight = fz['selected_sequence_weight']
        self.temperature = fz['temperature']
        self.trans, self.prior = trans, prior
        self.buffer = []            # raw epochs, (channels, samples), awaiting the warm-up
        self.scale_hist = []        # per-epoch per-channel median and scale, trailing window
        self.log_alpha = None       # forward HMM state
        self.n = 0
        self.sleep_epochs = 0
        self.emitted = []

    # -- causal normalisation ------------------------------------------------
    def _scale(self, epoch):
        med = np.median(epoch, axis=-1, keepdims=True)
        mad = np.median(np.abs(epoch - med), axis=-1, keepdims=True) * 1.4826
        self.scale_hist.append((med, np.maximum(mad, 1e-3)))
        self.scale_hist = self.scale_hist[-SCALE_EPOCHS:]
        m = np.mean([s[0] for s in self.scale_hist], axis=0)
        a = np.mean([s[1] for s in self.scale_hist], axis=0)
        return np.clip((epoch - m) / a, -CLIP, CLIP)

    # -- forward-only HMM ----------------------------------------------------
    def _filter(self, p):
        if self.weight == 0 or self.trans is None:
            return p
        logt = np.log(self.trans) * self.weight
        emission = np.log(p + 1e-12) - self.weight * np.log(self.prior)
        if self.log_alpha is None:
            self.log_alpha = np.log(self.prior) + emission
        else:
            from scipy.special import logsumexp
            self.log_alpha = emission + logsumexp(self.log_alpha[:, None] + logt, axis=0)
        a = self.log_alpha - self.log_alpha.max()
        out = np.exp(a) / np.exp(a).sum()
        self.log_alpha = np.log(out + 1e-12)      # renormalise to keep the recursion stable
        return out

    def push(self, epoch_uv, good=True):
        """epoch_uv: (channels, FS*EPOCH_S) microvolts. Returns the result for this epoch."""
        import jax.numpy as jnp
        self.n += 1
        x = self._scale(np.asarray(epoch_uv, dtype='float32'))
        if self.n <= WARMUP_EPOCHS:
            return {'epoch': self.n, 'status': 'warming up', 'stage': None}
        c = x.shape[0]
        probs = np.asarray(cc.predict_proba_batch(self.params,
                                                  jnp.asarray(x.reshape(c, 1, N_IN)))).mean(axis=0)
        probs = self._filter(probs)
        probs = np.exp(np.log(probs + 1e-12) / self.temperature)
        probs = probs / probs.sum()
        stage = int(probs.argmax())
        if good and stage > 0:
            self.sleep_epochs += 1
        self.emitted.append(stage)
        return {'epoch': self.n, 'status': 'scoring', 'stage': STAGES[stage],
                'stage_index': stage, 'confidence': round(float(probs[stage]), 3),
                'running_sleep_minutes': self.sleep_epochs * 0.5,
                'elapsed_minutes': self.n * 0.5}


PRIOR_FILE = OUT / 'transition_prior.json'


def load_trans_prior(cache=None):
    """The smoother's transition matrix and prior are a fixed function of the fitting people's
    labels. They are saved once into the package so the live path needs no training cache: a
    headset and this folder are enough."""
    if PRIOR_FILE.exists():
        d = json.loads(PRIOR_FILE.read_text())
        return np.array(d['transition']), np.array(d['prior'])
    if cache is None:
        raise SystemExit(f'{PRIOR_FILE} is missing and no --cache was given to rebuild it from.')
    trans, prior = fit_trans_prior(cache)
    PRIOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    PRIOR_FILE.write_text(json.dumps({
        'source': 'estimated on the 48 fitting people of the frozen v2 split, labels only',
        'transition': trans.tolist(), 'prior': prior.tolist()}, indent=2))
    return trans, prior


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


def evaluate(cache):
    """What does going causal cost? Same 20 people, same mask, same metric."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trans, prior = load_trans_prior(cache)
    frozen = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    offline = json.loads((ROOT / 'results/cnn/cnn_predictions.json').read_text())
    subs = sorted(frozen, key=lambda s: int(s.split('-')[1]))
    rows = []
    for sub in subs:
        z = np.load(Path(cache) / f'{sub}.npz')
        x = z['X'].astype('float32')
        y, good = z['y'].astype('int8'), z['good'].astype(bool)
        scorer = CausalScorer(trans=trans, prior=prior)
        live = np.full(len(y), -2, dtype='int8')
        for i in range(len(y)):
            r = scorer.push(x[i], good=bool(good[i]))
            if r['stage'] is not None:
                live[i] = r['stage_index']
        m = good & (y < 5) & (live >= 0)
        full = good & (y < 5)
        off = np.array(offline[sub])
        st_live = stage_metrics(y[m], live[m])
        st_off = stage_metrics(y[full], off[full])
        lt = float(np.sum((live > 0) & m) * .5); rt_m = float(np.sum((y > 0) & m) * .5)
        ot = float(np.sum((off > 0) & full) * .5); rt_f = float(np.sum((y > 0) & full) * .5)
        rows.append({'subject': sub, 'scored_epochs_live': int(m.sum()), 'scored_epochs_offline': int(full.sum()),
                     'warmup_epochs_skipped': WARMUP_EPOCHS,
                     'live_kappa': st_live['kappa'], 'offline_kappa': st_off['kappa'],
                     'live_macro_f1': st_live['macro_f1'], 'offline_macro_f1': st_off['macro_f1'],
                     'live_tst_error_minutes': abs(lt - rt_m),
                     'offline_tst_error_minutes': abs(ot - rt_f)})
        print(json.dumps({'subject': sub, 'live_kappa': round(st_live['kappa'], 3),
                          'offline_kappa': round(st_off['kappa'], 3)}), flush=True)

    def mean(k): return float(np.mean([r[k] for r in rows]))
    summary = {k: mean(k) for k in ['live_kappa', 'offline_kappa', 'live_macro_f1', 'offline_macro_f1',
                                    'live_tst_error_minutes', 'offline_tst_error_minutes']}
    summary['cost_of_going_live'] = {
        'kappa': summary['live_kappa'] - summary['offline_kappa'],
        'macro_f1': summary['live_macro_f1'] - summary['offline_macro_f1'],
        'sleep_time_error_minutes': summary['live_tst_error_minutes'] - summary['offline_tst_error_minutes']}
    report = {
        'what_this_measures': 'the same CNN scored causally, epoch by epoch, against its own retrospective '
                              'scoring on the same 20 people and the same mask',
        'causal_differences': ['running trailing-window normalisation instead of whole-recording',
                               f'the first {WARMUP_EPOCHS} epochs are not scored at all',
                               'forward-only HMM filtering instead of forward-backward smoothing'],
        'unchanged': ['model weights', 'sequence weight', 'temperature', 'evaluation mask', 'stage_metrics',
                      'sleep-time definition'],
        'scope': 'Development cohort, inspected at v1. This is the cost of the live path on the device the '
                 'model was trained for; on an unseen device the transfer failure measured elsewhere in '
                 'this project dominates and is not captured here.',
        'summary': summary, 'per_person': rows,
        'created_unix': time.time(), 'seconds': round(time.time() - t0, 1)}
    (OUT / 'live_vs_offline.json').write_text(json.dumps(report, indent=2))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / 'live_vs_offline.csv', index=False)
    print(json.dumps(summary, indent=2))
    return report


def replay(path, cache=None, realtime=False):
    trans, prior = load_trans_prior(cache)
    z = np.load(path)
    x = z['X'].astype('float32')
    scorer = CausalScorer(trans=trans, prior=prior)
    for i in range(len(x)):
        r = scorer.push(x[i])
        print(json.dumps(r), flush=True)
        if realtime:
            time.sleep(EPOCH_S)


def from_lsl(channels, cache=None):
    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError:
        sys.exit('pylsl is not installed here. Install it on the machine attached to the headset: '
                 'pip install pylsl')
    trans, prior = load_trans_prior(cache)
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        sys.exit('no EEG stream found on the network. Start the device\'s LSL outlet first.')
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    fs_in = info.nominal_srate()
    names = [info.desc().child('channels').child('channel').child_value('label')]
    print(json.dumps({'stream': info.name(), 'nominal_rate': fs_in, 'first_channel_label': names[0],
                      'requested_channels': channels,
                      'note': 'channel order and units come from the device; check them before trusting '
                              'any output'}), flush=True)
    from scipy.signal import resample_poly
    need = int(fs_in * EPOCH_S)
    buf = []
    scorer = CausalScorer(trans=trans, prior=prior)
    while True:
        chunk, _ = inlet.pull_chunk(timeout=1.0)
        if chunk:
            buf.extend(chunk)
        while len(buf) >= need:
            raw = np.asarray(buf[:need], dtype='float32').T      # (channels, samples)
            buf = buf[need:]
            up, down = FS, int(fs_in)
            ep = resample_poly(np.nan_to_num(raw), up=up, down=down, axis=-1)
            if ep.shape[-1] != N_IN:
                ep = ep[..., :N_IN] if ep.shape[-1] > N_IN else np.pad(ep, ((0, 0), (0, N_IN - ep.shape[-1])))
            print(json.dumps(scorer.push(ep)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluate', metavar='CACHE')
    ap.add_argument('--replay', metavar='NPZ')
    ap.add_argument('--lsl', action='store_true')
    ap.add_argument('--channels', default='')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--realtime', action='store_true')
    a = ap.parse_args()
    try:
        if a.evaluate:
            evaluate(a.evaluate)
        elif a.replay:
            replay(a.replay, a.cache, a.realtime)
        elif a.lsl:
            from_lsl([c for c in a.channels.split(',') if c], a.cache)
        else:
            ap.print_help()
    except BrokenPipeError:
        # A downstream reader closed the pipe first, as `| head -40` does. A normal Unix
        # tool stops quietly here; without this the interpreter prints a traceback and then
        # raises again while flushing stdout at exit. Redirect the remaining writes to
        # devnull, then leave without the final flush.
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        os._exit(0)
    except KeyboardInterrupt:
        # Ctrl-C during a live stream is how the operator stops it, not a failure.
        sys.stderr.write('\nstopped\n')
        sys.exit(130)

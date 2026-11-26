"""Shift ablation, A1 and A2 (exploratory; Sleep-EDF people were inspected at v2).

A2 for v2 runs from the shipped feature caches and needs no raw data. The v2 features are
log powers, relative powers, spectral entropy, log std and log mean absolute difference, so
multiplying the raw signal by a factor a shifts the log-power features by 2 ln a and the two
log-amplitude features by ln a, and leaves the rest unchanged. That lets us apply the scale
exactly, without re-reading any EDF.

A1 and A2 for the CNN need the 64 Hz epoch caches (not in the repo, about 723 MB):
  --boas-cache  folder with the BOAS test-night .npz files (X: n x 2 x 1920, microvolts)
  --ext-cache   folder with the Sleep-EDF .npz files (X: n x 1 x 1920, microvolts)

CNN normalisation variants (A1), all label-free unless marked oracle:
  whole_recording     median and MAD over every good epoch (what shipped)
  v2_night_window     median and MAD inside the window from the first to the last epoch the
                      frozen v2 model calls sleep, plus 30 minutes either side
  boas_population     one fixed median and MAD pooled over the BOAS fitting nights
  oracle_reference    median and MAD inside the reference-defined night window (uses labels;
                      an upper bound, not a usable method)

CNN scale variants (A2): the normalised input multiplied by 0.5, 0.75, 1.5 and 2 before the
clip. The network applies instance normalisation after every convolution, so it is close to
scale-invariant by construction; A2 measures how close, through the clip at 20 and the 1e-5
variance floor.

Usage:
  python analysis/ablation_shift.py                       # v2 scale ablation only
  python analysis/ablation_shift.py --boas-cache B --ext-cache E   # adds the CNN variants
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yasa_evaluate import score, summarize, windows  # noqa: E402
from v2_model import probabilities  # noqa: E402

OUT = ROOT / 'results' / 'ablation_shift'
FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]
LOG_POWER = [0, 1, 3, 5, 7, 9]   # log total power and log band powers, per channel
LOG_AMPLITUDE = [12, 13]         # log std and log mean absolute difference, per channel
PER_CHANNEL = 14


def scale_features(X, a):
    X = X.copy()
    channels = X.shape[1] // PER_CHANNEL
    for c in range(channels):
        base = c * PER_CHANNEL
        X[:, [base + i for i in LOG_POWER]] += 2 * np.log(a)
        X[:, [base + i for i in LOG_AMPLITUDE]] += np.log(a)
    return X


def v2_predict(frozen, z):
    p = probabilities(frozen['model'], z['X'], z['good'], frozen['transition'], frozen['prior'],
                      frozen['weight'], frozen['temperature'], z['onset'])
    return p.argmax(axis=1)


def score_people(per_person):
    out = {}
    for name in ['primary_complete_recording', 'secondary_reference_window']:
        rows = [r[name] for r in per_person if name in r]
        s = summarize(rows)
        s['wake_f1'] = float(np.mean([r['per_stage_f1'][0] for r in rows]))
        out[name] = s
    return out


def v2_scale_ablation():
    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    cohorts = {
        'sleep_edf': sorted((ROOT / 'data/external_sleepedf/features').glob('*.npz')),
        'boas_test': [ROOT / 'data/features' / f"{r['subject']}.npz"
                      for r in json.loads((ROOT / 'protocol.json').read_text())['records']
                      if r['split'] == 'test' and r['confirmatory_night']],
    }
    results = {}
    for cohort, files in cohorts.items():
        results[cohort] = {}
        for a in FACTORS:
            per_person = []
            for f in files:
                z = dict(np.load(f))
                z['X'] = scale_features(z['X'], a)
                y, good, onset = z['y'], z['good'].astype(bool), z['onset'].astype(float)
                guess = v2_predict(frozen, z)
                per_person.append({w: score(y, good, onset, guess, m) for w, m in windows(y, onset).items()})
            results[cohort][str(a)] = score_people(per_person)
            prim = results[cohort][str(a)]['primary_complete_recording']
            print(json.dumps({'model': 'v2', 'cohort': cohort, 'scale': a,
                              'kappa': round(prim['kappa'], 3), 'macro_f1': round(prim['macro_f1'], 3),
                              'wake_f1': round(prim['wake_f1'], 3),
                              'tst_error_minutes': round(prim['tst_error_minutes'], 1)}), flush=True)
    return results


def percentile_log_sd(z, q, mask=None):
    """q-th percentile of the per-epoch log standard deviation (feature 12 of each channel)."""
    good = z['good'].astype(bool) if mask is None else mask
    channels = z['X'].shape[1] // PER_CHANNEL
    cols = [c * PER_CHANNEL + 12 for c in range(channels)]
    return float(np.percentile(z['X'][good][:, cols].ravel(), q))


def v2_gain_calibration():
    """Label-free amplitude calibration for v2. Every rule tried is recorded, including the losers.

    per_recording_qXX: one gain per recording, matching that recording's q-th percentile of
                       epoch log-SD to the median over the BOAS fitting nights.
    dataset_qXX:       one gain per dataset, matching the median over that dataset's recordings.
                       On the training device this is about 1 by construction.
    The reference is always the BOAS fitting nights; no label from any cohort is used to set a gain.
    """
    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    records = json.loads((ROOT / 'protocol.json').read_text())['records']
    fit = [r['subject'] for r in records if r['split'] == 'fit']
    cohorts = {
        'sleep_edf': sorted((ROOT / 'data/external_sleepedf/features').glob('*.npz')),
        'boas_test': [ROOT / 'data/features' / f"{r['subject']}.npz" for r in records
                      if r['split'] == 'test' and r['confirmatory_night']],
        'boas_probability_validation': [ROOT / 'data/features' / f'{s}.npz' for s in
                                        sorted({r['subject'] for r in records if r['split'] == 'probability_validation'})],
    }
    fit_z = [dict(np.load(ROOT / 'data/features' / f'{s}.npz')) for s in fit]
    results = {}
    for cohort, files in cohorts.items():
        zs = [dict(np.load(f)) for f in files]
        results[cohort] = {}
        for q in [50, 75, 90]:
            ref = float(np.median([percentile_log_sd(z, q) for z in fit_z]))
            dataset_gain = float(np.exp(ref - np.median([percentile_log_sd(z, q) for z in zs])))
            for rule in ['per_recording', 'dataset']:
                per_person, gains = [], []
                for z in zs:
                    a = float(np.exp(ref - percentile_log_sd(z, q))) if rule == 'per_recording' else dataset_gain
                    gains.append(a)
                    zz = dict(z)
                    zz['X'] = scale_features(z['X'], a)
                    y, good, onset = z['y'], z['good'].astype(bool), z['onset'].astype(float)
                    guess = v2_predict(frozen, zz)
                    per_person.append({w: score(y, good, onset, guess, m) for w, m in windows(y, onset).items()})
                key = f'{rule}_q{q}'
                results[cohort][key] = {'gain_median': float(np.median(gains)),
                                        'gain_range': [float(min(gains)), float(max(gains))],
                                        **score_people(per_person)}
                prim = results[cohort][key]['primary_complete_recording']
                print(json.dumps({'model': 'v2', 'cohort': cohort, 'rule': key,
                                  'gain': round(float(np.median(gains)), 2), 'kappa': round(prim['kappa'], 3),
                                  'macro_f1': round(prim['macro_f1'], 3), 'wake_f1': round(prim['wake_f1'], 3),
                                  'tst_error_minutes': round(prim['tst_error_minutes'], 1)}), flush=True)
    return results


def cnn_ablation(boas_cache, ext_cache):
    import cnn_challenger as cc
    from cnn_challenger import night_probabilities, robust_scale, CLIP, N_CLASS
    from cnn_external import rebuild_params

    v2_ext = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    fit = [r['subject'] for r in rows if r['split'] == 'fit']

    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    pooled = []
    for sub in fit:
        z = np.load(Path(boas_cache) / f'{sub}.npz')
        y, good = z['y'].astype('int8'), z['good'].astype(bool)
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
        x = z['X'][good].astype('float32')
        pooled.append(x[:, :, ::16].ravel())   # thinned sample for a pooled median and MAD
    prior /= prior.sum()
    trans /= trans.sum(axis=1, keepdims=True)
    pooled = np.concatenate(pooled)
    pop_med = np.median(pooled)
    pop_mad = max(np.median(np.abs(pooled - pop_med)) * 1.4826, 1e-3)

    def stats_over(x, mask):
        g = x[mask] if mask.any() else x
        med = np.median(g, axis=(0, 2), keepdims=True)
        mad = np.median(np.abs(g - med), axis=(0, 2), keepdims=True) * 1.4826
        return med, np.maximum(mad, 1e-3)

    models = ['cnn', 'cnn_balanced', 'cnn_sqrtbalanced', 'cnn_augmented']
    results = {}
    for name in models:
        folder = ROOT / 'results' / name
        freeze = json.loads((folder / 'cnn_freeze_record.json').read_text())
        params = rebuild_params(folder / 'cnn_weights.json')
        weight, temperature = freeze['selected_sequence_weight'], freeze['temperature']
        variants = {}
        for sub, rec in sorted(v2_ext.items()):
            z = np.load(Path(ext_cache) / f'{sub}.npz')
            x = z['X'].astype('float32')
            good, y, onset = z['good'].astype(bool), z['y'].astype('int8'), z['onset'].astype(float)
            v2_guess = np.array(rec['model'])
            v2_sleep = onset[(v2_guess > 0) & (v2_guess < 5)]
            v2_night = (onset >= v2_sleep.min() - 1800) & (onset <= v2_sleep.max() + 1800) if len(v2_sleep) else good
            ref_night = windows(y, onset).get('secondary_reference_window', good)
            norms = {
                'whole_recording': robust_scale(x, good),
                'v2_night_window': stats_over(x, good & v2_night),
                'boas_population': (np.full((1, 1, 1), pop_med), np.full((1, 1, 1), pop_mad)),
                'oracle_reference': stats_over(x, good & ref_night),
            }
            plans = [(f'A1_{k}', med, mad, 1.0) for k, (med, mad) in norms.items()]
            med0, mad0 = norms['whole_recording']
            plans += [(f'A2_scale_{a}', med0, mad0, a) for a in FACTORS if a != 1.0]
            for label, med, mad, a in plans:
                xin = np.clip((x - med) / mad * a, -CLIP, CLIP)
                p = night_probabilities(params, xin, good, onset, trans, prior, weight, temperature)
                guess = p.argmax(axis=1)
                variants.setdefault(label, []).append(
                    {w: score(y, good, onset, guess, m) for w, m in windows(y, onset).items()})
        results[name] = {k: score_people(v) for k, v in variants.items()}
        for k, v in results[name].items():
            prim = v['primary_complete_recording']
            print(json.dumps({'model': name, 'variant': k, 'kappa': round(prim['kappa'], 3),
                              'macro_f1': round(prim['macro_f1'], 3), 'wake_f1': round(prim['wake_f1'], 3),
                              'tst_error_minutes': round(prim['tst_error_minutes'], 1)}), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache')
    ap.add_argument('--ext-cache')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'status': 'EXPLORATORY: Sleep-EDF people were inspected at the v2 evaluation',
              'refitted_anything': False,
              'v2_scale': v2_scale_ablation(),
              'v2_gain_calibration': v2_gain_calibration()}
    if args.boas_cache and args.ext_cache:
        report['cnn'] = cnn_ablation(args.boas_cache, args.ext_cache)
    else:
        report['cnn'] = 'not run: needs --boas-cache and --ext-cache (the 64 Hz epoch caches)'
    (OUT / 'ablation_shift_metrics.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

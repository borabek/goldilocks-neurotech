"""Does the CNN recognise wake by its high-frequency power? (exploratory)

A6 found that wake on the BOAS headband carries about twice the beta share (16-30 Hz) of wake
on Sleep-EDF Fpz-Cz, and on Sleep-EDF the CNNs call most wake N2. If the CNN's wake decision
rests on beta, removing 16-30 Hz from BOAS nights should make it call BOAS wake N2 as well,
while removing other bands should matter much less. The beta band is 14 Hz wide and the others
are 4 Hz, so beta is also removed in three 4-6 Hz slices (16-20, 20-24, 24-30) to compare like
with like.

Each condition applies a zero-phase 4th-order Butterworth band-stop to the 64 Hz signal before
the shipped robust scaling, then runs the frozen CNN exactly as cnn_external.py does. Nothing is
refitted. Bands: none, theta 4-8, alpha 8-12, sigma 12-16, beta 16-30 and its three slices.

On Sleep-EDF the script also relates each recording's wake beta share to the CNN's wake recall.

Usage: python analysis/beta_perturbation.py --boas-cache B --ext-cache E
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from cnn_challenger import night_probabilities, robust_scale, CLIP, N_CLASS  # noqa: E402
from cnn_external import rebuild_params  # noqa: E402
from yasa_evaluate import score, summarize, windows  # noqa: E402

FS = 64
BANDS = {'none': None, 'theta_4_8': (4, 8), 'alpha_8_12': (8, 12), 'sigma_12_16': (12, 16),
         'beta_16_30': (16, 30), 'beta_16_20': (16, 20), 'beta_20_24': (20, 24), 'beta_24_30': (24, 30)}
MODELS = ['cnn', 'cnn_balanced']
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def band_stop(x, band):
    if band is None:
        return x
    sos = butter(4, band, btype='bandstop', fs=FS, output='sos')
    return sosfiltfilt(sos, x, axis=-1).astype('float32')


def smoother_stats(boas_cache, fit):
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for sub in fit:
        z = np.load(Path(boas_cache) / f'{sub}.npz')
        y, good = z['y'].astype('int8'), z['good'].astype(bool)
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
    return trans / trans.sum(axis=1, keepdims=True), prior / prior.sum()


def run_model(name, files, trans, prior, band):
    folder = ROOT / 'results' / name
    freeze = json.loads((folder / 'cnn_freeze_record.json').read_text())
    params = rebuild_params(folder / 'cnn_weights.json')
    rows, wake_to = [], np.zeros(5)
    for f in files:
        z = np.load(f)
        x = band_stop(z['X'].astype('float32'), band)
        good, y, onset = z['good'].astype(bool), z['y'].astype('int8'), z['onset'].astype(float)
        med, mad = robust_scale(x, good)
        xin = np.clip((x - med) / mad, -CLIP, CLIP)
        p = night_probabilities(params, xin, good, onset, trans, prior,
                                freeze['selected_sequence_weight'], freeze['temperature'])
        guess = p.argmax(axis=1)
        s = score(y, good, onset, guess, windows(y, onset)['primary_complete_recording'])
        w = good & (y == 0)
        s['wake_recall'] = float(np.mean(guess[w] == 0)) if w.any() else float('nan')
        wake_to += np.bincount(guess[w].clip(0, 4), minlength=5)
        rows.append(s)
    out = summarize(rows)
    out['wake_recall'] = float(np.nanmean([r['wake_recall'] for r in rows]))
    out['wake_f1'] = float(np.mean([r['per_stage_f1'][0] for r in rows]))
    out['wake_predicted_as'] = {s: round(float(v), 3) for s, v in zip(STAGES, wake_to / wake_to.sum())}
    return out, rows


def wake_beta_share(f):
    z = np.load(f)
    x, y, good = z['X'].astype('float32'), z['y'], z['good'].astype(bool)
    m = good & (y == 0)
    freq, p = welch(x[m], fs=FS, nperseg=FS * 4, axis=-1)
    p = p.mean(axis=1)
    keep = (freq >= 0.5) & (freq <= 30)
    beta = (freq >= 16) & (freq <= 30)
    return float(np.median(p[:, beta].sum(axis=1) / p[:, keep].sum(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    args = ap.parse_args()
    records = json.loads((ROOT / 'protocol.json').read_text())['records']
    fit = [r['subject'] for r in records if r['split'] == 'fit']
    test = [r['subject'] for r in records if r['split'] == 'test' and r['confirmatory_night']]
    boas_files = [Path(args.boas_cache) / f'{s}.npz' for s in test]
    ext_subs = sorted(json.loads((ROOT / 'results/v2/external_predictions.json').read_text()))
    ext_files = [Path(args.ext_cache) / f'{s}.npz' for s in ext_subs]
    trans, prior = smoother_stats(args.boas_cache, fit)

    report = {'status': 'EXPLORATORY: diagnostic perturbation, nothing refitted or selected',
              'filter': '4th-order Butterworth band-stop, zero-phase (sosfiltfilt), 64 Hz, before robust scaling',
              'boas_test': {}, 'sleep_edf_wake_beta_vs_recall': {}}
    for name in MODELS:
        report['boas_test'][name] = {}
        for label, band in BANDS.items():
            res, _ = run_model(name, boas_files, trans, prior, band)
            report['boas_test'][name][label] = res
            print(json.dumps({'model': name, 'removed': label, 'wake_recall': round(res['wake_recall'], 3),
                              'macro_f1': round(res['macro_f1'], 3), 'kappa': round(res['kappa'], 3),
                              'wake_predicted_as': res['wake_predicted_as']}), flush=True)

    betas = [wake_beta_share(f) for f in ext_files]
    for name in MODELS:
        _, rows = run_model(name, ext_files, trans, prior, None)
        rec = [r['wake_recall'] for r in rows]
        rho = spearmanr(betas, rec)
        report['sleep_edf_wake_beta_vs_recall'][name] = {
            'spearman_rho': float(rho.statistic), 'p_value': float(rho.pvalue), 'people': len(rows),
            'per_recording': [{'recording': s, 'wake_beta_share': b, 'wake_recall': r}
                              for s, b, r in zip(ext_subs, betas, rec)]}
        print(json.dumps({'model': name, 'sleep_edf_rho_beta_vs_wake_recall': round(float(rho.statistic), 3),
                          'p': round(float(rho.pvalue), 3)}), flush=True)
    report['boas_test_wake_beta_share_median'] = float(np.median([wake_beta_share(f) for f in boas_files]))
    report['sleep_edf_wake_beta_share_median'] = float(np.median(betas))

    out = ROOT / 'results' / 'ablation_shift'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'beta_perturbation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""A6: is the 64 Hz CNN input the same kind of signal on both devices?

On Sleep-EDF the plain CNN labels 95% of N2, 85% of N3, 77% of REM and 67% of wake as N2
(research_outputs/day1/external_confusion.csv). Collapsing onto one class is what a model does
when its input looks nothing like training data, so before reading anything into the ablation
we check the boring explanations: epoch length, sampling grid, and where the spectral peaks sit.

For each cache it reports, per reference stage:
  - samples per epoch and the onset spacing (must be 1920 and 30 s on both)
  - median absolute amplitude after the shipped robust scaling
  - the frequency of the largest relative-power peak in 8-16 Hz (alpha in wake, spindles in N2)
  - relative power in the delta, theta, alpha, sigma and beta bands

A time-axis bug (for example 100 Hz data treated as 256 Hz) would move the peaks by a fixed
ratio; a filter difference would change the band shares without moving the peaks.

Usage: python analysis/a6_spectral_check.py --boas-cache B --ext-cache E
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from cnn_challenger import robust_scale  # noqa: E402

FS = 64
BANDS = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'sigma': (12, 16), 'beta': (16, 30)}
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def cohort_profile(cache, subjects):
    sums = {s: {'psd': None, 'n': 0, 'amp': []} for s in STAGES}
    checks = {'samples_per_epoch': set(), 'onset_spacing_s': set()}
    freq = None
    for sub in subjects:
        z = np.load(Path(cache) / f'{sub}.npz')
        x, y, good, onset = z['X'].astype('float32'), z['y'], z['good'].astype(bool), z['onset']
        checks['samples_per_epoch'].add(int(x.shape[-1]))
        checks['onset_spacing_s'].add(float(np.median(np.diff(onset))))
        med, mad = robust_scale(x, good)
        xs = (x - med) / mad
        freq, p = welch(xs, fs=FS, nperseg=FS * 4, axis=-1)
        p = p.mean(axis=1)  # average channels
        for t, name in enumerate(STAGES):
            m = good & (y == t)
            if not m.any():
                continue
            rel = p[m] / p[m][:, (freq >= 0.5) & (freq <= 30)].sum(axis=1, keepdims=True)
            sums[name]['psd'] = rel.sum(axis=0) if sums[name]['psd'] is None else sums[name]['psd'] + rel.sum(axis=0)
            sums[name]['n'] += int(m.sum())
            sums[name]['amp'].append(float(np.median(np.abs(xs[m]))))
    out = {'checks': {k: sorted(v) for k, v in checks.items()}, 'stages': {}}
    for name, s in sums.items():
        if not s['n']:
            continue
        psd = s['psd'] / s['n']
        band = (freq >= 8) & (freq <= 16)
        out['stages'][name] = {
            'epochs': s['n'],
            'median_abs_scaled_amplitude': float(np.median(s['amp'])),
            'peak_8_16_hz': float(freq[band][np.argmax(psd[band])]),
            'band_share': {b: float(psd[(freq >= lo) & (freq < hi)].sum()) for b, (lo, hi) in BANDS.items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    args = ap.parse_args()
    records = json.loads((ROOT / 'protocol.json').read_text())['records']
    boas = [r['subject'] for r in records if r['split'] == 'test' and r['confirmatory_night']]
    ext = sorted(json.loads((ROOT / 'results/v2/external_predictions.json').read_text()))
    report = {'boas_test': cohort_profile(args.boas_cache, boas),
              'sleep_edf': cohort_profile(args.ext_cache, ext)}
    flags = []
    for name in STAGES:
        a, b = report['boas_test']['stages'].get(name), report['sleep_edf']['stages'].get(name)
        if a and b and abs(a['peak_8_16_hz'] - b['peak_8_16_hz']) > 1.5:
            flags.append(f'{name}: 8-16 Hz peak at {a["peak_8_16_hz"]} Hz on BOAS vs {b["peak_8_16_hz"]} Hz on Sleep-EDF')
    for key in ['samples_per_epoch', 'onset_spacing_s']:
        if report['boas_test']['checks'][key] != report['sleep_edf']['checks'][key]:
            flags.append(f'{key} differs: {report["boas_test"]["checks"][key]} vs {report["sleep_edf"]["checks"][key]}')
    report['flags'] = flags or ['none']
    out = ROOT / 'results' / 'ablation_shift'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'a6_spectral_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['flags'], indent=2))


if __name__ == '__main__':
    main()

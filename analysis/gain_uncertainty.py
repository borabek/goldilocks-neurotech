"""How sure can we be about the v2 gain calibration? (exploratory, Sleep-EDF)

Two checks on the dataset-level rules that PREREGISTRATION.md carries forward (q75 primary,
q90 secondary):

1. Paired person-level bootstrap (10,000 resamples) of the change in kappa, macro-F1 and
   sleep-time error against uncalibrated v2, on the 20 Sleep-EDF people.
2. Split-half stability of the gain itself: estimate the dataset gain from a random half of the
   recordings, 1,000 times, and report the spread. DOD-H has 25 recordings, so this shows how
   much the gain can move when it's estimated from a dataset that small.

Usage: python analysis/gain_uncertainty.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from ablation_shift import percentile_log_sd, scale_features, v2_predict  # noqa: E402
from yasa_evaluate import score, windows  # noqa: E402

RESAMPLES = 10_000
SPLITS = 1_000
SEED = 20261126


def per_person(frozen, zs, gain):
    out = []
    for z in zs:
        zz = dict(z)
        zz['X'] = scale_features(z['X'], gain)
        y, good, onset = z['y'], z['good'].astype(bool), z['onset'].astype(float)
        s = score(y, good, onset, v2_predict(frozen, zz), windows(y, onset)['primary_complete_recording'])
        out.append([s['kappa'], s['macro_f1'], s['tst_error_minutes']])
    return np.array(out)


def main():
    rng = np.random.default_rng(SEED)
    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    records = json.loads((ROOT / 'protocol.json').read_text())['records']
    fit = [dict(np.load(ROOT / 'data/features' / f"{r['subject']}.npz")) for r in records if r['split'] == 'fit']
    zs = [dict(np.load(f)) for f in sorted((ROOT / 'data/external_sleepedf/features').glob('*.npz'))]
    base = per_person(frozen, zs, 1.0)
    report = {'status': 'EXPLORATORY: Sleep-EDF people were inspected at the v2 evaluation',
              'resamples': RESAMPLES, 'split_half_repeats': SPLITS, 'seed': SEED, 'rules': {}}
    for q in [75, 90]:
        ref = float(np.median([percentile_log_sd(z, q) for z in fit]))
        stats = np.array([percentile_log_sd(z, q) for z in zs])
        gain = float(np.exp(ref - np.median(stats)))
        diff = per_person(frozen, zs, gain) - base
        idx = rng.integers(0, len(zs), size=(RESAMPLES, len(zs)))
        boot = diff[idx].mean(axis=1)
        ci = {name: {'mean_change': float(diff[:, i].mean()),
                     'ci95': [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]}
              for i, name in enumerate(['kappa', 'macro_f1', 'tst_error_minutes'])}
        halves = []
        for _ in range(SPLITS):
            perm = rng.permutation(len(zs))
            a = perm[: len(zs) // 2]
            halves.append(float(np.exp(ref - np.median(stats[a]))))
        report['rules'][f'dataset_q{q}'] = {
            'gain_full': gain,
            'paired_change_vs_uncalibrated': ci,
            'gain_from_half_the_recordings': {'median': float(np.median(halves)),
                                              'p2_5': float(np.percentile(halves, 2.5)),
                                              'p97_5': float(np.percentile(halves, 97.5))}}
        print(json.dumps({'rule': f'q{q}', 'gain': round(gain, 2),
                          'kappa_change': [round(ci['kappa']['mean_change'], 3)] + [round(v, 3) for v in ci['kappa']['ci95']],
                          'macro_f1_change': [round(ci['macro_f1']['mean_change'], 3)] + [round(v, 3) for v in ci['macro_f1']['ci95']],
                          'tst_change': [round(ci['tst_error_minutes']['mean_change'], 1)] + [round(v, 1) for v in ci['tst_error_minutes']['ci95']],
                          'half_gain_95': [round(report['rules'][f'dataset_q{q}']['gain_from_half_the_recordings'][k], 2)
                                           for k in ['p2_5', 'p97_5']]}), flush=True)
    out = ROOT / 'results' / 'ablation_shift'
    (out / 'gain_uncertainty.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

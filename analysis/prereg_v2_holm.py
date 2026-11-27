"""Holm correction for the six prereg-v2 wake tests (addendum prereg-v2.1, item 2).

Reads the per-person rows saved by analysis/prereg_v2_eval.py, so nothing is re-predicted.
Two-sided bootstrap p-value: twice the smaller tail of the resampled mean changes, with the
same resampling scheme and seed as prereg-v2.

Usage: python analysis/prereg_v2_holm.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESAMPLES, SEED = 10_000, 20261126


def boot_p(d, rng):
    boot = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    tail = min(np.mean(boot <= 0), np.mean(boot >= 0))
    return float(min(1.0, 2 * max(tail, 1 / RESAMPLES))), float(d.mean())


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    src = json.loads((ROOT / 'results/prereg_v2/prereg_v2_results.json').read_text())
    rows = src['per_person']
    rng = np.random.default_rng(SEED)
    tests = []
    for new in ['cnn_lowpass16', 'cnn_hfdrop']:
        for ds in ['sleep_edf', 'dodh', 'hmc']:
            a = np.array([r['wake_recall'] for r in rows[ds]['cnn']], float)
            b = np.array([r['wake_recall'] for r in rows[ds][new]], float)
            d = (b - a)[np.isfinite(b - a)]
            p, m = boot_p(d, rng)
            tests.append({'model': new, 'dataset': ds, 'mean_change': m, 'p_two_sided': p,
                          'uncorrected_verdict': src['verdicts'][new][ds]['wake_prediction']})
    adj = holm(np.array([t['p_two_sided'] for t in tests]))
    for t, a in zip(tests, adj):
        t['p_holm'] = float(a)
        t['holm_significant_0.05'] = bool(a < 0.05 and t['mean_change'] > 0)
    out = {'addendum': 'PREREGISTRATION.md, prereg-v2.1 item 2', 'resamples': RESAMPLES,
           'seed': SEED, 'tests': tests}
    (ROOT / 'results/prereg_v2_1').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/prereg_v2_1/holm.json').write_text(json.dumps(out, indent=2))
    for t in tests:
        print(t['model'], t['dataset'], round(t['mean_change'], 3), t['p_two_sided'], round(t['p_holm'], 4),
              t['holm_significant_0.05'])


if __name__ == '__main__':
    main()

"""How much of the gap between variants is noise?

Every CNN number in this project comes from a single seed. The context model reads 0.618 macro-F1 and
the one on the square-root-balanced trunk reads 0.607; if retraining the same head with a different
seed moves it by that much, the ordering is not a finding and the plan must not present it as one.

The trunk is frozen, so only the temporal head is retrained — about 30 seconds a seed. The head's own
seed is all that changes: same trunk, same fitting people, same six selection people choosing the
number of passes and the sequence weight, same six calibration people fitting the temperature, same
mask and metric on the same 20 held-out people.

This cannot measure trunk variance, which needs full retraining at about 25 minutes a seed. The head
is the cheaper half and it is the half that differs between the context variants, so it is the right
thing to measure first. Stated as a limit, not glossed over.

Usage: python seed_variance.py <boas_cache> [trunk] [n_seeds]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cnn_context as cx
from pipeline import ROOT

OUT = ROOT / 'results/seed_variance'


def main(cache, trunk='results/cnn', n=5):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base_seed = cx.SEED
    runs = []
    for i in range(n):
        cx.SEED = base_seed + i                       # the head's initialisation and shuffling only
        cx.main(cache, trunk, 20)
        m = json.loads((ROOT / 'results/cnn_context' / 'context_metrics.json').read_text()) \
            if trunk == 'results/cnn' else \
            json.loads((ROOT / f"results/cnn_context_{Path(trunk).name.replace('cnn_', '')}"
                        / 'context_metrics.json').read_text())
        r = m['context_results']
        runs.append({'seed': cx.SEED, 'selected_pass': m['frozen']['selected_pass'],
                     'sequence_weight': m['frozen']['selected_sequence_weight'],
                     **{k: r[k] for k in ['kappa', 'macro_f1', 'accuracy', 'tst_error_minutes']}})
        print(json.dumps(runs[-1]), flush=True)
    cx.SEED = base_seed

    df = pd.DataFrame(runs)
    stats = {k: {'mean': float(df[k].mean()), 'sd': float(df[k].std(ddof=1)),
                 'min': float(df[k].min()), 'max': float(df[k].max()),
                 'spread': float(df[k].max() - df[k].min())}
             for k in ['kappa', 'macro_f1', 'accuracy', 'tst_error_minutes']}
    gaps = {'context_vs_context_sqrtbalanced_macro_f1': None}
    a = ROOT / 'results/cnn_context/context_metrics.json'
    b = ROOT / 'results/cnn_context_sqrtbalanced/context_metrics.json'
    if a.exists() and b.exists():
        ga = json.loads(a.read_text())['context_results']['macro_f1']
        gb = json.loads(b.read_text())['context_results']['macro_f1']
        gaps['context_vs_context_sqrtbalanced_macro_f1'] = float(ga - gb)
    verdict = None
    if gaps['context_vs_context_sqrtbalanced_macro_f1'] is not None:
        verdict = ('the gap between those two context variants is smaller than the head-seed spread, so it '
                   'is not a finding'
                   if abs(gaps['context_vs_context_sqrtbalanced_macro_f1']) < stats['macro_f1']['spread']
                   else 'the gap between those two context variants is larger than the head-seed spread')
    report = {'what_varies': 'the temporal head\'s seed only; the trunk is frozen and identical across runs',
              'trunk': trunk, 'seeds': n, 'runs': runs, 'statistics': stats,
              'observed_gaps': gaps, 'verdict': verdict,
              'not_measured': 'trunk variance, which needs full retraining at about 25 minutes a seed',
              'created_unix': time.time(), 'seconds': round(time.time() - t0, 1)}
    (OUT / 'seed_variance.json').write_text(json.dumps(report, indent=2))
    df.to_csv(OUT / 'seed_variance.csv', index=False)
    print(json.dumps({'statistics': stats, 'verdict': verdict}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'results/cnn',
         int(sys.argv[3]) if len(sys.argv) > 3 else 5)

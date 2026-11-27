"""Evaluate the two prereg-v2 CNNs once, against the original plain CNN (PREREGISTRATION.md,
addendum prereg-v2). Pre-specified, but DOD-H and HMC were already scored for the frozen models,
so the external numbers are not blind.

  1. BOAS test: kappa drop of at most 0.03 (mean person kappa, as in cnn_metrics.json).
  2. Wake recall up on Sleep-EDF, DOD-H and HMC: supported where the paired 95% bootstrap
     interval of the per-person change is entirely above 0.
  3. Secondary: kappa change on the three external datasets, with intervals.

Usage: python analysis/prereg_v2_eval.py --boas-cache B --ext-cache E --new-cache /tmp/goldilocks_external_cache
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cnn_challenger as cc  # noqa: E402
from cnn_challenger import lowpass16, night_probabilities, rebuild_free, robust_scale, CLIP  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402
from yasa_evaluate import score, windows  # noqa: E402

MODELS = ['cnn', 'cnn_lowpass16', 'cnn_hfdrop']
RESAMPLES, SEED = 10_000, 20261126


def cohort_items(args):
    ext = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    items = {'sleep_edf': [(s, Path(args.ext_cache) / f'{s}.npz', np.array(ext[s]['reference'])) for s in sorted(ext)]}
    for ds in ['dodh', 'hmc']:
        lab = json.loads((ROOT / f'results/external_{ds}/predictions/labels_and_grid.json').read_text())
        items[ds] = [(r, Path(args.new_cache) / ds / f'{r}.npz', np.array(lab[r]['reference'])) for r in sorted(lab)]
    return items


def per_person(name, items, trans, prior):
    src = ROOT / 'results' / name
    freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
    params = rebuild_free(src / 'cnn_weights.json')
    cc.TWOCHANNEL = False
    out = []
    for rec, path, y in items:
        z = np.load(path)
        x = z['X'].astype('float32')
        if name == 'cnn_lowpass16':
            x = lowpass16(x)
        good, onset = z['good'].astype(bool), z['onset'].astype(float)
        med, mad = robust_scale(x, good)
        p = night_probabilities(params, np.clip((x - med) / mad, -CLIP, CLIP), good, onset, trans, prior,
                                freeze['selected_sequence_weight'], freeze['temperature'])
        g = p.argmax(axis=1)
        s = score(y, good, onset, g, windows(y, onset)['primary_complete_recording'])
        m = good & (y == 0)
        out.append({'recording': rec, 'kappa': s['kappa'], 'macro_f1': s['macro_f1'],
                    'wake_recall': float(np.mean(g[m] == 0)) if m.any() else float('nan'),
                    'tst_error_minutes': s['tst_error_minutes']})
    return out


def paired(a, b, key, rng):
    d = np.array([x[key] for x in b], dtype=float) - np.array([x[key] for x in a], dtype=float)
    d = d[np.isfinite(d)]
    boot = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    return {'mean_change': float(d.mean()), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    ap.add_argument('--new-cache', required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    trans, prior = fit_trans_prior(args.boas_cache)
    boas = {m: json.loads((ROOT / 'results' / m / 'cnn_metrics.json').read_text())['cnn_results'] for m in MODELS}
    report = {'preregistration': 'PREREGISTRATION.md, addendum prereg-v2',
              'status': 'Pre-specified; DOD-H and HMC were already scored for the frozen models, so not blind',
              'boas_test': {m: boas[m] for m in MODELS}, 'external': {}, 'verdicts': {}}
    items = cohort_items(args)
    rows = {ds: {m: per_person(m, it, trans, prior) for m in MODELS} for ds, it in items.items()}
    for new in ['cnn_lowpass16', 'cnn_hfdrop']:
        drop = boas['cnn']['kappa'] - boas[new]['kappa']
        v = {'boas_kappa_drop': float(drop), 'boas_prediction_holds': bool(drop <= 0.03)}
        for ds in items:
            w = paired(rows[ds]['cnn'], rows[ds][new], 'wake_recall', rng)
            k = paired(rows[ds]['cnn'], rows[ds][new], 'kappa', rng)
            v[ds] = {'wake_recall_change': w, 'wake_prediction': 'supported' if w['ci95'][0] > 0 else (
                'contradicted' if w['ci95'][1] < 0 else 'not supported'), 'kappa_change': k}
        report['verdicts'][new] = v
    for ds in items:
        report['external'][ds] = {m: {k: float(np.nanmean([r[k] for r in rows[ds][m]]))
                                      for k in ['kappa', 'macro_f1', 'wake_recall', 'tst_error_minutes']}
                                  for m in MODELS}
    out = ROOT / 'results/prereg_v2'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prereg_v2_results.json').write_text(json.dumps({**report, 'per_person': rows}, indent=2))
    print(json.dumps({'boas_test': {m: round(boas[m]['kappa'], 3) for m in MODELS},
                      'external': {ds: {m: {k: round(v, 3) for k, v in r.items()} for m, r in d.items()}
                                   for ds, d in report['external'].items()},
                      'verdicts': report['verdicts']}, indent=1))


if __name__ == '__main__':
    main()

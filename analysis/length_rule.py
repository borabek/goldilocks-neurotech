"""A recording-length rule for the sleep-window detector (PREREGISTRATION.md, prereg-v2.1, item 4).

The frozen detector (W = 61, G = 60, from results/sleep_window/sleep_window_detector.json) is
applied only to recordings longer than T hours. T is fixed before any test set is touched, from
lengths alone: the midpoint between the longest BOAS fitting night and the shortest Sleep-EDF
development recording. It is then applied once to BOAS test, Sleep-EDF evaluation, DOD-H and HMC
for v2, the plain CNN and YASA. DOD-H and HMC have been seen, so the rule is not blind on them.

Usage: python analysis/length_rule.py --boas-cache B
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'analysis'))
from sleep_window_detector import CACHE as DEV_CACHE, detector  # noqa: E402
from yasa_evaluate import score, summarize, windows  # noqa: E402


def hours(n_epochs):
    return n_epochs * 30 / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    args = ap.parse_args()
    frozen = json.loads((ROOT / 'results/sleep_window/sleep_window_detector.json').read_text())['frozen_on_dev']
    w, g = frozen['W_epochs'], frozen['G_epochs']

    # T from lengths outside every test set.
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    fit = [r['subject'] for r in rows if r['split'] == 'fit']
    boas_fit_h = [hours(len(np.load(Path(args.boas_cache) / f'{s}.npz')['y'])) for s in fit]
    dev_h = [hours(len(np.load(f)['y'])) for f in sorted(DEV_CACHE.glob('SC4*.npz'))]
    assert len(fit) == 59 and len(dev_h) == 56, (len(fit), len(dev_h))
    t = (max(boas_fit_h) + min(dev_h)) / 2
    choice = {'longest_boas_fitting_night_h': max(boas_fit_h), 'shortest_sleepedf_dev_recording_h': min(dev_h),
              'T_hours': t, 'W_epochs': w, 'G_epochs': g}
    print(json.dumps(choice), flush=True)

    # Test sets: (reference, good, onset, {model: prediction}).
    sets = {}
    hist = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    cnn_boas = json.loads((ROOT / 'results/cnn/cnn_predictions.json').read_text())
    yasa = json.loads((ROOT / 'results/yasa/yasa_predictions.json').read_text())
    sets['boas'] = {s: (r['reference'], r['good'], r['onset'],
                        {'v2': r['model'], 'cnn': cnn_boas[s], 'yasa': yasa['historical_boas'][s]})
                    for s, r in hist.items()}
    ext = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    cnn_ext = json.loads((ROOT / 'results/cnn/cnn_external_predictions.json').read_text())
    sets['sleep_edf'] = {s: (r['reference'], r['good'], r['onset'],
                             {'v2': r['model'], 'cnn': cnn_ext[s], 'yasa': yasa['external'][s]}) for s, r in ext.items()}
    for ds in ['dodh', 'hmc']:
        folder = ROOT / f'results/external_{ds}/predictions'
        lab = json.loads((folder / 'labels_and_grid.json').read_text())
        preds = json.loads((folder / 'predictions.json').read_text())
        sets[ds] = {r: (v['reference'], v['good'], v['onset'], {m: preds[m][r] for m in ['v2', 'cnn', 'yasa']})
                    for r, v in lab.items()}

    out = {'addendum': 'PREREGISTRATION.md, prereg-v2.1 item 4', 'rule': choice, 'datasets': {}}
    for ds, recs in sets.items():
        lengths = {r: hours(len(v[0])) for r, v in recs.items()}
        applied = sorted(r for r, h in lengths.items() if h > t)
        res = {'recordings': len(recs), 'detector_applied_to': len(applied),
               'length_h_range': [min(lengths.values()), max(lengths.values())], 'models': {}}
        for m in ['v2', 'cnn', 'yasa']:
            without, never, rule, always = [], [], [], []
            for r, (y, gd, on, preds) in sorted(recs.items()):
                y, gd, on, p = np.array(y), np.array(gd).astype(bool), np.array(on, float), np.array(preds[m])
                win = windows(y, on)['primary_complete_recording']
                s0 = score(y, gd, on, p, win)
                s1 = score(y, gd, on, detector(p, w, g), win)
                without.append(s0); always.append(s1); rule.append(s1 if r in applied else s0)
            res['models'][m] = {k: {q: summarize(v)[q] for q in ['kappa', 'macro_f1', 'tst_error_minutes']}
                                for k, v in [('no_detector', without), ('detector_always', always),
                                             ('length_rule', rule)]}
        out['datasets'][ds] = res
        print(ds, res['detector_applied_to'], '/', res['recordings'],
              {m: {k: round(v['tst_error_minutes'], 1) for k, v in d.items()} for m, d in res['models'].items()},
              flush=True)
    (ROOT / 'results/prereg_v2_1').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/prereg_v2_1/length_rule.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

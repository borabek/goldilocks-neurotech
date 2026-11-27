"""A sleep-window detector for the 236-minute error (PREREGISTRATION.md, section 7; exploratory).

On Sleep-EDF's ~23-hour recordings v2 calls some daytime epochs sleep, and those add up to a
236-minute sleep-time error. The detector keeps only the main sleep period and calls everything
outside it wake:

  1. smooth the model's own sleep/wake calls with a centred moving average over W epochs,
  2. find the runs of smoothed sleep and merge runs separated by gaps of at most G epochs,
  3. keep the longest merged run; every epoch outside it becomes wake.

It uses only the model's predictions, never labels. W and G are chosen on the development
recordings (Sleep-EDF SC subjects 20 and up, first night, 56 recordings) by mean absolute
sleep-time error, frozen, and then applied once to the 20 evaluation people (subjects 00-19).
Those 20 were already inspected, so the result stays exploratory however well it does.

Usage: python analysis/sleep_window_detector.py --dev-dir D
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from ablation_shift import v2_predict  # noqa: E402
from pipeline import epoch_features  # noqa: E402
from yasa_evaluate import score, summarize, windows  # noqa: E402

W_GRID = [1, 5, 11, 21, 41, 61]
G_GRID = [0, 20, 60, 120, 240]
CACHE = Path('/tmp/goldilocks_sleepedf_dev_features')


def detector(pred, w, g):
    sleep = ((pred > 0) & (pred < 5)).astype(float)
    smooth = np.convolve(sleep, np.ones(w) / w, mode='same') > 0.5 if w > 1 else sleep > 0.5
    runs, start = [], None
    for i, s in enumerate(np.append(smooth, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append([start, i])
            start = None
    if not runs:
        return np.where(pred < 5, 0, pred)
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= g:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    a, b = max(merged, key=lambda r: r[1] - r[0])
    out = pred.copy()
    outside = np.ones(len(pred), dtype=bool)
    outside[a:b] = False
    out[outside & (pred >= 0) & (pred < 5)] = 0
    return out


def dev_features(dev_dir):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in sorted(Path(dev_dir).glob('SC4*.npz')):
        c = CACHE / f.name
        if not c.exists():
            z = np.load(f)
            sig = z['signal'].astype('float64')
            n = len(z['y'])
            ep = sig[:n * 3000].reshape(n, 1, 3000)
            good = np.isfinite(ep).all(axis=(1, 2)) & (np.std(ep, axis=-1).min(axis=1) > 1e-6)
            feats = np.concatenate([epoch_features(resample_poly(np.nan_to_num(ep[i:i + 64]), up=64, down=25, axis=-1), 256)
                                    for i in range(0, n, 64)])
            np.savez(c, X=feats, y=z['y'], good=good, onset=z['onset'])
        out[f.stem] = dict(np.load(c))
    return out


def tst_error(items, w, g):
    errs, rows = [], []
    for y, good, onset, pred in items:
        p = detector(pred, w, g)
        s = score(y, good, onset, p, windows(y, onset)['primary_complete_recording'])
        errs.append(abs(s['signed_tst_error_minutes']))
        rows.append(s)
    return float(np.mean(errs)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev-dir', required=True)
    args = ap.parse_args()
    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')

    dev = dev_features(args.dev_dir)
    dev_items = [(z['y'], z['good'].astype(bool), z['onset'].astype(float), v2_predict(frozen, z)) for z in dev.values()]
    grid = {}
    for w in W_GRID:
        for g in G_GRID:
            grid[f'W{w}_G{g}'] = tst_error(dev_items, w, g)[0]
    best = min(grid, key=grid.get)
    w_best, g_best = int(best.split('_')[0][1:]), int(best.split('_')[1][1:])
    baseline_dev = float(np.mean([abs(score(y, gd, on, p, windows(y, on)['primary_complete_recording'])['signed_tst_error_minutes'])
                                  for y, gd, on, p in dev_items]))
    frozen_choice = {'W_epochs': w_best, 'G_epochs': g_best, 'dev_mean_abs_tst_error_min': grid[best],
                     'dev_without_detector_min': baseline_dev, 'dev_recordings': len(dev_items),
                     'chosen_unix': time.time()}
    print(json.dumps({'frozen_on_dev': frozen_choice}), flush=True)

    # Applied once to the 20 evaluation people, with each model's own predictions.
    ext = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    models = {'v2': {s: r['model'] for s, r in ext.items()},
              'cnn': json.loads((ROOT / 'results/cnn/cnn_external_predictions.json').read_text())}
    yasa = json.loads((ROOT / 'results/yasa/yasa_predictions.json').read_text())
    yasa = yasa.get('external', yasa)
    if isinstance(next(iter(yasa.values())), dict):
        yasa = {s: r.get('model', r.get('prediction')) for s, r in yasa.items()}
    if set(yasa) == set(ext):
        models['yasa'] = yasa
    test = {}
    for name, pred in models.items():
        items = [(np.array(ext[s]['reference']), np.array(ext[s]['good']).astype(bool),
                  np.array(ext[s]['onset'], dtype=float), np.array(pred[s])) for s in sorted(ext)]
        _, with_rows = tst_error(items, w_best, g_best)
        without_rows = [score(y, gd, on, p, windows(y, on)['primary_complete_recording']) for y, gd, on, p in items]
        test[name] = {'without_detector': summarize(without_rows), 'with_detector': summarize(with_rows)}
        print(json.dumps({name: {k: {m: round(v[m], 3) for m in ['kappa', 'macro_f1', 'tst_error_minutes']}
                                 for k, v in test[name].items()}}), flush=True)

    # The same frozen detector on the night-only datasets, where it should change little.
    night = {}
    for ds in ['dodh', 'hmc']:
        folder = ROOT / f'results/external_{ds}/predictions'
        lab = json.loads((folder / 'labels_and_grid.json').read_text())
        preds = json.loads((folder / 'predictions.json').read_text())
        night[ds] = {}
        for name in ['v2', 'cnn', 'yasa']:
            items = [(np.array(lab[r]['reference']), np.array(lab[r]['good']).astype(bool),
                      np.array(lab[r]['onset'], dtype=float), np.array(preds[name][r])) for r in sorted(lab)]
            _, with_rows = tst_error(items, w_best, g_best)
            without_rows = [score(y, gd, on, p, windows(y, on)['primary_complete_recording']) for y, gd, on, p in items]
            night[ds][name] = {'without_detector': summarize(without_rows), 'with_detector': summarize(with_rows)}
    out = ROOT / 'results/sleep_window'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sleep_window_detector.json').write_text(json.dumps({
        'status': 'EXPLORATORY (PREREGISTRATION.md section 7): tuned on 56 development recordings, applied once '
                  'to the 20 already-inspected Sleep-EDF people',
        'frozen_on_dev': frozen_choice, 'dev_grid_mean_abs_tst_error_min': grid,
        'sleep_edf_20': test, 'night_only_datasets': night}, indent=2))


if __name__ == '__main__':
    main()

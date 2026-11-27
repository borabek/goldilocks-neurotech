"""Error analysis and calibration (PREREGISTRATION.md, addendum prereg-v2.1, item 5). Descriptive.

1. Row-normalised confusion matrices (epochs pooled over people, complete recording, good epochs)
   for v2, YASA, the plain CNN and cnn_hfdrop on BOAS test, Sleep-EDF evaluation, DOD-H and HMC.
2. Per-stage F1 for v2 (mean over people) on HMC against DOD-H, plus the stage shares v2 predicts
   against the reference, to see where H1 failed on HMC. H1 compares v2 + q75 gain calibration with
   uncalibrated v2, so the comparison that actually locates the failure is v2_gain_q75 against v2
   on DOD-H and HMC; both are reported (the second was added when writing the script).
3. Expected calibration error for the two CNNs: 10 equal-width bins on the top probability,
   epochs pooled. The CNN probabilities come from /tmp/goldilocks_probs, written by
   analysis/band_occlusion.py.

Usage: python analysis/error_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
PROBS = Path('/tmp/goldilocks_probs')
DATASETS = ['boas', 'sleep_edf', 'dodh', 'hmc']


def saved_predictions():
    """{dataset: {recording: (reference, good, {model: prediction})}} for v2 and YASA."""
    yasa = json.loads((ROOT / 'results/yasa/yasa_predictions.json').read_text())
    hist = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    ext = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    out = {'boas': {s: (r['reference'], r['good'], {'v2': r['model'], 'yasa': yasa['historical_boas'][s]})
                    for s, r in hist.items()},
           'sleep_edf': {s: (r['reference'], r['good'], {'v2': r['model'], 'yasa': yasa['external'][s]})
                         for s, r in ext.items()}}
    for ds in ['dodh', 'hmc']:
        folder = ROOT / f'results/external_{ds}/predictions'
        lab = json.loads((folder / 'labels_and_grid.json').read_text())
        preds = json.loads((folder / 'predictions.json').read_text())
        out[ds] = {r: (v['reference'], v['good'], {m: preds[m][r] for m in ['v2', 'v2_gain_q75', 'yasa']})
                   for r, v in lab.items()}
    for ds, recs in out.items():
        for r, (y, good, models) in recs.items():
            for m in ['cnn', 'cnn_hfdrop']:
                z = np.load(PROBS / m / ds / f'{r}.npz')
                assert len(z['y']) == len(y) and np.array_equal(z['y'], np.array(y)), (ds, r)
                models[m] = z['p'].argmax(axis=1).tolist()
    return out


def confusion(recs, model):
    c = np.zeros((5, 5))
    for y, good, preds in recs.values():
        y, good, p = np.array(y), np.array(good).astype(bool), np.array(preds[model])
        m = good & (y < 5) & (p >= 0) & (p < 5)
        np.add.at(c, (y[m], p[m]), 1)
    rows = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
    return {'counts': c.astype(int).tolist(), 'row_normalised': rows.round(4).tolist(),
            'reference_share': (c.sum(axis=1) / c.sum()).round(4).tolist(),
            'predicted_share': (c.sum(axis=0) / c.sum()).round(4).tolist()}


def per_stage_f1(recs, model):
    per = []
    for y, good, preds in recs.values():
        y, good, p = np.array(y), np.array(good).astype(bool), np.array(preds[model])
        m = good & (y < 5)
        f = f1_score(y[m], p[m], labels=list(range(5)), average=None, zero_division=np.nan)
        present = np.isin(np.arange(5), y[m])
        per.append(np.where(present, f, np.nan))
    return dict(zip(STAGES, np.nanmean(np.array(per, float), axis=0).round(4).tolist()))


def ece(ds, model, recs, bins=10):
    conf, ok = [], []
    for r, (y, good, _) in recs.items():
        z = np.load(PROBS / model / ds / f'{r}.npz')
        y, good = np.array(y), np.array(good).astype(bool)
        m = good & (y < 5)
        p = z['p'][m]
        conf.append(p.max(axis=1)); ok.append(p.argmax(axis=1) == y[m])
    conf, ok = np.concatenate(conf), np.concatenate(ok)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = sum(abs(ok[idx == b].mean() - conf[idx == b].mean()) * np.mean(idx == b) for b in range(bins) if np.any(idx == b))
    return {'ece': float(e), 'mean_confidence': float(conf.mean()), 'accuracy': float(ok.mean()), 'epochs': int(len(conf))}


def main():
    data = saved_predictions()
    models = ['v2', 'yasa', 'cnn', 'cnn_hfdrop']
    report = {'addendum': 'PREREGISTRATION.md, prereg-v2.1 item 5', 'stages': STAGES,
              'confusion': {ds: {m: confusion(data[ds], m) for m in models} for ds in DATASETS},
              'v2_per_stage_f1': {ds: per_stage_f1(data[ds], 'v2') for ds in DATASETS},
              'cnn_per_stage_f1': {ds: per_stage_f1(data[ds], 'cnn') for ds in DATASETS},
              'h1_v2_gain_q75_vs_v2': {ds: {'per_stage_f1_v2': per_stage_f1(data[ds], 'v2'),
                                            'per_stage_f1_v2_gain_q75': per_stage_f1(data[ds], 'v2_gain_q75'),
                                            'confusion_v2_gain_q75': confusion(data[ds], 'v2_gain_q75')}
                                       for ds in ['dodh', 'hmc']},
              'calibration': {ds: {m: ece(ds, m, data[ds]) for m in ['cnn', 'cnn_hfdrop']} for ds in DATASETS}}
    (ROOT / 'results/prereg_v2_1').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/prereg_v2_1/error_analysis.json').write_text(json.dumps(report, indent=2))
    for ds in DATASETS:
        print('==', ds)
        for m in models:
            c = report['confusion'][ds][m]
            print(f'  {m:11s} diag', [round(c['row_normalised'][i][i], 2) for i in range(5)],
                  'W->', [round(v, 2) for v in c['row_normalised'][0]], 'pred share', [round(v, 2) for v in c['predicted_share']])
        print('  ref share', [round(v, 2) for v in report['confusion'][ds]['v2']['reference_share']])
        print('  v2 F1', report['v2_per_stage_f1'][ds], 'cnn F1', report['cnn_per_stage_f1'][ds])
        if ds in report['h1_v2_gain_q75_vs_v2']:
            h = report['h1_v2_gain_q75_vs_v2'][ds]
            c = h['confusion_v2_gain_q75']
            print('  q75 F1', h['per_stage_f1_v2_gain_q75'], 'diag', [round(c['row_normalised'][i][i], 2) for i in range(5)],
                  'pred share', [round(v, 2) for v in c['predicted_share']])
        print('  ECE', {m: {k: round(v, 3) for k, v in d.items()} for m, d in report['calibration'][ds].items()})


if __name__ == '__main__':
    main()

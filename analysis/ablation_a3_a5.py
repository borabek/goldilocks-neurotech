"""A3 (recording length) and A5 (labelling scheme), from saved predictions (exploratory).

A3: score the same Sleep-EDF predictions inside three windows, with nothing re-run:
    whole recording (primary), a label-free night window (first to last epoch the frozen v2
    model calls sleep, +-30 min) and the reference night window (+-30 min; uses labels, an
    oracle). The label-free window is what a deployed system could actually use.
A5: the labelling scheme. Sleep-EDF is scored with R&K (stages 3 and 4 merged into N3 here);
    BOAS, DOD-H and HMC use AASM. Reports stage prevalence among scored epochs per cohort, and
    each model's N3 F1 and where reference N3 goes.

Usage: python analysis/ablation_a3_a5.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yasa_evaluate import score, summarize, windows  # noqa: E402

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def sleep_edf_models():
    v2 = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    models = {'v2': {s: r['model'] for s, r in v2.items()},
              'cnn': json.loads((ROOT / 'results/cnn/cnn_external_predictions.json').read_text()),
              'ensemble_of_7': json.loads((ROOT / 'results/external_all/ensemble_of_7_external_predictions.json').read_text())}
    yasa = json.loads((ROOT / 'results/yasa/yasa_predictions.json').read_text())
    if 'external' in yasa:
        yasa = yasa['external']
    if isinstance(next(iter(yasa.values())), dict):
        yasa = {s: r.get('model', r.get('prediction')) for s, r in yasa.items()}
    if set(yasa) == set(v2):
        models['yasa'] = yasa
    return v2, models


def a3():
    v2, models = sleep_edf_models()
    out = {}
    for name, pred in models.items():
        rows = {'whole_recording': [], 'label_free_v2_night_window': [], 'oracle_reference_window': []}
        for s, rec in v2.items():
            y = np.array(rec['reference'])
            good = np.array(rec['good']).astype(bool)
            onset = np.array(rec['onset'], dtype=float)
            g = np.array(pred[s])
            v2g = np.array(rec['model'])
            sl = onset[(v2g > 0) & (v2g < 5)]
            lf = (onset >= sl.min() - 1800) & (onset <= sl.max() + 1800) & (y < 5)
            w = windows(y, onset)
            rows['whole_recording'].append(score(y, good, onset, g, w['primary_complete_recording']))
            rows['label_free_v2_night_window'].append(score(y, good, onset, g, lf))
            rows['oracle_reference_window'].append(score(y, good, onset, g, w['secondary_reference_window']))
        out[name] = {}
        for k, r in rows.items():
            s = summarize(r)
            s['wake_f1'] = float(np.mean([x['per_stage_f1'][0] for x in r]))
            out[name][k] = s
    return out


def prevalence(y):
    y = np.asarray(y)
    m = y < 5
    return {st: round(float(np.mean(y[m] == i)), 3) for i, st in enumerate(STAGES)}


def n3_detail(ref, preds):
    out = {}
    for name, pred in preds.items():
        to = np.zeros(6)
        f1 = []
        for s in ref:
            y, g = np.array(ref[s]), np.array(pred[s])
            m = y == 3
            to += np.bincount(np.clip(g[m], -1, 4) + 1, minlength=6)
            tp, fp, fn = np.sum(m & (g == 3)), np.sum(~m & (y < 5) & (g == 3)), np.sum(m & (g != 3))
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        out[name] = {'n3_f1_person_mean': float(np.mean(f1)),
                     'reference_n3_predicted_as': {k: round(float(v), 3) for k, v in
                                                   zip(['abstain'] + STAGES, to / max(to.sum(), 1))}}
    return out


def a5():
    v2b = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    v2e, models = sleep_edf_models()
    report = {'stage_prevalence': {
        'boas_test (AASM)': prevalence(np.concatenate([r['reference'] for r in v2b.values()])),
        'sleep_edf (R&K, 3+4 -> N3)': prevalence(np.concatenate([r['reference'] for r in v2e.values()]))}}
    for name in ['dodh', 'hmc']:
        lab = json.loads((ROOT / f'results/external_{name}/predictions/labels_and_grid.json').read_text())
        report['stage_prevalence'][f'{name} (AASM)'] = prevalence(np.concatenate([r['reference'] for r in lab.values()]))
    report['sleep_edf_n3'] = n3_detail({s: r['reference'] for s, r in v2e.items()}, models)
    report['boas_test_n3'] = n3_detail({s: r['reference'] for s, r in v2b.items()},
                                       {'v2': {s: r['model'] for s, r in v2b.items()}})
    return report


def main():
    report = {'status': 'EXPLORATORY: Sleep-EDF people were inspected at the v2 evaluation; descriptive only',
              'A3_recording_length': a3(), 'A5_labelling_scheme': a5()}
    out = ROOT / 'results/ablation_shift'
    (out / 'a3_a5.json').write_text(json.dumps(report, indent=2))
    for name, r in report['A3_recording_length'].items():
        print(name, {k: (round(v['kappa'], 3), round(v['macro_f1'], 3), round(v['tst_error_minutes'], 1))
                     for k, v in r.items()})
    print(json.dumps(report['A5_labelling_scheme']['stage_prevalence'], indent=1))
    for name, r in report['A5_labelling_scheme']['sleep_edf_n3'].items():
        print('sleep_edf', name, round(r['n3_f1_person_mean'], 3), r['reference_n3_predicted_as'])
    print('boas v2', report['A5_labelling_scheme']['boas_test_n3'])


if __name__ == '__main__':
    main()

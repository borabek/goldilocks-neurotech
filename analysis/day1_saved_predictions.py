"""Day 1 checks that only need the saved predictions already in results/.

Everything here is exploratory. The 20 Sleep-EDF people were inspected at the v2 evaluation,
so none of this counts as a fresh test. It is here to decide which ablations are worth running
and what to write into the pre-registration before DOD-H and HMC are opened.

Outputs (research_outputs/day1/):
  recording_composition.csv   hours and wake share per recording, BOAS test nights vs Sleep-EDF
  external_confusion.csv      row-normalised confusion on Sleep-EDF for v2, YASA, CNN, ensemble
  wake_recall_by_window.csv   wake recall inside vs outside the reference night window
  day1_summary.json           the headline numbers quoted in PREREGISTRATION.md

Run:  python analysis/day1_saved_predictions.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yasa_evaluate import windows  # noqa: E402  same window definition as every other result

OUT = ROOT / 'research_outputs' / 'day1'
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def load(path):
    return json.loads((ROOT / path).read_text())


def arrays(rec):
    y = np.array(rec['reference'])
    good = np.array(rec['good']).astype(bool)
    onset = np.array(rec['onset'], dtype=float)
    return y, good, onset


def composition(preds):
    rows = []
    for sub, rec in preds.items():
        y, good, _ = arrays(rec)
        m = good & (y < 5)
        rows.append({'recording': sub, 'hours': round(len(y) * 0.5 / 60, 2),
                     'wake_share': round(float(np.mean(y[m] == 0)), 3)})
    return rows


def confusion(reference, predicted):
    """Rows = reference stage, columns = predicted stage (abstentions counted separately)."""
    counts = np.zeros((5, 6))
    for sub, rec in reference.items():
        y, good, _ = arrays(rec)
        p = np.array(predicted[sub])
        for t in range(5):
            m = good & (y == t)
            counts[t, 0] += np.sum(p[m] < 0)
            for q in range(5):
                counts[t, q + 1] += np.sum(p[m] == q)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def wake_recall(reference, predicted):
    inside, outside = [], []
    for sub, rec in reference.items():
        y, good, onset = arrays(rec)
        night = windows(y, onset)['secondary_reference_window']
        p = np.array(predicted[sub])
        w = good & (y == 0)
        if (w & night).any():
            inside.append(float(np.mean(p[w & night] == 0)))
        if (w & ~night).any():
            outside.append(float(np.mean(p[w & ~night] == 0)))
    return float(np.mean(inside)), float(np.mean(outside))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2_ext = load('results/v2/external_predictions.json')
    v2_boas = load('results/v2/historical_boas_predictions.json')

    models = {
        'v2': {s: r['model'] for s, r in v2_ext.items()},
        'cnn': load('results/cnn/cnn_external_predictions.json'),
        'ensemble_of_7': load('results/external_all/ensemble_of_7_external_predictions.json'),
        'cnn_balanced': load('results/external_all/cnn_balanced_external_predictions.json'),
    }
    yasa_path = ROOT / 'results/yasa/yasa_predictions.json'
    if yasa_path.exists():
        yasa = json.loads(yasa_path.read_text())
        if 'external' in yasa:
            yasa = yasa['external']
        first = next(iter(yasa.values()))
        if isinstance(first, dict) and 'model' in first:
            yasa = {s: r['model'] for s, r in yasa.items()}
        if set(yasa) == set(v2_ext):
            models['yasa'] = yasa

    comp_boas, comp_ext = composition(v2_boas), composition(v2_ext)
    with open(OUT / 'recording_composition.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cohort', 'recording', 'hours', 'wake_share'])
        w.writeheader()
        for cohort, rows in [('boas_test', comp_boas), ('sleep_edf', comp_ext)]:
            for r in rows:
                w.writerow({'cohort': cohort, **r})

    with open(OUT / 'external_confusion.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'reference_stage', 'abstain'] + [f'pred_{s}' for s in STAGES])
        conf = {}
        for name, pred in models.items():
            c = confusion(v2_ext, pred)
            conf[name] = c
            for t, s in enumerate(STAGES):
                w.writerow([name, s] + [round(x, 3) for x in c[t]])

    recall = {name: wake_recall(v2_ext, pred) for name, pred in models.items()}
    with open(OUT / 'wake_recall_by_window.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'wake_recall_inside_night_window', 'wake_recall_outside_night_window'])
        for name, (a, b) in recall.items():
            w.writerow([name, round(a, 3), round(b, 3)])

    boas_models = {
        'v2': {s: r['model'] for s, r in v2_boas.items()},
        'cnn': load('results/cnn/cnn_predictions.json'),
        'ensemble_of_7': load('results/cnn_posthoc/ensemble_predictions.json'),
    }
    boas_wake_recall = {}
    for name, pred in boas_models.items():
        vals = []
        for sub, rec in v2_boas.items():
            y, good, _ = arrays(rec)
            m = good & (y == 0)
            vals.append(float(np.mean(np.array(pred[sub])[m] == 0)))
        boas_wake_recall[name] = round(float(np.mean(vals)), 3)

    def med(rows, key):
        return float(np.median([r[key] for r in rows]))

    summary = {
        'status': 'EXPLORATORY: Sleep-EDF people were inspected at the v2 evaluation',
        'boas_test_hours_median': med(comp_boas, 'hours'),
        'boas_test_wake_share_median': med(comp_boas, 'wake_share'),
        'sleep_edf_hours_median': med(comp_ext, 'hours'),
        'sleep_edf_wake_share_median': med(comp_ext, 'wake_share'),
        'share_of_epochs_predicted_N2_by_reference_stage': {
            name: {s: round(float(c[t, 3]), 3) for t, s in enumerate(STAGES)} for name, c in conf.items()},
        'boas_test_wake_recall': boas_wake_recall,
        'wake_recall_inside_vs_outside_night_window': {
            name: {'inside': round(a, 3), 'outside': round(b, 3)} for name, (a, b) in recall.items()},
    }
    (OUT / 'day1_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

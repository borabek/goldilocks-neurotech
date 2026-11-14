"""Revision 16 Change 3: label-free main-sleep-period rule on the frozen v2 build.

1. Verify inference: frozen v2 on two BOAS test people must equal their saved predictions.
2. Predict (inference only) for the 20 BOAS summary-calibration people.
3. Tune gap/margin on those 20 only; freeze parameters with a timestamp.
4. Apply frozen parameters to saved v2 (and YASA) predictions on both test cohorts.
   Sleep-EDF results are EXPLORATORY DEVELOPMENT: those people were already inspected.
"""
import hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np, joblib

GL = Path(sys.argv[1]); OUTDIR = Path(sys.argv[2]); OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(GL))
import v2_model  # noqa: E402

frozen = joblib.load(GL / 'results/v2/frozen_model.joblib')
freeze = json.loads((GL / 'results/v2/freeze_record.json').read_text())
assert hashlib.sha256((GL / 'results/v2/frozen_model.joblib').read_bytes()).hexdigest() == freeze['model_sha256']

def v2_guess(z):
    p = v2_model.probabilities(frozen['model'], z['X'], z['good'], frozen['transition'], frozen['prior'], frozen['weight'], frozen['temperature'], z['onset'])
    return p.argmax(axis=1)

saved_hist = json.loads((GL / 'results/v2/historical_boas_predictions.json').read_text())
for sub in ['sub-1', 'sub-6']:
    z = dict(np.load(GL / 'data/features' / f'{sub}.npz'))
    assert np.array_equal(v2_guess(z), np.array(saved_hist[sub]['model'])), f'{sub}: inference does not reproduce saved v2'
print('inference check: v2 reproduces saved predictions for sub-1 and sub-6')

GAP_MIN = [10, 20, 30, 45, 60, 90]; MARGIN_MIN = [0, 15, 30]

def main_sleep_period(pred, gap_min, margin_min):
    sleep = (pred > 0) & (pred < 5); n = len(pred); idx = np.flatnonzero(sleep); mask = np.zeros(n, bool)
    if not len(idx): return mask
    gap, margin = int(gap_min * 2), int(margin_min * 2); blocks, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > gap: blocks.append((start, prev)); start = i
        prev = i
    blocks.append((start, prev))
    b = max(blocks, key=lambda b: (sleep[b[0]:b[1] + 1].sum(), -b[0]))
    mask[max(0, b[0] - margin): min(n, b[1] + margin + 1)] = True
    return mask

def errors(y, good, pred, gap, margin):
    m = good & (y < 5); w = main_sleep_period(pred, gap, margin)
    ref = np.sum((y > 0) & (y < 5) & m) * .5
    return {'rule': float(np.sum((pred > 0) & (pred < 5) & m & w) * .5 - ref), 'unwindowed': float(np.sum((pred > 0) & (pred < 5) & m) * .5 - ref)}

protocol = json.loads((GL / 'protocol.json').read_text())
calib = [r['subject'] for r in protocol['records'] if r['split'] == 'summary_calibration' and r['confirmatory_night']]
assert len(calib) == 20 and not set(calib) & set(saved_hist)
cal = {s: dict(np.load(GL / 'data/features' / f'{s}.npz')) for s in calib}
cal_pred = {s: v2_guess(z) for s, z in cal.items()}
grid = []
for gap, margin in itertools.product(GAP_MIN, MARGIN_MIN):
    e = [errors(z['y'], z['good'], cal_pred[s], gap, margin) for s, z in cal.items()]
    grid.append({'gap_min': gap, 'margin_min': margin, 'rule_mae': float(np.mean([abs(x['rule']) for x in e])), 'unwindowed_mae': float(np.mean([abs(x['unwindowed']) for x in e]))})
grid.sort(key=lambda r: (round(r['rule_mae'], 6), -r['gap_min'], -r['margin_min']))  # ties: most permissive window
best = grid[0]
params = {'rule': 'longest predicted-sleep block, gap-tolerant, with margins; uses model predictions only',
          'gap_min': best['gap_min'], 'margin_min': best['margin_min'], 'tuned_on': 'BOAS summary-calibration people (20, first night), v2 frozen predictions',
          'people': calib, 'model_sha256': freeze['model_sha256'], 'grid': grid,
          'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'tie_break': 'equal MAE -> larger gap, then larger margin (least restrictive)'}
(OUTDIR / 'window_params.json').write_text(json.dumps(params, indent=2))
print('frozen:', {k: params[k] for k in ['gap_min', 'margin_min', 'frozen_utc']}, 'calib MAE rule', round(best['rule_mae'], 1), 'unwindowed', round(best['unwindowed_mae'], 1))

# Apply (params now frozen above; nothing below changes them)
yasa = json.loads((GL / 'results/yasa/yasa_predictions.json').read_text())
saved_ext = json.loads((GL / 'results/v2/external_predictions.json').read_text())
results = {}
for cohort, saved, status in [('historical_boas', saved_hist, 'BOAS development comparison (inspected at v1)'), ('external', saved_ext, 'EXPLORATORY DEVELOPMENT: Sleep-EDF already inspected, not confirmation')]:
    out = {'status': status}
    for method in ['v2', 'yasa']:
        e = []
        for sub, p in saved.items():
            y, good = np.array(p['reference']), np.array(p['good']).astype(bool)
            pred = np.array(p['model']) if method == 'v2' else np.array(yasa[cohort][sub])
            e.append(errors(y, good, pred, params['gap_min'], params['margin_min']))
        out[method] = {'people': len(e), 'rule_mae_min': float(np.mean([abs(x['rule']) for x in e])), 'unwindowed_mae_min': float(np.mean([abs(x['unwindowed']) for x in e])), 'rule_signed_min': float(np.mean([x['rule'] for x in e]))}
    results[cohort] = out
(OUTDIR / 'window_results.json').write_text(json.dumps({'params_frozen_utc': params['frozen_utc'], 'results': results}, indent=2))
print(json.dumps(results, indent=2))

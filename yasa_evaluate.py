"""YASA external baseline on the frozen v2 cohorts (Revision 16, section 15, Change 2).

Written 7 Nov 2026 at the event. Never refits or changes v1/v2 or their saved results.
Uses YASA's documented path: no filtering, z-scoring or demographics; YASA resamples and
converts units itself. Channels fixed in advance: BOAS HB_1 (about AF7), Sleep-EDF EEG Fpz-Cz.
YASA prefers a central derivation, so both are channel-mismatched and labelled as such.

Scoring copies v2_evaluate.assess exactly: same people, same mask (reference 0-4 AND good
signal), same complete-recording and reference-window (+/-30 min) definitions, same
stage_metrics (copied verbatim from pipeline.py). Step 1 recomputes v2 from its saved
predictions and must reproduce the saved metrics before any YASA row is reported.
Output: results/yasa/ only.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/yasa'; OUT.mkdir(parents=True, exist_ok=True)
V2 = ROOT / 'results/v2'
CODE = {'W': 0, 'WAKE': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'REM': 4}

def stage_metrics(y, pred):  # verbatim from pipeline.py
    return {'accuracy': float(accuracy_score(y, pred)), 'kappa': float(cohen_kappa_score(y, pred, labels=list(range(5)) + [-2])), 'macro_f1': float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0)), 'per_stage_f1': f1_score(y, pred, labels=list(range(5)), average=None, zero_division=0).tolist(), 'coverage': float(np.mean(pred >= 0))}

def score(y, good, onset, pred, window):
    m = window & good & (y < 5)
    s = stage_metrics(y[m], pred[m])
    tst = float(np.sum((pred > 0) & (pred < 5) & m) * .5); ref = float(np.sum((y > 0) & (y < 5) & m) * .5)
    s.update(reference_tst_minutes=ref, model_tst_minutes=tst, tst_error_minutes=abs(tst - ref), signed_tst_error_minutes=tst - ref, available_epochs=int(m.sum()))
    return s

def windows(y, onset):
    w = {'primary_complete_recording': y < 5}
    sleep = onset[(y > 0) & (y < 5)]
    if len(sleep): w['secondary_reference_window'] = (y < 5) & (onset >= sleep.min() - 1800) & (onset <= sleep.max() + 1800)
    return w

def summarize(rows):
    keys = ['accuracy', 'kappa', 'macro_f1', 'tst_error_minutes', 'signed_tst_error_minutes', 'coverage']
    return {'people': len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}

def reproduce_v2():
    """Recompute v2 from its saved predictions; compare with saved metrics."""
    check = {}
    for cohort in ['historical_boas', 'external']:
        saved = json.loads((V2 / f'{cohort}_metrics.json').read_text())
        preds = json.loads((V2 / f'{cohort}_predictions.json').read_text())
        rows = {k: [] for k in ['primary_complete_recording', 'secondary_reference_window']}
        for sub, p in preds.items():
            y, pred, good, onset = (np.array(p[k]) for k in ['reference', 'model', 'good', 'onset']); good = good.astype(bool)
            for name, w in windows(y, onset).items(): rows[name].append(score(y, good, onset, pred, w))
        got = summarize(rows['primary_complete_recording'])
        pairs = {'kappa': (got['kappa'], saved['results']['kappa']), 'macro_f1': (got['macro_f1'], saved['results']['macro_f1']), 'tst_mae': (got['tst_error_minutes'], saved['results']['tst_error_minutes'])}
        if cohort == 'external':
            g2 = summarize(rows['secondary_reference_window']); s2 = saved['secondary_reference_defined_sleep_window']
            pairs.update({'secondary_kappa': (g2['kappa'], s2['kappa']), 'secondary_tst_mae': (g2['tst_error_minutes'], s2['tst_error_minutes'])})
        check[cohort] = {k: {'recomputed': round(a, 4), 'saved': round(b, 4), 'match': abs(a - b) < 1e-6} for k, (a, b) in pairs.items()}
    return check

def yasa_predict(raw, ch):
    import yasa
    raw = raw.copy().pick([ch])
    hyp = yasa.SleepStaging(raw, eeg_name=ch, eog_name=None, emg_name=None, metadata=None).predict()
    labels = hyp.hypno.astype(str).to_numpy() if hasattr(hyp, 'hypno') else np.asarray(hyp).astype(str)
    return np.array([CODE.get(s.upper(), -2) for s in labels], dtype='int8')  # ART/UNS -> abstain (-2), never wake

def boas_raw(sub):
    import mne
    z = np.load(ROOT / 'data/eeg_digital' / f'{sub}.npz')
    uv = (z['digital'] - z['digital_min'][:, None]) * (z['physical_max'] - z['physical_min'])[:, None] / (z['digital_max'] - z['digital_min'])[:, None] + z['physical_min'][:, None]
    return mne.io.RawArray(uv * 1e-6, mne.create_info(list(z['labels']), float(z['sfreq']), 'eeg'), verbose='ERROR')

def run_yasa():
    import mne, yasa
    preds = {}; rows = {'historical_boas': {}, 'external': {}}
    v2 = {c: json.loads((V2 / f'{c}_predictions.json').read_text()) for c in rows}
    sel = {r['subject']: r for r in json.loads((ROOT / 'data/external_sleepedf/selection.json').read_text())['records']}
    for cohort in rows:
        for sub, p in v2[cohort].items():
            t = time.time()
            if cohort == 'historical_boas': raw, ch = boas_raw(sub), 'HB_1'
            else: raw, ch = mne.io.read_raw_edf(ROOT / 'data/external_sleepedf' / sel[sub]['psg'], preload=False, verbose='ERROR').pick(['EEG Fpz-Cz']).load_data(verbose='ERROR'), 'EEG Fpz-Cz'
            y, good, onset = np.array(p['reference']), np.array(p['good']).astype(bool), np.array(p['onset'])
            pred = yasa_predict(raw, ch)
            assert len(pred) == len(y), f'{sub}: YASA {len(pred)} epochs vs frozen {len(y)}'
            assert np.allclose(onset, np.arange(len(y)) * 30), f'{sub}: frozen onsets are not k*30 from signal start'
            preds.setdefault(cohort, {})[sub] = pred.tolist()
            for name, w in windows(y, onset).items(): rows[cohort].setdefault(name, []).append({'subject': sub, **score(y, good, onset, pred, w)})
            print(json.dumps({'cohort': cohort, 'subject': sub, 'kappa': round(rows[cohort]['primary_complete_recording'][-1]['kappa'], 3), 'tst_error': rows[cohort]['primary_complete_recording'][-1]['tst_error_minutes'], 'seconds': round(time.time() - t, 1)}), flush=True)
    report = {'method': f'YASA {yasa.__version__} pre-trained SleepStaging, documented path, no metadata', 'channels': {'historical_boas': 'HB_1 (approx. AF7, forehead; channel mismatch: YASA prefers central)', 'external': 'EEG Fpz-Cz (frontal; channel mismatch: YASA prefers central)'}, 'scoring': 'Same people, masks, windows and stage_metrics as v2_evaluate.py', 'created_unix': time.time(), 'results': {c: {w: summarize(r) for w, r in v.items()} for c, v in rows.items()}}
    (OUT / 'yasa_metrics.json').write_text(json.dumps(report, indent=2)); (OUT / 'yasa_predictions.json').write_text(json.dumps(preds))
    pd.DataFrame([{'cohort': c, 'window': w, **r} for c, v in rows.items() for w, rs in v.items() for r in rs]).drop(columns='per_stage_f1').to_csv(OUT / 'yasa_people.csv', index=False)
    print(json.dumps(report['results'], indent=2))

if __name__ == '__main__':
    check = reproduce_v2(); (OUT / 'v2_reproduction_check.json').write_text(json.dumps(check, indent=2)); print(json.dumps(check, indent=2), flush=True)
    if not all(v['match'] for c in check.values() for v in c.values()): sys.exit('STOP: v2 saved metrics not reproduced; YASA not run')
    if '--check-only' not in sys.argv: run_yasa()

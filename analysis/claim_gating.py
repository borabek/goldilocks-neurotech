"""Claim gating, measured (exploratory; section D of the experiment design).

The viewer refuses claims it can't back up. This scores that idea two ways on all four cohorts
(BOAS test, Sleep-EDF, DOD-H, HMC), for v2 and the plain CNN:

1. Epoch level: risk-coverage curves. Rank epochs by the model's top probability, withhold the
   least confident, and track the error rate on what's left. AURC = area under that curve
   (lower is better); `oracle_aurc` is the best possible for the same error count.
2. Recording level, v2 only: withhold a recording's sleep-time claim when its own label-free gain
   estimate (q75 rule, as in PREREGISTRATION.md) says the amplitude is out of range,
   |ln gain| > ln 1.5. Counts how many large errors (> 60 min) that blocks and how many good
   claims (<= 30 min) it throws away. The 1.5 cut-off was chosen here, so this is exploratory.

Usage: python analysis/claim_gating.py --boas-cache B --ext-cache E --new-cache /tmp/goldilocks_external_cache
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from ablation_shift import percentile_log_sd  # noqa: E402
from cnn_challenger import night_probabilities, rebuild_free, robust_scale, CLIP  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402
from v2_model import probabilities  # noqa: E402
from yasa_evaluate import score, windows  # noqa: E402


def risk_coverage(conf, correct):
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    n = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / n
    aurc = float(risk.mean())
    errors = int((~correct).sum())
    opt = np.concatenate([np.zeros(len(c) - errors), np.ones(errors)])
    oracle = float((np.cumsum(opt) / n).mean())
    cov = [0.5, 0.8, 1.0]
    at = {f'risk_at_{int(k * 100)}pct_coverage': float(risk[max(0, int(k * len(c)) - 1)]) for k in cov}
    return {'aurc': aurc, 'oracle_aurc': oracle, 'epochs': int(len(c)), **at}


def cohorts(args):
    proto = json.loads((ROOT / 'protocol.json').read_text())['records']
    test = [r['subject'] for r in proto if r['split'] == 'test' and r['confirmatory_night']]
    out = {'boas_test': [(s, ROOT / 'data/features' / f'{s}.npz', Path(args.boas_cache) / f'{s}.npz', None)
                         for s in test]}
    ext = sorted(json.loads((ROOT / 'results/v2/external_predictions.json').read_text()))
    out['sleep_edf'] = [(s, ROOT / 'data/external_sleepedf/features' / f'{s}.npz', Path(args.ext_cache) / f'{s}.npz', None)
                        for s in ext]
    for name in ['dodh', 'hmc']:
        lab = json.loads((ROOT / f'results/external_{name}/predictions/labels_and_grid.json').read_text())
        out[name] = [(r, Path(args.new_cache) / name / f'{r}_v2.npz', Path(args.new_cache) / name / f'{r}.npz',
                      np.array(lab[r]['reference'], dtype='int8')) for r in sorted(lab)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    ap.add_argument('--new-cache', required=True)
    args = ap.parse_args()
    frozen = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    cnn_freeze = json.loads((ROOT / 'results/cnn/cnn_freeze_record.json').read_text())
    cnn_params = rebuild_free(ROOT / 'results/cnn/cnn_weights.json')
    trans, prior = fit_trans_prior(args.boas_cache)
    fit = [dict(np.load(ROOT / 'data/features' / f"{r['subject']}.npz"))
           for r in json.loads((ROOT / 'protocol.json').read_text())['records'] if r['split'] == 'fit']
    ref75 = float(np.median([percentile_log_sd(z, 75) for z in fit]))

    report = {'status': 'EXPLORATORY: gating cut-off chosen in this analysis', 'cohorts': {}}
    for cohort, items in cohorts(args).items():
        conf = {'v2': [], 'cnn': []}
        corr = {'v2': [], 'cnn': []}
        claims = []
        for rec, feat_path, cache_path, y_override in items:
            fz = dict(np.load(feat_path))
            cz = np.load(cache_path)
            y = y_override if y_override is not None else cz['y'].astype('int8')
            good = cz['good'].astype(bool)
            onset = cz['onset'].astype(float)
            pv = probabilities(frozen['model'], fz['X'], fz['good'].astype(bool), frozen['transition'], frozen['prior'],
                               frozen['weight'], frozen['temperature'], fz['onset'])
            x = cz['X'].astype('float32')
            med, mad = robust_scale(x, good)
            pc = night_probabilities(cnn_params, np.clip((x - med) / mad, -CLIP, CLIP), good, onset, trans, prior,
                                     cnn_freeze['selected_sequence_weight'], cnn_freeze['temperature'])
            m = good & (y < 5)
            for name, p in [('v2', pv), ('cnn', pc)]:
                conf[name].append(p[m].max(axis=1))
                corr[name].append(p[m].argmax(axis=1) == y[m])
            gain = float(np.exp(ref75 - percentile_log_sd(fz, 75)))
            s = score(y, good, onset, pv.argmax(axis=1), windows(y, onset)['primary_complete_recording'])
            claims.append({'recording': rec, 'label_free_gain_q75': gain,
                           'abs_tst_error_min': abs(s['signed_tst_error_minutes'])})
        res = {name: risk_coverage(np.concatenate(conf[name]), np.concatenate(corr[name])) for name in conf}
        withheld = [c for c in claims if abs(np.log(c['label_free_gain_q75'])) > np.log(1.5)]
        kept = [c for c in claims if c not in withheld]
        res['v2_recording_gate'] = {
            'recordings': len(claims), 'withheld': len(withheld),
            'large_errors_total': sum(c['abs_tst_error_min'] > 60 for c in claims),
            'large_errors_blocked': sum(c['abs_tst_error_min'] > 60 for c in withheld),
            'good_claims_withheld': sum(c['abs_tst_error_min'] <= 30 for c in withheld),
            'mean_abs_error_kept_min': float(np.mean([c['abs_tst_error_min'] for c in kept])) if kept else None,
            'mean_abs_error_all_min': float(np.mean([c['abs_tst_error_min'] for c in claims])),
            'per_recording': claims}
        report['cohorts'][cohort] = res
        print(json.dumps({'cohort': cohort, 'v2_aurc': round(res['v2']['aurc'], 3),
                          'cnn_aurc': round(res['cnn']['aurc'], 3),
                          'gate': {k: v for k, v in res['v2_recording_gate'].items() if k != 'per_recording'}}), flush=True)
    out = ROOT / 'results/claim_gating'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'claim_gating.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

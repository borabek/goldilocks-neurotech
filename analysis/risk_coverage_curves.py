"""Risk-coverage curves behind the claim-gating AURC values, saved for the paper figure.

Recomputes the epoch-level confidences of v2 and the plain CNN exactly as
analysis/claim_gating.py does, checks that the AURC matches results/claim_gating/claim_gating.json,
and stores each curve at 201 evenly spaced coverage points.

Usage: python analysis/risk_coverage_curves.py --boas-cache B --ext-cache E --new-cache N
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'analysis'))
from claim_gating import cohorts, risk_coverage  # noqa: E402
from cnn_challenger import CLIP, night_probabilities, rebuild_free, robust_scale  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402
from v2_model import probabilities  # noqa: E402

POINTS = np.linspace(0, 1, 201)


def curve(conf, correct):
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    n = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / n
    idx = np.clip((POINTS * len(c)).astype(int) - 1, 0, len(c) - 1)
    return risk[idx].tolist()


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
    saved = json.loads((ROOT / 'results/claim_gating/claim_gating.json').read_text())['cohorts']
    out = {'coverage': POINTS.tolist(), 'cohorts': {}}
    for cohort, items in cohorts(args).items():
        conf, corr = {'v2': [], 'cnn': []}, {'v2': [], 'cnn': []}
        for rec, feat_path, cache_path, y_override in items:
            fz = dict(np.load(feat_path))
            cz = np.load(cache_path)
            y = y_override if y_override is not None else cz['y'].astype('int8')
            good, onset = cz['good'].astype(bool), cz['onset'].astype(float)
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
        out['cohorts'][cohort] = {}
        for name in conf:
            c, k = np.concatenate(conf[name]), np.concatenate(corr[name])
            rc = risk_coverage(c, k)
            assert abs(rc['aurc'] - saved[cohort][name]['aurc']) < 1e-12, (cohort, name)
            out['cohorts'][cohort][name] = {'aurc': rc['aurc'], 'oracle_aurc': rc['oracle_aurc'], 'risk': curve(c, k)}
        print(cohort, {n: round(v['aurc'], 3) for n, v in out['cohorts'][cohort].items()}, flush=True)
    (ROOT / 'results/paper').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/paper/risk_coverage_curves.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()

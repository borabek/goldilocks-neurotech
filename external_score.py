"""Score the pre-registered external test once (run via `python external_eval.py score`).

Reads the predictions written by `external_eval.py predict`, applies yasa_evaluate.py's masks,
windows and metrics, and evaluates the hypotheses exactly as PREREGISTRATION.md states them
(prereg-v1 with amendment v1.1). Writes results/external_<dataset>/metrics.json and
results/external_test_verdicts.json.

Statistics: paired person-level bootstrap, 10,000 resamples, seed 20261126. For Holm, each test
gets a one-sided bootstrap p-value for its direction; a test with two conditions (H1b, H2) takes
the larger of its two p-values (intersection-union).
"""
import json
from pathlib import Path

import numpy as np

from yasa_evaluate import score, summarize, windows

ROOT = Path(__file__).resolve().parent
DATASETS = {'dodh': ROOT / 'results/external_dodh', 'hmc': ROOT / 'results/external_hmc'}
RESAMPLES, SEED = 10_000, 20261126


def per_person(labels, preds, extra, dataset):
    out = {}
    for rec, lab in labels.items():
        y = np.array(lab['reference'])
        good = np.array(lab['good']).astype(bool)
        onset = np.array(lab['onset'], dtype=float)
        win = windows(y, onset)
        if dataset == 'hmc':
            lo, hi = extra[rec]['lights_off'], extra[rec]['lights_on']
            win['secondary_lights_off_to_on'] = (onset >= lo) & (onset < hi) & (y < 5)
        guess = np.array(preds[rec])
        res = {}
        for w_name, w in win.items():
            s = score(y, good, onset, guess, w)
            m = good & w & (y == 0)
            s['wake_recall'] = float(np.mean(guess[m] == 0)) if m.any() else float('nan')
            res[w_name] = s
        out[rec] = res
    return out


def boot(values, rng):
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(RESAMPLES, len(v)))
    return np.nanmean(v[idx], axis=1)


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def holm(pvals):
    names = sorted(pvals, key=lambda k: pvals[k])
    adj, running = {}, 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * pvals[k]))
        adj[k] = running
    return adj


def main():
    rng = np.random.default_rng(SEED)
    verdicts, pvals = {}, {}
    for dataset, folder in DATASETS.items():
        pred_dir = folder / 'predictions'
        preds = json.loads((pred_dir / 'predictions.json').read_text())
        labels = json.loads((pred_dir / 'labels_and_grid.json').read_text())
        extra = json.loads((pred_dir / 'extra.json').read_text())
        recs = sorted(labels)
        scored = {m: per_person(labels, p, extra, dataset) for m, p in preds.items()}

        summary = {}
        for m, per in scored.items():
            summary[m] = {}
            for w in per[recs[0]]:
                rows = [per[r][w] for r in recs]
                s = summarize(rows)
                s['wake_f1'] = float(np.mean([r['per_stage_f1'][0] for r in rows]))
                s['wake_recall'] = float(np.nanmean([r['wake_recall'] for r in rows]))
                s['people'] = len(rows)
                summary[m][w] = s

        P = 'primary_complete_recording'
        get = lambda m, k: np.array([scored[m][r][P][k] for r in recs], dtype=float)

        # H1 as originally written: macro-F1, v2 + q75 against v2.
        d = get('v2_gain_q75', 'macro_f1') - get('v2', 'macro_f1')
        b = boot(d, rng)
        lo, hi = ci(b)
        h1 = 'supported' if (d.mean() >= 0.03 and lo > 0) else ('contradicted' if hi < 0 else 'not supported')
        pvals[f'{dataset}_H1'] = float(np.mean(b <= 0))
        # H1b (amendment v1.1): absolute sleep-time error down, kappa up.
        dt = np.abs(get('v2_gain_q75', 'signed_tst_error_minutes')) - np.abs(get('v2', 'signed_tst_error_minutes'))
        dk = get('v2_gain_q75', 'kappa') - get('v2', 'kappa')
        bt, bk = boot(dt, rng), boot(dk, rng)
        h1b = 'supported' if (ci(bt)[1] < 0 and ci(bk)[0] > 0) else (
            'contradicted' if (ci(bt)[0] > 0 or ci(bk)[1] < 0) else 'not supported')
        pvals[f'{dataset}_H1b'] = float(max(np.mean(bt >= 0), np.mean(bk <= 0)))
        # H2: plain CNN wake recall below 0.50 while v2's is at least 0.60.
        wc, wv = get('cnn', 'wake_recall'), get('v2', 'wake_recall')
        bc, bv = boot(wc, rng), boot(wv, rng)
        cnn_w, v2_w = float(np.nanmean(wc)), float(np.nanmean(wv))
        h2 = 'supported' if (cnn_w < 0.5 and v2_w >= 0.6) else ('contradicted' if cnn_w >= 0.7 else 'mixed')
        pvals[f'{dataset}_H2'] = float(max(np.mean(bc >= 0.5), np.mean(bv < 0.6)))
        # Secondary: the family gap in macro-F1.
        g = get('v2', 'macro_f1') - get('cnn', 'macro_f1')
        bg = boot(g, rng)
        glo, ghi = ci(bg)
        gap = 'closes' if (g.mean() < 0.05 and glo <= 0 <= ghi) else (
            'persists' if (g.mean() > 0.15 and glo > 0.05) else 'mixed')

        verdicts[dataset] = {
            'people': len(recs),
            'H1_macro_f1': {'mean_change': float(d.mean()), 'ci95': [lo, hi], 'verdict': h1},
            'H1b_tst_and_kappa': {'abs_tst_error_change_min': float(dt.mean()), 'tst_ci95': ci(bt),
                                  'kappa_change': float(dk.mean()), 'kappa_ci95': ci(bk), 'verdict': h1b},
            'H2_cnn_wake': {'cnn_wake_recall': cnn_w, 'cnn_ci95': ci(bc), 'v2_wake_recall': v2_w,
                            'v2_ci95': ci(bv), 'verdict': h2},
            'secondary_family_gap_macro_f1': {'mean': float(g.mean()), 'ci95': [glo, ghi], 'verdict': gap},
        }
        metrics = {'status': 'CONFIRMATORY for the pre-registered tests; every other number is descriptive',
                   'preregistration': 'PREREGISTRATION.md, tag prereg-v1, amendment v1.1',
                   'gains': json.loads((pred_dir / 'gains.json').read_text()),
                   'summary': summary,
                   'per_person': {m: {r: per[r][P] for r in recs} for m, per in scored.items()}}

        if dataset == 'dodh':
            ceiling = {}
            for r in recs:
                for s, dct in extra[r]['ceiling'].items():
                    y = np.array(dct['reference'])
                    good = np.ones(len(y), dtype=bool)
                    onset = np.arange(len(y)) * 30.0
                    ceiling.setdefault(s, []).append(score(y, good, onset, np.array(dct['scorer']),
                                                           windows(y, onset)[P]))
            metrics['human_ceiling_each_scorer_vs_other_four'] = {s: summarize(rows) for s, rows in ceiling.items()}
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    adj = holm(pvals)
    for k, p in adj.items():
        dataset, test = k.split('_', 1)
        key = {'H1': 'H1_macro_f1', 'H1b': 'H1b_tst_and_kappa', 'H2': 'H2_cnn_wake'}[test]
        verdicts[dataset][key]['bootstrap_p'] = pvals[k]
        verdicts[dataset][key]['holm_adjusted_p'] = p
    out = {'preregistration': 'PREREGISTRATION.md, tag prereg-v1, amendment v1.1',
           'resamples': RESAMPLES, 'seed': SEED, 'verdicts': verdicts}
    (ROOT / 'results/external_test_verdicts.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

"""Band occlusion on the frozen plain CNN (PREREGISTRATION.md, addendum prereg-v2.1, item 3).

At test time only, one frequency band at a time is removed from every epoch (FFT bins with
lo <= f < hi set to zero) before the shipped robust scaling. For each band and dataset the script
reports the change in wake recall and in mean wake probability on reference-wake epochs, per
person and averaged. The baseline probabilities of `cnn` and `cnn_hfdrop` are also saved to
/tmp/goldilocks_probs for the calibration and error analysis, which reuse them.

Prediction: the summed drop in wake probability from the 16-20, 20-25 and 25-30 Hz bands is
larger on BOAS than on each of Sleep-EDF, DOD-H and HMC.

Added after the first run, and labelled post hoc in the output: for each band, the change in
wake probability on reference-sleep epochs and the change in kappa, so a band whose removal
raises wake recall can be told apart from one that simply turns everything into wake; and the
share of 0.5-4 Hz power in reference-wake epochs of each dataset.

Usage: python analysis/band_occlusion.py --boas-cache B --ext-cache E --new-cache /tmp/goldilocks_external_cache
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
import cnn_challenger as cc  # noqa: E402
from cnn_challenger import FS, CLIP, night_probabilities, rebuild_free, robust_scale  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402
from prereg_v2_eval import cohort_items  # noqa: E402

BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30)]
BETA = ['16-20', '20-25', '25-30']
PROBS = Path('/tmp/goldilocks_probs')
RESAMPLES, SEED = 10_000, 20261126


def band_name(b):
    return f'{b[0]:g}-{b[1]:g}'


def occlude(x, band):
    spec = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    spec[..., (f >= band[0]) & (f < band[1])] = 0
    return np.fft.irfft(spec, n=x.shape[-1], axis=-1).astype('float32')


def probs(params, freeze, x, good, onset, trans, prior):
    med, mad = robust_scale(x, good)
    return night_probabilities(params, np.clip((x - med) / mad, -CLIP, CLIP), good, onset, trans, prior,
                               freeze['selected_sequence_weight'], freeze['temperature'])


def wake_stats(p, y, good):
    m = good & (y == 0)
    if not m.any():
        return float('nan'), float('nan')
    return float(np.mean(p[m].argmax(axis=1) == 0)), float(np.mean(p[m, 0]))


def post_hoc_stats(p, y, good):
    """Wake probability on reference-sleep epochs, and kappa over all scored epochs."""
    from sklearn.metrics import cohen_kappa_score
    s = good & (y > 0) & (y < 5)
    a = good & (y < 5)
    return float(np.mean(p[s, 0])), float(cohen_kappa_score(y[a], p[a].argmax(axis=1), labels=list(range(5))))


def delta_share(x, y, good):
    """Median over reference-wake epochs (and channels) of the 0.5-4 Hz share of 0.5-30 Hz power."""
    m = good & (y == 0)
    if not m.any():
        return float('nan')
    spec = np.abs(np.fft.rfft(x[m], axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    tot = spec[..., (f >= 0.5) & (f < 30)].sum(axis=-1)
    low = spec[..., (f >= 0.5) & (f < 4)].sum(axis=-1)
    return float(np.median(low / np.maximum(tot, 1e-12)))


def boot_ci(d, rng):
    d = d[np.isfinite(d)]
    b = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    ap.add_argument('--new-cache', required=True)
    args = ap.parse_args()
    cc.TWOCHANNEL = False
    trans, prior = fit_trans_prior(args.boas_cache)
    models = {m: (rebuild_free(ROOT / 'results' / m / 'cnn_weights.json'),
                  json.loads((ROOT / 'results' / m / 'cnn_freeze_record.json').read_text()))
              for m in ['cnn', 'cnn_hfdrop']}
    boas_test = sorted(json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text()),
                       key=lambda s: int(s.split('-')[1]))
    items = {'boas': [(s, Path(args.boas_cache) / f'{s}.npz', None) for s in boas_test], **cohort_items(args)}
    rng = np.random.default_rng(SEED)
    report = {'addendum': 'PREREGISTRATION.md, prereg-v2.1 item 3', 'model': 'cnn (frozen)',
              'bands_hz': [band_name(b) for b in BANDS], 'datasets': {}}
    per_person = {}
    for ds, its in items.items():
        rows = []
        for rec, path, y in its:
            z = np.load(path)
            x = z['X'].astype('float32')
            good, onset = z['good'].astype(bool), z['onset'].astype(float)
            if y is None:
                y = z['y'].astype(int)
            for m, (params, freeze) in models.items():
                p = probs(params, freeze, x, good, onset, trans, prior)
                (PROBS / m / ds).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(PROBS / m / ds / f'{rec}.npz', p=p.astype('float32'), y=y, good=good,
                                    onset=onset)
                if m == 'cnn':
                    base, base_ph = wake_stats(p, y, good), post_hoc_stats(p, y, good)
            row = {'recording': rec, 'wake_recall': base[0], 'wake_prob': base[1],
                   'post_hoc_sleep_epoch_wake_prob': base_ph[0], 'post_hoc_kappa': base_ph[1],
                   'post_hoc_wake_epoch_delta_share': delta_share(x, y, good)}
            params, freeze = models['cnn']
            for b in BANDS:
                po = probs(params, freeze, occlude(x, b), good, onset, trans, prior)
                (r, q), (sq, k) = wake_stats(po, y, good), post_hoc_stats(po, y, good)
                row[band_name(b)] = {'wake_recall_change': r - base[0], 'wake_prob_change': q - base[1],
                                     'post_hoc_sleep_epoch_wake_prob_change': sq - base_ph[0],
                                     'post_hoc_kappa_change': k - base_ph[1]}
            rows.append(row)
            print(ds, rec, round(base[0], 3), {k: round(row[k]['wake_prob_change'], 3) for k in BETA}, flush=True)
        per_person[ds] = rows
        summary = {'people': len(rows), 'baseline_wake_recall': float(np.nanmean([r['wake_recall'] for r in rows])),
                   'baseline_wake_prob': float(np.nanmean([r['wake_prob'] for r in rows])), 'bands': {}}
        for b in BANDS:
            n = band_name(b)
            for k in ['wake_recall_change', 'wake_prob_change', 'post_hoc_sleep_epoch_wake_prob_change',
                      'post_hoc_kappa_change']:
                d = np.array([r[n][k] for r in rows], float)
                summary['bands'].setdefault(n, {})[k] = {'mean': float(np.nanmean(d)), 'ci95': boot_ci(d, rng)}
        summary['post_hoc_baseline_kappa'] = float(np.nanmean([r['post_hoc_kappa'] for r in rows]))
        summary['post_hoc_wake_epoch_delta_share_median'] = float(np.nanmedian(
            [r['post_hoc_wake_epoch_delta_share'] for r in rows]))
        beta = np.array([sum(r[n]['wake_prob_change'] for n in BETA) for r in rows], float)
        summary['beta_16_30_summed_wake_prob_change'] = {'mean': float(np.nanmean(beta)), 'ci95': boot_ci(beta, rng)}
        report['datasets'][ds] = summary
    b0 = report['datasets']['boas']['beta_16_30_summed_wake_prob_change']['mean']
    report['prediction'] = {
        'statement': 'summed 16-30 Hz drop in wake probability larger on BOAS than on each external dataset',
        'per_dataset': {ds: {'external_drop': -report['datasets'][ds]['beta_16_30_summed_wake_prob_change']['mean'],
                             'boas_drop': -b0,
                             'boas_larger': bool(-b0 > -report['datasets'][ds]['beta_16_30_summed_wake_prob_change']['mean'])}
                        for ds in ['sleep_edf', 'dodh', 'hmc']}}
    report['prediction']['holds'] = all(v['boas_larger'] for v in report['prediction']['per_dataset'].values())
    out = ROOT / 'results/prereg_v2_1'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'band_occlusion.json').write_text(json.dumps({**report, 'per_person': per_person}, indent=2))
    print(json.dumps({ds: {'base_recall': round(s['baseline_wake_recall'], 3),
                           **{n: round(v['wake_prob_change']['mean'], 3) for n, v in s['bands'].items()},
                           'beta_sum': round(s['beta_16_30_summed_wake_prob_change']['mean'], 3)}
                      for ds, s in report['datasets'].items()}, indent=1))
    print(json.dumps(report['prediction'], indent=1))


if __name__ == '__main__':
    main()

"""A4, channel and montage, on the 20 original Sleep-EDF recordings (exploratory).

Also closes the open check from day 1: the Sleep-EDF branch of build_epoch_cache.py is compared
array by array with the 64 Hz cache every Sleep-EDF CNN result came from.

Variants, all with frozen models and the same masks and windows:
  - Fpz-Cz (what shipped) and Pz-Oz, each alone, for v2 and the channel-shared CNNs
  - both derivations as two channels, for v2 and the channel-shared CNNs (probabilities averaged
    over channels, as on the two-channel BOAS headband)
  - the two-channel CNN, fed [Fpz-Cz, Fpz-Cz] and [Fpz-Cz, Pz-Oz]

Usage: python analysis/a4_montage.py --psg-dir D --boas-cache B --ext-cache E
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'analysis'))
import cnn_challenger as cc  # noqa: E402
from build_epoch_cache import sleepedf_night  # noqa: E402
from ablation_shift import v2_predict  # noqa: E402
from cnn_challenger import night_probabilities, rebuild_free, robust_scale, CLIP  # noqa: E402
from cnn_posthoc import fit_trans_prior  # noqa: E402
from pipeline import epoch_features  # noqa: E402
from yasa_evaluate import score, summarize, windows  # noqa: E402

CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']


def v2_features(psg_dir, sub, frozen_rec, channels):
    import mne
    psg = sorted(Path(psg_dir).glob(f'{sub}*-PSG.edf'))[0]
    raw = mne.io.read_raw_edf(psg, preload=False, verbose='ERROR')
    starts = np.rint(np.array(frozen_rec['onset'], dtype=float) * 100).astype(int)
    feats = []
    for i in range(0, len(starts), 32):
        ep = np.stack([raw.get_data(picks=channels, start=int(a), stop=int(a + 3000)) * 1e6 for a in starts[i:i + 32]])
        ep = resample_poly(np.nan_to_num(ep), up=64, down=25, axis=-1)
        feats.append(epoch_features(ep, 256))
    return np.concatenate(feats)


def metrics(frozen, preds):
    rows = []
    for s, rec in frozen.items():
        y = np.array(rec['reference'])
        good = np.array(rec['good']).astype(bool)
        onset = np.array(rec['onset'], dtype=float)
        g = np.array(preds[s])
        r = score(y, good, onset, g, windows(y, onset)['primary_complete_recording'])
        m = good & (y == 0)
        r['wake_recall'] = float(np.mean(g[m] == 0))
        rows.append(r)
    out = summarize(rows)
    out['wake_recall'] = float(np.mean([r['wake_recall'] for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--psg-dir', required=True)
    ap.add_argument('--boas-cache', required=True)
    ap.add_argument('--ext-cache', required=True)
    args = ap.parse_args()
    frozen = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    subs = sorted(frozen)

    # 1. Rebuild check for the Sleep-EDF branch, and the Pz-Oz epochs.
    check, eps = {}, {c: {} for c in CHANNELS}
    for s in subs:
        for c in CHANNELS:
            eps[c][s] = sleepedf_night(args.psg_dir, s, frozen[s], channel=c)
        old = np.load(Path(args.ext_cache) / f'{s}.npz')
        new = eps['EEG Fpz-Cz'][s]
        same = {k: bool(np.array_equal(new[k], old[k])) for k in ['X', 'y', 'good', 'onset']}
        if not same['X'] and new['X'].shape == old['X'].shape:
            d = np.abs(new['X'].astype('float32') - old['X'].astype('float32'))
            same['X_max_abs_diff_uV'] = float(d.max())
            same['X_share_differing'] = float(np.mean(d > 0))
        check[s] = same
    print(json.dumps({'rebuild_identical_all': all(all(v for k, v in c.items() if k in ('X', 'y', 'good', 'onset'))
                                                     for c in check.values())}), flush=True)

    # 2. v2 on each montage and on both.
    frozen_v2 = joblib.load(ROOT / 'results/v2/frozen_model.joblib')
    results = {}
    for label, chans in [('v2 Fpz-Cz', ['EEG Fpz-Cz']), ('v2 Pz-Oz', ['EEG Pz-Oz']),
                         ('v2 Fpz-Cz + Pz-Oz', CHANNELS)]:
        preds = {}
        for s in subs:
            z = {'X': v2_features(args.psg_dir, s, frozen[s], chans),
                 'good': np.array(frozen[s]['good']).astype(bool), 'onset': np.array(frozen[s]['onset'], dtype=float)}
            preds[s] = v2_predict(frozen_v2, z)
        results[label] = metrics(frozen, preds)
        print(json.dumps({label: {k: round(results[label][k], 3) for k in ['kappa', 'macro_f1', 'wake_recall', 'tst_error_minutes']}}), flush=True)

    # 3. CNNs.
    trans, prior = fit_trans_prior(args.boas_cache)

    def run(name, inputs, two_channel=False):
        src = ROOT / 'results' / name
        freeze = json.loads((src / 'cnn_freeze_record.json').read_text())
        params = rebuild_free(src / 'cnn_weights.json')
        cc.TWOCHANNEL = two_channel
        preds = {}
        for s in subs:
            x = np.concatenate([eps[c][s]['X'].astype('float32') for c in inputs], axis=1)
            good = eps['EEG Fpz-Cz'][s]['good']
            med, mad = robust_scale(x, good)
            p = night_probabilities(params, np.clip((x - med) / mad, -CLIP, CLIP), good,
                                    np.array(frozen[s]['onset'], dtype=float), trans, prior,
                                    freeze['selected_sequence_weight'], freeze['temperature'])
            preds[s] = p.argmax(axis=1)
        cc.TWOCHANNEL = False
        return metrics(frozen, preds)

    for name in ['cnn', 'cnn_balanced']:
        for label, inputs in [('Fpz-Cz', ['EEG Fpz-Cz']), ('Pz-Oz', ['EEG Pz-Oz']), ('Fpz-Cz + Pz-Oz', CHANNELS)]:
            key = f'{name} {label}'
            results[key] = run(name, inputs)
            print(json.dumps({key: {k: round(results[key][k], 3) for k in ['kappa', 'macro_f1', 'wake_recall', 'tst_error_minutes']}}), flush=True)
    if (ROOT / 'results/cnn_twochannel/cnn_weights.json').exists():
        for label, inputs in [('[Fpz-Cz, Fpz-Cz]', ['EEG Fpz-Cz', 'EEG Fpz-Cz']), ('[Fpz-Cz, Pz-Oz]', CHANNELS)]:
            key = f'cnn_twochannel {label}'
            results[key] = run('cnn_twochannel', inputs, two_channel=True)
            print(json.dumps({key: {k: round(results[key][k], 3) for k in ['kappa', 'macro_f1', 'wake_recall', 'tst_error_minutes']}}), flush=True)

    out = ROOT / 'results/ablation_shift'
    (out / 'a4_montage.json').write_text(json.dumps({
        'status': 'EXPLORATORY: Sleep-EDF people were inspected at the v2 evaluation',
        'sleep_edf_cache_rebuild_check': check, 'results': results}, indent=2))


if __name__ == '__main__':
    main()

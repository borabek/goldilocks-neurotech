"""Build the 64 Hz epoch caches the CNN scripts read (cnn_challenger.py, cnn_external.py, ...).

The original builder wasn't committed; this one is rebuilt from the settings recorded in the two
cache manifests and checked against the cached files (see `--verify`).

BOAS headband (cache_manifest.json):
  HB_1 and HB_2 from <sub>_task-Sleep_acq-headband_eeg.edf, read with MNE and converted to
  microvolts, cut into the 30-second epochs given by the PSG events file, then decimated
  256 -> 64 Hz with scipy.signal.resample_poly(up=1, down=4). Labels are stage_hum.
Sleep-EDF (external_cache_manifest.json):
  EEG Fpz-Cz at 100 Hz, cut on the frozen v2 epoch grid (onsets, reference and quality mask
  from results/v2/external_predictions.json), resampled 100 -> 64 Hz with
  resample_poly(up=16, down=25).
Both: quality = finite and not flat, as in pipeline.py; X stored as float16, shape
(epochs, channels, 1920).

Usage:
  python build_epoch_cache.py boas <BOAS root> <out dir> [sub-1 sub-6 ...]
  python build_epoch_cache.py sleepedf <folder with SC4*-PSG.edf> <out dir>
  python build_epoch_cache.py verify <new .npz> <cached .npz>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent


def quality(epochs):
    return np.isfinite(epochs).all(axis=(1, 2)) & (np.std(epochs, axis=-1).min(axis=1) > 1e-6)


def boas_night(boas_root, sub):
    import mne
    import pandas as pd
    base = Path(boas_root) / sub / 'eeg' / f'{sub}_task-Sleep_acq-'
    human = pd.read_csv(str(base) + 'psg_events.tsv', sep='\t')
    raw = mne.io.read_raw_edf(str(base) + 'headband_eeg.edf', preload=False, verbose='ERROR')
    fs = float(raw.info['sfreq'])
    assert fs == 256, (sub, fs)
    picks = [raw.ch_names.index(n) for n in ['HB_1', 'HB_2']]
    starts = np.rint(human.onset.to_numpy() * fs).astype(int)
    xs, good = [], []
    for i in range(0, len(starts), 32):
        ep = np.stack([raw.get_data(picks=picks, start=int(a), stop=int(a + 30 * fs)) * 1e6
                       for a in starts[i:i + 32]])
        good.extend(quality(ep).tolist())
        xs.append(resample_poly(np.nan_to_num(ep), up=1, down=4, axis=-1))
    X = np.concatenate(xs)
    assert X.shape[-1] == 1920
    return {'X': X.astype('float16'), 'y': human.stage_hum.to_numpy().astype('int8'),
            'good': np.array(good), 'onset': human.onset.to_numpy()}


def sleepedf_night(folder, sub, frozen):
    import mne
    psg = sorted(Path(folder).glob(f'{sub}*-PSG.edf'))[0]
    raw = mne.io.read_raw_edf(psg, preload=False, verbose='ERROR')
    assert raw.info['sfreq'] == 100
    onset = np.array(frozen['onset'], dtype=float)
    starts = np.rint(onset * 100).astype(int)
    xs = []
    for i in range(0, len(starts), 32):
        ep = np.stack([raw.get_data(picks=['EEG Fpz-Cz'], start=int(a), stop=int(a + 3000)) * 1e6
                       for a in starts[i:i + 32]])
        xs.append(resample_poly(np.nan_to_num(ep), up=16, down=25, axis=-1))
    X = np.concatenate(xs)
    assert X.shape[-1] == 1920
    return {'X': X.astype('float16'), 'y': np.array(frozen['reference'], dtype='int8'),
            'good': np.array(frozen['good']).astype(bool), 'onset': onset}


def verify(new, old):
    a, b = np.load(new), np.load(old)
    report = {}
    for k in ['X', 'y', 'good', 'onset']:
        same_shape = a[k].shape == b[k].shape
        exact = bool(same_shape and np.array_equal(a[k], b[k]))
        entry = {'shape_new': list(a[k].shape), 'shape_cached': list(b[k].shape), 'identical': exact}
        if same_shape and not exact and k == 'X':
            d = np.abs(a[k].astype('float32') - b[k].astype('float32'))
            entry.update(max_abs_diff_uV=float(d.max()), share_of_values_differing=float(np.mean(d > 0)))
        report[k] = entry
    print(json.dumps(report, indent=2))
    return all(v['identical'] for v in report.values())


def main():
    mode = sys.argv[1]
    if mode == 'verify':
        sys.exit(0 if verify(sys.argv[2], sys.argv[3]) else 1)
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if mode == 'boas':
        subs = sys.argv[4:] or [r['subject'] for r in json.loads((ROOT / 'protocol.json').read_text())['records']]
        for sub in subs:
            np.savez_compressed(out / f'{sub}.npz', **boas_night(sys.argv[2], sub))
            print('cached', sub, flush=True)
    elif mode == 'sleepedf':
        frozen = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
        for sub in sorted(frozen):
            np.savez_compressed(out / f'{sub}.npz', **sleepedf_night(sys.argv[2], sub, frozen[sub]))
            print('cached', sub, flush=True)


if __name__ == '__main__':
    main()

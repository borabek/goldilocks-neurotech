"""Shrink the 25 pre-registered HMC recordings to what the evaluation uses.

For each recording in data/external_hmc/selection.json this keeps EEG F4-M1 in microvolts at the
native rate, the 30-second scoring grid (onsets and stages) and the lights-off / lights-on
times, and writes one compressed .npz. Nothing is scored here.

Stage codes follow the rest of the repo: W 0, N1 1, N2 2, N3 3, REM 4, anything else 8 (excluded).

  python scripts/compact_hmc.py            # reads data/external_hmc/raw, writes data/external_hmc/compact
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/external_hmc/raw'
OUT = ROOT / 'data/external_hmc/compact'
CHANNEL = 'EEG F4-M1'
STAGE = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}


def compact(sub):
    import mne
    raw = mne.io.read_raw_edf(RAW / f'{sub}.edf', preload=False, verbose='ERROR')
    assert CHANNEL in raw.ch_names, (sub, raw.ch_names)
    fs = float(raw.info['sfreq'])
    signal = (raw.get_data(picks=[CHANNEL])[0] * 1e6).astype('float32')
    ann = mne.read_annotations(RAW / f'{sub}_sleepscoring.edf')
    onset, stage, lights_off, lights_on, other = [], [], None, None, set()
    for o, d, desc in zip(ann.onset, ann.duration, ann.description):
        if desc in STAGE:
            assert abs(d - 30) < 1e-6, (sub, desc, d)
            onset.append(float(o))
            stage.append(STAGE[desc])
        elif desc.lower().startswith('lights off'):
            lights_off = float(o)
        elif desc.lower().startswith('lights on'):
            lights_on = float(o)
        else:
            other.add(desc)
    onset, stage = np.array(onset), np.array(stage, dtype='int8')
    order = np.argsort(onset)
    onset, stage = onset[order], stage[order]
    keep = onset + 30 <= len(signal) / fs
    np.savez_compressed(OUT / f'{sub}.npz', signal=signal, fs=fs, channel=CHANNEL,
                        onset=onset[keep], y=stage[keep],
                        lights_off=np.nan if lights_off is None else lights_off,
                        lights_on=np.nan if lights_on is None else lights_on)
    return {'subject': sub, 'fs': fs, 'samples': int(len(signal)), 'epochs': int(keep.sum()),
            'epochs_dropped_past_signal_end': int((~keep).sum()),
            'lights_off_found': lights_off is not None, 'lights_on_found': lights_on is not None,
            'other_annotation_types': sorted(other)}


def main():
    import mne
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ROOT / 'data/external_hmc/selection.json').read_text())['records']
    manifest_path = OUT / 'manifest.json'
    done = {m['subject']: m for m in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    subs = sys.argv[1:] or [r['subject'] for r in sel]
    for sub in subs:
        if sub in done and (OUT / f'{sub}.npz').exists():
            continue
        done[sub] = compact(sub)
        done[sub]['mne_version'] = mne.__version__
        manifest_path.write_text(json.dumps(sorted(done.values(), key=lambda m: m['subject']), indent=2))
        print(json.dumps(done[sub]), flush=True)


if __name__ == '__main__':
    main()

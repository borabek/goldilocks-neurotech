"""Shrink the Sleep-EDF development recordings (SC subjects 20 and up, first night) for the
sleep-window detector (PREREGISTRATION.md, section 7).

Keeps EEG Fpz-Cz in microvolts at 100 Hz and the 30-second stage grid, built exactly as
v2_evaluate.extract_external builds it for the 20 evaluation people: R&K 3 and 4 map to N3,
movement and unscored epochs get code 8, the grid starts at the first raw sample.

  python scripts/compact_sleepedf_dev.py   # reads data/external_sleepedf_dev/raw, writes .../compact
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/external_sleepedf_dev/raw'
OUT = ROOT / 'data/external_sleepedf_dev/compact'
MAPPING = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 'Sleep stage 3': 3,
           'Sleep stage 4': 3, 'Sleep stage R': 4}


def compact(psg):
    import mne
    sub = psg.name[:5]
    hyp = sorted(RAW.glob(f'{psg.name[:6]}*-Hypnogram.edf'))[0]
    raw = mne.io.read_raw_edf(psg, preload=False, verbose='ERROR')
    assert raw.info['sfreq'] == 100 and 'EEG Fpz-Cz' in raw.ch_names
    raw.set_annotations(mne.read_annotations(hyp), verbose='ERROR')
    events, _ = mne.events_from_annotations(raw, event_id=MAPPING, chunk_duration=30, verbose='ERROR')
    starts = events[:, 0] - raw.first_samp
    assert np.all(starts % 3000 == 0)
    grid = np.arange(raw.n_times // 3000) * 3000
    y = np.full(len(grid), 8, dtype='int8')
    idx = starts // 3000
    ok = (idx >= 0) & (idx < len(grid))
    y[idx[ok]] = events[ok, 2]
    signal = (raw.get_data(picks=['EEG Fpz-Cz'])[0] * 1e6).astype('float32')
    np.savez_compressed(OUT / f'{sub}.npz', signal=signal, fs=100.0, y=y, onset=grid / 100.0)
    return {'subject': sub, 'psg': psg.name, 'hypnogram': hyp.name, 'epochs': int(len(y)),
            'hours': round(len(signal) / 100 / 3600, 2)}


def main():
    import mne
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    done = {m['subject']: m for m in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    for psg in sorted(RAW.glob('SC4*-PSG.edf')):
        sub = psg.name[:5]
        if sub in done and (OUT / f'{sub}.npz').exists():
            continue
        done[sub] = {**compact(psg), 'mne_version': mne.__version__}
        manifest_path.write_text(json.dumps(sorted(done.values(), key=lambda m: m['subject']), indent=2))
        print(json.dumps(done[sub]), flush=True)


if __name__ == '__main__':
    sys.exit(main())

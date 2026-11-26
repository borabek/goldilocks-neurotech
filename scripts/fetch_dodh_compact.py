"""Fetch DOD-H from Zenodo one record at a time and keep only what the pre-registered test uses.

DOD-H now lives on Zenodo (record 15900394, DOI 10.5281/zenodo.15900394, MIT licence) as a
single 21.9 GB zip. Rather than download all of it, this reads the zip remotely with HTTP range
requests, pulls one .h5 record at a time, hashes it, copies out the hypnogram and the frontal
derivations into a small .npz, and deletes the .h5. Peak extra disk use is about one record.

Nothing here scores anything. The hypnogram is copied, not read.

  python -m venv .venv && .venv/bin/pip install remotezip h5py numpy
  .venv/bin/python scripts/fetch_dodh_compact.py

Output: data/external_dodh/compact/<record>.npz and data/external_dodh/compact/manifest.json.
Safe to re-run; finished records are skipped.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://zenodo.org/records/15900394/files/dodh.zip?download=1'
OUT = ROOT / 'data/external_dodh/compact'
KEEP = ['FP1_F3', 'FP2_F4']   # the two frontal derivations named in PREREGISTRATION.md


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def find_fs(dataset, n_epochs):
    for key in ('fs', 'sampling_rate', 'frequency', 'sfreq'):
        if key in dataset.attrs:
            return float(dataset.attrs[key])
    return dataset.shape[-1] / (n_epochs * 30.0)   # fall back on the epoch count


def main():
    import h5py
    import numpy as np
    from remotezip import RemoteZip

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {m['member'] for m in manifest}
    tmp = OUT / '_tmp'
    tmp.mkdir(exist_ok=True)

    with RemoteZip(URL) as z:
        members = sorted(n for n in z.namelist() if n.endswith('.h5'))
        print(f'{len(members)} records in the zip', flush=True)
        for i, member in enumerate(members, 1):
            if member in done:
                continue
            record = Path(member).stem
            print(f'[{i}/{len(members)}] {record}', flush=True)
            path = Path(z.extract(member, path=tmp))
            digest, size = sha256(path), path.stat().st_size
            with h5py.File(path, 'r') as f:
                hyp = np.array(f['hypnogram'])
                eeg = f['signals/eeg']
                available = sorted(eeg.keys())
                keep = [c for c in KEEP if c in eeg] or [c for c in available if c.upper().startswith('FP')]
                if not keep:
                    raise RuntimeError(f'{record}: no frontal EEG channel among {available}')
                fs = find_fs(eeg[keep[0]], len(hyp))
                signals = np.stack([np.array(eeg[c], dtype='float32') for c in keep])
            np.savez_compressed(OUT / f'{record}.npz', signals=signals, channels=np.array(keep),
                                fs=fs, hypnogram=hyp.astype('int8'))
            os.remove(path)
            manifest.append({'member': member, 'record': record, 'h5_bytes': size, 'h5_sha256': digest,
                             'channels_kept': keep, 'eeg_channels_available': available, 'fs': fs,
                             'epochs': int(len(hyp)), 'source': URL, 'fetched_unix': time.time()})
            manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'done: {len(manifest)} records in {OUT}')


if __name__ == '__main__':
    sys.exit(main())

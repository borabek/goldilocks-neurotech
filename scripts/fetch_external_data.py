"""Download the datasets the pre-registered tests need, and hash every file on arrival.

Run this on a machine that can reach PhysioNet. It only downloads and hashes.
It never reads a signal or a label, so running it doesn't break the pre-registration.

  python scripts/fetch_external_data.py hmc          # 25 HMC recordings, about 2.5 GB
  python scripts/fetch_external_data.py dodh         # moved: see scripts/fetch_dodh_compact.py
  python scripts/fetch_external_data.py sleepedf-dev # Sleep-EDF SC subjects 20 and up, first night
  python scripts/fetch_external_data.py all

Raw BOAS headband files come from the existing script:  node acquire.mjs

Everything here is the standard library.
Each dataset gets a manifest (file, bytes, sha256, url, downloaded_unix) next to its files.
Already-downloaded files with a matching size are skipped, so it's safe to re-run.
"""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHYSIONET = 'https://physionet.org/files'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, attempts=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(part, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            part.rename(dest)
            return
        except Exception as e:  # network hiccups: retry a few times, then give up loudly
            if attempt == attempts - 1:
                raise RuntimeError(f'failed after {attempts} tries: {url}: {e}')
            time.sleep(5 * (attempt + 1))


def write_manifest(folder, rows):
    (folder / 'download_manifest.json').write_text(json.dumps(rows, indent=2))
    print(f'{folder}: {len(rows)} files hashed')


def hmc():
    sel = json.loads((ROOT / 'data/external_hmc/selection.json').read_text())
    folder = ROOT / 'data/external_hmc/raw'
    rows = []
    for rec in sel['records']:
        for name in [rec['psg'], rec['hypnogram']]:
            url = f'{PHYSIONET}/hmc-sleep-staging/1.1/recordings/{name}'
            dest = folder / name
            if not dest.exists():
                print('downloading', name, flush=True)
                fetch(url, dest)
            rows.append({'file': name, 'bytes': dest.stat().st_size, 'sha256': sha256(dest),
                         'url': url, 'downloaded_unix': time.time()})
    write_manifest(folder, rows)


def dodh():
    """DOD-H left Dreem's S3 bucket (it now returns NoSuchBucket) and moved to Zenodo as a
    21.9 GB zip. Use scripts/fetch_dodh_compact.py, which streams it one record at a time."""
    raise SystemExit('DOD-H moved to Zenodo: run  python scripts/fetch_dodh_compact.py  instead')


def sleepedf_dev():
    """First-night Sleep-EDF SC recordings for subjects 20 and up (development data for the
    sleep-window detector). The file list comes from PhysioNet's published SHA256SUMS, which
    also gives the expected hash of every file."""
    folder = ROOT / 'data/external_sleepedf_dev/raw'
    folder.mkdir(parents=True, exist_ok=True)
    sums_path = folder / 'SHA256SUMS.txt'
    if not sums_path.exists():
        fetch(f'{PHYSIONET}/sleep-edfx/1.0.0/SHA256SUMS.txt', sums_path)
    published = {}
    for line in sums_path.read_text().splitlines():
        digest, _, name = line.partition(' ')
        published[name.strip().lstrip('*')] = digest
    wanted = sorted(n for n in published if n.startswith('sleep-cassette/SC4')
                    and int(n.split('SC4')[1][:2]) >= 20 and n.split('SC4')[1][2] == '1')
    rows = []
    for name in wanted:
        dest = folder / Path(name).name
        url = f'{PHYSIONET}/sleep-edfx/1.0.0/{name}'
        if not dest.exists() or sha256(dest) != published[name]:
            print('downloading', name, flush=True)
            fetch(url, dest)
        got = sha256(dest)
        if got != published[name]:
            raise RuntimeError(f'checksum mismatch: {name}')
        rows.append({'file': name, 'bytes': dest.stat().st_size, 'sha256': got,
                     'url': url, 'downloaded_unix': time.time()})
    write_manifest(folder, rows)


JOBS = {'hmc': hmc, 'dodh': dodh, 'sleepedf-dev': sleepedf_dev}

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, job in JOBS.items():
        if which in (name, 'all'):
            job()

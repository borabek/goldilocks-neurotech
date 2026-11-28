"""Compare result JSON files in the working tree with the committed versions, to within 1e-9.

A rerun on another machine can differ in the last bit of a float; anything larger is a real change.
Exit status 1 if any value differs by more than the tolerance or any key appears or disappears.

Usage: python scripts/compare_json.py <file> [<file> ...]
"""
import json
import subprocess
import sys

TOL = 1e-9


def flat(o, p=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f'{p}/{k}'))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, f'{p}[{i}]'))
    else:
        out[p] = o
    return out


def main(paths):
    bad = 0
    for path in paths:
        old = flat(json.loads(subprocess.run(['git', 'show', f'HEAD:{path}'], check=True,
                                             capture_output=True, text=True).stdout))
        new = flat(json.load(open(path)))
        diffs = []
        for k in sorted(set(old) | set(new)):
            a, b = old.get(k, '<missing>'), new.get(k, '<missing>')
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                if abs(a - b) > TOL:
                    diffs.append((k, a, b))
            elif a != b:
                diffs.append((k, a, b))
        print(f'{path}: {len(old)} values, {len(diffs)} differ')
        for d in diffs[:10]:
            print('   ', d)
        bad += len(diffs)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

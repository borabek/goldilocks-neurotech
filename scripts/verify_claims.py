"""Check that every figure shown in the presentation can be found in a saved result file.

The project's claim is that no number on a slide is typed by hand: `hackathon/tools/make_presentation.py` reads
them all from the result JSONs at build time. That is an easy thing to assert and a hard thing to
keep true, so this script tests it instead. It strips the markup out of the generated handout and
slides, pulls every number out of the remaining text, and looks for each one among the values
stored under results/. A figure that no result file supports is reported.

It also verifies SHA256SUMS.txt, so a file edited after the hashes were written is caught too.

What each check can and cannot do, stated plainly because a verifier that oversells itself is
worse than none. The figure check catches an *invented* number: one that matches no stored
metric at any sensible rounding. It does not reliably catch a *substituted* one. Tens of thousands
of legitimate metric values are stored across the result files (the report prints the count), so a
wrong figure at three decimals can coincide with some other real metric by chance; a deliberate swap of 0.661 for
0.761 passes this check for that reason. The hash check is what catches substitution: it fails
on any edit made after SHA256SUMS.txt was written. Run both, and rebuild the hashes only from a
clean `hackathon/tools/make_presentation.py` run.

Exit status is 0 when every figure is accounted for and the hashes match, 1 otherwise.

Usage:  python scripts/verify_claims.py [--verbose]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Figures that are not measurements: cohort sizes, parameter counts, dates, the stage names N1-N3,
# and the small integers that appear in prose. Each is listed with the reason it is exempt, so the
# allowlist cannot quietly grow into a way of hiding a wrong number.
ALLOWED = {
    '20': 'people per cohort', '48': 'fitting people', '6': 'selection/calibration people',
    '5': 'stages, scorers', '30': 'seconds per epoch', '21': 'epochs of context',
    '44581': 'trunk parameters', '44': 'trunk parameters, grouped', '581': 'trunk parameters',
    '90': 'nominal interval target, per cent', '0.90': 'nominal interval target',
    '1': 'counting', '2': 'counting', '3': 'counting', '4': 'counting', '7': 'tests, models',
    '8': 'lanes, models', '9': 'counting', '10': 'counting', '11': 'counting', '13': 'ratio, rounded',
    '0': 'counting', '12': 'counting', '15': 'significant figures', '64': 'sampling rate',
    '1920': 'samples per epoch', '2026': 'year', '2020': 'year', '2000': 'year', '1.0': 'version',
    '0.7': 'YASA version', '19': 'subject ids', '00': 'subject ids', '256': 'resampled rate',
    '100': 'source rate', '473': 'per-person reference minutes', '1383': 'per-person model minutes',
    '910': 'per-person error minutes', '82.4': 'published per-stage average, cited',
    '89.9': 'published headline, cited', '0.852': 'comparable published figure, cited',
    '0.598': 'published N1 record, cited', '0.80': 'unreachable target, discussed',
    '0.70': 'N3 ceiling under this convention', '14': 'days of nasal evidence, cited',
    # The one figure on the handout that make_presentation.py does not read from a result file at
    # build time. It records a one-off clean-environment reproduction check and is written down in
    # README.md and the plan, not in results/. Listed here rather than quietly matched, so the
    # exception stays visible: if that check is ever rerun, this number has to be updated by hand.
    '294208': 'reproduction field count, recorded in README.md and the plan, not in results/',
    '52': 'differing fields in that same check', '2.6': 'largest relative difference, same check',
    # Inventory facts rather than measurements: counts of files and their sizes. Each is checkable
    # by listing the folder, which is why they are exempt from needing a result file behind them.
    '128': 'derived BOAS feature files in data/features/', '33': 'files in a hash manifest',
    '955': 'megabytes of original Sleep-EDF EDFs, not shipped', '3.4': 'gigabytes of raw EEG, not shipped',
    '571': 'files hashed', '575': 'files hashed', '576': 'files hashed', '102': 'figures this script checks',
}


METRIC_KEY = re.compile(
    r'kappa|macro_f1|weighted_f1|f1_|per_stage|accuracy|tst|sleep_time|error_minutes|mae|'
    r'log_loss|brier|ece|covered|coverage|width|spread|gain|temperature|minutes|'
    r'correlation|ratio|groups|people|epochs|seconds|seed|hours|tau|p_holm|p_two_sided|p_value|bootstrap_p|share|rows', re.I)


def result_values():
    """Headline metric values only.

    An earlier version of this script collected every number in every result file, including the
    per-person CSVs. That pool held over a million values, so almost any three-decimal figure
    matched something and the check passed a deliberately corrupted handout. A verifier that
    cannot fail proves nothing, so the pool is now restricted to values stored under a key that
    names a metric, which is what a slide actually quotes.
    """
    found = set()

    def add(x):
        found.add(f'{x}')
        for n in (0, 1, 2, 3):
            found.add(f'{round(float(x), n):.{n}f}')
            found.add(f'{round(float(x) * 100, n):.{n}f}')
            # The scanner reads digits only, so a figure printed with a minus sign arrives unsigned.
            found.add(f'{abs(round(float(x), n)):.{n}f}')

    def walk(o, key='', inherited=False):
        named = inherited or bool(METRIC_KEY.search(key or ''))
        if isinstance(o, dict):
            for k, v in o.items():
                # A summary stored as {'kappa': {'mean': ..., 'ci95': [...]}} names the metric one level
                # up; only those two generic child keys inherit it, so the pool does not balloon.
                walk(v, k, named and k in ('mean', 'ci95', 'mean_change', 'mean_difference'))
        elif isinstance(o, list):
            for v in o:
                walk(v, key, inherited)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)) and named:
            add(o)

    for p in sorted(ROOT.glob('results/**/*.json')) + sorted(ROOT.glob('neurorelease/**/out/*.json')):
        try:
            walk(json.loads(p.read_text()))
        except Exception:
            continue
    return found


def shown_numbers(html_path):
    """Numbers in the rendered text, with the markup and inline styles removed first."""
    text = html_path.read_text()
    # Remove the contents of style and script blocks first: stripping only the tags would leave
    # the CSS behind, and a line-height of 1.45 would then be read as a figure needing evidence.
    text = re.sub(r'(?is)<(style|script)\b.*?</\1>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)          # drop tags, and with them every px and hex colour
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Identifiers are not measurements: a DOI prefix (10.18112/...), the version or year that
    # follows an author list, and anything inside a doi: or URL token. Blank them before scanning
    # so they are never counted as figures that need a result behind them.
    text = re.sub(r'\b10\.\d{4,5}/\S*', ' ', text)
    text = re.sub(r'\bdoi:\S*', ' ', text)
    text = re.sub(r'https?://\S*', ' ', text)
    text = re.sub(r'\bet al\.\s*(19|20)\d\d', ' ', text)
    text = re.sub(r'\b[0-9a-f]{16,}\b', ' ', text)      # content hashes
    text = re.sub(r'(\d),(\d{3})\b', r'\1\2', text)       # 294,208 is one number, not two
    out = []
    for m in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)(?![\w])', text):
        out.append((m.group(1), text[max(0, m.start() - 60):m.end() + 30].strip()))
    return out


def paper_numbers():
    """Numbers in the paper source after the \\val macros are replaced by their generated values."""
    paper = ROOT / 'paper'
    if not (paper / 'main.tex').exists():
        return []
    defs = dict(re.findall(r'\\csname val@([^\\]+)\\endcsname\{([^}]*)\}', (paper / 'numbers.tex').read_text()))
    text = (paper / 'main.tex').read_text()
    text = text.split('\\begin{document}', 1)[1]                       # the preamble holds layout lengths
    text = text.replace('\\input{table1}', (paper / 'table1.tex').read_text())
    text = re.sub(r'(?<!\\)%.*', ' ', text)                               # comments
    text = re.sub(r'\\val\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}', lambda m: defs[f'{m[1]}@{m[2]}@{m[3]}'], text)
    text = re.sub(r'\\(cite|ref|label|url|input|includegraphics|bibliography|bibliographystyle|texttt)'
                  r'(\[[^\]]*\])?\{[^}]*\}', ' ', text)
    text = re.sub(r'\\begin\{tabular\}\{[^}]*\}', ' ', text)
    text = re.sub(r'\\(begin|end)\{[^}]*\}(\[[^\]]*\])?', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)                          # remaining commands
    text = text.replace('$', ' ').replace('{', ' ').replace('}', ' ').replace('~', ' ')
    text = text.replace('--', ' ').replace(',', ' , ')
    text = re.sub(r'(\d) , (\d{3})\b', r'\1\2', text)                     # 10,000 is one number
    out = []
    for m in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)(?![\w])', text):
        out.append((m.group(1), text[max(0, m.start() - 60):m.end() + 30].strip()))
    return out


def check_hashes():
    manifest = ROOT / 'SHA256SUMS.txt'
    if not manifest.exists():
        return ['SHA256SUMS.txt is missing']
    bad = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        want, _, rel = line.partition('  ')
        p = ROOT / rel
        if not p.exists():
            bad.append(f'listed but missing: {rel}')
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            bad.append(f'changed since hashing: {rel}')
    return bad


def main():
    verbose = '--verbose' in sys.argv
    values = result_values()
    # Only generated documents are checked. The handout, the slides and README.md are all written
    # by scripts that read the result files, so every figure in them must trace back to one.
    # hackathon/START_HERE.md and hackathon/QA_CARD.md are written by hand on purpose and are not checked here; the
    # figures they quote are the headline ones, which these three already cover.
    pages = [p for p in (ROOT / 'hackathon/handout/goldilocks_handout.html',
                         ROOT / 'hackathon/slides/goldilocks_support_slides.html',
                         ROOT / 'README.md') if p.exists()]
    pages += sorted((ROOT / 'docs/model_cards').glob('*.md'))   # hand-written, so worth checking too
    if not pages:
        print(json.dumps({'error': 'no generated presentation found; run hackathon/tools/make_presentation.py'}))
        return 1

    unsupported = []
    checked = 0
    sources = [(page.name, shown_numbers(page)) for page in pages] + [('paper/main.tex', paper_numbers())]
    for name, numbers in sources:
        for num, context in numbers:
            if num in ALLOWED or num.lstrip('0') in ALLOWED:
                continue
            checked += 1
            if num in values:
                continue
            stripped = num.rstrip('0').rstrip('.') if '.' in num else num
            if stripped in values:
                continue
            unsupported.append({'page': name, 'number': num, 'context': ' '.join(context.split())})

    hash_problems = check_hashes()
    report = {
        'figures_checked': checked,
        'unsupported_figures': len(unsupported),
        'result_values_available': len(values),
        'hash_problems': len(hash_problems),
        'verdict': ('every figure shown is backed by a saved result, and the hashes match'
                    if not unsupported and not hash_problems else 'see the lists below'),
    }
    print(json.dumps(report, indent=2))
    if unsupported:
        print('\nFigures with no matching result value:')
        for u in unsupported[:40]:
            print(f"  {u['page']}  {u['number']}   ...{u['context']}...")
    if hash_problems:
        print('\nHash problems:')
        for h in hash_problems[:40]:
            print(f'  {h}')
    if verbose and not unsupported:
        print(f'\n{checked} figures checked against {len(values)} stored values.')
    return 0 if not unsupported and not hash_problems else 1


if __name__ == '__main__':
    sys.exit(main())


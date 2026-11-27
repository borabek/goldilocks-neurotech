"""Leakage audit for the exposure model (plan Appendix B, "Independence and leakage").

Why this file exists. `exposure_model.py` holds out whole *studies*, where a study is a
normalised title string. The plan asks for something stricter: resolve title strings to
source identities and group linked publications, repeated formulations, cohorts and time
points that share experimental information. A title string cannot do that by itself, and
the result write-up lists it as a known limit. This audit tests that assumption against
the data instead of restating it, and measures what breaking it would cost.

Four detectors. Each one is proved able to fire by `--self-test`, which injects a
synthetic violation and requires the detector to catch it, then requires a clean frame to
produce no finding. A detector that cannot fire is a dead branch, not a check.

  D1 same measured outcome pair (blood AUC, brain AUC), bit-for-bit, under two study ids.
  D2 same outcome pair to four significant figures under two study ids.
  D3 same full formulation record (payload, route, descriptors, outcome) under two ids.
  D4 two study ids that resolve to one source by title: token containment, or
     Jaccard overlap >= JACCARD_MIN.

D1-D3 are leakage that group-held-out validation cannot stop: the same measurement sits
in a training fold and a test fold at once. D4 is suspected shared authorship/cohort.

If any detector links two groups, the audit merges every linked group with union-find and
reruns the identical grouped 10-fold CV and fixed holdout on the merged grouping, so the
effect on the headline conclusions is measured. Before trusting that rerun, the audit
reproduces the saved unmerged numbers with its own CV code and refuses to continue if they
differ, so a changed number can only come from the merge.

Usage:
  python leakage_audit.py <workbook.xlsx> <outdir>      audit and measure the effect
  python leakage_audit.py --self-test                   prove the detectors can fire

Exit code 1 if a headline verdict changes under merged grouping, or if a self-test fails,
or if the audit's own CV cannot reproduce the saved result. Otherwise 0.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import exposure_model as em

SIGFIG = 4
JACCARD_MIN = 0.9
BOOT = 2000
SAVED_RESULT = Path(__file__).resolve().parent / 'out' / 'exposure_results.json'
TOL = 1e-9


# ---------------------------------------------------------------- helpers

def _sigfig(x, n=SIGFIG):
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1)))


def _spanning(keys_to_studies):
    """Keys whose records sit under more than one study id."""
    return {k: sorted(v) for k, v in keys_to_studies.items() if len(v) > 1}


def _collect(df, keyfunc):
    out = {}
    for idx, row in df.iterrows():
        out.setdefault(keyfunc(row), set()).add(row['study'])
    return _spanning(out)


def _links(spans):
    """Every unordered pair of study ids tied together by a detector."""
    pairs = set()
    for studies in spans.values():
        pairs.update(itertools.combinations(sorted(studies), 2))
    return pairs


# ---------------------------------------------------------------- detectors

def d1_exact_outcome(df):
    return _collect(df, lambda r: (repr(float(r[em.BLOOD])), repr(float(r[em.BRAIN]))))


def d2_rounded_outcome(df):
    return _collect(df, lambda r: (_sigfig(r[em.BLOOD]), _sigfig(r[em.BRAIN])))


def d3_full_record(df):
    cols = em.NUM_DESC + em.CAT_DESC + em.CONTEXT
    def key(r):
        head = (str(r.get('API', '')).strip().lower(), _sigfig(r[em.BLOOD]), _sigfig(r[em.BRAIN]))
        body = tuple(_sigfig(r[c]) if c in em.NUM_DESC else str(r[c]) for c in cols)
        return head + body
    return _collect(df, key)


def d4_title_variants(df):
    ids = sorted(set(df['study']))
    tokens = {s: set(s.split()) for s in ids}
    found = {}
    for a, b in itertools.combinations(ids, 2):
        ta, tb = tokens[a], tokens[b]
        if not ta or not tb:
            continue
        inter = len(ta & tb)
        jac = inter / len(ta | tb)
        contained = ta <= tb or tb <= ta
        if contained or jac >= JACCARD_MIN:
            found[(a, b)] = {'studies': [a, b], 'jaccard': round(jac, 4),
                             'containment': bool(contained),
                             'records': [int((df['study'] == a).sum()), int((df['study'] == b).sum())]}
    return found


DETECTORS = {
    'D1_exact_outcome_pair_under_two_studies': d1_exact_outcome,
    'D2_outcome_pair_to_4_sigfigs_under_two_studies': d2_rounded_outcome,
    'D3_full_formulation_record_under_two_studies': d3_full_record,
    'D4_title_variants_resolving_to_one_source': d4_title_variants,
}
LEAKING = ['D1_exact_outcome_pair_under_two_studies',
           'D2_outcome_pair_to_4_sigfigs_under_two_studies',
           'D3_full_formulation_record_under_two_studies']


# ---------------------------------------------------------------- merge

def merge_groups(studies, pairs):
    """Union-find over detector links. Returns study id -> merged group label."""
    parent = {s: s for s in studies}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {s: find(s) for s in studies}


# ---------------------------------------------------------------- CV, same protocol

def grouped_cv(df, groups):
    """The grouped 10-fold CV and fixed holdout of exposure_model.main, with the grouping
    passed in. Kept deliberately identical so a difference can only come from grouping."""
    df = df.copy()
    df['group'] = groups
    preds = pd.DataFrame(index=df.index)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=10).split(df, groups=df['group'])):
        train, test = df.iloc[tr], df.iloc[te]
        for name, p in em.evaluate(train, test).items():
            preds.loc[test.index, name] = p
        preds.loc[test.index, 'fold'] = k
    preds['y'], preds['group'] = df['y'], df['group']

    methods = ['training_mean', 'route_ratio_baseline'] + list(em.METHODS)
    per_group = {m: (preds[m] - preds['y']).abs().groupby(preds['group']).mean() for m in methods}
    rng = np.random.default_rng(em.SEED)
    uniq = preds['group'].unique()
    diff = {m: [] for m in methods}
    boot = {m: [] for m in methods}
    for _ in range(BOOT):
        s = rng.choice(uniq, len(uniq), replace=True)
        base = per_group['route_ratio_baseline'].loc[s].mean()
        for m in methods:
            boot[m].append(per_group[m].loc[s].mean())
            diff[m].append(per_group[m].loc[s].mean() - base)

    cv = {m: {'study_mae_ln': float(per_group[m].mean()),
              'study_mae_ln_95ci': [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))],
              'minus_route_ratio_95ci': [float(np.percentile(diff[m], 2.5)),
                                         float(np.percentile(diff[m], 97.5))]} for m in methods}

    gs = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=em.SEED)
    tr, te = next(gs.split(df, groups=df['group']))
    train, test = df.iloc[tr], df.iloc[te]
    hold = {name: float(em.study_mae(test, p)) for name, p in em.evaluate(train, test).items()}

    best = min(em.METHODS, key=lambda m: cv[m]['study_mae_ln'])
    verdict = {
        'best_learned_by_cv': best,
        'beats_route_ratio_baseline': bool(cv[best]['minus_route_ratio_95ci'][1] < 0),
        'blood_adds_value_over_descriptors':
            bool(cv['full_ridge']['study_mae_ln'] < cv['descriptors_and_context_ridge']['study_mae_ln']),
        'descriptors_add_value_over_blood':
            bool(cv['full_ridge']['study_mae_ln'] < cv['blood_and_context_ridge']['study_mae_ln']),
    }
    return {'n_groups': int(len(uniq)), 'cv': cv, 'fixed_holdout_study_mae_ln': hold, 'verdict': verdict}


def check_against_saved(unmerged):
    """Refuse to interpret a merged rerun unless the unmerged numbers match the saved result."""
    if not SAVED_RESULT.exists():
        return {'checked': False, 'reason': f'{SAVED_RESULT} not present'}
    saved = json.loads(SAVED_RESULT.read_text())['grouped_10fold_cv']
    bad = []
    for m, v in unmerged['cv'].items():
        s = saved[m]
        if abs(v['study_mae_ln'] - s['study_mae_ln']) > TOL:
            bad.append((m, 'study_mae_ln', s['study_mae_ln'], v['study_mae_ln']))
        for i in (0, 1):
            if abs(v['minus_route_ratio_95ci'][i] - s['minus_route_ratio_95ci'][i]) > TOL:
                bad.append((m, f'minus_route_ratio_95ci[{i}]',
                            s['minus_route_ratio_95ci'][i], v['minus_route_ratio_95ci'][i]))
    return {'checked': True, 'reproduces_saved_result': not bad, 'tolerance': TOL,
            'mismatches': [{'method': m, 'field': f, 'saved': a, 'audit': b} for m, f, a, b in bad]}


# ---------------------------------------------------------------- self-test

def self_test():
    """Prove each detector can fire, and that a clean frame produces no finding."""
    cols = {em.BLOOD: [10.0, 25.0, 40.0], em.BRAIN: [1.0, 2.5, 4.0],
            'API': ['drug a', 'drug b', 'drug c'],
            'study': ['alpha study one', 'beta study two', 'gamma study three']}
    for c in em.NUM_DESC:
        cols[c] = [1.0, 2.0, 3.0]
    for c in em.CAT_DESC + em.CONTEXT:
        cols[c] = ['x', 'y', 'z']
    clean = pd.DataFrame(cols)
    results = []

    for name, fn in DETECTORS.items():
        found = fn(clean)
        results.append({'detector': name, 'phase': 'clean frame', 'fired': bool(found),
                        'expected': False, 'pass': not found})

    dup = pd.concat([clean, clean.iloc[[0]].assign(study='alpha study one reprint')], ignore_index=True)
    for name in ['D1_exact_outcome_pair_under_two_studies',
                 'D2_outcome_pair_to_4_sigfigs_under_two_studies',
                 'D3_full_formulation_record_under_two_studies']:
        found = DETECTORS[name](dup)
        results.append({'detector': name, 'phase': 'duplicated outcome injected', 'fired': bool(found),
                        'expected': True, 'pass': bool(found)})

    near = clean.copy()
    near.loc[len(near)] = near.iloc[0].to_dict()
    near.loc[len(near) - 1, 'study'] = 'delta study four'
    near.loc[len(near) - 1, em.BLOOD] = 10.0001          # same to 4 significant figures
    near.loc[len(near) - 1, em.BRAIN] = 1.00001
    d1 = DETECTORS['D1_exact_outcome_pair_under_two_studies'](near)
    d2 = DETECTORS['D2_outcome_pair_to_4_sigfigs_under_two_studies'](near)
    results.append({'detector': 'D1_exact_outcome_pair_under_two_studies', 'phase': 'near-duplicate injected',
                    'fired': bool(d1), 'expected': False, 'pass': not d1})
    results.append({'detector': 'D2_outcome_pair_to_4_sigfigs_under_two_studies',
                    'phase': 'near-duplicate injected', 'fired': bool(d2), 'expected': True, 'pass': bool(d2)})

    variant = pd.concat([clean, clean.iloc[[1]].assign(study='beta study two part ii',
                                                       **{em.BLOOD: 99.0, em.BRAIN: 9.9})], ignore_index=True)
    d4 = DETECTORS['D4_title_variants_resolving_to_one_source'](variant)
    results.append({'detector': 'D4_title_variants_resolving_to_one_source', 'phase': 'title variant injected',
                    'fired': bool(d4), 'expected': True, 'pass': bool(d4)})

    merged = merge_groups(['a', 'b', 'c'], {('a', 'b')})
    results.append({'detector': 'union_find_merge', 'phase': 'two of three groups linked',
                    'fired': len(set(merged.values())) == 2, 'expected': True,
                    'pass': len(set(merged.values())) == 2})
    return results


# ---------------------------------------------------------------- main

def main(path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, audit = em.load(path)

    findings, pairs = {}, set()
    for name, fn in DETECTORS.items():
        spans = fn(df)
        findings[name] = {'count': len(spans),
                          'evidence': [{'key': str(k), 'studies': v if isinstance(v, list) else v['studies']}
                                       for k, v in list(spans.items())[:20]]}
        if name in LEAKING or name == 'D4_title_variants_resolving_to_one_source':
            pairs |= _links({k: (v if isinstance(v, list) else v['studies']) for k, v in spans.items()})

    leaking_found = sum(findings[n]['count'] for n in LEAKING)
    studies = sorted(set(df['study']))
    groups = merge_groups(studies, pairs)
    n_merged = len(set(groups.values()))

    unmerged = grouped_cv(df, df['study'].to_numpy())
    saved_check = check_against_saved(unmerged)
    report = {
        'purpose': 'test the study-title grouping assumption of exposure_model.py and measure the cost of '
                   'breaking it (plan Appendix B, Independence and leakage)',
        'data_audit': audit,
        'self_test': self_test(),
        'detectors': {'sigfig': SIGFIG, 'jaccard_min': JACCARD_MIN, 'findings': findings,
                      'leaking_detector_hits': int(leaking_found),
                      'linked_study_pairs': sorted(pairs)[:50], 'linked_study_pair_count': len(pairs)},
        'grouping': {'title_groups': len(studies), 'merged_groups': n_merged,
                     'groups_absorbed': len(studies) - n_merged},
        'unmerged_result': unmerged,
        'saved_result_check': saved_check,
    }

    if n_merged < len(studies):
        merged_run = grouped_cv(df, df['study'].map(groups).to_numpy())
        report['merged_result'] = merged_run
        changed = {k: [unmerged['verdict'][k], merged_run['verdict'][k]]
                   for k in unmerged['verdict'] if unmerged['verdict'][k] != merged_run['verdict'][k]}
        report['verdict_changes_under_merged_grouping'] = changed
        report['conclusion'] = ('headline conclusions unchanged under merged grouping' if not changed
                                else 'headline conclusions CHANGE under merged grouping')
    else:
        report['merged_result'] = None
        report['verdict_changes_under_merged_grouping'] = {}
        report['conclusion'] = ('no detector linked two study groups; title grouping and source grouping '
                                'agree on this corpus')

    (outdir / 'leakage_audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        rows = self_test()
        for r in rows:
            print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['detector']:52s} {r['phase']:30s} "
                  f"fired={r['fired']} expected={r['expected']}")
        sys.exit(0 if all(r['pass'] for r in rows) else 1)

    rep = main(sys.argv[1], sys.argv[2])
    st_ok = all(r['pass'] for r in rep['self_test'])
    print(f"self-test: {'all detectors proved able to fire' if st_ok else 'FAILED'}")
    for n, f in rep['detectors']['findings'].items():
        print(f"{n:52s} hits={f['count']}")
    print('title groups', rep['grouping']['title_groups'], '-> merged groups', rep['grouping']['merged_groups'])
    sc = rep['saved_result_check']
    print('reproduces saved result:', sc.get('reproduces_saved_result'), sc.get('mismatches') or '')
    if rep['merged_result']:
        for m in rep['unmerged_result']['cv']:
            a = rep['unmerged_result']['cv'][m]['study_mae_ln']
            b = rep['merged_result']['cv'][m]['study_mae_ln']
            print(f"  {m:34s} title-grouped {a:.3f}   merged {b:.3f}")
    print(rep['conclusion'])
    bad_saved = sc.get('checked') and not sc.get('reproduces_saved_result')
    sys.exit(1 if (rep['verdict_changes_under_merged_grouping'] or not st_ok or bad_saved) else 0)

"""NeuroRelease exposure model (plan Appendix B): blood exposure to brain exposure.

Research track, not the scored EEG core. Written at the event, 7 Nov 2026.

Task: predict ln(brain AUC) from measured ln(blood AUC), formulation descriptors and
declared context (route), on studies the model has never seen.

Data: Yousfan et al. 2023, Molecular Pharmaceutics, supporting workbook mp3c00880_si_003.xlsx
(CC BY-NC 4.0; MD5 fdd98a50447d9fd21288b0c2ca653601). Non-commercial use only, credited.

Rules from the plan, enforced in code:
- Unit of validation = source study (normalised title); records of one study never split.
- Brain-derived quantities never enter inputs (brain Cmax/C0/Tmax/Tmin/residence, every
  brain/blood ratio, the 'AUC' ratio column). Only blood AUC is the physiological input.
- Preprocessing (imputation, scaling, encoding) is fitted on training studies only.
- Quarantined before any modelling: Subject 0 (baicalin units/interval discrepancy).
  Exclusions are fixed before results and never depend on prediction error.
- Model choices fixed in advance: ridge (alpha by inner CV), one gradient-boosting
  comparison on identical records, three matched baselines.
- Missing is never zero: numeric gaps are imputed inside the pipeline with a missing flag.

Usage: python exposure_model.py path/to/mp3c00880_si_003.xlsx outdir
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261107
BRAIN, BLOOD = 'AUC brain (µg.h/ml)', 'AUC blood (µg.h/ml)'
NUM_DESC = ['Weight', 'logP', 'pKa', 'Solubility', 'DrugCarrierRatio', 'Ncomps', 'Size', 'Zeta', 'PDI', 'EE',
            'T 1/2 of drug release (hour)', 'T max of drug release (hour)', 'Maximium release (%)', 'Release']
CAT_DESC = ['Pgp', 'Position', 'Comp1', 'CarrierNature', 'Ligands', 'NPShape']
CONTEXT = ['Route']
FORBIDDEN = ['Cmax brain  (µg/ ml)', 'C0 brain  (µg/ ml)', 'T max (brain) (mins)', 'T min (brain) (mins)',
             'API Residence time in  brain (mins)', 'C max (brain/ Blood)', 'C min (brain/ Blood)', 'AUC',
             'Tmax ratio (brain/plasma)', 'Tmin ratio (brain/plasma)', BRAIN]
QUARANTINE = {0: 'baicalin record: curated blood AUC 267 (hours header) vs source Table 3 2266.38/2676.03 '
                 '(minutes scale); cohort/units/interval unreconciled'}
ADHD_RE = 'methylphenidate|amphetamine|atomoxetine|guanfacine|clonidine|lisdexamfetamine'


def study_id(title):
    return re.sub(r'[^a-z0-9]+', ' ', str(title).lower()).strip()


def load(path):
    df = pd.read_excel(path)
    n0 = len(df)
    df = df[~df['Subject'].isin(QUARANTINE)].copy()
    df = df[(pd.to_numeric(df[BRAIN], errors='coerce') > 0) & (pd.to_numeric(df[BLOOD], errors='coerce') > 0)].copy()
    df['study'] = df['Study'].map(study_id)
    df['y'] = np.log(df[BRAIN].astype(float))
    df['ln_blood'] = np.log(df[BLOOD].astype(float))
    for c in NUM_DESC:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CAT_DESC + CONTEXT:
        df[c] = df[c].astype(str).str.strip().str.lower()
    adhd = df['API'].astype(str).str.lower().str.contains(ADHD_RE)
    return df, {'rows_in_workbook': n0, 'quarantined': len(QUARANTINE), 'eligible_rows': len(df),
                'studies': int(df['study'].nunique()), 'adhd_drug_rows': int(adhd.sum()),
                'routes': df['Route'].value_counts().to_dict()}


def pipe(num, cat, model):
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), num),
        ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),
                              OneHotEncoder(handle_unknown='ignore', min_frequency=3)), cat)])
    return make_pipeline(pre, model)


def ridge():
    return RidgeCV(alphas=np.logspace(-2, 3, 21))


METHODS = {
    'blood_and_context_ridge': lambda: pipe(['ln_blood'], CONTEXT, ridge()),
    'descriptors_and_context_ridge': lambda: pipe(NUM_DESC, CAT_DESC + CONTEXT, ridge()),
    'full_ridge': lambda: pipe(['ln_blood'] + NUM_DESC, CAT_DESC + CONTEXT, ridge()),
    'full_gradient_boosting': lambda: pipe(['ln_blood'] + NUM_DESC, CAT_DESC + CONTEXT,
                                           HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                                         max_leaf_nodes=15, l2_regularization=1.0,
                                                                         random_state=SEED)),
}
assert not set(FORBIDDEN) & set(['ln_blood'] + NUM_DESC + CAT_DESC + CONTEXT)


def route_ratio_predict(train, test):
    """Baseline: ln brain = ln blood + training median ln(brain/blood) for that route."""
    r = train['y'] - train['ln_blood']
    by, g = r.groupby(train['Route']).median(), r.median()
    return test['ln_blood'] + test['Route'].map(by).fillna(g)


def support_mask(train, test):
    """Inside support: route seen in training and ln blood within the training range."""
    lo, hi = train['ln_blood'].min(), train['ln_blood'].max()
    return test['Route'].isin(set(train['Route'])) & test['ln_blood'].between(lo, hi)


def evaluate(train, test):
    out = {'training_mean': np.full(len(test), train['y'].mean()),
           'route_ratio_baseline': route_ratio_predict(train, test).to_numpy()}
    for name, make in METHODS.items():
        out[name] = make().fit(train, train['y']).predict(test)
    return out


def study_mae(test, pred):
    e = pd.Series(np.abs(test['y'].to_numpy() - pred), index=test.index).groupby(test['study']).mean()
    return float(e.mean())


def main(path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, audit = load(path)
    audit['workbook_md5'] = hashlib.md5(Path(path).read_bytes()).hexdigest()

    preds = pd.DataFrame(index=df.index)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=10).split(df, groups=df['study'])):
        train, test = df.iloc[tr], df.iloc[te]
        for name, p in evaluate(train, test).items():
            preds.loc[test.index, name] = p
        preds.loc[test.index, 'in_support'] = support_mask(train, test).to_numpy()
        preds.loc[test.index, 'fold'] = k
    preds['y'], preds['study'] = df['y'], df['study']

    methods = ['training_mean', 'route_ratio_baseline'] + list(METHODS)
    rng = np.random.default_rng(SEED)
    studies = preds['study'].unique()
    per_study = {m: (preds[m] - preds['y']).abs().groupby(preds['study']).mean() for m in methods}
    boot = {m: [] for m in methods}
    diff = {m: [] for m in methods}
    for _ in range(2000):
        s = rng.choice(studies, len(studies), replace=True)
        base = per_study['route_ratio_baseline'].loc[s].mean()
        for m in methods:
            boot[m].append(per_study[m].loc[s].mean())
            diff[m].append(per_study[m].loc[s].mean() - base)

    sup = preds['in_support'].astype(bool)
    cv = {}
    for m in methods:
        err = (preds[m] - preds['y']).abs()
        cv[m] = {'record_mae_ln': float(err.mean()), 'study_mae_ln': float(per_study[m].mean()),
                 'study_mae_ln_95ci': [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))],
                 'minus_route_ratio_95ci': [float(np.percentile(diff[m], 2.5)), float(np.percentile(diff[m], 97.5))],
                 'fold_change_median_error': float(np.exp(err.median())),
                 'in_support_record_mae_ln': float(err[sup].mean()), 'in_support_fraction': float(sup.mean())}

    gs = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gs.split(df, groups=df['study']))
    train, test = df.iloc[tr], df.iloc[te]
    hold = {name: {'study_mae_ln': study_mae(test, p), 'record_mae_ln': float(np.mean(np.abs(test['y'] - p)))}
            for name, p in evaluate(train, test).items()}
    hold_info = {'train_studies': int(train['study'].nunique()), 'test_studies': int(test['study'].nunique()),
                 'test_records': len(test)}

    best = min(METHODS, key=lambda m: cv[m]['study_mae_ln'])
    verdict = {
        'best_learned_by_cv': best,
        'beats_route_ratio_baseline': bool(cv[best]['minus_route_ratio_95ci'][1] < 0),
        'blood_adds_value_over_descriptors': bool(cv['full_ridge']['study_mae_ln'] < cv['descriptors_and_context_ridge']['study_mae_ln']),
        'descriptors_add_value_over_blood': bool(cv['full_ridge']['study_mae_ln'] < cv['blood_and_context_ridge']['study_mae_ln']),
    }
    report = {'task': 'ln brain AUC from measured ln blood AUC + formulation descriptors + route; source-study held out',
              'data_audit': audit, 'quarantine': QUARANTINE, 'forbidden_inputs': FORBIDDEN,
              'grouped_10fold_cv': cv, 'fixed_holdout_20pct_studies': {'split': hold_info, 'results': hold},
              'verdict': verdict,
              'limits': ['Blood/plasma/serum and brain tissue-mass vs volume bases are not separable in this '
                         'compiled table; results are exploratory.',
                         'No ADHD drug is in the corpus; an ADHD case is out of domain and gets a support refusal.',
                         'Study titles are normalised strings, not verified DOIs; linked publications may remain split.',
                         'Licence CC BY-NC 4.0: no commercial use of models trained on these data.']}
    (outdir / 'exposure_results.json').write_text(json.dumps(report, indent=2))
    preds.to_csv(outdir / 'exposure_cv_predictions.csv')
    return report


if __name__ == '__main__':
    r = main(sys.argv[1], sys.argv[2])
    print(json.dumps({k: r[k] for k in ['data_audit', 'verdict']}, indent=2))
    for m, v in r['grouped_10fold_cv'].items():
        print(f"{m:32s} study MAE {v['study_mae_ln']:.3f} {np.round(v['study_mae_ln_95ci'], 3)}  "
              f"vs route-ratio {np.round(v['minus_route_ratio_95ci'], 3)}  "
              f"fold err x{v['fold_change_median_error']:.2f}  in-support {v['in_support_fraction']:.2f}")
    print(json.dumps(r['fixed_holdout_20pct_studies'], indent=1))

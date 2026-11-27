"""Formulation and host-factor evidence layer (plan Appendices K-M). Research backlog.

Written at the event, 7 Nov 2026. NOT part of the scored EEG core: no neural or
physiological signal enters here, and nothing in this module may appear in the pitch as
a result. It exists so the team's delivery research is queryable under explicit rules,
and so the app can REFUSE questions the evidence cannot answer.

Three tables:
  data/hydrogel_formulations.csv  39 formulation rows from the two hydrogel source docs
  data/host_factor_evidence.csv   56 lifestyle/genotype/demographic barrier findings
  data/coating_core_numeric.csv   61 coating/core values (Appendix K), optional

Rules enforced here, each taken from the sources themselves:
  R1 Route decides whether the barrier is involved. Intranasal, intracerebroventricular,
     intraparenchymal and depot routes do not demonstrate barrier transport.
  R2 Never pool across pooling_group. Nose-to-brain intact-barrier rows, direct-brain
     rows, other sites (spinal cord, retina, tumour), in-vitro-only rows and
     material-only rows are separate families.
  R3 Disrupted-barrier rows (injury, stroke, tumour) are excluded from any
     intact-barrier statement.
  R4 Comparator must match. A fold value against an oral comparator is not comparable
     with a fold value against another gel.
  R5 Evidence tier gates the claim: 'A' (abstract/excerpt) and below cannot support a
     numeric claim without opening the full text; 'patent' and 'news' can support none.
  R6 Cationic surface charge is a flagged risk (Nat Commun charge series), not a
     disqualifier, and the flag is reported, never silently applied.
  R7 Host factors: a factor that is not person_measurable cannot personalise anything.
  R8 No host factor may be turned into an uptake coefficient: only one row in the host
     table is paired with a particle outcome, and it is qualitative.
  R9 Tracer scale: a leak measured with a small tracer does not license a claim about a
     60-200 nm particle. Only pore-diameter rows reach particle scale.
  R10 Chemical-induced models (scopolamine, trimethyltin, streptozotocin, aluminium
     chloride, LPS) are memory-loss models, not Alzheimer pathology; flagged on output.

Usage:
    python formulation_evidence.py <data_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

NON_BARRIER_ROUTES = {'intranasal', 'intracerebroventricular', 'intraparenchymal', 'subcutaneous',
                      'intramuscular', 'intravitreal'}
DISRUPTED = {'disrupted', 'bypassed', 'both'}
NUMERIC_TIERS = {'A'}          # may carry a number, flagged as excerpt-level
NO_CLAIM_TIERS = {'patent', 'news'}
CHEMICAL_MODELS = {'chemical_induced'}
PARTICLE_NM = (60, 200)        # the size range this project would deliver


class Refusal(Exception):
    """Raised when the evidence cannot support the question asked."""


def load(data_dir: Path):
    gels = pd.read_csv(data_dir / 'hydrogel_formulations.csv')
    host = pd.read_csv(data_dir / 'host_factor_evidence.csv')
    for frame, name in ((gels, 'hydrogel_formulations'), (host, 'host_factor_evidence')):
        if frame.empty:
            raise ValueError(f'{name} is empty')
    return gels, host


# ---------------------------------------------------------------- hydrogels

def barrier_transport_rows(gels: pd.DataFrame) -> pd.DataFrame:
    """R1+R3: rows that could speak to crossing an intact barrier. Expected: none."""
    return gels[(~gels['route'].isin(NON_BARRIER_ROUTES)) & (~gels['barrier_state'].isin(DISRUPTED))
                & (gels['pooling_group'] == 'nose_to_brain_intact')]


def comparable_set(gels: pd.DataFrame, pooling_group: str, comparator: str | None = None) -> pd.DataFrame:
    """R2+R4: one pooling group, optionally one comparator. Anything else is refused."""
    if pooling_group not in set(gels['pooling_group']):
        raise Refusal(f'unknown pooling group {pooling_group!r}')
    out = gels[gels['pooling_group'] == pooling_group]
    if comparator is not None:
        out = out[out['comparator'].str.contains(comparator, case=False, na=False)]
    return out


def rank_formulations(gels: pd.DataFrame, pooling_group: str, comparator: str | None = None):
    """R4+R5: refuse to rank unless at least two rows share a comparator and carry numbers."""
    rows = comparable_set(gels, pooling_group, comparator)
    usable = rows[rows['value'].notna() & rows['evidence_status'].isin(NUMERIC_TIERS)]
    if len(usable) < 2:
        raise Refusal(
            f'cannot rank {pooling_group}: {len(usable)} rows carry a number at an admissible evidence tier '
            f'(need 2 or more sharing a comparator). Request: open the full texts, or measure one panel in one assay.')
    units = set(usable['unit'])
    if len(units) > 1:
        raise Refusal(f'cannot rank {pooling_group}: units differ {sorted(units)}; no conversion is supported')
    comparators = set(usable['comparator'])
    if comparator is None and len(comparators) > 1:
        raise Refusal(f'cannot rank {pooling_group}: comparators differ {sorted(comparators)} (R4)')
    return usable.sort_values('value', ascending=False)[['formulation_id', 'material', 'cargo', 'value', 'unit', 'comparator', 'caveat']]


def flags(row: pd.Series) -> list[str]:
    out = []
    if str(row.get('surface_charge', '')).startswith('cationic'):
        out.append('cationic surface: flagged tissue-reaction risk (R6)')
    if row.get('model_type') in CHEMICAL_MODELS:
        out.append('chemically induced memory-loss model, not Alzheimer pathology (R10)')
    if row.get('evidence_status') in NO_CLAIM_TIERS:
        out.append(f"evidence tier {row['evidence_status']}: supports no result claim (R5)")
    if row.get('route') in NON_BARRIER_ROUTES:
        out.append(f"route {row['route']}: does not demonstrate barrier transport (R1)")
    if row.get('barrier_state') in DISRUPTED:
        out.append('barrier disrupted or bypassed in this model (R3)')
    if str(row.get('species', '')).startswith('none') or row.get('model_type') == 'in_vitro':
        out.append('no animal data (R5)')
    return out


def card(gels: pd.DataFrame, formulation_id: str) -> dict:
    hit = gels[gels['formulation_id'] == formulation_id]
    if hit.empty:
        raise Refusal(f'no formulation {formulation_id!r}')
    r = hit.iloc[0]
    return {'formulation_id': r['formulation_id'], 'material': r['material'], 'cargo': r['cargo'],
            'route': r['route'], 'pooling_group': r['pooling_group'],
            'outcome': {'measure': r['outcome_measure'], 'value': None if pd.isna(r['value']) else r['value'],
                        'unit': r['unit'], 'comparator': r['comparator'], 'direction': r['direction']},
            'evidence_status': r['evidence_status'], 'read_status': r['read_status'],
            'flags': flags(r), 'caveat': r['caveat'], 'citation': r['citation'], 'url': r['url'],
            'supports_barrier_transport_claim': False}


# -------------------------------------------------------------- host factors

def person_measurable(host: pd.DataFrame) -> pd.DataFrame:
    return host[host['person_measurable'] == 'yes']


def host_uptake_coefficient(host: pd.DataFrame, factor: str):
    """R8: always refuses. Kept as a callable so the refusal is testable, not just prose."""
    rows = host[host['factor'] == factor]
    if rows.empty:
        raise Refusal(f'no host evidence for {factor!r}')
    paired = rows[rows['paired_with_particle_outcome'] == 'yes']
    raise Refusal(
        f'cannot produce an uptake coefficient for {factor!r}: {len(paired)} of {len(rows)} rows are paired with a '
        f'particle outcome, and the whole table contains one such row (qualitative, mouse). '
        f'Required: a particle panel measured in people or animals whose host factor was measured.')


def particle_scale_rows(host: pd.DataFrame) -> pd.DataFrame:
    """R9: rows whose measurement scale could admit a 60-200 nm particle."""
    return host[host['quantity'].astype(str).str.contains('pore_diameter', na=False)]


def individual_variation_bound(host: pd.DataFrame) -> dict:
    """The ceiling on personalisation: receptor abundance varies ~13-14x between people and
    age and sex explain none of it."""
    rows = host[host['factor'] == 'inter_individual_variability_not_explained_by_age_or_sex']
    explained = rows[rows['quantity'] == 'age_and_sex_explain_transporter_abundance']
    return {'glut1_fold_range': 57.8 / 4.05, 'mct1_fold_range': 14.10 / 1.05,
            'abc_fold_range': '3 to 11', 'slc_fold_range': '2 to 14',
            'age_and_sex_significant_associations': int(explained['value'].iloc[0]) if len(explained) else None,
            'source': 'Al-Majdoub et al. 2019, 22 donors, frontal cortex microvessels, targeted mass spectrometry',
            'meaning': 'The receptor a ligand targets varies about 13-14 fold between people and none of that '
                       'variation was explained by age or sex. Lifestyle and demographic features cannot stand in '
                       'for measuring the receptor.'}


def summarise(gels: pd.DataFrame, host: pd.DataFrame) -> dict:
    gel_tiers = gels['evidence_status'].value_counts().to_dict()
    return {
        'hydrogels': {
            'rows': len(gels),
            'pooling_groups': gels['pooling_group'].value_counts().to_dict(),
            'routes': gels['route'].value_counts().to_dict(),
            'gelation_triggers': gels['gelation_trigger'].value_counts().to_dict(),
            'surface_charge': gels['surface_charge'].value_counts().to_dict(),
            'evidence_tiers': gel_tiers,
            'rows_with_a_number': int(gels['value'].notna().sum()),
            'human_rows': int((gels['species'] == 'human').sum()),
            'rows_demonstrating_intact_barrier_transport': len(barrier_transport_rows(gels)),
            'chemical_model_rows': int((gels['model_type'] == 'chemical_induced').sum()),
            'cationic_rows': int(gels['surface_charge'].astype(str).str.startswith('cationic').sum()),
        },
        'host_factors': {
            'rows': len(host),
            'factors': host['factor'].nunique(),
            'person_measurable_rows': int((host['person_measurable'] == 'yes').sum()),
            'rows_paired_with_particle_outcome': int((host['paired_with_particle_outcome'] == 'yes').sum()),
            'rows_at_particle_scale': len(particle_scale_rows(host)),
            'unread_sources': int((host['read_status'].astype(str).str.startswith('not read')).sum()),
            'individual_variation_bound': individual_variation_bound(host),
        },
    }


def main(data_dir: str, out_dir: str) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gels, host = load(data_dir)
    report = summarise(gels, host)
    refusals = {}
    for group in sorted(set(gels['pooling_group'])):
        try:
            ranked = rank_formulations(gels, group)
            refusals[group] = {'ranked': ranked.to_dict('records')}
        except Refusal as e:
            refusals[group] = {'refused': str(e)}
    report['ranking_attempts'] = refusals
    try:
        host_uptake_coefficient(host, 'cigarette_smoking')
    except Refusal as e:
        report['host_coefficient_attempt'] = str(e)
    (out_dir / 'formulation_evidence_report.json').write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == '__main__':
    rep = main(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=2, default=str))

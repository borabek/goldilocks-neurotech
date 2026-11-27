"""Tests for the formulation and host-factor evidence layer.

Each test pins one rule from the source documents, so a later edit that loosens a rule
fails here instead of reaching a slide.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from formulation_evidence import (  # noqa: E402
    DISRUPTED, NON_BARRIER_ROUTES, Refusal, barrier_transport_rows, card, comparable_set,
    flags, host_uptake_coefficient, individual_variation_bound, load, particle_scale_rows,
    person_measurable, rank_formulations, summarise)

DATA = HERE / 'data'


@pytest.fixture(scope='module')
def tables():
    return load(DATA)


def test_tables_load_with_expected_shape(tables):
    gels, host = tables
    assert len(gels) == 39 and len(host) == 56
    assert gels['formulation_id'].is_unique and host['finding_id'].is_unique


def test_no_hydrogel_row_demonstrates_intact_barrier_transport(tables):
    """R1+R3: every gel row is intranasal, direct-brain, depot, another site or in vitro."""
    gels, _ = tables
    assert len(barrier_transport_rows(gels)) == 0


def test_no_human_hydrogel_row(tables):
    """The Alzheimer's doc's headline: no human hydrogel trial was found."""
    gels, _ = tables
    assert (gels['species'] == 'human').sum() == 0


def test_pooling_groups_are_disjoint_and_named(tables):
    gels, _ = tables
    expected = {'nose_to_brain_intact', 'direct_brain', 'systemic_depot', 'other_site',
                'in_vitro_only', 'material_only', 'unassigned_route'}
    assert set(gels['pooling_group']) <= expected


def test_ranking_refuses_when_comparators_or_units_differ(tables):
    """R4: the nose-to-brain group has numbers, but in different units and against
    different comparators, so no ranking is possible. Units are checked first."""
    gels, _ = tables
    with pytest.raises(Refusal) as e:
        rank_formulations(gels, 'nose_to_brain_intact')
    msg = str(e.value).lower()
    assert any(k in msg for k in ('units differ', 'comparators differ', 'rows carry a number'))


def test_ranking_refuses_unknown_group(tables):
    gels, _ = tables
    with pytest.raises(Refusal):
        comparable_set(gels, 'not_a_group')


def test_ranking_refuses_across_groups_even_with_numbers(tables):
    """M01's 25-fold (oral comparator) must never be ranked beside H02's 107 percent."""
    gels, _ = tables
    m01 = gels[gels['formulation_id'] == 'M01'].iloc[0]
    h02 = gels[gels['formulation_id'] == 'H02'].iloc[0]
    assert m01['pooling_group'] == h02['pooling_group'] == 'nose_to_brain_intact'
    assert m01['unit'] != h02['unit'] and m01['comparator'] != h02['comparator']
    with pytest.raises(Refusal):
        rank_formulations(gels, 'nose_to_brain_intact')


def test_cationic_rows_are_flagged(tables):
    gels, _ = tables
    cat = gels[gels['surface_charge'].astype(str).str.startswith('cationic')]
    assert len(cat) >= 3
    for _, r in cat.iterrows():
        assert any('cationic' in f for f in flags(r))


def test_chemical_models_are_flagged(tables):
    gels, _ = tables
    chem = gels[gels['model_type'] == 'chemical_induced']
    assert len(chem) >= 6
    for _, r in chem.iterrows():
        assert any('chemically induced' in f for f in flags(r))


def test_patent_and_news_rows_support_no_claim(tables):
    gels, _ = tables
    low = gels[gels['evidence_status'].isin(['patent', 'news'])]
    assert len(low) >= 3
    for _, r in low.iterrows():
        assert any('supports no result claim' in f for f in flags(r))


def test_card_never_supports_a_barrier_claim(tables):
    gels, _ = tables
    for fid in ['H01', 'H16', 'M01', 'M22']:
        c = card(gels, fid)
        assert c['supports_barrier_transport_claim'] is False
        assert c['flags']


def test_direct_brain_row_is_flagged_as_bypassing_the_barrier(tables):
    gels, _ = tables
    c = card(gels, 'H16')
    assert any('bypassed' in f or 'disrupted' in f for f in flags(gels[gels.formulation_id == 'H16'].iloc[0]))
    assert c['route'] == 'intracerebroventricular'


def test_host_uptake_coefficient_always_refuses(tables):
    _, host = tables
    for factor in ['cigarette_smoking', 'chronic_alcohol_use', 'APOE4_genotype', 'sleep_fragmentation']:
        with pytest.raises(Refusal) as e:
            host_uptake_coefficient(host, factor)
        assert 'cannot produce an uptake coefficient' in str(e.value)


def test_only_one_host_row_is_paired_with_a_particle_outcome(tables):
    _, host = tables
    paired = host[host['paired_with_particle_outcome'] == 'yes']
    assert len(paired) == 1 and paired.iloc[0]['finding_id'] == 'D04'
    assert paired.iloc[0]['status'] == 'qualitative'


def test_non_person_measurable_factors_are_excluded_from_personalisation(tables):
    _, host = tables
    pm = person_measurable(host)
    assert len(pm) == 49
    excluded = set(host['factor']) - set(pm['factor'])
    assert 'inter_individual_variability_not_explained_by_age_or_sex' in excluded


def test_particle_scale_rule(tables):
    """R9: only the pore-diameter row reaches 60-200 nm; tracers are 0.376-70 kDa."""
    _, host = tables
    rows = particle_scale_rows(host)
    assert len(rows) == 1 and rows.iloc[0]['finding_id'] == 'A02'
    tracers = pd.to_numeric(host['tracer_kda'], errors='coerce').dropna()
    assert tracers.max() <= 70


def test_individual_variation_bound_is_the_personalisation_ceiling(tables):
    _, host = tables
    b = individual_variation_bound(host)
    assert 13 < b['glut1_fold_range'] < 15
    assert 12 < b['mct1_fold_range'] < 14
    assert b['age_and_sex_significant_associations'] == 0


def test_summary_counts_are_self_consistent(tables):
    gels, host = tables
    s = summarise(gels, host)
    assert s['hydrogels']['rows_demonstrating_intact_barrier_transport'] == 0
    assert s['hydrogels']['human_rows'] == 0
    assert s['host_factors']['rows_paired_with_particle_outcome'] == 1
    assert sum(s['hydrogels']['pooling_groups'].values()) == len(gels)

# Exposure model result (research track, not the scored EEG core)

Run 7 November 2026 at the event. Data: Yousfan et al. 2023, *Molecular Pharmaceutics*, supporting workbook `mp3c00880_si_003.xlsx` (MD5 `fdd98a50447d9fd21288b0c2ca653601`, CC BY-NC 4.0, non-commercial use only).

**Task.** Predict ln(brain AUC) from measured ln(blood AUC), formulation descriptors and route, on **source studies the model has never seen**.

**Data after the fixed exclusions.** 403 rows in the workbook; 1 quarantined (the baicalin record, whose blood AUC cannot be reconciled with the source table's units and interval); 402 eligible rows across **159 study-title groups**, which the leakage audit below resolves to **157 source groups**. Routes: IV 174, intranasal 157, oral 48, intraperitoneal 23. **No ADHD drug appears in the corpus** (0 rows), which is why an ADHD case is out of domain here.

## Grouped 10-fold cross-validation, held out by source study

Error is mean absolute error on the natural-log scale, averaged per study. "vs baseline" is the 95% bootstrap interval of the difference against the route-ratio baseline (negative favours the model).

| Method | Study MAE (ln) | 95% CI | vs baseline | Median error as a fold |
|---|---:|---|---|---:|
| Training mean | 3.050 | 2.77–3.35 | +1.33 to +2.02 | ×16.6 |
| Route brain/blood ratio (baseline) | 1.372 | 1.21–1.55 | — | ×2.58 |
| **Blood + route (ridge)** | **1.316** | 1.16–1.49 | **−0.11 to −0.00** | ×2.55 |
| Formulation descriptors + route (ridge) | 3.308 | 2.97–3.66 | +1.58 to +2.30 | ×13.99 |
| Blood + descriptors (ridge) | 1.423 | 1.24–1.63 | −0.07 to +0.19 | ×2.51 |
| Blood + descriptors (gradient boosting) | 1.519 | 1.35–1.68 | +0.00 to +0.29 | ×3.40 |

Fixed untouched split (20% of studies: 127 train, 32 test studies, 93 records) reproduces the ordering: blood + route 1.321, route baseline 1.383, descriptors only 3.279, blood + descriptors 1.509.

## What this says

1. **Measured blood exposure carries nearly all the signal, and the one apparent win does not survive a leakage audit.** Under title grouping, blood + route beat the route-ratio baseline by a hair: the difference interval was −0.11 to −0.00, touching zero. The leakage audit below found two pairs of study titles that report the same measurement, merged them into single groups, and the interval became −0.10 to +0.02. The claim "beats the route-specific ratio baseline" therefore **fails**. The honest reading is "no better than a route-specific ratio".
2. **Formulation descriptors do not transfer across studies.** On their own they are barely better than predicting the mean (×14 median error). *Added to blood exposure they make the model worse.* This is the negative result that matters: the premise that particle design predicts brain delivery across the literature does not hold once whole studies are held out.
3. **Nothing here is accurate enough to act on.** The best model's median error is a factor of 2.5 on brain AUC.
4. **Why the published work is not contradicted.** Yousfan et al. report strong fits with linear mixed-effects models that include a study random intercept; that estimates within-study structure. Held out by whole study, the between-study offset is exactly what cannot be estimated. The two results are compatible and measure different things.

## Leakage audit of the study grouping

`leakage_audit.py` (run 7 November 2026, output `out/leakage_audit.json`) tests the assumption that a normalised
title string identifies a source study, which is what the grouped validation depends on. Four detectors, each
proved able to fire by an injected synthetic violation before the real data is touched (`--self-test`, 11 checks,
all passing):

| Detector | What it looks for | Hits |
|---|---|---:|
| D1 | The same measured (blood AUC, brain AUC) pair, bit-for-bit, under two study titles | 1 |
| D2 | The same outcome pair to four significant figures under two titles | 2 |
| D3 | The same full formulation record (payload, route, descriptors, outcome) under two titles | 0 |
| D4 | Two titles resolving to one source by token containment or Jaccard ≥ 0.9 | 0 |

The two duplicated measurements are real literature duplication, not a coding artefact:

- blood 2612.31 / brain 2389.17 appears under both the haloperidol solid-lipid-nanoparticle HPLC method study and
  the haloperidol nose-to-brain solid-lipid-nanoparticle pharmacokinetics study.
- blood 2179 / brain 3740 (four significant figures) appears under both the lectin-conjugated PEG-PLA intranasal
  nanoparticle study and the UEA-I-bearing intranasal nanoparticle study.

Each duplication puts one measurement in a training fold and a test fold at the same time, which group-held-out
validation cannot prevent when the groups are titles.

**Effect, measured not asserted.** The audit merges linked titles with union-find (159 title groups → 157 source
groups) and reruns the identical grouped 10-fold cross-validation and fixed holdout. Before interpreting that
rerun it reproduces the saved title-grouped numbers with its own code to within 1e-9 and refuses to continue
otherwise, so any difference can only come from the grouping. Study MAE on the natural-log scale:

| Method | Title-grouped | Source-merged |
|---|---:|---:|
| Training mean | 3.050 | 3.057 |
| Route brain/blood ratio (baseline) | 1.372 | 1.372 |
| Blood + route (ridge) | 1.316 | 1.335 |
| Formulation descriptors + route (ridge) | 3.308 | 3.362 |
| Blood + descriptors (ridge) | 1.423 | 1.372 |
| Blood + descriptors (gradient boosting) | 1.519 | 1.565 |

One headline verdict changes: `beats_route_ratio_baseline` goes from true (95% interval −0.109 to −0.004) to
**false** (−0.097 to +0.015). The two findings that matter for the pitch do not change: blood exposure still
carries the signal, and formulation descriptors still do not transfer across studies. The audit exits non-zero
when a verdict changes, so this cannot be missed in a rerun.

## Limits

- Blood/plasma/serum bases and brain tissue-mass vs volume bases are not separable in this compiled table, so all of it is exploratory.
- Study identity is a normalised title string, not a verified DOI. This is now audited rather than assumed: `leakage_audit.py` found two duplicated measurements split across folds and the headline comparison changes once they are merged (see the leakage audit section). Resolving identities by DOI remains outstanding work; the remaining 157 groups are unverified beyond the audit's four detectors.
- CC BY-NC 4.0: no commercial use of models trained on these data.
- Brain-derived quantities (brain Cmax, C0, Tmax, Tmin, residence time, every brain/blood ratio, and the workbook's own `AUC` ratio column) are excluded from the inputs by an assertion in the code, not by convention.

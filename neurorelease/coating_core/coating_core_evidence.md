# Coating and core choice: what can be predicted from a property of a person's cells or blood

Question: can a measurable property pick polyethylene glycol versus heparosan, or iron versus gold (including gas-core nanobubbles)?

Short answer: one candidate property has a mechanism and numbers (anti-polyethylene glycol antibody titer in blood), one has a clear cell-level split in vitro (phagocytic capacity), and for iron versus gold there is no matched evidence at all. No matched pairs with a varying host property exist yet, so no recommendation can be made. `bbb/choice.py` refuses until they do.

## What the evidence supports

| Decision | Candidate property | What was found | Status |
|---|---|---|---|
| Polyethylene glycol versus a non-polyethylene glycol coat | Anti-polyethylene glycol antibody titer, from a blood test | 83% of 500 healthy German donors carry immunoglobulin G or M against polyethylene glycol; plasma pools span below 0.5 to 110.5 ug/mL immunoglobulin G; binding is weak at 2 kDa (2690 nM) and strong at 10 kDa (24 nM); antibody in the particle corona raised macrophage uptake in vitro (C03). Other cohorts report 24% to 72%, so the number depends on the assay. | Measurable per person, mechanism plausible. No study has set titer against particle outcome in matched pairs. No clearance data in these studies. |
| Polyethylene glycol versus heparosan | Phagocytic capacity of the cell | Heparosan-coated gold was taken up up to 230x (RAW264.7), 70x (J774A.1) and 45x (DC2.4) more than polyethylene glycol-coated gold; no difference in endothelial cells or three non-phagocytic lines (C01). | Strongest cell-level split found. In vitro, one core, mechanism unknown. |
| Polyethylene glycol versus heparosan | Brain cell type | Polyethylene glycol 2 kDa lowered uptake in microglia, astrocytes, oligodendrocyte precursors and neural stem cells; neurons took up under 0.5% either way. Order: microglia > astrocytes > oligodendrocyte precursors > neural stem cells > neurons (C04). The control coat was dextran, not heparosan. | No heparosan arm in brain cells. |
| Polyethylene glycol versus heparosan | Brain endothelial glycocalyx density | Brain endothelium: 40.1% surface coverage and 301 nm thickness, versus heart 15.1% and 136 nm, lung 3.7% and 65 nm (mouse). Heparan sulfate proteoglycans mediated crossing of neural stem cell exosomes in vitro (C10). | Hypothesis only. No particle data. |
| Iron versus gold | Anything | No study found with the same size, coat and cell model and only the core changed. Gold at 80 nm entered brain endothelial cells poorly versus silica-polymer particles, but size, charge and coat all differed (C07). Iron oxide uptake and neuron damage depended on coat and size and on whether glia were present (C05). | Not answerable. |
| Nanobubbles (iron or gold) | Anything | Only an iron oxide microbubble study with focused ultrasound in glioma rats (C09). Focused ultrasound plus tumour means a disrupted barrier, which the screener excludes by rule. No gold nanobubble study was readable. | Out of scope for barrier transport. |

## What this means for a heparosan design

1. No study found shows heparosan circulating longer than polyethylene glycol. The heparosan liposome paper did not measure circulation half-life or repeat-dose accelerated clearance (C02).
2. The only head-to-head says heparosan is taken up much more by macrophages (C01). If that carries to animals, expect faster capture by liver and spleen macrophages and more microglial uptake. That helps if microglia are the target and hurts if they are not.
3. Heparosan showed no antibody response in three rats over 63 weeks (C02). Human anti-heparosan antibodies have not been measured in anything read here. Anti-polyethylene glycol titer only argues against polyethylene glycol, not for heparosan.
4. Endothelium (the first cell a particle meets at the barrier) showed no coating difference in the one comparison available, so the barrier crossing step may not be where the choice matters.

## The experiment that would make the claim testable

A crossed panel, one study design, many hosts:
- Particles: one size within 25% (about 60 nm), grafting density matched, coating arms polyethylene glycol 5 kDa and heparosan at matched chain mass; core arms iron oxide and gold crossed with both coats, so coating-by-core interaction is estimable. Add a matched uncoated arm.
- Hosts: at least 10 donors with plasma and with microglia and brain endothelial cells (induced pluripotent stem cell derived is acceptable). Measure anti-polyethylene glycol immunoglobulin G and M in each donor's plasma before the run.
- Readouts, same donor for both particles of a pair: uptake per cell by inductively coupled plasma mass spectrometry in each cell type, in the donor's own plasma and in serum-free medium; transwell permeability for the endothelial model.
- Record `swap_group`, `coating_class`, `core_material` and the host property per row. `choose_coating.py` then reports the pairs, the gate, the interaction slope and a recommendation only if at least 10 studies or donors and 8 distinct host values (continuous) or 5 studies per group (binary) are present.

## Files
- `coating_core_evidence.csv`: ten papers, what each can and cannot support, read status.
- `bbb/choice.py`, `choose_coating.py`: the pairing, gate, interaction fit and recommendation.

## Found but not read
Blocked or unreachable pages: PMC5881748 (heparosan liposomes, PubMed Central copy), PMC6874909, PMC5595925 (anti-polyethylene glycol prevalence), PMC4364595 (iron oxide in brain cells), PMC4955054, PMC10463668, the multitheragnostic nanobubble paper (hosted at cgmh.org.tw, blocked), thno.org v14p4076, the August 2023 Journal of Nanobiotechnology result, the Wilhelm lab thesis on shareok.org. Titles unknown for PMC7904660, PMC12388889, PMC5881802, PMC6223124 (returned by the heparosan search, rate limited when fetched). Nothing from these is used above.

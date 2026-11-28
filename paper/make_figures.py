"""Every figure in the paper, from result files in the repository only.

Usage: python paper/make_figures.py      (writes paper/figures/*.pdf)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yasa_evaluate import windows  # noqa: E402

OUT = ROOT / 'paper/figures'
YASA, V2, CNN = '#257de0', '#f47230', '#00b894'     # fixed family colours (validated, light surface)
INK, MUTED, GRID = '#1f1f1f', '#6b6b6b', '#e4e4e4'
DATASETS = ['boas', 'sleep_edf', 'dodh', 'hmc']
NAMES = {'boas': 'BOAS\n(headband)', 'sleep_edf': 'Sleep-EDF', 'dodh': 'DOD-H', 'hmc': 'HMC'}
STAGE_ROWS = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}          # W on top, then REM, N1, N2, N3
STAGE_TICKS = ['N3', 'N2', 'N1', 'REM', 'W']

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def load(p):
    return json.loads((ROOT / p).read_text())


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_ranking():
    b = load('results/paper/benchmark.json')['table']
    headband = load('results/paper/benchmark.json')['ranking']['models']
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    x = np.arange(len(DATASETS))
    labelled = {'yasa': ('YASA (PSG-trained)', YASA), 'v2': ('v2 (features)', V2),
                'cnn_context_sqrtbalanced': ('best CNN on BOAS', CNN), 'cnn_balanced': ('worst CNN on BOAS', CNN)}
    for m in headband + ['yasa']:
        y = [b[ds][m]['kappa']['mean'] for ds in DATASETS]
        if m in labelled:
            name, col = labelled[m]
            ls = '--' if m == 'cnn_balanced' else '-'
            ax.plot(x, y, color=col, lw=2, ls=ls, marker='o', ms=4, mec='white', mew=0.8, zorder=3)
            ax.annotate(name, (x[-1], y[-1]), xytext=(5, 0), textcoords='offset points', va='center',
                        fontsize=6.5, color=INK)
        else:
            ax.plot(x, y, color=CNN, lw=0.8, alpha=0.45, zorder=2)
    ax.plot([], [], color=CNN, lw=0.8, alpha=0.45, label='other CNN variants')
    ax.legend(loc='lower left', frameon=False)
    ax.set_xticks(x, [NAMES[d] for d in DATASETS])
    ax.set_xlim(-0.2, len(DATASETS) - 1 + 1.35)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Cohen's kappa (mean over people)")
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / 'fig1_ranking.pdf')
    plt.close(fig)


def boas_wake_recall(model_dir):
    hist = load('results/v2/historical_boas_predictions.json')
    pred = load(f'results/{model_dir}/cnn_predictions.json')
    vals = []
    for s, r in hist.items():
        y, good = np.array(r['reference']), np.array(r['good']).astype(bool)
        m = good & (y == 0) & windows(y, np.array(r['onset'], float))['primary_complete_recording']
        if m.any():
            vals.append(np.mean(np.array(pred[s])[m] == 0))
    return float(np.mean(vals))


def fig_mechanism():
    seeds = load('results/prereg_v2_1/seeds_matched_passes.json')
    occ = load('results/prereg_v2_1/band_occlusion.json')
    fig, (a, c) = plt.subplots(1, 2, figsize=(7.1, 2.35), gridspec_kw={'width_ratios': [1.15, 1]})
    recipes = [('cnn', 'plain CNN', CNN, 'o'), ('cnn_hfdrop', '16-30 Hz dropout', INK, 's'),
               ('cnn_lowpass16', '16 Hz low-pass', MUTED, '^')]
    seed_list = seeds['seeds']
    for j, (rec, label, col, mk) in enumerate(recipes):
        for k, ds in enumerate(DATASETS):
            vals = []
            for s in seed_list:
                if rec == 'cnn':
                    name = f'cnn_seed{s}'                      # 10-pass plain CNN at every seed
                else:
                    name = rec if s == seed_list[0] else f'{rec}_seed{s}'
                vals.append(boas_wake_recall(name) if ds == 'boas' else seeds['mean_wake_recall'][name][ds])
            xs = k + (j - 1) * 0.22 + np.array([-0.05, 0, 0.05])
            a.plot(xs, vals, ls='none', marker=mk, ms=4.2, color=col, mec='white', mew=0.6,
                   label=label if k == 0 else None, zorder=3)
            a.plot([k + (j - 1) * 0.22 - 0.08, k + (j - 1) * 0.22 + 0.08], [np.mean(vals)] * 2, color=col, lw=1.2)
    a.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS])
    a.set_ylim(0, 1)
    a.set_ylabel('Wake recall (three seeds each)')
    a.legend(loc='upper right', frameon=False, ncol=1)
    a.set_title('(a) Wake recall across training seeds', loc='left')
    grid(a)

    bands = occ['bands_hz']
    xb = np.arange(len(bands))
    styles = {'boas': (CNN, '-', 'BOAS'), 'sleep_edf': (V2, '-', 'Sleep-EDF'), 'dodh': (YASA, '--', 'DOD-H'),
              'hmc': (INK, ':', 'HMC')}
    for ds, (col, ls, lab) in styles.items():
        bd = occ['datasets'][ds]['bands']
        wake = [bd[b]['wake_prob_change']['mean'] for b in bands]
        c.plot(xb, wake, color=col, ls=ls, lw=1.6, marker='o', ms=3.2, label=lab)
    c.axhline(0, color=MUTED, lw=0.7)
    c.set_xticks(xb, [b.replace('-', '–') for b in bands], rotation=0)
    c.set_xlabel('Band removed at test time (Hz)')
    c.set_ylabel('Change in wake probability\non wake epochs (frozen CNN)')
    c.set_yscale('symlog', linthresh=0.1)
    c.set_yticks([-0.1, -0.05, 0, 0.05, 0.1, 0.3, 0.6], ['−0.10', '−0.05', '0', '0.05', '0.10', '0.30', '0.60'])
    c.legend(loc='upper right', frameon=False)
    c.set_title('(b) Band occlusion on the frozen CNN', loc='left')
    grid(c)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(OUT / 'fig2_mechanism.pdf')
    plt.close(fig)


def fig_failure(rec='SC401'):
    ext = load('results/v2/external_predictions.json')[rec]
    cnn = load('results/external_all/cnn_external_predictions.json')[rec]
    yasa = load('results/yasa/yasa_predictions.json')['external'][rec]
    y, good = np.array(ext['reference']), np.array(ext['good']).astype(bool)
    hours = (np.array(ext['onset'], float) - ext['onset'][0]) / 3600
    rows = [('Reference', y), ('v2', np.array(ext['model'])), ('CNN', np.array(cnn)), ('YASA', np.array(yasa))]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.1, 2.9), sharex=True)
    for ax, (name, s) in zip(axes, rows):
        m = good & (y < 5) & (s >= 0) & (s < 5)
        level = np.full(len(s), np.nan)
        level[m] = [STAGE_ROWS[int(v)] for v in s[m]]
        ax.step(hours, level, where='post', color=INK if name == 'Reference' else MUTED, lw=0.6)
        sleep_min = int(np.sum((s[m] > 0) & (s[m] < 5)) * 0.5)
        ax.set_yticks(range(5), STAGE_TICKS)
        ax.set_ylim(-0.6, 4.6)
        ax.text(1.005, 0.5, f'{name}\n{sleep_min} min sleep', transform=ax.transAxes, va='center', fontsize=6.5)
        ax.tick_params(axis='y', labelsize=5.5)
        grid(ax, 'x')
    axes[-1].set_xlabel(f'Hours from the start of recording {rec} (Sleep-EDF)')
    fig.tight_layout(pad=0.3, h_pad=0.2, rect=(0, 0, 0.9, 1))
    fig.savefig(OUT / 'fig3_failure_case.pdf')
    plt.close(fig)


def fig_risk_coverage():
    rc = load('results/paper/risk_coverage_curves.json')
    cov = np.array(rc['coverage'])
    titles = {'boas_test': 'BOAS', 'sleep_edf': 'Sleep-EDF', 'dodh': 'DOD-H', 'hmc': 'HMC'}
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.85), sharey=True)
    for ax, (cohort, t) in zip(axes, titles.items()):
        for name, col, lab in [('v2', V2, 'v2'), ('cnn', CNN, 'CNN')]:
            d = rc['cohorts'][cohort][name]
            ax.plot(cov[1:], d['risk'][1:], color=col, lw=1.6, label=f"{lab} (AURC {d['aurc']:.2f})")
        ax.set_title(t, loc='left')
        ax.set_xlabel('Coverage')
        ax.set_ylim(0, 0.85)
        ax.legend(loc='upper left', frameon=False, fontsize=6)
        grid(ax)
    axes[0].set_ylabel('Error rate among\nepochs kept')
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / 'fig4_risk_coverage.pdf')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_ranking()
    fig_mechanism()
    fig_failure()
    if (ROOT / 'results/paper/risk_coverage_curves.json').exists():
        fig_risk_coverage()
    print('figures written to', OUT)


if __name__ == '__main__':
    main()

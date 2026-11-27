"""Three things the delivery pack still needed: the vendor comparison, confusion matrices, and a
failure gallery for the model that now leads.

1. Vendor-covered mask. The released Bitbrain scorer ships with BOAS and abstains on some epochs.
   v2 was compared to it on its covered mask; the models trained tonight never were. This is the
   only like-for-like comparison available — same hardware, same people, same epochs — and it is a
   far more defensible reference point than a published figure from full polysomnography.

2. Confusion matrices, pooled over the 20 held-out people, for the frozen v2 model, YASA, the
   context model and the ensemble. Where N1 goes is the first question a sleep researcher asks.

3. Failure gallery. The three held-out nights where the leading model's total-sleep estimate is
   worst, each with its reference and predicted hypnogram, and a cautious sentence about what went
   wrong. The plan lists this as priority 10 of the delivery pack and it existed only for v1.

Nothing here refits or reselects anything. Usage: python src/vendor_and_failures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/vendor_and_failures'
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']

MODELS = {
    'Frozen v2 feature model': ('results/v2/historical_boas_predictions.json', 'model'),
    'YASA 0.7.0': ('results/yasa/yasa_predictions.json', 'historical_boas'),
    'CNN challenger': ('results/cnn/cnn_predictions.json', None),
    'Context model': ('results/cnn_context/context_predictions.json', None),
    'Ensemble of seven models': ('results/cnn_posthoc/ensemble_predictions.json', None),
}


def load(spec):
    path, key = spec
    p = ROOT / path
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if key == 'model':
        return {s: np.array(v['model']) for s, v in d.items()}
    if key:
        d = d.get(key)
        if d is None:
            return None
    return {s: np.array(v) for s, v in d.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads((ROOT / 'results/v2/historical_boas_predictions.json').read_text())
    subs = sorted(ref, key=lambda s: int(s.split('-')[1]))
    feats = {s: dict(np.load(ROOT / 'data/features' / f'{s}.npz')) for s in subs}
    preds = {k: load(v) for k, v in MODELS.items()}
    preds = {k: v for k, v in preds.items() if v}

    # ---------------- 1. vendor-covered mask ----------------
    rows, cover = [], []
    for name in list(preds) + ['Released Bitbrain scorer']:
        per = []
        for s in subs:
            y = np.array(ref[s]['reference']); good = np.array(ref[s]['good']).astype(bool)
            vend = feats[s]['vendor']
            m = good & (y < 5) & (vend >= 0) & (vend < 5)
            if not m.any():
                continue
            g = vend if name == 'Released Bitbrain scorer' else preds[name][s]
            mt = float(np.sum((g > 0) & (g < 5) & m) * .5); rt = float(np.sum((y > 0) & m) * .5)
            per.append({'subject': s, 'kappa': cohen_kappa_score(y[m], g[m]),
                        'macro_f1': f1_score(y[m], g[m], labels=list(range(5)), average='macro',
                                             zero_division=0),
                        'tst_error_minutes': abs(mt - rt),
                        'covered_fraction': float(m.sum() / (good & (y < 5)).sum())})
        rows.append({'model': name, 'people': len(per),
                     **{k: float(np.mean([p[k] for p in per]))
                        for k in ['kappa', 'macro_f1', 'tst_error_minutes', 'covered_fraction']}})
        cover += [{'model': name, **p} for p in per]
    vendor_table = pd.DataFrame(rows)
    vendor_table.to_csv(OUT / 'vendor_mask_comparison.csv', index=False)
    pd.DataFrame(cover).to_csv(OUT / 'vendor_mask_per_person.csv', index=False)

    # ---------------- 2. confusion matrices ----------------
    conf = {}
    for name, p in preds.items():
        Y, G = [], []
        for s in subs:
            y = np.array(ref[s]['reference']); good = np.array(ref[s]['good']).astype(bool)
            m = good & (y < 5)
            Y.append(y[m]); G.append(p[s][m])
        Y, G = np.concatenate(Y), np.concatenate(G)
        cm = confusion_matrix(Y, G, labels=list(range(5)))
        conf[name] = {'counts': cm.tolist(),
                      'row_normalised': (cm / cm.sum(axis=1, keepdims=True)).round(4).tolist(),
                      'stages': STAGES}
    (OUT / 'confusion_matrices.json').write_text(json.dumps(conf, indent=2))

    # ---------------- 3. failure gallery ----------------
    lead = 'Ensemble of seven models' if 'Ensemble of seven models' in preds else list(preds)[0]
    errs = []
    for s in subs:
        y = np.array(ref[s]['reference']); good = np.array(ref[s]['good']).astype(bool)
        m = good & (y < 5); g = preds[lead][s]
        mt = float(np.sum((g > 0) & (g < 5) & m) * .5); rt = float(np.sum((y > 0) & m) * .5)
        errs.append({'subject': s, 'signed_error_minutes': mt - rt, 'abs_error_minutes': abs(mt - rt),
                     'kappa': float(cohen_kappa_score(y[m], g[m])), 'epochs': int(m.sum()),
                     'reference_tst_minutes': rt, 'model_tst_minutes': mt})
    worst = sorted(errs, key=lambda r: -r['abs_error_minutes'])[:3]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=False)
    gallery = []
    for ax, w in zip(axes, worst):
        s = w['subject']
        y = np.array(ref[s]['reference']); good = np.array(ref[s]['good']).astype(bool)
        m = good & (y < 5); g = preds[lead][s]
        t = np.arange(len(y)) / 120.0
        yy = np.where(m, y, np.nan); gg = np.where(m, g, np.nan)
        ax.step(t, yy, where='post', lw=1.1, color='#14181f', label='human reference')
        ax.step(t, gg, where='post', lw=1.1, color='#9b2c2c', alpha=.8, label='model')
        ax.set_yticks(range(5)); ax.set_yticklabels(STAGES, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('hours from recording start', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{s}: sleep time off by {w['signed_error_minutes']:+.0f} min "
                     f"(reference {w['reference_tst_minutes']:.0f}, model {w['model_tst_minutes']:.0f}), "
                     f"kappa {w['kappa']:.2f}", fontsize=9, loc='left')
        ax.grid(alpha=.15)
        cm = confusion_matrix(y[m], g[m], labels=list(range(5)))
        with np.errstate(invalid='ignore', divide='ignore'):
            rown = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        worst_stage = int(np.argmin([rown[i, i] if cm[i].sum() else 9 for i in range(5)]))
        taken_by = int(np.argmax([rown[worst_stage, j] if j != worst_stage else -1 for j in range(5)]))
        gallery.append({**w, 'hardest_stage': STAGES[worst_stage],
                        'recall_of_hardest_stage': float(rown[worst_stage, worst_stage]),
                        'mostly_called': STAGES[taken_by],
                        'reading': f"On this night the model recovers {rown[worst_stage, worst_stage]:.0%} of "
                                   f"{STAGES[worst_stage]} epochs and calls most of the rest "
                                   f"{STAGES[taken_by]}. This is one night out of twenty; it illustrates the "
                                   f"error, it does not explain the aggregate."})
    axes[0].legend(fontsize=8, ncol=2, loc='upper right', framealpha=.9)
    fig.suptitle(f'Failure gallery: the three worst held-out nights for the {lead.lower()}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / 'failure_gallery.png', dpi=150)
    plt.close(fig)

    (OUT / 'failure_gallery.json').write_text(json.dumps(
        {'model': lead, 'cohort': 'historical_boas, 20 held-out people, confirmatory nights',
         'scope': 'Development cohort, already inspected at v1. Three nights chosen by absolute total-sleep '
                  'error, which is the summary the product reports; they are not the three worst by stage '
                  'agreement and are not representative of the aggregate.',
         'all_people': errs, 'worst_three': gallery}, indent=2))

    print(vendor_table.to_string(index=False))
    print()
    for g in gallery:
        print(f"{g['subject']}: {g['signed_error_minutes']:+.0f} min, kappa {g['kappa']:.2f}, "
              f"hardest {g['hardest_stage']} ({g['recall_of_hardest_stage']:.0%} recall), "
              f"mostly called {g['mostly_called']}")
    print(json.dumps({'written': str(OUT)}))


if __name__ == '__main__':
    main()

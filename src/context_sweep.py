"""Backlog items 6 and 7: how wide should the context window be, and how big the temporal head?

The context head uses a 15-epoch window and two 64-channel convolutions. The window was chosen from
a literature prior (published work plateaus around seven epochs of context) and the head size was
chosen by nothing at all. Both are cheap to sweep because the trunk stays frozen.

Selection rule, unchanged: each configuration is scored on the six selection people and the best is
chosen by that alone. Every configuration's test numbers are printed too, so the reader can see
whether the selection people pointed at the right one — but they are reported, never used to choose.

The shipped context model is backed up before the sweep and restored afterwards, so this changes no
published result.

Usage: python src/context_sweep.py <boas_cache> [trunk]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

import cnn_context as cx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared core modules sit at the top level
from pipeline import ROOT

OUT = ROOT / 'results/context_sweep'
WINDOWS = [3, 5, 7, 11]
HEADS = {'64x2 (shipped)': [(128, 64, 5), (64, 64, 5)],
         '128x3 (wider)': [(128, 128, 5), (128, 128, 5), (128, 64, 5)]}


def run(cache, trunk, context, head_name, head, tag):
    cx.CONTEXT, cx.HEAD = context, head
    produced = ROOT / ('results/cnn_context' if trunk == 'results/cnn'
                       else f"results/cnn_context_{Path(trunk).name.replace('cnn_', '')}")
    cx.main(cache, trunk, 20)
    m = json.loads((produced / 'context_metrics.json').read_text())
    dst = OUT / tag
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(produced, dst)
    sel = max(m['frozen']['selection_candidates'], key=lambda r: r['mean_person_kappa'])
    r = m['context_results']
    return {'tag': tag, 'context_epochs_each_side': context, 'window_epochs': 2 * context + 1,
            'head': head_name, 'selected_pass': m['frozen']['selected_pass'],
            'sequence_weight': m['frozen']['selected_sequence_weight'],
            'selection_people_kappa': sel['mean_person_kappa'],
            'test_kappa': r['kappa'], 'test_macro_f1': r['macro_f1'],
            'test_accuracy': r['accuracy'], 'test_tst_error_minutes': r['tst_error_minutes']}


def main(cache, trunk='results/cnn'):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    produced = ROOT / 'results/cnn_context'
    backup = ROOT / 'results/_cnn_context_shipped'
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(produced, backup)
    base_c, base_h = cx.CONTEXT, cx.HEAD

    rows = []
    try:
        for w in WINDOWS:
            rows.append(run(cache, trunk, w, '64x2 (shipped)', HEADS['64x2 (shipped)'], f'ctx{w}_head64x2'))
            print(json.dumps(rows[-1]), flush=True)
        best_w = max(rows, key=lambda r: r['selection_people_kappa'])['context_epochs_each_side']
        rows.append(run(cache, trunk, best_w, '128x3 (wider)', HEADS['128x3 (wider)'],
                        f'ctx{best_w}_head128x3'))
        print(json.dumps(rows[-1]), flush=True)
    finally:
        cx.CONTEXT, cx.HEAD = base_c, base_h
        shutil.rmtree(produced)
        shutil.copytree(backup, produced)
        shutil.rmtree(backup)

    chosen = max(rows, key=lambda r: r['selection_people_kappa'])
    best_test = max(rows, key=lambda r: r['test_macro_f1'])
    shipped = json.loads((produced / 'context_metrics.json').read_text())['context_results']
    sv = ROOT / 'results/seed_variance/seed_variance.json'
    spread = json.loads(sv.read_text())['statistics']['macro_f1']['spread'] if sv.exists() else None
    gain = chosen['test_macro_f1'] - shipped['macro_f1']
    report = {
        'question': 'how wide a context window, and how large a temporal head, with the trunk frozen',
        'selection_rule': 'mean person kappa on the six selection people; test numbers reported only',
        'shipped_configuration': {'context_epochs_each_side': base_c, 'head': '64x2',
                                  'test_macro_f1': shipped['macro_f1'], 'test_kappa': shipped['kappa']},
        'runs': rows, 'chosen_by_selection_people': chosen,
        'best_on_test_for_reference_only': best_test,
        'selection_picked_the_test_winner': chosen['tag'] == best_test['tag'],
        'gain_over_shipped_macro_f1': gain,
        'head_seed_spread_macro_f1': spread,
        'verdict': ('the gain is smaller than the head-seed spread, so it is not a finding'
                    if spread is not None and abs(gain) < spread else
                    'the gain exceeds the head-seed spread'),
        'note': 'The shipped context model was restored after the sweep; no published result changed.',
        'created_unix': time.time(), 'seconds': round(time.time() - t0, 1)}
    (OUT / 'context_sweep.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(OUT / 'context_sweep.csv', index=False)
    print()
    print(pd.DataFrame(rows)[['tag', 'window_epochs', 'head', 'selection_people_kappa', 'test_kappa',
                              'test_macro_f1', 'test_tst_error_minutes']].to_string(index=False))
    print(json.dumps({k: report[k] for k in ['chosen_by_selection_people', 'gain_over_shipped_macro_f1',
                                             'head_seed_spread_macro_f1', 'verdict',
                                             'selection_picked_the_test_winner']}, indent=2, default=str))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'results/cnn')

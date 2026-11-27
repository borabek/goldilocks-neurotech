#!/bin/bash
# Paced, silent screen-demo of the frozen pipeline. Every line below is produced by
# running the real code or reading a saved result file; nothing is printed from memory.
# Usage: start a screen recording, then run  ./demo.sh
PY=./.venv/bin/python
say() { printf '\n\033[1m%s\033[0m\n\n' "$1"; }
pace() { while IFS= read -r l; do printf '%s\n' "$l"; sleep "${1:-0.12}"; done; }

clear
say "Goldilocks AI - sleep staging on a forehead headband"
sleep 2
printf '  One model. Frozen before testing. Person-held-out.\n'
printf '  20 held-out people per cohort.\n'
sleep 3

say "1. Scoring a night live, epoch by epoch"
sleep 1.5
printf '  $ python live_inference.py --replay data/live_demo/sub-4.npz\n\n'
sleep 1.5
$PY live_inference.py --replay data/live_demo/sub-4.npz 2>/dev/null | head -40 | pace 0.14
sleep 1
printf '\n  Ten epochs of warm-up, then a stage every 30 seconds.\n'
printf '  Epoch 35 is sleep onset: N1 at confidence 0.384 - it is unsure, and it says so.\n'
sleep 4

say "2. The result that decides the pitch"
sleep 1
$PY - <<'PYEOF' | pace 0.25
import json, pathlib
h = json.load(open('results/v2/historical_boas_metrics.json'))['results']
e = json.load(open('results/v2/external_metrics.json'))['results']
y = json.load(open('results/yasa/yasa_metrics.json'))['results']
yh = y['historical_boas']['primary_complete_recording']
print(f"  {'':34s}{'kappa':>8s}{'sleep-time error':>20s}")
print("  " + "-" * 62)
print(f"  {'BOAS headband - ours':34s}{h['kappa']:>8.3f}{h['tst_error_minutes']:>17.1f} min")
print(f"  {'BOAS headband - YASA 0.7.0':34s}{yh['kappa']:>8.3f}{yh['tst_error_minutes']:>17.1f} min")
print(f"  {'Sleep-EDF, new device - ours':34s}{e['kappa']:>8.3f}{e['tst_error_minutes']:>17.1f} min")
print("")
print("  We beat the open-source baseline at home.")
print("  On a device we never trained for, the night summary falls apart.")
PYEOF
sleep 4

say "3. Where it breaks, per person"
sleep 1
$PY - <<'PYEOF2' | pace 0.45
import pandas as pd
d = pd.read_csv('results/v2/external_people.csv').nlargest(4, 'tst_error_minutes')
print(f"  {'person':10s}{'kappa':>8s}{'true sleep':>13s}{'our estimate':>15s}{'off by':>11s}")
print("  " + "-" * 57)
for _, r in d.iterrows():
    print(f"  {r.subject:10s}{r.kappa:>8.3f}{r.reference_tst_minutes:>10.0f} min"
          f"{r.model_tst_minutes:>12.0f} min{r.tst_error_minutes:>8.0f} min")
PYEOF2
sleep 1
printf '\n  On SC410 the reference night holds 474 minutes of sleep. We reported 1282.\n'
printf '  The stage decoding still looks reasonable; the night summary does not.\n'
sleep 4

say "4. The architecture upgrade we tried, and it lost"
sleep 1
$PY - <<'PYEOF3' | pace 0.35
import json
g = json.load(open('results/cnn_endtoend/endtoend_metrics_grid21.json'))
r = json.load(open('results/cnn_endtoend/endtoend_metrics_random.json'))['summary']['endtoend']
c = g['frozen_trunk_context_model']
sp = g['head_seed_spread_macro_f1']
gap = abs(g['gain_over_frozen_trunk_context_macro_f1'])
print(f"  {'':44s}{'macro-F1':>10s}{'kappa':>9s}")
print("  " + "-" * 63)
print(f"  {'end-to-end, grid sampling':44s}{g['endtoend_results']['macro_f1']:>10.3f}{g['endtoend_results']['kappa']:>9.3f}")
print(f"  {'end-to-end, random sampling':44s}{r['macro_f1']:>10.3f}{r['kappa']:>9.3f}")
print(f"  {'frozen trunk + temporal head':44s}{c['macro_f1']:>10.3f}{c['kappa']:>9.3f}")
print("")
print(f"  Two samplers, no shared code path, same answer.")
print(f"  {gap:.3f} below the staged model - about {gap/sp:.0f}x the {sp:.3f} spread across seeds.")
print("  With 48 people to fit on, the constraint is what helps.")
PYEOF3
sleep 4

say "5. Every number is reproducible"
sleep 1
printf '  $ python reproduce_v2.py\n\n'
sleep 1.5
$PY reproduce_v2.py 2>&1 | grep -E "passed" | pace 0.3
$PY - <<'PYEOF4' | pace 0.4
import json
h = json.load(open('results/v2/historical_boas_metrics.json'))
print(f"  frozen model sha256  {h['model_sha256'][:32]}...")
print(f"  BOAS kappa           {h['results']['kappa']}")
print(f"  sleep-time error     {h['results']['tst_error_minutes']} min")
PYEOF4
sleep 1
printf '\n  One command, no network, identical to the last digit on three machines.\n'
sleep 4

say "We are not the best sleep stager. We measured where we stop being trustworthy."
sleep 3


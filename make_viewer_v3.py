"""Add the models trained on 7–8 November to the validation explorer, as a new file.

The demo a judge clicks still showed the frozen v2 model and YASA only, so the strongest evidence of
the night — that the CNN family wins at home and collapses away — was not in the thing being
demonstrated. This builds a v3 copy of the v2.1 explorer with three more lanes (the CNN challenger,
the context model and the seven-model ensemble) and their rows in the results table.

It edits nothing in place: `results/yasa/Goldilocks_Validation_Explorer_v2_1_yasa.html` and the v2
explorer are left exactly as they are, and the new file is written beside them.

Usage: python make_viewer_v3.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'results/yasa/Goldilocks_Validation_Explorer_v2_1_yasa.html'
DST = ROOT / 'results/cnn/Goldilocks_Validation_Explorer_v3.html'

LANES = [
    ('cnn', 'CNN challenger (trained 7 Nov, same frozen split)', '#9b2c2c',
     'results/cnn/cnn_predictions.json', 'results/external_all/cnn_external_predictions.json'),
    ('context', 'Context model (CNN trunk plus a 15-epoch temporal head)', '#1f6f4a',
     'results/cnn_context/context_predictions.json',
     'results/external_all/cnn_context_external_predictions.json'),
    ('ensemble', 'Ensemble of seven models', '#5b3fa8',
     'results/cnn_posthoc/ensemble_predictions.json',
     'results/external_all/ensemble_of_7_external_predictions.json'),
]


def read_json(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def split(html):
    i = html.index('<script>const DATA=')
    start = i + len('<script>const DATA=')
    depth, j = 0, start
    while True:
        if html[j] == '{':
            depth += 1
        elif html[j] == '}':
            depth -= 1
            if depth == 0:
                break
        j += 1
    return html[:start], html[start:j + 1], html[j + 1:]


def main():
    head, data_text, tail = split(SRC.read_text())
    data = json.loads(data_text)
    cohort_key = {}
    for name, c in data['cohorts'].items():
        cohort_key[name] = 'historical' if str(c.get('cohort', name)).startswith('historical') \
            or 'BOAS' in name else 'external'

    added = {}
    for key, label, colour, boas_rel, ext_rel in LANES:
        b, e = read_json(boas_rel), read_json(ext_rel)
        if b is None:
            print(json.dumps({'skipped': key, 'why': f'{boas_rel} missing'}))
            continue
        n = 0
        for name, c in data['cohorts'].items():
            src = b if cohort_key[name] == 'historical' else e
            if src is None:
                continue
            for sub, z in c['predictions'].items():
                if sub in src and len(src[sub]) == len(z['reference']):
                    z[key] = src[sub]
                    n += 1
        if n:
            added[key] = {'label': label, 'colour': colour, 'records': n}
            print(json.dumps({'lane': key, 'records': n}))

    keys = list(added)
    if not keys:
        raise SystemExit('no lane could be added')

    # 1. the lane list, both branches of the reveal test
    old = "z.vendor?['reference','vendor','v1','yasa','model']:['reference','yasa','model']"
    new = ("z.vendor?['reference','vendor','v1','yasa','model'," + ','.join(f"'{k}'" for k in keys) +
           "]:['reference','yasa','model'," + ','.join(f"'{k}'" for k in keys) + "]")
    assert old in tail, 'lane list not found'
    tail = tail.replace(old, new, 1)

    # 2. the lane labels
    old = "model:'Frozen v2 model'}"
    new = "model:'Frozen v2 model'," + ','.join(f"{k}:'{added[k]['label']}'" for k in keys) + "}"
    assert old in tail, 'label map not found'
    tail = tail.replace(old, new, 1)

    # 3. a colour per lane
    old = "stroke=\"'+(key==='model'?'#af6c28':'#18364a')+'\""
    colour_js = "'+({" + ','.join(f"{k}:'{added[k]['colour']}'" for k in keys) + \
                ",model:'#af6c28'}[key]||'#18364a')+'"
    assert old in tail, 'stroke colour not found'
    tail = tail.replace(old, "stroke=\"" + colour_js + "\"", 1)

    # 4. only hide epochs on the frozen-model lanes that carry a quality mask
    old = "((key==='model'||key==='v1')&&!z.good[i])"
    new = "((key==='model'||key==='v1'||" + '||'.join(f"key==='{k}'" for k in keys) + ")&&!z.good[i])"
    assert old in tail, 'quality mask test not found'
    tail = tail.replace(old, new, 1)

    # 5. extra rows in the results table
    rows = []
    summ = read_json('results/summary/metrics_summary.json')
    ext_all = read_json('results/external_all/external_all_metrics.json') or {}
    name_map = {'cnn': 'CNN challenger', 'context': 'Context model', 'ensemble': 'Ensemble of seven models'}
    ext_map = {'cnn': 'cnn', 'context': 'cnn_context', 'ensemble': 'ensemble_of_7'}
    if summ:
        by = {(r['model'], r['cohort']): r for r in summ['rows']}
        for k in keys:
            r = by.get((name_map[k], 'historical_boas'))
            if r:
                rows.append([f"BOAS headband, {name_map[k]} (development)", 20,
                             r['kappa_pooled'], None, None, r['macro_f1_person_averaged']])
    for k in keys:
        e = (ext_all.get('results', {}) or {}).get(ext_map[k], {}).get('primary_complete_recording')
        if e:
            rows.append([f"Sleep-EDF complete recording, {name_map[k]} (exploratory)", 20,
                         e['kappa'], e['tst_error_minutes'], None, e['macro_f1']])
    # fill the sleep-time column for the BOAS rows from the per-person files
    for row in rows:
        if row[3] is None:
            k = next(kk for kk in keys if name_map[kk] in row[0])
            src = {'cnn': 'results/cnn/cnn_people.csv', 'context': 'results/cnn_context/context_people.csv',
                   'ensemble': None}[k]
            if src and (ROOT / src).exists():
                import csv
                vals = [float(d['tst_error_minutes']) for d in
                        csv.DictReader((ROOT / src).read_text().splitlines())]
                row[3] = sum(vals) / len(vals)
            else:
                ph = read_json('results/cnn_posthoc/posthoc_metrics.json') or {}
                for lab, res in ph.get('results', {}).items():
                    if lab.startswith('ensemble_of_'):
                        row[3] = res['test_alpha_0_no_adjustment']['tst_error_minutes']
    # the row shape is [label, people, kappa, sleep-time MAE, global coverage, scaled coverage].
    # These models have no interval coverage, so both coverage slots are null and the label carries
    # the macro-F1 instead of inventing a column.
    js_rows = ''.join(
        f",['{r[0]} — macro-F1 {r[5]:.3f}',{r[1]},{r[2]:.6f},{(r[3] or 0):.4f},null,null]" for r in rows)
    anchor = "]];$('metrics').innerHTML="
    assert anchor in tail, 'metrics rows not found'
    tail = tail.replace(anchor, "]" + js_rows + "];$('metrics').innerHTML=", 1)

    # the global-coverage column is undefined for these rows: print a dash rather than 0%
    old = "+(100*r[4]).toFixed(0)+'% / '"
    new = "+(r[4]===null?'—':(100*r[4]).toFixed(0)+'%')+' / '"
    assert old in tail, 'coverage cell not found'
    tail = tail.replace(old, new, 1)

    title_old = 'Goldilocks v2.1 | Validation explorer with YASA'
    html = (head + json.dumps(data) + tail).replace(
        title_old, 'Goldilocks v3 | Validation explorer with the 7 November models')
    note = ('<small>The CNN challenger, the context model and the ensemble were trained on 7 November on '
            'the same frozen split and are scored with the same masks and metrics. They are development '
            'results on a cohort already inspected at v1, not fresh confirmation. On Sleep-EDF every one of '
            'them collapses; that is shown here rather than omitted.</small>')
    html = html.replace('</section> <section><h2>Inspect frozen predictions</h2>',
                        note + '</section> <section><h2>Inspect frozen predictions</h2>', 1)
    DST.write_text(html)
    print(json.dumps({'written': str(DST.relative_to(ROOT)), 'megabytes': round(len(html) / 1e6, 2),
                      'lanes_added': list(added), 'table_rows_added': len(rows)}, indent=2))


if __name__ == '__main__':
    main()


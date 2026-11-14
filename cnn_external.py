"""Run the frozen CNN challenger, unchanged, on the 20 external Sleep-EDF records.

The question this answers: the headline finding is that stage accuracy survives a move to a new
device while the night summary does not. Is that a property of the system, or just of the one
feature model? Here the CNN that won on the headband meets the same domain shift.

Status: EXPLORATORY DEVELOPMENT. These 20 external people were already inspected when v2 was
evaluated, so this is not a fresh independent test for any model, including this one. The CNN
weights, sequence weight and temperature are the ones frozen by `cnn_challenger.py`; nothing is
refitted, reselected or tuned here. Windows, masks, metrics and the sleep-time definition are
`yasa_evaluate.py`'s, which copied `v2_evaluate.assess`, so all three systems are scored
identically.

Usage: python cnn_external.py <boas_cache_dir> <external_cache_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

import cnn_challenger as cc
from cnn_challenger import (N_CLASS, load_night, night_probabilities, robust_scale, CLIP)
from yasa_evaluate import score, summarize, windows
from pipeline import ROOT


def rebuild_params(path):
    flat = json.loads(Path(path).read_text())
    n = max(int(k.split('/')[0]) for k in flat) + 1
    return [{k.split('/')[1]: jnp.asarray(np.array(v, dtype='float32'))
             for k, v in flat.items() if int(k.split('/')[0]) == i} for i in range(n)]


def load_external(cache, sub):
    z = np.load(Path(cache) / f'{sub}.npz')
    x = z['X'].astype('float32')                                  # (n, 1, 1920)
    good, y, onset = z['good'].astype(bool), z['y'].astype('int8'), z['onset'].astype(float)
    med, mad = robust_scale(x, good)
    return np.clip((x - med) / mad, -CLIP, CLIP), y, good, onset


def main(boas_cache, ext_cache, out=None):
    t0 = time.time()
    OUT = out or cc.OUT
    freeze = json.loads((OUT / 'cnn_freeze_record.json').read_text())
    params = rebuild_params(OUT / 'cnn_weights.json')
    weight, temperature = freeze['selected_sequence_weight'], freeze['temperature']

    # The smoother's transition matrix and prior are a deterministic function of the fitting
    # nights' labels; recompute them exactly as cnn_challenger.py did. No external label is used.
    rows = json.loads((ROOT / 'protocol.json').read_text())['records']
    trans, prior = np.ones((N_CLASS, N_CLASS)), np.ones(N_CLASS)
    for r in rows:
        if r['split'] != 'fit':
            continue
        z = np.load(Path(boas_cache) / f"{r['subject']}.npz")
        y, good = z['y'].astype('int8'), z['good'].astype(bool)
        m = good & (y < 5)
        prior += np.bincount(y[m], minlength=N_CLASS)
        adj = m[1:] & m[:-1]
        np.add.at(trans, (y[:-1][adj], y[1:][adj]), 1)
    prior /= prior.sum(); trans /= trans.sum(axis=1, keepdims=True)

    frozen_ext = json.loads((ROOT / 'results/v2/external_predictions.json').read_text())
    per_window, preds = {}, {}
    for sub in sorted(frozen_ext):
        x, y, good, onset = load_external(ext_cache, sub)
        fz = frozen_ext[sub]
        assert np.array_equal(np.array(fz['reference']), y), f'{sub}: reference differs from the frozen file'
        assert np.array_equal(np.array(fz['good']).astype(bool), good), f'{sub}: quality mask differs'
        assert np.allclose(np.array(fz['onset'], dtype=float), onset), f'{sub}: onsets differ'
        p = night_probabilities(params, x, good, onset, trans, prior, weight, temperature)
        guess = p.argmax(axis=1)
        preds[sub] = guess.tolist()
        for name, w in windows(y, onset).items():
            per_window.setdefault(name, []).append({'subject': sub, **score(y, good, onset, guess, w)})
        print(json.dumps({'evaluated': sub,
                          'cnn_kappa': round(per_window['primary_complete_recording'][-1]['kappa'], 3),
                          'cnn_tst_error': per_window['primary_complete_recording'][-1]['tst_error_minutes']}),
              flush=True)

    cnn = {w: summarize(r) for w, r in per_window.items()}
    v2 = json.loads((ROOT / 'results/v2/external_metrics.json').read_text())
    yasa = json.loads((ROOT / 'results/yasa/yasa_metrics.json').read_text())['results']['external']
    v2_primary, v2_secondary = v2['results'], v2['secondary_reference_defined_sleep_window']
    boas = json.loads((OUT / 'cnn_metrics.json').read_text())

    verdict = {
        'cnn_kappa_primary': cnn['primary_complete_recording']['kappa'],
        'v2_kappa_primary': v2_primary['kappa'],
        'yasa_kappa_primary': yasa['primary_complete_recording']['kappa'],
        'cnn_tst_error_primary': cnn['primary_complete_recording']['tst_error_minutes'],
        'v2_tst_error_primary': v2_primary['tst_error_minutes'],
        'yasa_tst_error_primary': yasa['primary_complete_recording']['tst_error_minutes'],
        'cnn_beats_v2_on_kappa': bool(cnn['primary_complete_recording']['kappa'] > v2_primary['kappa']),
        'cnn_beats_yasa_on_kappa': bool(cnn['primary_complete_recording']['kappa']
                                        > yasa['primary_complete_recording']['kappa']),
        'cnn_sleep_time_error_still_large': bool(cnn['primary_complete_recording']['tst_error_minutes'] > 60),
        'cnn_headband_to_external_kappa_drop': (boas['cnn_results']['kappa']
                                                - cnn['primary_complete_recording']['kappa']),
        'cnn_headband_to_external_sleep_time_change': (cnn['primary_complete_recording']['tst_error_minutes']
                                                       - boas['cnn_results']['tst_error_minutes']),
    }
    report = {
        'method': 'frozen CNN challenger from cnn_challenger.py, applied unchanged to Sleep-EDF Fpz-Cz',
        'status': 'EXPLORATORY DEVELOPMENT: these external people were already inspected at the v2 '
                  'evaluation, so this is not a fresh independent test for any model.',
        'input': '100 Hz Fpz-Cz resampled to 64 Hz, frozen epoch grid, reference and quality mask from '
                 'results/v2/external_predictions.json',
        'channel_note': 'The CNN was fitted on two forehead headband channels at 64 Hz and is applied here '
                        'to a single frontal PSG channel: a device, montage and scorer shift.',
        'frozen_weights_sha256': freeze['weights_sha256'], 'sequence_weight': weight,
        'temperature': temperature, 'refitted_anything': False,
        'cnn_results': cnn,
        'v2_results': {'primary_complete_recording': v2_primary,
                       'secondary_reference_window': v2_secondary},
        'yasa_results': yasa,
        'cnn_results_on_headband_for_reference': boas['cnn_results'],
        'verdict': verdict, 'created_unix': time.time(), 'seconds': round(time.time() - t0, 1)}
    (OUT / 'cnn_external_metrics.json').write_text(json.dumps(report, indent=2))
    (OUT / 'cnn_external_predictions.json').write_text(json.dumps(preds))
    pd.DataFrame([{'window': w, **r} for w, rs in per_window.items() for r in rs]).drop(
        columns='per_stage_f1').to_csv(OUT / 'cnn_external_people.csv', index=False)
    print(json.dumps({'cnn': cnn, 'verdict': verdict}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], out)

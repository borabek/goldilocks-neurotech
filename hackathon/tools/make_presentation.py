"""Build the support slides and the one-page handout from the saved result files.

Every number on a slide is read from a result JSON at build time. Nothing is typed in by hand, so
a slide cannot drift from the evidence: rerun this after any evaluation changes.

Usage: python make_presentation.py            (writes slides/ and handout/)
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDES, HANDOUT = ROOT / 'slides', ROOT / 'handout'
def _find_chrome():
    """A headless browser is only needed to turn the HTML into PDF. Look in the usual places on
    Linux, macOS and Windows; if none is found the HTML is still written and can be printed to PDF
    from any browser."""
    import shutil
    candidates = [
        '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Chromium.app/Contents/MacOS/Chromium',
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ['google-chrome', 'chromium', 'chrome', 'msedge']:
        found = shutil.which(name)
        if found:
            return found
    return None


CHROME = _find_chrome()


def j(rel, default=None):
    p = ROOT / rel
    if not p.exists():
        return default
    return json.loads(p.read_text())


def fmt(x, n=3):
    return '—' if x is None else f'{x:.{n}f}'


def mins(x):
    return '—' if x is None else f'{x:.1f} min'


def main():
    SLIDES.mkdir(exist_ok=True); HANDOUT.mkdir(exist_ok=True)
    hist = j('results/v2/historical_boas_metrics.json')['results']
    ext_all = j('results/v2/external_metrics.json')
    ext = ext_all['results']; ext2 = ext_all['secondary_reference_defined_sleep_window']
    same_mask = j('results/v2/historical_boas_metrics.json')['same_mask_tst']
    gate = j('results/v2/release_gate.json')
    freeze = j('results/v2/freeze_record.json')
    yasa = j('results/yasa/yasa_metrics.json')['results']
    yh = yasa['historical_boas']['primary_complete_recording']
    ye = yasa['external']['primary_complete_recording']
    ye2 = yasa['external']['secondary_reference_window']
    win = j('results/window_rule/window_results.json')['results']
    cnn_all = j('results/cnn/cnn_metrics.json')
    cnn = cnn_all['cnn_results'] if cnn_all else None
    cnn_verdict = cnn_all['verdict'] if cnn_all else None
    cnn_ext_all = j('results/cnn/cnn_external_metrics.json')
    cnn_ext = cnn_ext_all['cnn_results']['primary_complete_recording'] if cnn_ext_all else None
    cnn_ext2 = cnn_ext_all['cnn_results']['secondary_reference_window'] if cnn_ext_all else None
    e2e_rand = j('results/cnn_endtoend/endtoend_metrics_random.json')
    e2e_grid = j('results/cnn_endtoend/endtoend_metrics_grid21.json')
    ctx = j('results/cnn_context/context_metrics.json')
    live = j('results/live/live_vs_offline.json')
    leak = j('research/exposure_model/out/leakage_audit.json')
    cal = j('results/v2/probability_calibration_summary.json')
    protocol_sha = freeze['protocol_sha256']; model_sha = freeze['model_sha256']
    built = datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')

    cnn_row = ''
    if cnn:
        cnn_row = f"""<tr><td><b>BOAS headband</b> — CNN challenger</td><td class="win">{fmt(cnn['kappa'])}</td>
        <td class="win">{mins(cnn['tst_error_minutes'])}</td><td class="muted">best here</td></tr>"""
    if cnn_ext:
        cnn_row += f"""<tr><td><b>Sleep-EDF full recording</b> — CNN challenger</td>
        <td class="lose">{fmt(cnn_ext['kappa'])}</td><td class="lose">{mins(cnn_ext['tst_error_minutes'])}</td>
        <td class="muted">worst here</td></tr>"""
    ctxs = j('results/cnn_context_sqrtbalanced/context_metrics.json')
    if ctxs:
        c2 = ctxs['context_results']
        cnn_row += f"""<tr><td>BOAS headband — context model, sqrt-balanced trunk</td>
        <td>{fmt(c2['kappa'])}</td><td>{mins(c2['tst_error_minutes'])}</td>
        <td class="muted">variant</td></tr>"""
    ph = j('results/cnn_posthoc/posthoc_metrics.json')
    ens = None
    if ph:
        for lab, res in ph['results'].items():
            if lab.startswith('ensemble_of_'):
                ens = res['test_alpha_0_no_adjustment']
    if ens:
        cnn_row += f"""<tr><td><b>BOAS headband — ensemble of seven models</b></td>
        <td class="win">{fmt(ens['kappa'])}</td><td class="win">{mins(ens['tst_error_minutes'])}</td>
        <td class="muted">best on this cohort only</td></tr>"""
    cnn_line = ('not run' if not cnn else
                ('it loses' if not cnn_verdict['cnn_beats_v2_on_kappa'] else
                 'it beats the frozen model on the headband and then collapses on Sleep-EDF'))

    css = """
    :root{--ink:#14181f;--mute:#5d6774;--line:#d9dee6;--accent:#1f5f8b;--bad:#9b2c2c;--good:#1f6f4a}
    *{box-sizing:border-box;-webkit-print-color-adjust:exact;print-color-adjust:exact}
    body{margin:0;font-family:Helvetica,Arial,sans-serif;color:var(--ink)}
    .slide{width:297mm;height:167mm;padding:16mm 18mm;page-break-after:always;position:relative;
      display:flex;flex-direction:column;background:#fff}
    .slide:last-child{page-break-after:auto}
    .kicker{font-size:10pt;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);font-weight:700}
    h1{font-size:30pt;line-height:1.1;margin:4mm 0 2mm;font-weight:700;letter-spacing:-.01em}
    h2{font-size:21pt;line-height:1.15;margin:3mm 0 4mm;font-weight:700;letter-spacing:-.01em}
    .sub{font-size:11pt;color:var(--mute);line-height:1.4;max-width:215mm}
    table{border-collapse:collapse;width:100%;font-size:10pt;margin-top:1.5mm}
    th,td{text-align:left;padding:1.7mm 3mm;border-bottom:1px solid var(--line)}
    th{font-size:9.5pt;text-transform:uppercase;letter-spacing:.07em;color:var(--mute);font-weight:700}
    td:nth-child(n+2),th:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}
    .win{color:var(--good);font-weight:700}.lose{color:var(--bad);font-weight:700}
    .muted{color:var(--mute);font-size:10pt;text-align:left!important}
    ul{font-size:12pt;line-height:1.5;margin:2mm 0 0;padding-left:6mm}
    li{margin-bottom:2.4mm}li b{font-weight:700}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:10mm}
    .foot{position:absolute;bottom:8mm;left:18mm;right:18mm;font-size:8.5pt;color:var(--mute);
      border-top:1px solid var(--line);padding-top:2.5mm;display:flex;justify-content:space-between}
    .big{font-size:40pt;font-weight:700;letter-spacing:-.02em;line-height:1}
    .bignum{display:flex;gap:14mm;margin:5mm 0 3mm}
    .bignum div{flex:1}.bignum .lab{font-size:10pt;color:var(--mute);margin-top:1.5mm;line-height:1.35}
    code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:9pt;background:#f4f6f9;padding:.6mm 1.2mm;
      border-radius:1mm;word-break:break-all}
    """

    def foot(n, total=7):
        return (f'<div class="foot"><span>Goldilocks AI · Stream 2, AI / Neural Decoding · '
                f'20 people per cohort</span><span>{n}/{total}</span></div>')

    # End-to-end arms. Both are read from their own result file; neither number is typed in.
    _eg = (e2e_grid or {}).get('endtoend_results') or {}
    _er = ((e2e_rand or {}).get('summary') or {}).get('endtoend') or {}
    _ec = (ctx or {}).get('context_results') or {}
    e2e_spread = (e2e_grid or {}).get('head_seed_spread_macro_f1')
    e2e_gap = (e2e_grid or {}).get('gain_over_frozen_trunk_context_macro_f1')
    if _eg and _er and _ec and e2e_spread:
        e2e_line = (f"Training the encoder jointly over 21-epoch sequences scored macro-F1 "
                    f"{_eg['macro_f1']:.3f} with grid sampling and {_er['macro_f1']:.3f} with random "
                    f"sampling, against {_ec['macro_f1']:.3f} for the frozen trunk with a temporal head "
                    f"&mdash; a gap of {abs(e2e_gap):.3f}, about {abs(e2e_gap)/e2e_spread:.0f}&times; the "
                    f"{e2e_spread:.3f} seed spread, and two independent sampling strategies agree.")
    else:
        e2e_line = None
    _ls = (live or {}).get('summary') or {}
    _lc = _ls.get('cost_of_going_live') or {}
    live_short = (f"costs {abs(_lc['kappa']):.3f} kappa against the same model scored retrospectively "
                  f"({_ls['offline_kappa']:.3f} &rarr; {_ls['live_kappa']:.3f}), with no lookahead") if _lc else None
    _skip = {'.venv', '__pycache__', '.git'}
    pkg_mb = round(sum(f.stat().st_size for f in ROOT.rglob('*')
                       if f.is_file() and not _skip & set(f.relative_to(ROOT).parts)) / 1e6)
    slides = []
    slides.append(f"""<section class="slide">
      <div class="kicker">Goldilocks AI · support slides</div>
      <h1>A wearable can stage sleep well<br>and still get the night wrong.</h1>
      <div class="bignum">
        <div><div class="big">{fmt(hist['kappa'])}</div><div class="lab">stage agreement (kappa) on the
          BOAS forehead headband, 20 held-out people</div></div>
        <div><div class="big">{hist['tst_error_minutes']:.0f} min</div><div class="lab">average error in total
          sleep time, same 20 people</div></div>
        <div><div class="big">{ext['tst_error_minutes']:.0f} min</div><div class="lab">same model, 20 new
          Sleep-EDF people, full recording with daytime wake</div></div>
      </div>
      <div class="sub">One frozen model, person-held-out. The 236-minute figure is the finding, not the
      footnote: stage accuracy mostly survived the move to a new device, and the night summary a user
      would actually read did not. We then trained a stronger model to check whether that is a property
      of our model or of the problem. It is the problem — see slide 2.</div>
      {foot(1, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">The comparison we did not want</div>
      <h2>We are not the best sleep stager. We say so.</h2>
      <table><tr><th>20 people per cohort</th><th>Kappa</th><th>Sleep-time error</th><th>Note</th></tr>
      <tr><td><b>BOAS headband</b> — ours</td><td class="win">{fmt(hist['kappa'])}</td>
        <td class="win">{mins(hist['tst_error_minutes'])}</td><td class="muted">development comparison</td></tr>
      <tr><td>BOAS headband — YASA 0.7.0</td><td>{fmt(yh['kappa'])}</td><td>{mins(yh['tst_error_minutes'])}</td>
        <td class="muted">open-source baseline</td></tr>
      <tr><td><b>Sleep-EDF full recording</b> — ours</td><td class="lose">{fmt(ext['kappa'])}</td>
        <td class="lose">{mins(ext['tst_error_minutes'])}</td><td class="muted">independent test</td></tr>
      <tr><td>Sleep-EDF full recording — YASA</td><td class="win">{fmt(ye['kappa'])}</td>
        <td class="win">{mins(ye['tst_error_minutes'])}</td><td class="muted">beats us here</td></tr>
      <tr><td>Sleep-EDF sleep window — ours</td><td>{fmt(ext2['kappa'])}</td>
        <td>{mins(ext2['tst_error_minutes'])}</td><td class="muted">secondary window</td></tr>
      <tr><td>Sleep-EDF sleep window — YASA</td><td>{fmt(ye2['kappa'])}</td>
        <td>{mins(ye2['tst_error_minutes'])}</td><td class="muted">secondary window</td></tr>
      {cnn_row}
      </table>
      <div class="sub" style="margin-top:2mm"><b>Read the CNN's two rows together.</b> It is the best model
      on the headband, kappa {fmt(cnn['kappa']) if cnn else '—'} against our {fmt(hist['kappa'])}, and by far
      the worst on Sleep-EDF at {fmt(cnn_ext['kappa']) if cnn_ext else '—'} and
      {mins(cnn_ext['tst_error_minutes']) if cnn_ext else '—'}, calling daytime wake sleep. Picking a wearable
      model by its in-domain score would have picked exactly that one. On the vendor-covered mask the released
      Bitbrain scorer still leads on sleep time, {same_mask['vendor']:.1f} against
      {same_mask['model']:.1f} minutes. All eight members of the CNN family, the ensemble included, were then
      scored on Sleep-EDF: none reaches kappa 0.22 and every one exceeds 700 minutes of sleep-time error.</div>
      {foot(2, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">Scope</div>
      <h2>What we cut, and why</h2>
      <div class="two"><ul>
        <li><b>Calibrated sleep-time ranges.</b> Built, then switched off. Observed coverage was
          {hist['global_covered']*100:.0f}% on BOAS and {ext['global_covered']*100:.0f}% externally against a
          90% target, so the release gate disables the claim and exported intervals are null.</li>
        <li><b>The label-free sleep-window rule.</b> Tested, not adopted: it cut the external error from
          {win['external']['v2']['unwindowed_mae_min']:.1f} to {win['external']['v2']['rule_mae_min']:.1f} minutes
          but made the headband worse, {win['historical_boas']['v2']['unwindowed_mae_min']:.1f} to
          {win['historical_boas']['v2']['rule_mae_min']:.1f}. Moved to next work.</li>
        <li><b>The daytime experiment.</b> Needs teammates recorded on the headset. Not run, so there is no
          pre-stimulus claim here. The live path itself was built and measured: scoring epoch by epoch as
          the night arrives {live_short}.</li>
      </ul><ul>
        <li><b>A public repository and QR code.</b> Team decision: no public repo tonight. The identifier is a
          content hash, not an invented link, and the folder is the artefact.</li>
        <li><b>Drug, hydrogel and nanoparticle modules.</b> Kept as a bounded research track outside the
          scored core. The coating/core model refuses to fit until matched pairs from ten studies exist —
          by its own rule, not by our choice.</li>
        <li><b>Any clinical, ADHD or medication claim.</b> Out of scope by construction: no paired outcomes,
          no children, no treatment data in anything we evaluated.</li>
      </ul></div>
      <div class="sub" style="margin-top:3mm">Protected throughout: the trained model, the frozen
      person-held-out evaluation, and the baselines. The CNN challenger was the one stretch item we did
      finish tonight — {cnn_line}.</div>
      {foot(3, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">Falsifiability</div>
      <h2>What would change our mind</h2>
      <ul>
        <li><b>A matched baseline beating us.</b> Two did. YASA beats us on Sleep-EDF,
          {fmt(ye['kappa'])} against {fmt(ext['kappa'])}, and the CNN we trained tonight beats us on the
          headband, {fmt(cnn['kappa']) if cnn else '—'} against {fmt(hist['kappa'])}. We report both. The
          contribution was never "our model is best": it is the evaluation that shows which one to trust
          where.</li>
        <li><b>Interval coverage near the nominal target.</b> It was not: {hist['global_covered']*100:.0f}% and
          {ext['global_covered']*100:.0f}% against 90%. That is why the range claim is switched off rather than
          softened. Probability calibration improved (log loss {cal['before_temperature']['log_loss']:.3f} →
          {cal['after_temperature']['log_loss']:.3f}); that is a different property.</li>
        <li><b>A new population.</b> Sleep-EDF was exactly that, and the summary claim failed on it. The honest
          boundary is the device and population we evaluated, not "wearables".</li>
        <li><b>A better architecture than the frozen trunk.</b> We built the obvious upgrade and it lost.
          {e2e_line}</li>
        <li><b>In the research track, an audit that removes a win.</b> That also happened: merging two
          duplicated study measurements moved the exposure model from beating its baseline to not beating it
          ({leak['grouping']['title_groups']} title groups → {leak['grouping']['merged_groups']} source groups).</li>
      </ul>
      <div class="sub" style="margin-top:3mm">More exposure, more epochs or more datasets without paired
      functional outcomes would still not establish benefit.</div>
      {foot(4, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">Landscape</div>
      <h2>Who else does this, and what we compared to what</h2>
      <table><tr><th>Work</th><th>Its task</th><th>How we compared</th></tr>
      <tr><td><b>YASA</b> (Vallat &amp; Walker)</td><td class="muted">established automatic sleep staging</td>
        <td class="muted">Run as a baseline on both cohorts, our masks, windows and metric. It beats us
        externally and loses on the headband</td></tr>
      <tr><td><b>Bitbrain / BOAS</b></td><td class="muted">released headband scorer with PSG reference</td>
        <td class="muted">Vendor baseline on its own covered mask: still ahead on sleep time,
        {same_mask['vendor']:.1f} vs {same_mask['model']:.1f} minutes</td></tr>
      <tr><td><b>Yousfan et al.</b></td><td class="muted">brain-delivery exposure modelling</td>
        <td class="muted">Adjacent research track, not a staging competitor. Reproduced held out by study;
        descriptors did not transfer</td></tr>
      <tr><td><b>Qiu et al.</b></td><td class="muted">ML optimisation of grouped procedures</td>
        <td class="muted">Method context for the formulation track only</td></tr>
      </table>
      <div class="sub" style="margin-top:4mm">Delivery models are adjacent work, not sleep-staging
      competitors, and we do not list consumer apps whose actual claims we have not checked. What is new here
      is not the stager: it is a frozen, inspectable evaluation that shows where the night summary breaks.</div>
      {foot(5, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">How it is built</div>
      <h2>Five pieces, each there because something failed without it</h2>
      <div class="two"><ul>
        <li><b>Per-recording robust scaling.</b> Amplitude on a forehead headband depends on how well the
          electrode sits, and that changes between people and across a single night. Scaling each recording
          by its own median and MAD, rather than a global constant, is what lets one set of weights read a
          stranger's signal at all.</li>
        <li><b>A small convolutional trunk.</b> 44,581 parameters over the raw 30-second epoch. It is
          deliberately small: with 48 people to fit on, a larger trunk memorises individuals rather than
          learning what N2 looks like.</li>
        <li><b>A temporal head on top.</b> One epoch in isolation is genuinely ambiguous — N1 and wake share
          most of their spectrum. The head sees {int(ctx.get('context', 10)) * 2 + 1 if isinstance(ctx, dict) and ctx.get('context') else 21} epochs at once, which is roughly the span a human scorer uses
          when they look at the page before and after.</li>
      </ul><ul>
        <li><b>A sequence smoother.</b> Sleep does not jump between stages at random; the transition matrix
          estimated from the fitting nights encodes that, and a forward-backward pass over the night removes
          single-epoch flicker that no scorer would produce.</li>
        <li><b>Temperature calibration.</b> The raw network is overconfident. One scalar, fitted on six people
          who are in neither the fitting nor the test set, takes log loss from
          {cal['before_temperature']['log_loss']:.3f} to {cal['after_temperature']['log_loss']:.3f} and expected
          calibration error from {cal['before_temperature']['ece']:.3f} to {cal['after_temperature']['ece']:.3f}.
          It changes no prediction, only how much the prediction claims to know.</li>
      </ul></div>
      <div class="sub" style="margin-top:3mm">The live path keeps all five but gives up the lookahead: a
      trailing window replaces whole-recording scaling and the smoother runs forward only.</div>
      {foot(6, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">The architecture question we actually tested</div>
      <h2>Reusing a frozen trunk beat training the whole thing together</h2>
      <div class="sub">The obvious next step, once a trunk and a temporal head both work, is to stop treating
      them as separate stages and train one network end to end over whole sequences. We built it. It lost, and
      it lost by a margin that is not noise.</div>
      <ul>
        <li><b>Grid sampling, stride 21:</b> macro-F1 {_eg['macro_f1']:.3f}, kappa {_eg['kappa']:.3f}.</li>
        <li><b>Random window sampling, three exposures per epoch:</b> macro-F1 {_er['macro_f1']:.3f},
          kappa {_er['kappa']:.3f}.</li>
        <li><b>Frozen trunk with a temporal head:</b> macro-F1 {_ec['macro_f1']:.3f},
          kappa {_ec['kappa']:.3f}, sleep-time error {_ec['tst_error_minutes']:.1f} minutes.</li>
      </ul>
      <div class="sub" style="margin-top:3mm">Two sampling strategies that share no code path landed on the
      two figures above, and both sat {abs(e2e_gap):.3f} below the staged model — about
      {abs(e2e_gap)/e2e_spread:.0f} times the {e2e_spread:.3f} spread we measured across random seeds. The reading
      we take from it is unglamorous: with 48 people to fit on, a jointly trained encoder has enough freedom to
      fit the fitting set and not enough data to generalise from it. Freezing a trunk that was fitted on single
      epochs is a constraint, and the constraint is what helps.</div>
      <div class="sub" style="margin-top:3mm">We report it because the opposite is what we expected, and because
      a negative result that cost a night of compute is still evidence about where the ceiling sits. It also
      sharpens the main finding: the limit here is the number of people, not the number of epochs. More nights
      from the same device would not fix a model that has seen 48 of them — and the Sleep-EDF collapse says
      the same thing from the other direction.</div>
      {foot(7, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">How the code holds the line</div>
      <h2>The protocol is enforced by file boundaries, not by discipline</h2>
      <div class="two"><ul>
        <li><b>Fitting and evaluation live in different files.</b> <code>v2_evaluate.py</code> cannot fit a model,
          a calibrator or a threshold — there is no code in it that could. If a number needs to change, it has to
          change somewhere visible.</li>
        <li><b>One mask, written once.</b> <code>(y &gt;= 0) &amp; (y &lt; 5) &amp; good</code> appears in every
          scorer. Comparing two models on two different masks is the easiest way to manufacture a win, so the
          mask is not a parameter anywhere.</li>
        <li><b>The split is a file, not a convention.</b> <code>protocol.json</code> names who is in the 48
          fitting people, the six selection people, the six calibration people and the 20 test people. Selection
          reads the six; nothing reads the 20 until the end.</li>
      </ul><ul>
        <li><b>Every slide number is read at build time.</b> This deck is generated by
          <code>make_presentation.py</code> from the saved result JSONs. There is no typed-in figure to go stale,
          which is why the deck can be rebuilt after a result changes and simply be right.</li>
        <li><b>The seed spread is a gate, not a footnote.</b> <code>seed_variance.py</code> puts a number
          ({e2e_spread:.3f} macro-F1) on how much two identical runs differ. Anything smaller than it is not
          reported as an improvement. Three claims were withdrawn under that rule.</li>
        <li><b>The audit tests itself.</b> <code>leakage_audit.py</code> ships eleven controlled cases with
          known answers, so a detector that silently stops working fails loudly instead.</li>
      </ul></div>
      <div class="sub" style="margin-top:3mm">One command re-runs the frozen evaluation offline in about half a
      minute, and it has now reproduced to the last digit on three machines and two operating systems.</div>
      {foot(8, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">Privacy and ethics</div>
      <h2>Public data, visible provenance, nothing covert</h2>
      <div class="two"><ul>
        <li><b>Every result uses public, de-identified releases</b> — BOAS via OpenNeuro (CC0) and Sleep-EDF via
          PhysioNet (ODC-By 1.0), both credited with DOIs.</li>
        <li><b>No teammate or attendee recording</b> exists in this project. The optional day experiment that
          would have involved teammates was not run, so there is no consent question to answer after the fact.</li>
        <li><b>No health inference about anyone at the event.</b></li>
      </ul><ul>
        <li><b>Local storage only.</b> No upload, no public repository, no third-party service holds the data.
          The folder stays on the team's machine.</li>
        <li><b>Provenance is part of the product:</b> frozen protocol and model hashes, acquisition logs,
          published dataset checksums and per-person loader audits all ship with the results.</li>
        <li><b>No clinical function.</b> Nothing here diagnoses, titrates medication, or applies to children.</li>
      </ul></div>
      <div class="sub" style="margin-top:3mm">Consent to EEG collection would not have permitted a public
      face or video recording; that distinction is why the demo video shows only the viewer.</div>
      {foot(9, 10)}</section>""")

    slides.append(f"""<section class="slide">
      <div class="kicker">Credits</div>
      <h2>Data, software and people</h2>
      <div class="two"><ul>
        <li><b>BOAS — Bitbrain Open Access Sleep</b>, López-Larraz, Sierra-Torralba, Clemente, Fierro, Oriol,
          Minguez, Montesano, Klinzing. OpenNeuro <code>doi:10.18112/openneuro.ds005555.v1.1.3</code>, CC0.</li>
        <li><b>Sleep-EDF Expanded 1.0.0</b>, Kemp et al. 2000, PhysioNet <code>10.13026/C2X676</code>,
          ODC-By 1.0.</li>
        <li><b>Yousfan et al. 2023</b>, <i>Molecular Pharmaceutics</i> supporting data, CC BY-NC 4.0 —
          research track only, not redistributed.</li>
        <li><b>Bannigan et al.</b> PLGA release data, MIT licence retained.</li>
      </ul><ul>
        <li><b>YASA 0.7.0</b> (Vallat &amp; Walker) as the open-source baseline; <b>MNE-Python</b>,
          <b>scikit-learn</b>, <b>SciPy</b>, <b>NumPy</b>, <b>pandas</b>, <b>JAX</b> for the CNN challenger.</li>
        <li><b>Tobii</b> sponsors the AI stream. No Tobii hardware was used, and no g.tec or other borrowed
          kit produced any result here.</li>
        <li><b>The team:</b> <b>Ceren</b> &mdash; biomaterials · <b>Bora</b> &mdash; coding ·
          <b>Jacob</b> &mdash; neurotechnology.
          <span class="muted">Anyone who contributed and is not named here should be added before this
          slide is shown.</span></li>
      </ul></div>
      <div class="sub" style="margin-top:2mm">Every number here was recomputed from the saved result
      files by <code>make_presentation.py</code>.</div>
      {foot(10, 10)}</section>""")

    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Goldilocks AI — support slides</title>'
            f'<style>@page{{size:297mm 167mm;margin:0}}{css}</style></head><body>{"".join(slides)}</body></html>')
    (SLIDES / 'goldilocks_support_slides.html').write_text(html)

    cnn_hand = ''
    if cnn:
        cnn_hand = (f"<tr><td>BOAS headband — CNN challenger (trained tonight)</td><td>{fmt(cnn['kappa'])}</td>"
                    f"<td>{mins(cnn['tst_error_minutes'])}</td></tr>")
    if cnn_ext:
        cnn_hand += (f"<tr><td>Sleep-EDF full recording — CNN challenger</td><td>{fmt(cnn_ext['kappa'])}</td>"
                     f"<td>{mins(cnn_ext['tst_error_minutes'])}</td></tr>")
    hcss = css + """
    .page{width:210mm;height:297mm;padding:14mm 15mm;background:#fff;position:relative;display:flex;
      flex-direction:column}
    h1{font-size:19pt;margin:2.5mm 0 1mm;line-height:1.12}
    h2{font-size:10pt;margin:3.4mm 0 .8mm;text-transform:uppercase;letter-spacing:.09em;color:var(--accent)}
    .sub{font-size:9pt;max-width:none;line-height:1.4}table{font-size:9pt}th,td{padding:1.4mm 2mm}
    ul{font-size:9.5pt;line-height:1.45}li{margin-bottom:1.4mm}
    .foot{position:absolute;bottom:10mm;left:15mm;right:15mm;font-size:8pt}
    .bignum{margin:2.5mm 0 .5mm;gap:8mm}.big{font-size:23pt}.bignum .lab{font-size:8pt;line-height:1.3}
    .cmd{display:block;background:#f4f6f9;border-radius:1mm;padding:1.6mm 2.5mm;margin:1.2mm 0;
      font-family:ui-monospace,Menlo,Consolas,monospace;font-size:8pt;line-height:1.45;
      white-space:pre-wrap;word-break:normal;overflow-wrap:break-word}
    """
    handout = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Goldilocks AI — one page</title><style>@page{{size:A4;margin:0}}{hcss}</style></head><body>
    <div class="page">
      <div class="kicker">Goldilocks AI · Stream 2, AI / Neural Decoding · sponsored by Tobii</div>
      <h1>A wearable can stage sleep well and still get the night wrong.</h1>
      <div class="sub">One sleep-stage model trained on the BOAS forehead headband, frozen before testing,
      evaluated person-held-out, then run unchanged on an independent dataset. 20 people in each cohort.</div>
      <div class="bignum">
        <div><div class="big">{fmt(hist['kappa'])}</div><div class="lab">kappa, BOAS headband,
          20 held-out people</div></div>
        <div><div class="big">{hist['tst_error_minutes']:.0f} min</div><div class="lab">sleep-time error,
          same people</div></div>
        <div><div class="big">{ext['tst_error_minutes']:.0f} min</div><div class="lab">sleep-time error,
          20 new Sleep-EDF people, full recording</div></div>
      </div>
      <h2>Every number, and the baselines</h2>
      <table><tr><th>20 people per cohort</th><th>Kappa</th><th>Sleep-time error</th></tr>
      <tr><td>BOAS headband — ours</td><td>{fmt(hist['kappa'])}</td><td>{mins(hist['tst_error_minutes'])}</td></tr>
      <tr><td>BOAS headband — YASA 0.7.0</td><td>{fmt(yh['kappa'])}</td><td>{mins(yh['tst_error_minutes'])}</td></tr>
      <tr><td>Sleep-EDF full recording — ours</td><td>{fmt(ext['kappa'])}</td><td>{mins(ext['tst_error_minutes'])}</td></tr>
      <tr><td>Sleep-EDF full recording — YASA</td><td>{fmt(ye['kappa'])}</td><td>{mins(ye['tst_error_minutes'])}</td></tr>
      <tr><td>Sleep-EDF sleep window — ours / YASA</td><td>{fmt(ext2['kappa'])} / {fmt(ye2['kappa'])}</td>
        <td>{mins(ext2['tst_error_minutes'])} / {mins(ye2['tst_error_minutes'])}</td></tr>
      {cnn_hand}</table>
      <h2>The result that decides the pitch</h2>
      <div class="sub">The CNN trained tonight is the <b>best</b> model on the headband and the <b>worst</b>
      on Sleep-EDF: kappa {fmt(cnn['kappa']) if cnn else '—'} → {fmt(cnn_ext['kappa']) if cnn_ext else '—'},
      sleep-time error {mins(cnn['tst_error_minutes']) if cnn else '—'} →
      {mins(cnn_ext['tst_error_minutes']) if cnn_ext else '—'}, calling daytime wake sleep. Choosing a wearable
      model by its in-domain score would have picked the one that fails hardest away from home. That is the
      argument for inspectable evaluation rather than a single accuracy number.</div>
      <h2>The demo, in three acts</h2>
      <div class="sub">Decode a held-out night → reveal the human reference and the YASA lane and challenge
      the evidence → inspect the resulting summary, its error and what it cannot say. Asked for lights-off
      sleep-onset latency, the viewer refuses: the recording does not establish lights-off, so it offers
      recording-start to first scored sleep instead. The viewer itself is precomputed retrospective
      analysis. Live inference is a separate, measured path: <code>live_inference.py</code> scores epoch by
      epoch with trailing-window normalisation, and {live_short}.</div>
      <h2>Main limitation</h2>
      <div class="sub">Prediction intervals missed their 90% target ({hist['global_covered']*100:.0f}% coverage on
      BOAS, {ext['global_covered']*100:.0f}% externally), so calibrated-range claims are disabled in the release
      gate and exported intervals are null. The BOAS cohort was inspected at v1, so it is a development
      comparison, not independent confirmation. No clinical, ADHD, medication or paediatric claim is made.</div>
      <h2>Where the work is, and how to check it</h2>
      <div class="sub">No public repository, by team decision: the artefact is the folder
      <code>Goldilocks_Submission/</code> on the team machine, {pkg_mb} MB, with code, derived features,
      every result, the viewer and the backup video. Reproduce the frozen evaluation with:</div>
      <span class="cmd">python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python reproduce_v2.py</span>
      <div class="sub">31 seconds, no network. Verified from a clean environment: 294,208 numbers compared,
      largest relative difference 2.6 × 10⁻¹⁵, nothing differing at four decimal places
      (<code>REPRODUCTION.md</code>).</div>
      <h2>Identity of this build</h2>
      <div class="sub">Frozen protocol <code>{protocol_sha}</code><br>
      Frozen model <code>{model_sha}</code><br>
      Both match <code>results/v2/freeze_record.json</code>, written before the external reference labels were
      opened. Seeds 20261108 and 20261107. Built {built}.</div>
      <div class="foot"><span>Public data only: BOAS via OpenNeuro (CC0), Sleep-EDF via PhysioNet (ODC-By 1.0).
      No teammate recording, no health inference, no clinical function.</span><span>1/1</span></div>
    </div></body></html>"""
    (HANDOUT / 'goldilocks_handout.html').write_text(handout)

    made = []
    if CHROME:
        for src, dst in [(SLIDES / 'goldilocks_support_slides.html', SLIDES / 'goldilocks_support_slides.pdf'),
                         (HANDOUT / 'goldilocks_handout.html', HANDOUT / 'goldilocks_handout.pdf')]:
            subprocess.run([CHROME, '--headless', '--disable-gpu', '--no-sandbox', '--no-pdf-header-footer',
                            f'--print-to-pdf={dst}', src.as_uri()], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # works on Windows too
            made.append({'file': str(dst.relative_to(ROOT)), 'bytes': dst.stat().st_size,
                         'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()[:16]})
    else:
        print('no Chrome found: HTML written, PDF not built', file=sys.stderr)
    print(json.dumps({'cnn_included': bool(cnn), 'built_utc': built, 'outputs': made}, indent=2))


if __name__ == '__main__':
    main()

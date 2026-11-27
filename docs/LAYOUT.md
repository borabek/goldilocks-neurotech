# Where files moved

On 27 November 2026 the repository was reorganised. Contents did not change, except for the
import-path and repository-root lines that code needs after a move and the command lines in
docstrings and write-ups that name the new paths (git records these as
renames, so `git log --follow <new path>` shows the full history of each file). The last commit
with the old flat layout is `dad97a1`.

`PREREGISTRATION.md` and everything under `hackathon/` were written for the old layout, so they
name files by their old paths. This table maps them.

Not moved, because the pre-registration hashes or the imports of frozen code depend on their
place: `pipeline.py`, `v2_model.py`, `yasa_evaluate.py`, `protocol.json`, `v2_protocol.json`,
`preregistration_hashes.json`, and everything under `analysis/`, `data/` and `results/` (apart from
the hackathon demo video, which went to `hackathon/demo/`).

Removed: `plan/GOLDILOCKS_Plan_Rev17.md` and `plan/GOLDILOCKS_Plan_Rev18.md` (superseded by
revision 19, now `hackathon/plan/GOLDILOCKS_Plan_Rev19.md`), and `upload.log` / `upload.exit`
(a failed upload attempt with no other content). All four are still in the git history.

| Old path | New path |
| --- | --- |
| `DATA_SOURCES.md` | `docs/DATA_SOURCES.md` |
| `BACKLOG.md` | `hackathon/BACKLOG.md` |
| `BUILD_REPORT.md` | `hackathon/BUILD_REPORT.md` |
| `BUILD_REPORT_V2.md` | `hackathon/BUILD_REPORT_V2.md` |
| `QA_CARD.md` | `hackathon/QA_CARD.md` |
| `README_V2.md` | `hackathon/README_V2.md` |
| `START_HERE.md` | `hackathon/START_HERE.md` |
| `results/yasa/Goldilocks_backup_demo.mp4` | `hackathon/demo/Goldilocks_backup_demo.mp4` |
| `demo.sh` | `hackathon/demo/demo.sh` |
| `demo_video.html` | `hackathon/demo/demo_video.html` |
| `demo_video_v2.html` | `hackathon/demo/demo_video_v2.html` |
| `handout/goldilocks_handout.html` | `hackathon/handout/goldilocks_handout.html` |
| `handout/goldilocks_handout.pdf` | `hackathon/handout/goldilocks_handout.pdf` |
| `release_protocol.json` | `hackathon/manifests/release_protocol.json` |
| `source_sha256.json` | `hackathon/manifests/source_sha256.json` |
| `source_sha256_v2.json` | `hackathon/manifests/source_sha256_v2.json` |
| `plan/GOLDILOCKS_Plan_Rev19.md` | `hackathon/plan/GOLDILOCKS_Plan_Rev19.md` |
| `v2_desktop.png` | `hackathon/screenshots/v2_desktop.png` |
| `v2_mobile.png` | `hackathon/screenshots/v2_mobile.png` |
| `v2_mobile_controls.png` | `hackathon/screenshots/v2_mobile_controls.png` |
| `v2_mobile_top.png` | `hackathon/screenshots/v2_mobile_top.png` |
| `slides/goldilocks_support_slides.html` | `hackathon/slides/goldilocks_support_slides.html` |
| `slides/goldilocks_support_slides.pdf` | `hackathon/slides/goldilocks_support_slides.pdf` |
| `build_report.py` | `hackathon/tools/build_report.py` |
| `extract_watch.py` | `hackathon/tools/extract_watch.py` |
| `make_plan_rev19.py` | `hackathon/tools/make_plan_rev19.py` |
| `make_presentation.py` | `hackathon/tools/make_presentation.py` |
| `make_viewer_v3.py` | `hackathon/tools/make_viewer_v3.py` |
| `package_build.py` | `hackathon/tools/package_build.py` |
| `package_v2.py` | `hackathon/tools/package_v2.py` |
| `release_audit.py` | `hackathon/tools/release_audit.py` |
| `reproduce.py` | `hackathon/tools/reproduce.py` |
| `update_plan_v2.py` | `hackathon/tools/update_plan_v2.py` |
| `update_readme.py` | `hackathon/tools/update_readme.py` |
| `window_rule_run.py` | `hackathon/tools/window_rule_run.py` |
| `plan/coating_core_evidence.md` | `neurorelease/coating_core/coating_core_evidence.md` |
| `plan/coating_core_numeric.csv` | `neurorelease/coating_core/coating_core_numeric.csv` |
| `acquire.mjs` | `scripts/acquire.mjs` |
| `acquire_external.mjs` | `scripts/acquire_external.mjs` |
| `verify_claims.py` | `scripts/verify_claims.py` |
| `build_epoch_cache.py` | `src/build_epoch_cache.py` |
| `cnn_ablation.py` | `src/cnn_ablation.py` |
| `cnn_challenger.py` | `src/cnn_challenger.py` |
| `cnn_context.py` | `src/cnn_context.py` |
| `cnn_endtoend.py` | `src/cnn_endtoend.py` |
| `cnn_external.py` | `src/cnn_external.py` |
| `cnn_posthoc.py` | `src/cnn_posthoc.py` |
| `confidence_intervals.py` | `src/confidence_intervals.py` |
| `context_sweep.py` | `src/context_sweep.py` |
| `diagnostics.py` | `src/diagnostics.py` |
| `external_all.py` | `src/external_all.py` |
| `external_eval.py` | `src/external_eval.py` |
| `external_score.py` | `src/external_score.py` |
| `live_inference.py` | `src/live_inference.py` |
| `metrics_summary.py` | `src/metrics_summary.py` |
| `reproduce_v2.py` | `src/reproduce_v2.py` |
| `seed_variance.py` | `src/seed_variance.py` |
| `v2_evaluate.py` | `src/v2_evaluate.py` |
| `v2_report.py` | `src/v2_report.py` |
| `vendor_and_failures.py` | `src/vendor_and_failures.py` |
| `test_pipeline.py` | `tests/test_pipeline.py` |
| `test_v2.py` | `tests/test_v2.py` |
| `research/` (whole folder) | `neurorelease/` (same structure inside) |
| run logs and exit codes at the top level (`*.log`, `*.exit`) | `hackathon/logs/` |

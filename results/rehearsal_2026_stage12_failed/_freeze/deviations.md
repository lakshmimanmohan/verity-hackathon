# Deviations (rehearsal, seed 2026)

Every stage failure, guard that fired, restore after a crash, resume and rerun is listed here with its time. Nothing here is edited by hand afterwards.

- 2026-11-25T12:59:05Z  stage 12  failed: cost_model exited 1; see results/rehearsal_2026/_freeze/logs/stage_12.log
- 2026-11-25T13:05:20Z  stage 12  manifest rewritten: cost/cost_model.py and eval/freeze_run.py changed after stage 11 (the stage 12 fix); the old manifest is kept as freeze_manifest_before_stage12_fix.json; changed frozen files: cost/cost_model.py, eval/freeze_run.py; model-side assets unchanged
- 2026-11-25T13:05:24Z  stage start  resume: resumed; stages already done: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- 2026-11-25T13:05:24Z  stage 0  guard fired (outputs unchanged since the stage finished): changed or missing: results/rehearsal_2026/_freeze/freeze_manifest.json

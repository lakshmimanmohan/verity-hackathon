# Deviations (rehearsal, seed 2026)

Every stage failure, guard that fired, restore after a crash, resume and rerun is listed here with its time. Nothing here is edited by hand afterwards.

- 2026-11-25T12:29:48Z  stage start  resume: resumed; stages already done: [0, 1, 2, 3, 4, 5, 6]
- 2026-11-25T12:29:55Z  stage 7  archive skipped: freeze_run.py stage 7 archives calls.jsonl, which does not exist when every B answer came from the cache (0 live calls): the stage raised FileNotFoundError after the B step. Finished with a wrapper (not committed) that skips the archive.
- 2026-11-25T12:31:05Z  stage 9  failed: decoy_check_report exited 1; see results/rehearsal_2026/_freeze/logs/stage_09.log
- 2026-11-25T12:31:17Z  stage start  resume: resumed; stages already done: [0, 1, 2, 3, 4, 5, 6, 7, 8]
- 2026-11-25T12:31:23Z  stage 9  decoy report skipped: eval/decoy_check_report.py formats meta['prompt_cache']['hit_ratio_excl_first'], which is None when a run makes 0 live calls (all answers cached): TypeError. Skipped in the wrapper; not run.
- 2026-11-25T12:31:39Z  stage 10  failed: b_sample_report exited 1; see results/rehearsal_2026/_freeze/logs/stage_10.log

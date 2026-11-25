# Latency, final figures from the freeze run (seed 7919)

Read from files under `results/eval_7919` and `results/freeze_2026-11-25`; the one dev input is labeled dev. Nothing was rerun, no call was made, no truth or data file was opened.

## 1. What the freeze run saved

### C per-pair latency arrays

**No per-pair array was saved** for C in the freeze run: `options/latency/` holds only `latency_summary.json` and `latency_summary.txt`, and no `.npz`, `.npy` or per-pair file exists under the C model folder or the run folder. The stage 6 and stage 8 latency bench keeps summary statistics only.

What exists for C (source: `results/eval_7919/options/latency/latency_summary.json`, 44,022 test band pairs, one core, features included, one pair at a time):

| repeat | p50 ms | p95 ms | max ms | mean ms | pairs | total s |
|---|---|---|---|---|---|---|
| 1 | 0.0826 | 0.2871 | 66.0761 | 0.113 | 44022 | 5.017 |
| 2 | 0.0831 | 0.2791 | 2.1451 | 0.1027 | 44022 | 4.564 |
| 3 | 0.0813 | 0.2777 | 1.5294 | 0.1016 | 44022 | 4.513 |
| median of repeats | 0.0826 | 0.2791 | 2.1451 | 0.1027 |  |  |

### B call latencies

| calls file | calls | p50 s | p95 s | max s | source |
|---|---|---|---|---|---|
| sample5000 | 4996 | 1.045 | 1.330 | 9.280 | results/eval_7919/options/b_llm/openai/sample5000/calls.jsonl |
| dfull | 358 | 1.047 | 1.340 | 4.049 | results/eval_7919/options/b_llm/openai/dfull/calls.jsonl |
| decoycheck | 52 | 1.088 | 1.410 | 2.226 | results/eval_7919/options/b_llm/openai/decoycheck/calls.jsonl |
| sample5000 archive | 4996 | 1.045 | 1.330 | 9.280 | results/freeze_2026-11-25/b_archive/calls.jsonl.gz |

Percentiles are computed here from the `latency_s` of each live call (linear interpolation). `sample5000/calls.jsonl` is a local file (not committed); the committed copy of the same calls is the freeze archive `calls.jsonl.gz`.

## 2. D latency with the LLM fallback (simulation)

**Not run.** The simulation draws a C latency per pair, and the freeze run did not save C's per-pair latency array (section 1). Dev data was not substituted: no dev array, no dev simulation values and no summary statistics were used in place of the missing array.

Consequences:

- There is no freeze-run value for D's p99, p99.5, p99.9, p99.95 or max with the fallback.
- The "D fallback tail" row of `head_to_head_v2` is **not superseded**: it still shows the dev simulation recorded in `cost_final/assumptions.csv` (p99.9 0.5671 ms, p99.95 921.2648 ms, from a routed share of 0.085%), and it is a dev figure, not a freeze-run figure.
- What would be needed: a rerun of the latency bench that also saves C's per-pair latencies (stage 8 of the runbook writes summary statistics only). That rerun was not made.

## 3. Per record

### D's routed pairs, per record of the test group (no truth)

| quantity | value | how |
|---|---|---|
| test group records | 4,500 | `split_records.csv`, side = test |
| routed pairs (D, rule v2) | 426 of 44,022 test band pairs (0.968%) | `routed_fullband_pairs.csv`, rows of C's `decisions.csv` |
| mean routed pairs per record (each pair counted for both of its records) | 0.1893 | 2 x routed pairs / records |
| mean routed pairs per record (each pair counted once) | 0.0947 | routed pairs / records |
| records with at least one routed pair | 518 of 4,500 = 11.51% | distinct records in the routed list |
| most routed pairs on one record | 12 | routed list |
| records by number of routed pairs (n pairs: records) | 1: 343, 2: 98, 3: 37, 4: 19, 5: 12, 6: 5, 7: 1, 8: 1, 9: 1, 12: 1 | routed list |
| routed records outside the test group | 0 | must be 0 |
| share expected if routed pairs were independent, exponent 20.05 (band pairs per record, as specified) | 17.71% | 1 - (1 - 426/44,022)^20.05; band pairs per record from `results/eval_7919/band_summary.json` |
| same, exponent 19.57 (test band pairs a test record takes part in: 2 x 44,022 / 4,500) | 17.33% | the per-record degree inside the test group |

Sources: `results/eval_7919/split_records.csv`, `results/eval_7919/options/d_hybrid/routed_fullband_pairs.csv`, `results/eval_7919/options/c_model/c_v1_lgbm/decisions.csv`.

### B per record

B makes one call per band pair: 20.05 calls per record (`results/eval_7919/band_summary.json`). At 10 workers, the time per record is calls x latency / 10 (mean) or ceil(calls / 10) x latency (whole rounds); queueing is not included.

| call latency used | latency s | mean time per record at 10 workers, s | whole rounds, s |
|---|---|---|---|
| freeze run, call p50 (2 workers) | 1.045 | 2.10 | 3.14 |
| freeze run, call p95 (2 workers) | 1.330 | 2.67 | 3.99 |
| DEV: call p50 at 10 workers | 1.041 | 2.09 | 3.12 |
| DEV: call p95 at 10 workers | 1.320 | 2.65 | 3.96 |

Sources: freeze run call latency from `results/eval_7919/options/b_llm/openai/sample5000/timings.json` (the run used 2 workers); DEV concurrency measurement from `results/dev_2026_run11/options/b_llm/openai/concurrency/concurrency.json` (dev run, labeled dev: it was not measured in the freeze run).

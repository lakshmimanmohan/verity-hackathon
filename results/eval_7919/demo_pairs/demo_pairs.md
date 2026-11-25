# Three demo pairs from the freeze run (seed 7919)

Rule (applied to the frozen outputs, test-group pairs only, sorted by (record_a, record_b), ids in sorted order inside each pair; selection uses no truth, the truth file is read afterwards only to mark each pair correct or not):

1. the first pair whose `decision_status` in `scored_pairs.csv` is `match` (Verity decides, outside the band);
2. the first band pair that C decides `same` and D does not route to the LLM (route `c_same` in D's decisions);
3. the first pair D routes to the LLM (route `llm`).

## Pair 1

| field | record A (R00001e929c) | record B (R5aa4754d82) |
|---|---|---|
| name | Falcon Evergreen Manufacturing Co | Falcon Evergreen Manufacturing Co |
| address | 9154 River Court | 9154 River Court |
| city | Tampa | Tampa |
| state | FL | FL |
| ZIP | 33689 | 33689 |
| phone | 3246665918 | +1 324 666 5918 |
| website | falconevergreen.com | www.falconevergreen.com |

| item | value |
|---|---|
| route | Verity decides (outside the band) |
| Verity's score and status | 0.970414 (match, match type weak_signal) |
| C's calibrated p and decision | not applicable (the pair is outside the band; Verity decides) |
| B's p_same and decision | not called for this pair |
| final answer | same |
| correct | yes |
| time | 0.2257 ms (Verity's per-pair scoring p50, median over all 216,092 scored pairs; stage verity.score of run_info.json) |
| cost | $0 |

## Pair 2

| field | record A (R009d5c8244) | record B (Rbab5603f83) |
|---|---|---|
| name | Redstone Granite Medical Supply, Llc | Redstone Granite Medical Supply LLC |
| address | PO Box 15532 | 8526 Ridge Lane |
| city | Milwaukee | Milwaukee |
| state | WI | WI |
| ZIP | 53207-1254 | 53207 |
| phone |  | +1 650 589 8462 |
| website |  | redstonegranite.com |

| item | value |
|---|---|
| route | c_same (band pair, C decides without the LLM) |
| Verity's score and status | 0.846448 (review, match type insufficient_evidence) |
| C's calibrated p and decision | 1.000000 (same) |
| B's p_same and decision | not called for this pair |
| final answer | same |
| correct | yes |
| time | 0.0826 ms (C's median-of-repeats p50, median over all test band pairs; one core, features included) |
| cost | $0 |

## Pair 3

| field | record A (R009d5c8244) | record B (R05988575b6) |
|---|---|---|
| name | Redstone Granite Medical Supply, Llc | Redstone Granite Me |
| address | PO Box 15532 | 8526 Ridge Lane |
| city | Milwaukee | Milwaukee |
| state | WI | WI |
| ZIP | 53207-1254 | 53207 |
| phone |  | (650) 589-8462 |
| website |  | redstonegranite.com |

| item | value |
|---|---|
| route | llm (band pair routed to the LLM) |
| Verity's score and status | 0.813628 (review, match type below_threshold) |
| C's calibrated p and decision | 0.085708 (different) |
| B's p_same and decision | 0.8800 (same) |
| final answer | same |
| correct | yes |
| time | 0.964 s (this pair's live call) |
| cost | $0.001505 (this pair's live call) |

N_CORRECT = 3 of 3

Sources: `results/eval_7919/verity_data/scored_pairs.csv`, `results/eval_7919/options/d_hybrid/decisions_fullband.csv`, `results/eval_7919/split_records.csv`, `orgs_raw.csv` (record fields), `options/c_model/c_v1_lgbm/decisions.csv`, B's `decisions.csv` files, `options/b_llm/openai/*/calls.jsonl`, `verity_data/run_info.json`, `options/latency/latency_summary.json`; `orgs_raw_truth.csv` only for the correct / not correct mark.

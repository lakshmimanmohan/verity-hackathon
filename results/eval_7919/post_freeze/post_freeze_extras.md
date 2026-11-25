EXPLORATORY: not a protocol result; no setting was chosen from these numbers.

# Post-freeze extras (seed 7919 freeze run)

Everything below is read from the frozen outputs (files under `results/eval_7919`, plus `results/dstudy_summary` and the development learning curve for sections 2 and 4, both marked development). Nothing was rerun, no call was made, and no setting was chosen from these numbers.

## 1. False merges

False-merge rate = false merges / predicted matches, overall in the test group (Verity's own decisions outside the band, the option's decisions inside it). Counts are read from the score reports. Intervals: exact (Clopper-Pearson) on the full band; bootstrap (1,000 resamples, seed 0, sample weights kept, the exact counts outside the band added to every resample) on the sample basis. The truth file was read only to flag each sampled pair's false merges and, in section 2, to count the errors among D's routed pairs.

### Full band (exact)

| option | predicted matches | false merges | false-merge rate | 95% interval (exact binomial) | decoy pairs | decoy false merges | source |
|---|---|---|---|---|---|---|---|
| A | 2,909 | 0 | 0.000% | [0.000%, 0.127%] | 111 | 0 | results/eval_7919/options/a_threshold/score_report.txt |
| C | 3,239 | 9 | 0.278% | [0.127%, 0.527%] | 111 | 0 | results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt |
| D | 3,255 | 9 | 0.276% | [0.127%, 0.524%] | 111 | 0 | results/eval_7919/options/d_hybrid/score_report_fullband.txt |
| D (orig) | 3,239 | 7 | 0.216% | [0.087%, 0.445%] | 111 | 0 | results/eval_7919/options/d_hybrid/score_report_fullband_original.txt |

### 5,000-pair sample (weighted estimates)

| option | predicted matches | false merges | false-merge rate | 95% interval (bootstrap) | decoy pairs | decoy false merges | report | decisions |
|---|---|---|---|---|---|---|---|---|
| A | 2,909.0 | 0.0 | 0.000% | [0.000%, 0.000%] | 157.9 | 0.0 | results/eval_7919/options/a_threshold/score_report_sample5000.txt | results/eval_7919/options/a_threshold/decisions.csv |
| B | 3,255.4 | 9.4 | 0.289% | [0.000%, 0.867%] | 157.9 | 0.0 | results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt | results/eval_7919/options/b_llm/openai/sample5000/decisions.csv |
| C | 3,239.4 | 9.4 | 0.290% | [0.000%, 0.871%] | 157.9 | 0.0 | results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt | results/eval_7919/options/c_model/c_v1_lgbm/decisions.csv |
| D | 3,264.8 | 18.8 | 0.576% | [0.000%, 1.430%] | 157.9 | 0.0 | results/eval_7919/options/d_hybrid/score_report_sample5000.txt | results/eval_7919/options/d_hybrid/decisions.csv |

The sample-basis false merges recomputed from the decisions and the truth reproduce the report's figures for every option: A yes, B yes, C yes, D yes (the reports print weighted counts to one decimal, so the match is to within 0.05). The two bases are not comparable with each other.

## 2. Review queue (if D's routed pairs went to a human steward instead of the LLM)

Band pairs per record: 20.0499 (`results/eval_7919/cost_final/assumptions.csv`). Daily queue = records a day x band pairs per record x routed share.

| setting | routed pairs | band pairs | routed per 1,000 band pairs | queue per day at 100K records | at 1M records | at 10M records | source |
|---|---|---|---|---|---|---|---|
| D | 426 | 44,022 | 9.677 | 19,402 | 194,022 | 1,940,225 | results/eval_7919/options/d_hybrid/routed_fullband_pairs.csv |
| D (orig) | 16 | 44,022 | 0.363 | 729 | 7,287 | 72,872 | results/eval_7919/options/d_hybrid/routed_fullband_pairs_original.csv |

### Oracle steward (upper bound): full-band quality if every routed pair were decided correctly

An upper bound, not an estimate of what stewards would achieve: the routed pairs' errors are removed as if every steward decision were right.

| setting | routed errors (false merges / misses) | band F1 now | band F1 oracle | band precision now | band precision oracle | band recall now | band recall oracle | overall precision now | overall precision oracle | overall recall now | overall recall oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 2 / 0 | 0.9862 | 0.9889 | 0.9754 | 0.9808 | 0.9972 | 0.9972 | 0.9972 | 0.9978 | 0.5713 | 0.5713 |
| D (orig) | 0 / 0 | 0.9689 | 0.9689 | 0.9800 | 0.9800 | 0.9581 | 0.9581 | 0.9978 | 0.9978 | 0.5688 | 0.5688 |

Sources: the full-band score reports (counts) and D's full-band decisions with the routed lists (the routed pairs' errors; truth read only for those pairs).

### Development-seed trade-off (seeds 11, 12, 13, 42 and 2026 pooled; existing file, not recomputed)

| lo (hi 0.9) | pooled LLM share of the test band | pooled band F1 | precision | recall | false merges | misses | routed pairs |
|---|---|---|---|---|---|---|---|
| 0.005 | 0.260% | 0.9898 | 0.9915 | 0.9881 | 15 | 21 | 576 |
| 0.01 | 0.170% | 0.9892 | 0.9914 | 0.9869 | 15 | 23 | 378 |
| 0.05 | 0.138% | 0.9863 | 0.9914 | 0.9813 | 15 | 33 | 307 |
| 0.1 | 0.114% | 0.9831 | 0.9913 | 0.9750 | 15 | 44 | 254 |
| 0.5 | 0.070% | 0.9754 | 0.9850 | 0.9659 | 26 | 60 | 155 |

Source: `results/dstudy_summary/d_select_lo.csv`. Development seeds: these rows were the basis of the lo choice made before the freeze; they are shown for the trade-off only.

## 3. Explanation exhibit for the three demo pairs

Evidence stored in `scored_pairs.csv` for each demo pair (per-field similarity and weight, the evidence mass and conflict mass, the evidence gate result) and C's probability. The weighted column is similarity x weight (derived). No new model run.

### Pair 1: R00001e929c / R5aa4754d82

| field | similarity | weight | similarity x weight (derived) | evidence similarity |
|---|---|---|---|---|
| address_line | 1.0000 | 0.2572 | 0.2572 | 1.0000 |
| city | 1.0000 | 0.0009 | 0.0009 | 1.0000 |
| company_name | 1.0000 | 0.2830 | 0.2830 | 1.0000 |
| phone | 0.8000 | 0.2447 | 0.1958 | 0.8000 |
| postal_code | 1.0000 | 0.1156 | 0.1156 | 1.0000 |
| state | 1.0000 | 0.0008 | 0.0008 | 1.0000 |
| website | 0.9231 | 0.0978 | 0.0903 | 0.6667 |

| item | value |
|---|---|
| score (final) / score uncapped | 0.970414 / 0.970414 |
| match type | weak_signal |
| decision status | match |
| evidence mass | 0.657507 |
| conflict mass | 0.000000 |
| evidence gate result (evidence_passes) | true |
| applied / effective threshold | 0.820000 / 0.820000 |
| C's calibrated probability | not applicable (the pair is outside the band; Verity decides) |
| route | Verity decides (outside the band) |

### Pair 2: R009d5c8244 / Rbab5603f83

| field | similarity | weight | similarity x weight (derived) | evidence similarity |
|---|---|---|---|---|
| address_line | 0.1579 | 0.3912 | 0.0618 | 0.1481 |
| city | 1.0000 | 0.0014 | 0.0014 | 1.0000 |
| company_name | 0.9444 | 0.4303 | 0.4064 | 0.9859 |
| postal_code | 0.7143 | 0.1758 | 0.1256 | 0.0000 |
| state | 1.0000 | 0.0012 | 0.0012 | 1.0000 |

| item | value |
|---|---|
| score (final) / score uncapped | 0.846448 / 0.846448 |
| match type | insufficient_evidence |
| decision status | review |
| evidence mass | 0.001717 |
| conflict mass | 0.000000 |
| evidence gate result (evidence_passes) | false |
| applied / effective threshold | 0.820000 / 0.820000 |
| C's calibrated probability | 1.000000 |
| route | c_same (band pair, C decides without the LLM) |

### Pair 3: R009d5c8244 / R05988575b6

| field | similarity | weight | similarity x weight (derived) | evidence similarity |
|---|---|---|---|---|
| address_line | 0.1579 | 0.3912 | 0.0618 | 0.1481 |
| city | 1.0000 | 0.0014 | 0.0014 | 1.0000 |
| company_name | 0.6667 | 0.4303 | 0.2869 | 0.6909 |
| postal_code | 0.7143 | 0.1758 | 0.1256 | 0.0000 |
| state | 1.0000 | 0.0012 | 0.0012 | 1.0000 |

| item | value |
|---|---|
| score (final) / score uncapped | 0.813628 / 0.813628 |
| match type | below_threshold |
| decision status | review |
| evidence mass | 0.001717 |
| conflict mass | 0.000000 |
| evidence gate result (evidence_passes) | false |
| applied / effective threshold | 0.820000 / 0.820000 |
| C's calibrated probability | 0.085708 |
| route | llm (band pair routed to the LLM) |

Sources: `results/eval_7919/demo_pairs/demo_pairs.csv`, `results/eval_7919/verity_data/scored_pairs.csv`.

## 4. Effort to quality (development seed 2026)

Labeling effort at 5 hours per 1,000 labeled pairs (`results/eval_7919/cost_final/assumptions.csv`). Caveat from that file: cost/effort_log.csv row 'Hand check of 100 pairs' (0.5 h); sensitivity 2 and 10; the hand check was blind agreement checking, not a labeling workflow. Mean band F1 is the mean over the sampling seeds of the development learning curve (lc_v2), **development seed 2026, not the freeze run**.

| labels | hours | sampling seeds | mean band F1 | min | max |
|---|---|---|---|---|---|
| 100 | 0.50 | 5 | 0.5097 | 0.2234 | 0.8525 |
| 250 | 1.25 | 5 | 0.7648 | 0.5255 | 0.9427 |
| 500 | 2.50 | 5 | 0.7771 | 0.5504 | 0.9103 |
| 1,000 | 5.00 | 5 | 0.8376 | 0.6093 | 0.9569 |
| 2,000 | 10.00 | 5 | 0.8999 | 0.8507 | 0.9489 |
| 63,665 | 318.32 | 1 | 0.9653 | 0.9653 | 0.9653 |

The last row is the point of the same file at all train pairs (one run, so one seed): the whole labeled train band of the development run.

Source: `results/dev_2026_run11/options/c_model/lc_v2/learning_curve.csv`.

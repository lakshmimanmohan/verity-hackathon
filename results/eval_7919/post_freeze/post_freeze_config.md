EXPLORATORY: not a protocol result; no setting was chosen from these numbers.

# Learned configuration and setup time (freeze run, seed 7919)

Read from the freeze run's Verity outputs. Nothing was rerun. Every value is marked **learned** (computed from this run's data) or **fixed** (a setting that is the same for every run, or chosen before the freeze).

## 1. Learned configuration

### Per column

| column | semantic type | uniqueness ratio | null rate | discriminative score | reliability | normalized discriminative weight | composite weight | origin |
|---|---|---|---|---|---|---|---|---|
| company_name | NAME | 0.7908 | 0.0000 | 0.7908 | 0.1000 | 0.282954 | 0.278972 | learned from the data |
| address_line | STREET_ADDRESS | 0.7189 | 0.0000 | 0.7189 | 0.1000 | 0.257228 | 0.253608 | learned from the data |
| phone | QUASI_IDENTIFIER | 0.7455 | 0.0826 | 0.6839 | 0.1000 | 0.244712 | 0.241268 | learned from the data |
| postal_code | DESCRIPTIVE | 0.3231 | 0.0000 | 0.3231 | 0.1000 | 0.115608 | 0.113981 | learned from the data |
| website | DESCRIPTIVE | 0.3356 | 0.1857 | 0.2733 | 0.1000 | 0.097781 | 0.096405 | learned from the data |
| city | DESCRIPTIVE | 0.0026 | 0.0000 | 0.0026 | 0.8728 | 0.000930 | 0.008006 | learned from the data |
| state | DESCRIPTIVE | 0.0022 | 0.0000 | 0.0022 | 1.0000 | 0.000787 | 0.007761 | learned from the data |

Source: `results/eval_7919/verity_data/attribute_weights.json` (10,000 rows; reliability and the threshold use 1,000 random row pairs). discriminative score = uniqueness ratio x (1 - null rate); the normalized weights sum to 1; the composite weight merges the normalized weight with the cross-field reliability. The score used **normalized_discriminative_weight** as the field weights (checked on the first scored pair's `field_weights` in `scored_pairs.csv`).

Columns left out of learning (classified from the data, not named by hand):

| column | decision and reason (from the classifier) |
|---|---|
| record_id | excluded: [primary_key] primary key (uniqueness ratio 100.00% ≥ 99%) \| [system_process_field] system process field (name contains: id; only flagged because values are 100% unique) |
| source_system | excluded: [near_constant_field] near-constant field, 4 distinct values (<5) — system label / category regardless of dataset size |

Source: `results/eval_7919/verity_data/column_classification.json` (record id column: `record_id`).

### Thresholds, margin and blend weights

| item | value | learned or fixed | how |
|---|---|---|---|
| learned threshold | 0.455581 | learned from the data | KMeans (k=2) on 1,000 random row pairs: lower cluster (486 pairs) mean 0.3613 + 2 x sd 0.0471 |
| standard-deviation multiplier | 2 | fixed setting | part of the threshold rule |
| pairs sampled for the threshold and the weights | 1,000 | fixed setting | same for every run |
| effective threshold | 0.82 | fixed floor applied to the learned value | the larger of the learned threshold (0.455581) and the fixed floor 0.82; here the floor applies |
| review margin | 0.05 | fixed setting | review when \|score - effective threshold\| <= 0.05 (the band) |
| embedding blend weight (record cosine) | 0.65 | fixed setting | score = embedding blend x record cosine + field blend x weighted field similarities |
| field blend weight | 0.35 | fixed setting | same formula |
| within-field blend (NAME, STREET_ADDRESS, DESCRIPTIVE) | 0.65 cosine + 0.35 character Jaccard | fixed setting | per field, for the types listed in the formula |
| D routing, lo | 0.005 | fixed setting, chosen before the freeze | rule `selection_rule_v2.txt` applied to development seeds 11, 12, 13, 42, 2026; not learned on seed 7919 |
| D routing, hi | 0.9 | fixed setting, chosen before the freeze | unchanged by the v2 rule |

Sources: `results/eval_7919/verity_data/learned_threshold.json`, `results/eval_7919/verity_data/scored_pairs_columns.json` (effective threshold, review margin, score formula), `results/eval_7919/verity_data/learned_rules.md` (the same threshold and blend, in words), `results/eval_7919/options/d_hybrid/frozen_settings_v2.json` (D's lo and hi).

## 2. Setup time

From the input file to the learned configuration and the matches, one run, all threads (`results/eval_7919/verity_data/run_info.json`, timings; interpreter start-up excluded):

| step | seconds |
|---|---|
| start-up (imports, data folder, store check) | 20.6 |
| rule learning (column types, weights, threshold) | 13.2 |
| graph build | 1.6 |
| embedding | 47.3 |
| matching (neighbors, scoring, decisions, outputs) | 80.3 |
| merge | 2.1 |
| explorer files | 0.5 |
| **total (verity.total)** | **165.6** |

The steps listed add up to 165.58 s of the 165.61 s total (0.03 s untimed). The learned configuration (column types, weights, threshold) is ready after start-up and rule learning (33.8 s); the 216,092 scored pairs are written at the end of matching. 10,000 records, neighbor search `exact`.

**No column mapping or parameter was set by hand.** The command line of the run (`results/freeze_2026-11-25/logs/stage_02.log`) is `C:\verity-hackathon\.venv\Scripts\python.exe verity/src/main.py --nn-mode exact --input C:\verity-hackathon\results\eval_7919\orgs_raw.csv`: an input file and the neighbor-search mode, nothing else. The column types, the weights and the learned threshold above were derived from the data; the fixed settings in the table are the same for every run.

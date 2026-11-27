EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.

# Product-schema run, code path v2 (products_semantic, exploratory)

A Verity run on a fourth, unlabeled input file (product records with a description column) with a different schema, after the revised code path. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. Nothing about accuracy can be said: no labels exist for this file.

## a. Input facts

| item | value |
|---|---|
| rows (data rows, header excluded) | 508 |
| columns | 5: `product_id`, `manufacturer_part_number`, `upc_ean`, `description`, `source_system` |
| file sha256 | `05ca430f8086c4e051acf9672779d0f72e5ca48bd1ffdda257ffde48223ee39a` |
| file size | 32,677 bytes |

Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): 

`python verity/src/main.py --nn-mode exact --input "(local path removed)\products_semantic.csv"   [VERITY_DATA_DIR=results/product_v4_semantic/verity_data]`

The only arguments are the input path and `--nn-mode exact`: no column mapping and no parameter was given. The directory of the input path is replaced by "(local path removed)" in this report.

## b. Classification

| column | role | semantic type | classifier's reason |
|---|---|---|---|
| description | content | DESCRIPTIVE | included: 91.3% unique, 464 distinct; human-readable text signal |
| manufacturer_part_number | content | DESCRIPTIVE | included: 38.4% unique, 195 distinct; no metadata rule matched (entity-describing field) |
| upc_ean | content | DESCRIPTIVE | included: 42.3% unique, 215 distinct; no metadata rule matched (entity-describing field) |
| product_id | excluded |  | excluded: [primary_key] primary key (uniqueness ratio 100.00% ≥ 99%) \| [system_process_field] system process field (name contains: id; only flagged because values are 100% unique) |
| source_system | excluded |  | excluded: [near_constant_field] near-constant field, 3 distinct values (<5) — system label / category regardless of dataset size |

Record id column: `product_id`.

## c. Weights

| column | semantic type | uniqueness ratio | null rate | discriminative score | normalized weight |
|---|---|---|---|---|---|
| description | DESCRIPTIVE | 0.9134 | 0.0827 | 0.8379 | 0.529871 |
| manufacturer_part_number | DESCRIPTIVE | 0.3839 | 0.0118 | 0.3793 | 0.239886 |
| upc_ean | DESCRIPTIVE | 0.4232 | 0.1398 | 0.3641 | 0.230243 |

## d. Threshold

| item | value |
|---|---|
| learned threshold | 0.493871 |
| effective threshold | 0.82 |
| review margin | 0.05 |
| rule | effective threshold = the larger of the learned threshold and the fixed floor 0.82 |

## e. Counts

Scored pairs: **10,730** (`scored_pairs_rows` in run_info: 10,730); records in the index: 508; neighbor search `exact`.

| neighbor-search counter (run_info.json) | pairs |
|---|---|
| neighbor_pairs | 10,730 |
| skipped_missing_store_context | 0 |
| skipped_missing_embeddings | 0 |
| skipped_no_semantic_field | 0 |
| skipped_single_token_semantic_fields | 1,655 |
| scored_neighbor_pairs | 10,730 |
| exact_duplicate_pairs | 6 |
| exact_only_pairs | 0 |
| text_path_pairs | 9,075 |
| code_path_pairs | 1,655 |
| skipped_no_code_field | 0 |

| decision status | pairs |
|---|---|
| match | 377 |
| non_match | 6,919 |
| review | 3,434 |

| match type | pairs |
|---|---|
| below_threshold | 326 |
| code_conflict | 1,547 |
| code_signal | 104 |
| full_signal | 6 |
| insufficient_evidence | 8,323 |
| weak_signal | 424 |

| decision status | match type | pairs |
|---|---|---|
| match | code_signal | 104 |
| match | full_signal | 6 |
| match | weak_signal | 267 |
| non_match | below_threshold | 229 |
| non_match | code_conflict | 1,503 |
| non_match | insufficient_evidence | 5,187 |
| review | below_threshold | 97 |
| review | code_conflict | 44 |
| review | insufficient_evidence | 3,136 |
| review | weak_signal | 157 |

Merge groups: **246 golden records** (`golden_records.csv`). `match_summary.csv` has 251 distinct group ids: 91 multi-record groups and 160 unmatched single records. The two files give the same number of groups at the sizes 2, 3, 4, 5, 7 and differ at the sizes 1, 6 (not investigated).

| group size | multi-record groups (match_summary) | golden records (member_count) |
|---|---|---|
| 1 | 0 | 154 |
| 2 | 14 | 14 |
| 3 | 24 | 24 |
| 4 | 30 | 30 |
| 5 | 14 | 14 |
| 6 | 5 | 6 |
| 7 | 4 | 4 |

### Neighbor pairs scored, by path

Of 10,730 neighbor pairs, **10,730 were scored** and 0 were skipped before scoring (skipped missing store context: 0, skipped missing embeddings: 0, skipped no semantic field: 0, skipped no code field: 0). **Text path:** 9,075 pairs (embedding cosine blended with field similarity, the original scoring). **Code path:** 1,655 pairs that fail the multi-word test and were scored on code-like columns only (no embedding cosine); code-like columns chosen from the data: `manufacturer_part_number`. The exact-duplicate pairs (6) are scored 1.0 by rule. No labels exist, so nothing here says whether a pair is right.

## f. Timing (run_info.json)

| stage | parent | seconds | items |
|---|---|---|---|
| verity.startup |  | 7.815 |  |
| verity.rule_learning |  | 3.039 |  |
| verity.build_graph |  | 0.042 |  |
| verity.embed.model_load | verity.embed | 0.504 |  |
| verity.embed.encode | verity.embed | 0.759 | 757 |
| verity.embed |  | 1.315 |  |
| verity.embed.other | verity.embed | 0.052 |  |
| verity.neighbors.build | verity.match | 0.019 | 508 |
| verity.neighbors.query | verity.match | 0.081 | 508 |
| verity.score | verity.match | 0.846 | 10730 |
| verity.write_outputs | verity.match | 0.237 | 11264 |
| verity.match |  | 1.329 |  |
| verity.match.other | verity.match | 0.146 |  |
| verity.merge |  | 0.089 |  |
| verity.explore |  | 0.018 |  |
| verity.total |  | 13.648 |  |

Total (`verity.total`): 13.6 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## e2. Code conflicts

`code_conflict` pairs: **1,547** (a code-like field present in both records differs after normalization, so the pair cannot be a match). Decision status: review when the pair would otherwise have passed (score at or above the effective threshold and evidence mass at or above the minimum) or when its score is inside the review band; otherwise non_match. For these rows `evidence_passes` is true exactly when the pair would otherwise have passed.

| decision status | code_conflict pairs |
|---|---|
| non_match | 1,503 |
| review | 44 |

## g1. Explanation card: the first code_signal match, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

| field | similarity | weight | evidence similarity |
|---|---|---|---|
| manufacturer_part_number | 1.0000 | 0.5103 | 1.0000 |
| upc_ean | 1.0000 | 0.4897 | 1.0000 |

| item | value |
|---|---|
| score | 1.000000 |
| match_type | code_signal |
| decision_status | match |
| evidence_mass | 1.000000 |
| conflict_mass | 0.000000 |
| evidence_passes | true |

## g2. Explanation card: the first code_conflict pair, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

| field | similarity | weight | evidence similarity |
|---|---|---|---|
| manufacturer_part_number | 0.2500 | 0.5103 | 0.2500 |
| upc_ean | 0.3333 | 0.4897 | 0.3333 |

| item | value |
|---|---|
| score | 0.290812 |
| match_type | code_conflict |
| decision_status | non_match |
| evidence_mass | 0.000000 |
| conflict_mass | 0.510256 |
| evidence_passes | false |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

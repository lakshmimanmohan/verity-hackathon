EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.

# Product-schema run, code path v3 (products_semantic, exploratory)

A Verity run on a fourth, unlabeled input file (product records with a description column) with a different schema, after the second revision of the code path. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. Nothing about accuracy can be said: no labels exist for this file.

## a. Input facts

| item | value |
|---|---|
| rows (data rows, header excluded) | 508 |
| columns | 5: `product_id`, `manufacturer_part_number`, `upc_ean`, `description`, `source_system` |
| file sha256 | `05ca430f8086c4e051acf9672779d0f72e5ca48bd1ffdda257ffde48223ee39a` |
| file size | 32,677 bytes |

Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): 

`python verity/src/main.py --nn-mode exact --input "(local path removed)\products_semantic.csv"   [VERITY_DATA_DIR=results/product_v5_semantic/verity_data]`

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
| below_threshold | 1,827 |
| code_conflict | 44 |
| code_signal | 104 |
| full_signal | 6 |
| insufficient_evidence | 8,325 |
| weak_signal | 424 |

| decision status | match type | pairs |
|---|---|---|
| match | code_signal | 104 |
| match | full_signal | 6 |
| match | weak_signal | 267 |
| non_match | below_threshold | 1,730 |
| non_match | insufficient_evidence | 5,189 |
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
| verity.startup |  | 7.748 |  |
| verity.rule_learning |  | 3.183 |  |
| verity.build_graph |  | 0.046 |  |
| verity.embed.model_load | verity.embed | 0.478 |  |
| verity.embed.encode | verity.embed | 0.743 | 757 |
| verity.embed |  | 1.270 |  |
| verity.embed.other | verity.embed | 0.049 |  |
| verity.neighbors.build | verity.match | 0.020 | 508 |
| verity.neighbors.query | verity.match | 0.109 | 508 |
| verity.score | verity.match | 0.807 | 10730 |
| verity.write_outputs | verity.match | 0.252 | 11264 |
| verity.match |  | 1.348 |  |
| verity.match.other | verity.match | 0.160 |  |
| verity.merge |  | 0.101 |  |
| verity.explore |  | 0.020 |  |
| verity.total |  | 13.719 |  |

Total (`verity.total`): 13.7 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## e2. Code conflicts

`code_conflict` pairs: **44** (the conflict rule: an identifying field present in both records differs after normalization, and the conflict blocks a pair that would otherwise have matched or fallen in the review band). Decision status: review (the pair is in the band, or would otherwise have passed: score strictly above the effective threshold and evidence mass at or above the minimum). For these rows `evidence_passes` is true exactly when the pair would otherwise have passed. A pair with a conflict that would not have passed keeps its ordinary label.

| decision status | code_conflict pairs |
|---|---|
| review | 44 |

## e3. v4 against v5

v5 limits the conflict rule to identifying fields: code-like columns whose learned normalized weight is at least 1/(2n), n being the number of content columns used for scoring (3 here, cut 0.166667). A conflict is labeled code_conflict only when it blocks a pair that would otherwise have matched or fallen in the review band. Identifying fields in this run: `manufacturer_part_number`. Code-like columns: `manufacturer_part_number`.

| count | v4 | v5 |
|---|---|---|
| code_signal matches | 104 | 104 |
| code_conflict pairs, total | 1,547 | 44 |
| code_conflict pairs, match | 0 | 0 |
| code_conflict pairs, review | 44 | 44 |
| code_conflict pairs, non_match | 1,503 | 0 |
| total matches (decision_status match) | 377 | 377 |

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
| manufacturer_part_number | 0.9333 | 0.5103 | 0.9333 |
| upc_ean | 1.0000 | 0.4897 | 1.0000 |

| item | value |
|---|---|
| score | 0.965983 |
| match_type | code_conflict |
| decision_status | review |
| evidence_mass | 0.489744 |
| conflict_mass | 0.510256 |
| evidence_passes | true |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

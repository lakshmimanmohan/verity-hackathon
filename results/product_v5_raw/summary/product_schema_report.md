EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.

# Product-schema run, code path v3 (products_raw, exploratory)

A Verity run on a third, unlabeled input file (product records) with a different schema, after the second revision of the code path. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. Nothing about accuracy can be said: no labels exist for this file.

## a. Input facts

| item | value |
|---|---|
| rows (data rows, header excluded) | 500 |
| columns | 7: `product_id`, `manufacturer_part_number`, `member_part_number`, `upc_ean`, `manufacturer_code`, `brand_code`, `source_system` |
| file sha256 | `cf602121b06e079bbb98f0df59b997313d35957bd6905c17ebd18e9cf7fcbd44` |
| file size | 33,057 bytes |

Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): 

`python verity/src/main.py --nn-mode exact --input "(local path removed)\products_raw.csv"   [VERITY_DATA_DIR=results/product_v5_raw/verity_data]`

The only arguments are the input path and `--nn-mode exact`: no column mapping and no parameter was given. The directory of the input path is replaced by "(local path removed)" in this report.

## b. Classification

| column | role | semantic type | classifier's reason |
|---|---|---|---|
| brand_code | content | DESCRIPTIVE | included: 3.0% unique, 15 distinct; no metadata rule matched (entity-describing field) |
| manufacturer_code | content | DESCRIPTIVE | included: 4.0% unique, 20 distinct; no metadata rule matched (entity-describing field) |
| manufacturer_part_number | content | DESCRIPTIVE | included: 52.6% unique, 263 distinct; no metadata rule matched (entity-describing field) |
| member_part_number | content | IDENTIFIER | included: 92.4% unique, 462 distinct; no metadata rule matched (entity-describing field) |
| upc_ean | content | QUASI_IDENTIFIER | included: 67.8% unique, 339 distinct; no metadata rule matched (entity-describing field) |
| product_id | excluded |  | excluded: [primary_key] primary key (uniqueness ratio 100.00% ≥ 99%) \| [system_process_field] system process field (name contains: id; only flagged because values are 100% unique) |
| source_system | excluded |  | excluded: [near_constant_field] near-constant field, 3 distinct values (<5) — system label / category regardless of dataset size |

Record id column: `product_id`.

## c. Weights

| column | semantic type | uniqueness ratio | null rate | discriminative score | normalized weight |
|---|---|---|---|---|---|
| member_part_number | IDENTIFIER | 0.9240 | 0.0000 | 0.9240 | 0.437101 |
| upc_ean | QUASI_IDENTIFIER | 0.6780 | 0.1240 | 0.5939 | 0.280959 |
| manufacturer_part_number | DESCRIPTIVE | 0.5260 | 0.0000 | 0.5260 | 0.248826 |
| manufacturer_code | DESCRIPTIVE | 0.0400 | 0.0000 | 0.0400 | 0.018922 |
| brand_code | DESCRIPTIVE | 0.0300 | 0.0000 | 0.0300 | 0.014192 |

## d. Threshold

| item | value |
|---|---|
| learned threshold | 0.204313 |
| effective threshold | 0.82 |
| review margin | 0.05 |
| rule | effective threshold = the larger of the learned threshold and the fixed floor 0.82 |

## e. Counts

Scored pairs: **11,602** (`scored_pairs_rows` in run_info: 11,602); records in the index: 500; neighbor search `exact`.

| neighbor-search counter (run_info.json) | pairs |
|---|---|
| neighbor_pairs | 11,584 |
| skipped_missing_store_context | 0 |
| skipped_missing_embeddings | 0 |
| skipped_no_semantic_field | 0 |
| skipped_single_token_semantic_fields | 11,584 |
| scored_neighbor_pairs | 11,584 |
| exact_duplicate_pairs | 321 |
| exact_only_pairs | 18 |
| text_path_pairs | 0 |
| code_path_pairs | 11,584 |
| skipped_no_code_field | 0 |

| decision status | pairs |
|---|---|
| match | 321 |
| non_match | 11,270 |
| review | 11 |

| match type | pairs |
|---|---|
| below_threshold | 11,228 |
| code_conflict | 11 |
| full_signal | 321 |
| insufficient_evidence | 42 |

| decision status | match type | pairs |
|---|---|---|
| match | full_signal | 321 |
| non_match | below_threshold | 11,228 |
| non_match | insufficient_evidence | 42 |
| review | code_conflict | 11 |

Merge groups: **251 golden records** (`golden_records.csv`). `match_summary.csv` has 251 distinct group ids: 178 multi-record groups and 73 unmatched single records. The two files give the same number of groups at the sizes 2, 3, 4 and differ at the sizes 1 (not investigated).

| group size | multi-record groups (match_summary) | golden records (member_count) |
|---|---|---|
| 1 | 0 | 73 |
| 2 | 108 | 108 |
| 3 | 69 | 69 |
| 4 | 1 | 1 |

### Neighbor pairs scored, by path

Of 11,584 neighbor pairs, **11,584 were scored** and 0 were skipped before scoring (skipped missing store context: 0, skipped missing embeddings: 0, skipped no semantic field: 0, skipped no code field: 0). **Text path:** 0 pairs (embedding cosine blended with field similarity, the original scoring). **Code path:** 11,584 pairs that fail the multi-word test and were scored on code-like columns only (no embedding cosine); code-like columns chosen from the data: `brand_code`, `manufacturer_code`, `manufacturer_part_number`, `member_part_number`, `upc_ean`. The exact-duplicate pairs (321) are scored 1.0 by rule. No labels exist, so nothing here says whether a pair is right.

## f. Timing (run_info.json)

| stage | parent | seconds | items |
|---|---|---|---|
| verity.startup |  | 7.909 |  |
| verity.rule_learning |  | 3.553 |  |
| verity.build_graph |  | 0.057 |  |
| verity.embed.model_load | verity.embed | 0.531 |  |
| verity.embed.encode | verity.embed | 1.369 | 1098 |
| verity.embed |  | 1.954 |  |
| verity.embed.other | verity.embed | 0.054 |  |
| verity.neighbors.build | verity.match | 0.016 | 500 |
| verity.neighbors.query | verity.match | 0.142 | 500 |
| verity.score | verity.match | 0.829 | 11584 |
| verity.write_outputs | verity.match | 0.310 | 11923 |
| verity.match |  | 1.453 |  |
| verity.match.other | verity.match | 0.156 |  |
| verity.merge |  | 0.094 |  |
| verity.explore |  | 0.013 |  |
| verity.total |  | 15.036 |  |

Total (`verity.total`): 15.0 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## e2. Code conflicts

`code_conflict` pairs: **11** (the conflict rule: an identifying field present in both records differs after normalization, and the conflict blocks a pair that would otherwise have matched or fallen in the review band). Decision status: review (the pair is in the band, or would otherwise have passed: score strictly above the effective threshold and evidence mass at or above the minimum). For these rows `evidence_passes` is true exactly when the pair would otherwise have passed. A pair with a conflict that would not have passed keeps its ordinary label.

| decision status | code_conflict pairs |
|---|---|
| review | 11 |

## e3. v4 against v5

v5 limits the conflict rule to identifying fields: code-like columns whose learned normalized weight is at least 1/(2n), n being the number of content columns used for scoring (5 here, cut 0.100000). A conflict is labeled code_conflict only when it blocks a pair that would otherwise have matched or fallen in the review band. Identifying fields in this run: `manufacturer_part_number`, `member_part_number`, `upc_ean`. Code-like columns: `brand_code`, `manufacturer_code`, `manufacturer_part_number`, `member_part_number`, `upc_ean`.

| count | v4 | v5 |
|---|---|---|
| code_signal matches | 0 | 0 |
| code_conflict pairs, total | 11,281 | 11 |
| code_conflict pairs, match | 0 | 0 |
| code_conflict pairs, review | 11 | 11 |
| code_conflict pairs, non_match | 11,270 | 0 |
| total matches (decision_status match) | 321 | 321 |

## g1. Explanation card: the first code_signal match, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

There is no such pair in this run (no code_signal match), so there is no card to show. Exact duplicates are scored 1.0 by rule and carry no per-field data (section e).

| item | value |
|---|---|
| result | no code_signal match |

## g2. Explanation card: the first code_conflict pair, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

| field | similarity | weight | evidence similarity |
|---|---|---|---|
| brand_code | 1.0000 | 0.0142 | 1.0000 |
| manufacturer_code | 1.0000 | 0.0189 | 1.0000 |
| manufacturer_part_number | 0.9474 | 0.2488 | 0.9474 |
| member_part_number | 1.0000 | 0.4371 | 1.0000 |
| upc_ean | 0.8889 | 0.2810 | 0.8889 |

| item | value |
|---|---|
| score | 0.955686 |
| match_type | code_conflict |
| decision_status | review |
| evidence_mass | 0.470215 |
| conflict_mass | 0.529785 |
| evidence_passes | true |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

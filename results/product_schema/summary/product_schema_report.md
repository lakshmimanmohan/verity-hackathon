EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.

# Product-schema run (exploratory)

A Verity run on a third, unlabeled input file (product records) with a different schema. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. Nothing about accuracy can be said: no labels exist for this file.

## a. Input facts

| item | value |
|---|---|
| rows (data rows, header excluded) | 500 |
| columns | 7: `product_id`, `manufacturer_part_number`, `member_part_number`, `upc_ean`, `manufacturer_code`, `brand_code`, `source_system` |
| file sha256 | `cf602121b06e079bbb98f0df59b997313d35957bd6905c17ebd18e9cf7fcbd44` |
| file size | 33,057 bytes |

Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): 

`python verity/src/main.py --nn-mode exact --input "(local path removed)\products_raw.csv"   [VERITY_DATA_DIR=results/product_schema/verity_data]`

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

Scored pairs: **321** (`scored_pairs_rows` in run_info: 321); records in the index: 500; neighbor search `exact`.

| neighbor-search counter (run_info.json) | pairs |
|---|---|
| neighbor_pairs | 11,584 |
| skipped_missing_store_context | 0 |
| skipped_missing_embeddings | 0 |
| skipped_no_semantic_field | 0 |
| skipped_single_token_semantic_fields | 11,584 |
| scored_neighbor_pairs | 0 |
| exact_duplicate_pairs | 321 |
| exact_only_pairs | 321 |

| decision status | pairs |
|---|---|
| match | 321 |

| match type | pairs |
|---|---|
| full_signal | 321 |

| decision status | match type | pairs |
|---|---|---|
| match | full_signal | 321 |

Merge groups: **251 golden records** (`golden_records.csv`). `match_summary.csv` has 251 distinct group ids: 178 multi-record groups and 73 unmatched single records. The two files give the same number of groups at the sizes 2, 3, 4 and differ at the sizes 1 (not investigated).

| group size | multi-record groups (match_summary) | golden records (member_count) |
|---|---|---|
| 1 | 0 | 73 |
| 2 | 108 | 108 |
| 3 | 69 | 69 |
| 4 | 1 | 1 |

### Finding: no neighbor pair was scored

Of 11,584 neighbor pairs, 11,584 were skipped before scoring by the single-token guard (`verity/src/match.py` lines 2036-2040: `has_a = any(" " in str(aa.get(k, "")) for k in sem_fields) ...`: both records need a space in at least one NAME, DESCRIPTIVE or STREET_ADDRESS value), 0 by the no-semantic-field guard, and 0 were scored. The only pairs in the output are the exact duplicates (321), which Verity scores 1.0 by rule. So this run did no similarity scoring: there is no review band, no non-match row and no per-field evidence.

Key lists (names only; the two forms agree, so the column-key mismatch of the second-schema run is not the cause here):

| list | keys |
|---|---|
| semantic-type keys as match.py builds them (`col_key` form) | `brand_code`, `manufacturer_code`, `manufacturer_part_number`, `member_part_number`, `upc_ean` |
| of those, the keys the guard looks up (NAME, DESCRIPTIVE, STREET_ADDRESS) | `brand_code`, `manufacturer_code`, `manufacturer_part_number` |
| attribute types in the record store (lower-cased) | `brand_code`, `manufacturer_code`, `manufacturer_part_number`, `member_part_number`, `upc_ean` |

Counts: 500 records in the store; 500 have a value in at least one guard field; 0 have a space in at least one such value. Likely cause (a reading of the counters, not tested by a fix): the guard fields hold no multi-word values, so every pair fails the space test. The identifier and quasi-identifier columns are not guard fields. No change was made to Verity.

## f. Timing (run_info.json)

| stage | parent | seconds | items |
|---|---|---|---|
| verity.startup |  | 7.586 |  |
| verity.rule_learning |  | 3.876 |  |
| verity.build_graph |  | 0.088 |  |
| verity.embed.model_load | verity.embed | 0.711 |  |
| verity.embed.encode | verity.embed | 1.503 | 1098 |
| verity.embed |  | 2.283 |  |
| verity.embed.other | verity.embed | 0.068 |  |
| verity.neighbors.build | verity.match | 0.019 | 500 |
| verity.neighbors.query | verity.match | 0.021 | 500 |
| verity.score | verity.match | 0.147 | 0 |
| verity.write_outputs | verity.match | 0.009 | 642 |
| verity.match |  | 0.316 |  |
| verity.match.other | verity.match | 0.120 |  |
| verity.merge |  | 0.170 |  |
| verity.explore |  | 0.029 |  |
| verity.total |  | 14.351 |  |

Total (`verity.total`): 14.4 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## g. Explanation card: the first matched pair that is not an exact duplicate, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

No matched pair that is not an exact duplicate exists in this run, so there is no card to show. Exact duplicates are scored 1.0 by rule and carry no per-field data (section e).

| item | value |
|---|---|
| result | no matched pair that is not an exact duplicate |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

## h. Explorer and group-size checks (counts only; nothing was fixed)

This section was appended by hand to the generated report.

| check | result |
|---|---|
| matched-pair cards in `match_explorer.html` | 200 (each has a "Why it matched" block) |
| cards whose "Why it matched" table has at least one field row | **0 of 200** (every table body is empty: the matched pairs are exact duplicates, which carry no per-field data) |
| groups of each size, `golden_records.csv` (`member_count`) vs `match_summary.csv` (`group_size`, one count per `match_id`) | **agree at every size**: size 1: 73 and 73; size 2: 108 and 108; size 3: 69 and 69; size 4: 1 and 1 |

The groups agree in this run, so there is no disagreement to trace here. For information, the earlier second-schema run (`results/second_schema_v2`) does disagree at 35 sizes (for example size 2: 3,965 golden records against 4,053 groups in `match_summary.csv`, size 1: 857 against 2,054). The two files come from two separate code paths: `golden_records.csv` is built in `verity/src/merge.py` (`main`, starting line 930; `member_count` set at lines 1150, 1165 and 1173; large communities split by `split_oversized_communities`, line 402, called at line 1067), and `match_summary.csv` is built in `verity/src/explore_matches.py` (`run_match_summary`, line 748; its own `recluster_component`, line 1145; `group_size` set at line 1285, with `U` ids for single records). Neither file uses `col_key` (0 occurrences in `merge.py` and in `explore_matches.py`). Whether the two splitting rules explain the disagreement was not investigated.

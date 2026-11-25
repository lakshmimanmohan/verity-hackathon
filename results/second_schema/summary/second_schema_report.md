EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.

# Second-schema run (exploratory)

A Verity run on a second, unlabeled input file with a different schema. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. Nothing about accuracy can be said: no labels exist for this file.

## a. Input facts

| item | value |
|---|---|
| rows (data rows, header excluded) | 14,570 |
| columns | 7: `Id`, `Alias Account`, `Street Name`, `House Number1`, `Box1 Number`, `Postal Code`, `City Name` |
| file sha256 | `f3a06f55cbfdbb5a6c40d98efe55be184f33b3fef2c1c5073654bdc8434b0d23` |
| file size | 1,432,564 bytes |

Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): 

`python verity/src/main.py --nn-mode exact --input "(local path removed)\Trillium_Sample_Std_Data_15k.csv"   [VERITY_DATA_DIR=results/second_schema/verity_data]`

The only arguments are the input path and `--nn-mode exact`: no column mapping and no parameter was given. The directory of the input path is replaced by "(local path removed)" in this report.

## b. Classification

| column | role | semantic type | classifier's reason |
|---|---|---|---|
| Alias Account | content | NAME | included: 62.6% unique, 9115 distinct; human-readable text signal |
| Box1 Number | content | DESCRIPTIVE | included: 3.3% unique, 480 distinct; no metadata rule matched (entity-describing field) |
| City Name | content | DESCRIPTIVE | included: 21.2% unique, 3091 distinct; human-readable text signal |
| House Number1 | content | DESCRIPTIVE | included: 20.2% unique, 2947 distinct; no metadata rule matched (entity-describing field) |
| Postal Code | content | DESCRIPTIVE | included: 44.7% unique, 6506 distinct; no metadata rule matched (entity-describing field) |
| Street Name | content | DESCRIPTIVE | included: 24.0% unique, 3498 distinct; human-readable text signal |
| Id | excluded |  | excluded: [primary_key] primary key (uniqueness ratio 100.00% ≥ 99%) \| [system_process_field] system process field (name contains: id; only flagged because values are 100% unique) |

Record id column: `Id`.

## c. Weights

| column | semantic type | uniqueness ratio | null rate | discriminative score | normalized weight |
|---|---|---|---|---|---|
| Alias Account | NAME | 0.6256 | 0.0000 | 0.6256 | 0.362878 |
| Postal Code | DESCRIPTIVE | 0.4465 | 0.0004 | 0.4464 | 0.258904 |
| Street Name | DESCRIPTIVE | 0.2401 | 0.0036 | 0.2392 | 0.138752 |
| City Name | DESCRIPTIVE | 0.2121 | 0.0000 | 0.2121 | 0.123056 |
| House Number1 | DESCRIPTIVE | 0.2023 | 0.0218 | 0.1978 | 0.114763 |
| Box1 Number | DESCRIPTIVE | 0.0329 | 0.9138 | 0.0028 | 0.001647 |

Against the normalized weights of an earlier run of the same file (values given in the request):

| column | this run | earlier run | difference | differs |
|---|---|---|---|---|
| Alias Account | 0.362878 | 0.362878 | +0.000000 | no (within 6-decimal rounding) |
| Street Name | 0.138752 | 0.138752 | +0.000000 | no (within 6-decimal rounding) |
| House Number1 | 0.114763 | 0.114763 | +0.000000 | no (within 6-decimal rounding) |
| Box1 Number | 0.001647 | 0.001647 | +0.000000 | no (within 6-decimal rounding) |
| Postal Code | 0.258904 | 0.258904 | +0.000000 | no (within 6-decimal rounding) |
| City Name | 0.123056 | 0.123056 | +0.000000 | no (within 6-decimal rounding) |

## d. Threshold

| item | value |
|---|---|
| learned threshold | 0.414924 |
| effective threshold | 0.82 |
| review margin | 0.05 |
| rule | effective threshold = the larger of the learned threshold and the fixed floor 0.82 |

## e. Counts

Scored pairs: **6,387** (`scored_pairs_rows` in run_info: 6,387); records in the index: 14,570; neighbor search `exact`.

| neighbor-search counter (run_info.json) | pairs |
|---|---|
| neighbor_pairs | 363,550 |
| skipped_missing_store_context | 0 |
| skipped_missing_embeddings | 0 |
| skipped_no_semantic_field | 363,550 |
| skipped_single_token_semantic_fields | 0 |
| scored_neighbor_pairs | 0 |
| exact_duplicate_pairs | 6,387 |
| exact_only_pairs | 6,387 |

| decision status | pairs |
|---|---|
| match | 6,387 |

| match type | pairs |
|---|---|
| full_signal | 6,387 |

| decision status | match type | pairs |
|---|---|---|
| match | full_signal | 6,387 |

Merge groups: **10,282 golden records** (`golden_records.csv`). `match_summary.csv` has 10,279 distinct group ids: 3,590 multi-record groups and 6,689 unmatched single records. The two files give the same number of groups at the sizes 2, 3, 4, 5, 6, 7, 8, 9, 10 and differ at the sizes 1, 14, 15, 16, 22, 29, 37 (not investigated).

| group size | multi-record groups (match_summary) | golden records (member_count) |
|---|---|---|
| 1 | 0 | 6,690 |
| 2 | 3,138 | 3,138 |
| 3 | 348 | 348 |
| 4 | 68 | 68 |
| 5 | 17 | 17 |
| 6 | 4 | 4 |
| 7 | 7 | 7 |
| 8 | 2 | 2 |
| 9 | 1 | 1 |
| 10 | 1 | 1 |
| 14 | 1 | 2 |
| 15 | 0 | 3 |
| 16 | 1 | 0 |
| 22 | 0 | 1 |
| 29 | 1 | 0 |
| 37 | 1 | 0 |

### Finding: no neighbor pair was scored

Of 363,550 neighbor pairs, 363,550 were skipped before scoring by the guard "a record has no NAME/DESCRIPTIVE/STREET_ADDRESS value" (`verity/src/match.py` lines 2028-2035: `if not any(k in aa for k in sem_fields) ...`), and 0 were scored. The only pairs in the output are the exact duplicates (6,387), which Verity scores 1.0 by rule. So this run did no similarity scoring: there is no review band, no non-match row and no per-field evidence.

Likely cause (not tested by a fix, and a hypothesis from names only): the guard looks up the semantic-type keys, which are the lower-cased input column names with their spaces (for example `alias account`), in each record's attributes, and the record store keeps the attributes under snake_case keys (for example `alias_account`; the store's attribute types are upper-case snake_case). With names that contain a space the two never match, so every pair looks as if it had no such value. The first dataset's column names have no spaces and are unaffected. No change was made to Verity.

## f. Timing (run_info.json)

| stage | parent | seconds | items |
|---|---|---|---|
| verity.startup |  | 18.793 |  |
| verity.rule_learning |  | 12.816 |  |
| verity.build_graph |  | 1.868 |  |
| verity.embed.model_load | verity.embed | 0.778 |  |
| verity.embed.encode | verity.embed | 51.725 | 41023 |
| verity.embed |  | 53.952 |  |
| verity.embed.other | verity.embed | 1.449 |  |
| verity.neighbors.build | verity.match | 0.020 | 14570 |
| verity.neighbors.query | verity.match | 4.094 | 14570 |
| verity.score | verity.match | 2.843 | 0 |
| verity.write_outputs | verity.match | 0.087 | 12774 |
| verity.match |  | 8.189 |  |
| verity.match.other | verity.match | 1.144 |  |
| verity.merge |  | 1.780 |  |
| verity.explore |  | 0.160 |  |
| verity.total |  | 97.583 |  |

Total (`verity.total`): 97.6 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## g. Explanation card: the first matched pair by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

The first matched pair by sorted record ids is an exact-duplicate pair (`pair_source` exact). Exact duplicates are scored 1.0 by rule and carry no per-field similarity, weight or evidence, so there is nothing to list per field. Every one of the scored pairs in this run is of this kind (section e).

| item | value |
|---|---|
| score | 1.000000 |
| match_type | full_signal |
| decision_status | match |
| pair_source | exact |
| per-field similarities, weights and evidence | not stored: exact-duplicate pairs are scored 1.0 by rule |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

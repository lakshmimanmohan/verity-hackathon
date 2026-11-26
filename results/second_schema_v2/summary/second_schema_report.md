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

`python verity/src/main.py --nn-mode exact --input "(local path removed)\Trillium_Sample_Std_Data_15k.csv"   [VERITY_DATA_DIR=results/second_schema_v2/verity_data]`

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

Scored pairs: **323,409** (`scored_pairs_rows` in run_info: 323,409); records in the index: 14,570; neighbor search `exact`.

| neighbor-search counter (run_info.json) | pairs |
|---|---|
| neighbor_pairs | 363,550 |
| skipped_missing_store_context | 0 |
| skipped_missing_embeddings | 0 |
| skipped_no_semantic_field | 0 |
| skipped_single_token_semantic_fields | 40,581 |
| scored_neighbor_pairs | 322,969 |
| exact_duplicate_pairs | 6,387 |
| exact_only_pairs | 440 |

| decision status | pairs |
|---|---|
| match | 16,217 |
| non_match | 15,281 |
| review | 291,911 |

| match type | pairs |
|---|---|
| below_threshold | 7,458 |
| full_signal | 6,387 |
| insufficient_evidence | 293,415 |
| weak_signal | 16,149 |

| decision status | match type | pairs |
|---|---|---|
| match | full_signal | 6,387 |
| match | weak_signal | 9,830 |
| non_match | below_threshold | 160 |
| non_match | insufficient_evidence | 15,121 |
| review | below_threshold | 7,298 |
| review | insufficient_evidence | 278,294 |
| review | weak_signal | 6,319 |

Merge groups: **6,034 golden records** (`golden_records.csv`). `match_summary.csv` has 7,216 distinct group ids: 5,162 multi-record groups and 2,054 unmatched single records. The two files give the same number of groups at the sizes (none) and differ at the sizes 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 25, 27, 29, 31, 33, 34, 42, 43, 44, 48, 55, 58, 79, 96, 100 (not investigated).

| group size | multi-record groups (match_summary) | golden records (member_count) |
|---|---|---|
| 1 | 0 | 857 |
| 2 | 4,053 | 3,965 |
| 3 | 639 | 620 |
| 4 | 267 | 290 |
| 5 | 99 | 105 |
| 6 | 44 | 59 |
| 7 | 25 | 35 |
| 8 | 8 | 17 |
| 9 | 10 | 14 |
| 10 | 3 | 9 |
| 11 | 2 | 11 |
| 12 | 2 | 6 |
| 13 | 2 | 7 |
| 14 | 1 | 4 |
| 15 | 1 | 21 |
| 16 | 0 | 1 |
| 18 | 0 | 1 |
| 19 | 0 | 1 |
| 20 | 1 | 0 |
| 21 | 0 | 1 |
| 25 | 0 | 1 |
| 27 | 1 | 0 |
| 29 | 1 | 0 |
| 31 | 1 | 0 |
| 33 | 0 | 1 |
| 34 | 0 | 1 |
| 42 | 0 | 1 |
| 43 | 0 | 1 |
| 44 | 1 | 0 |
| 48 | 0 | 1 |
| 55 | 1 | 0 |
| 58 | 0 | 1 |
| 79 | 0 | 1 |
| 96 | 0 | 1 |
| 100 | 0 | 1 |

### Neighbor pairs scored

Of 363,550 neighbor pairs, **322,969 were scored** and 40,581 were skipped before scoring (skipped missing store context: 0, skipped missing embeddings: 0, skipped no semantic field: 0, skipped single token semantic fields: 40,581). The exact-duplicate pairs (6,387) are scored 1.0 by rule. The first run of this file (`results/second_schema/summary`) scored none of the neighbor pairs, because column names with a space did not match the record store's snake_case keys; this run follows the key fix (`col_key`). No labels exist, so nothing here says whether a pair is right.

## f. Timing (run_info.json)

| stage | parent | seconds | items |
|---|---|---|---|
| verity.startup |  | 8.510 |  |
| verity.rule_learning |  | 10.486 |  |
| verity.build_graph |  | 1.310 |  |
| verity.embed.model_load | verity.embed | 0.497 |  |
| verity.embed.encode | verity.embed | 49.061 | 41023 |
| verity.embed |  | 50.919 |  |
| verity.embed.other | verity.embed | 1.362 |  |
| verity.neighbors.build | verity.match | 0.021 | 14570 |
| verity.neighbors.query | verity.match | 3.804 | 14570 |
| verity.score | verity.match | 35.662 | 322969 |
| verity.write_outputs | verity.match | 10.174 | 345945 |
| verity.match |  | 52.278 |  |
| verity.match.other | verity.match | 2.618 |  |
| verity.merge |  | 2.602 |  |
| verity.explore |  | 0.432 |  |
| verity.total |  | 126.562 |  |

Total (`verity.total`): 126.6 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## g. Explanation card: the first matched pair that is not an exact duplicate, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

| field | similarity | weight | evidence similarity |
|---|---|---|---|
| alias_account | 0.8000 | 0.3635 | 0.8333 |
| city_name | 1.0000 | 0.1233 | 1.0000 |
| house_number1 | 1.0000 | 0.1150 | 1.0000 |
| postal_code | 1.0000 | 0.2593 | 1.0000 |
| street_name | 1.0000 | 0.1390 | 1.0000 |

| item | value |
|---|---|
| score | 0.985108 |
| match_type | weak_signal |
| decision_status | match |
| evidence_mass | 0.635475 |
| conflict_mass | 0.000000 |
| evidence_passes | true |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

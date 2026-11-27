EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.

# Second-schema run, code path v3 (address file, exploratory)

A Verity run on a second, unlabeled input file with a different schema, after the second revision of the code path. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. Nothing about accuracy can be said: no labels exist for this file.

## a. Input facts

| item | value |
|---|---|
| rows (data rows, header excluded) | 14,570 |
| columns | 7: `Id`, `Alias Account`, `Street Name`, `House Number1`, `Box1 Number`, `Postal Code`, `City Name` |
| file sha256 | `f3a06f55cbfdbb5a6c40d98efe55be184f33b3fef2c1c5073654bdc8434b0d23` |
| file size | 1,432,564 bytes |

Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): 

`python verity/src/main.py --nn-mode exact --input "(local path removed)\Trillium_Sample_Std_Data_15k.csv"   [VERITY_DATA_DIR=results/second_schema_v5/verity_data]`

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

## d. Threshold

| item | value |
|---|---|
| learned threshold | 0.414924 |
| effective threshold | 0.82 |
| review margin | 0.05 |
| rule | effective threshold = the larger of the learned threshold and the fixed floor 0.82 |

## e. Counts

Scored pairs: **363,583** (`scored_pairs_rows` in run_info: 363,583); records in the index: 14,570; neighbor search `exact`.

| neighbor-search counter (run_info.json) | pairs |
|---|---|
| neighbor_pairs | 363,550 |
| skipped_missing_store_context | 0 |
| skipped_missing_embeddings | 0 |
| skipped_no_semantic_field | 0 |
| skipped_single_token_semantic_fields | 40,581 |
| scored_neighbor_pairs | 363,547 |
| exact_duplicate_pairs | 6,387 |
| exact_only_pairs | 36 |
| text_path_pairs | 322,969 |
| code_path_pairs | 40,578 |
| skipped_no_code_field | 3 |

| decision status | pairs |
|---|---|
| match | 16,362 |
| non_match | 54,779 |
| review | 292,442 |

| match type | pairs |
|---|---|
| below_threshold | 47,198 |
| code_conflict | 104 |
| code_signal | 329 |
| full_signal | 6,387 |
| insufficient_evidence | 293,416 |
| weak_signal | 16,149 |

| decision status | match type | pairs |
|---|---|---|
| match | code_signal | 145 |
| match | full_signal | 6,387 |
| match | weak_signal | 9,830 |
| non_match | below_threshold | 39,657 |
| non_match | insufficient_evidence | 15,122 |
| review | below_threshold | 7,541 |
| review | code_conflict | 104 |
| review | code_signal | 184 |
| review | insufficient_evidence | 278,294 |
| review | weak_signal | 6,319 |

Merge groups: **5,825 golden records** (`golden_records.csv`). `match_summary.csv` has 7,032 distinct group ids: 5,274 multi-record groups and 1,758 unmatched single records. The two files give the same number of groups at the sizes (none) and differ at the sizes 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 25, 27, 29, 31, 33, 35, 42, 43, 44, 49, 55, 58, 79, 96, 102 (not investigated).

| group size | multi-record groups (match_summary) | golden records (member_count) |
|---|---|---|
| 1 | 0 | 533 |
| 2 | 4,115 | 4,028 |
| 3 | 668 | 649 |
| 4 | 282 | 304 |
| 5 | 107 | 113 |
| 6 | 45 | 60 |
| 7 | 22 | 32 |
| 8 | 8 | 19 |
| 9 | 10 | 15 |
| 10 | 3 | 9 |
| 11 | 2 | 11 |
| 12 | 2 | 6 |
| 13 | 2 | 6 |
| 14 | 1 | 5 |
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
| 35 | 0 | 1 |
| 42 | 0 | 1 |
| 43 | 0 | 1 |
| 44 | 1 | 0 |
| 49 | 0 | 1 |
| 55 | 1 | 0 |
| 58 | 0 | 1 |
| 79 | 0 | 1 |
| 96 | 0 | 1 |
| 102 | 0 | 1 |

### Neighbor pairs scored, by path

Of 363,550 neighbor pairs, **363,547 were scored** and 3 were skipped before scoring (skipped missing store context: 0, skipped missing embeddings: 0, skipped no semantic field: 0, skipped no code field: 3). **Text path:** 322,969 pairs (embedding cosine blended with field similarity, the original scoring). **Code path:** 40,578 pairs that fail the multi-word test and were scored on code-like columns only (no embedding cosine); code-like columns chosen from the data: `box1_number`, `house_number1`, `postal_code`. The exact-duplicate pairs (6,387) are scored 1.0 by rule. No labels exist, so nothing here says whether a pair is right.

## f. Timing (run_info.json)

| stage | parent | seconds | items |
|---|---|---|---|
| verity.startup |  | 8.036 |  |
| verity.rule_learning |  | 10.305 |  |
| verity.build_graph |  | 1.293 |  |
| verity.embed.model_load | verity.embed | 0.482 |  |
| verity.embed.encode | verity.embed | 45.381 | 41023 |
| verity.embed |  | 47.164 |  |
| verity.embed.other | verity.embed | 1.301 |  |
| verity.neighbors.build | verity.match | 0.020 | 14570 |
| verity.neighbors.query | verity.match | 3.608 | 14570 |
| verity.score | verity.match | 37.623 | 363547 |
| verity.write_outputs | verity.match | 11.481 | 386448 |
| verity.match |  | 55.634 |  |
| verity.match.other | verity.match | 2.902 |  |
| verity.merge |  | 2.553 |  |
| verity.explore |  | 0.441 |  |
| verity.total |  | 125.450 |  |

Total (`verity.total`): 125.5 s. Host: Intel(R) Core(TM) Ultra 7 265U, 14 logical cores.

## e2. Code conflicts

`code_conflict` pairs: **104** (the conflict rule: an identifying field present in both records differs after normalization, and the conflict blocks a pair that would otherwise have matched or fallen in the review band). Decision status: review (the pair is in the band, or would otherwise have passed: score strictly above the effective threshold and evidence mass at or above the minimum). For these rows `evidence_passes` is true exactly when the pair would otherwise have passed. A pair with a conflict that would not have passed keeps its ordinary label.

| decision status | code_conflict pairs |
|---|---|
| review | 104 |

## e3. v4 against v5

v5 limits the conflict rule to identifying fields: code-like columns whose learned normalized weight is at least 1/(2n), n being the number of content columns used for scoring (6 here, cut 0.083333). A conflict is labeled code_conflict only when it blocks a pair that would otherwise have matched or fallen in the review band. Identifying fields in this run: `house_number1`, `postal_code`. Code-like columns: `box1_number`, `house_number1`, `postal_code`.

| count | v4 | v5 |
|---|---|---|
| code_signal matches | 145 | 145 |
| code_conflict pairs, total | 39,423 | 104 |
| code_conflict pairs, match | 0 | 0 |
| code_conflict pairs, review | 104 | 104 |
| code_conflict pairs, non_match | 39,319 | 0 |
| total matches (decision_status match) | 16,362 | 16,362 |

## g1. Explanation card: the first code_signal match, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

| field | similarity | weight | evidence similarity |
|---|---|---|---|
| alias_account | 0.7000 | 0.3635 | 0.7000 |
| city_name | 1.0000 | 0.1233 | 1.0000 |
| house_number1 | 1.0000 | 0.1150 | 1.0000 |
| postal_code | 1.0000 | 0.2593 | 1.0000 |
| street_name | 1.0000 | 0.1390 | 1.0000 |

| item | value |
|---|---|
| score | 0.890957 |
| match_type | code_signal |
| decision_status | match |
| evidence_mass | 0.636523 |
| conflict_mass | 0.000000 |
| evidence_passes | true |

## g2. Explanation card: the first code_conflict pair, by sorted record ids

Field names, similarities, weights and evidence similarities only; no values.

| field | similarity | weight | evidence similarity |
|---|---|---|---|
| alias_account | 1.0000 | 0.3635 | 1.0000 |
| city_name | 1.0000 | 0.1233 | 1.0000 |
| house_number1 | 0.7500 | 0.1150 | 0.7500 |
| postal_code | 1.0000 | 0.2593 | 1.0000 |
| street_name | 1.0000 | 0.1390 | 1.0000 |

| item | value |
|---|---|
| score | 0.971262 |
| match_type | code_conflict |
| decision_status | review |
| evidence_mass | 0.885048 |
| conflict_mass | 0.114952 |
| evidence_passes | true |

## Copies in this folder

Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of `source_csv` and `column_classification_path` were replaced by "(local path removed)". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. Nothing else was changed.

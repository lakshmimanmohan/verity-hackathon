dev_2026_run9: no diversity bonus, embedding_only only across input files, no hardcoded names
============================================================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-24 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run9/verity_data
Mode:            --nn-mode exact (resolved exact; Annoy installed: no), n = 10,000

Code
----
run_info.json records HEAD 38cd6dc (git_commit 38cd6dcea3e634c9a959c6b69e49595cef3c5faf,
git_src_uncommitted_changes false). HEAD is a docs commit; verity/ is identical to commit
0683757: git diff 0683757 38cd6dc -- verity/ is empty. The Verity changes since run 8:
  3839572  remove the diversity bonus from the pair score
  23b9ea5  record each record's input file; allow embedding_only only across input files
  0683757  remove the hardcoded source_system and true_entity_id column names
Each was checked with a full seed-2026 run in a scratch folder (deleted afterwards):
  3839572: precision 0.7694, recall 0.5114 (TP 6334, FP 1898)
  23b9ea5: precision 1.0000, recall 0.5114 (TP 6334, FP 0); 1,898 embedding_only pairs blocked
  0683757: precision 1.0000, recall 0.5114 (TP 6334, FP 0); match_candidates.csv byte-identical
           to 23b9ea5's; the only output change apart from per-run values is the classifier's
           reasoning text for source_system (now metadata by the near-constant rule alone)

How it was run (from the repo root)
-----------------------------------
  export PYTHONHASHSEED=0 PYTHONUTF8=1
  export VERITY_DATA_DIR=results/dev_2026_run9/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run9/orgs_raw.csv
  python verity/src/main.py --nn-mode exact --input "$PWD/results/dev_2026_run9/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run9/orgs_raw_truth.csv       --matches results/dev_2026_run9/verity_data/match_candidates.csv
  python eval/check_scored_pairs.py results/dev_2026_run9
  python eval/band.py results/dev_2026_run9 --truth results/dev_2026_run9/orgs_raw_truth.csv

orgs_raw.csv is byte-identical to dev_2026_run8's.

Phase 1 learns without embeddings
---------------------------------
Phase 1 (rule_learner) reads embeddings_global.pkl before phase 3 (embed) writes it.
In a fresh data folder that file does not exist, so the weights and threshold are
learned without embeddings: NAME and DESCRIPTIVE fields are scored by Jaccard only.
This is Verity's current behaviour and was not changed.

Single input file
-----------------
Every record comes from orgs_raw.csv, so the cross-file rule never lets an embedding_only
pair through (run_info.json: embedding_only_blocked_same_file 1898). The lineage log shows
"Same-file pairs: 216458" (its label still says "registry origin"; the counts now come
from the input files recorded in the store).

Evaluator (evaluator_output.txt)
--------------------------------
  True pairs: 12386   Matched pairs: 6334
  TP 6334  FP 0  FN 6052
  Precision 1.0000  Recall 0.5114  F1 0.6767

  Recall by hard-case slice (pairs where either record carries the tag):
    acronym         876 / 1997   recall 0.4387
    truncation     1114 / 2141   recall 0.5203
    po_box          431 / 1895   recall 0.2274
    easy           4054 / 6798   recall 0.5964

  Decoy pairs (same name, different entity): 482
    wrongly matched 0  (0.00%)

check_scored_pairs.py (exit 0)
------------------------------
  Run: C:\verity-hackathon\results\dev_2026_run9

  PASS  one row per unordered pair  (216458 rows, 0 duplicates)
  PASS  record_a < record_b in every row
  PASS  run_info.json has the scored-pair and skip counts  (skipped_missing_store_context=0, skipped_missing_embeddings=0, skipped_no_semantic_field=0, skipped_single_token_semantic_fields=0)
  PASS  row count equals run_info scored_pairs_rows  (216458 vs 216458)
  PASS  row count equals scored_neighbor_pairs + exact_only_pairs  (216458 vs 216458 + 0)
  PASS  written-type rows equal match_candidates.csv (pairs, score, match_type)  (6334 vs 6334 rows; pairs in only one: 0; value mismatches: 0)
  PASS  rebuilt score_uncapped within 1e-05  (216458 rows, max abs error 1.26e-06)
  PASS  min(rebuilt, 1) equals score within 1e-05 (pair_source neighbor)  (max abs error 1.26e-06)

  Information
    rows by pair_source: {'both': 729, 'neighbor': 215729}
    rows by match_type:  {'below_threshold': 33299, 'full_signal': 4681, 'insufficient_evidence': 176825, 'weak_signal': 1653}
    rows with score_uncapped > 1 (score capped): 0 of 216458

  RESULT: all checks passed

band.py (exit 0)
----------------
  Run: C:\verity-hackathon\results\dev_2026_run9
  Band: effective_threshold +/- 0.05 per row, inclusive
  Wrote 200854 band rows to results\dev_2026_run9\band_pairs.csv

  Pairs by region, match_type and truth (true = same entity)
    region match_type                true    false    total
    below  below_threshold              0      949      949
    below  insufficient_evidence        0     1857     1857
    below  ALL                          0     2806     2806
    in     below_threshold             47    32303    32350
    in     full_signal                 31        0       31
    in     insufficient_evidence      768   167678   168446
    in     weak_signal                 27        0       27
    in     ALL                        873   199981   200854
    above  full_signal               4650        0     4650
    above  insufficient_evidence     4751     1771     6522
    above  weak_signal               1626        0     1626
    above  ALL                      11027     1771    12798

  Missed true pairs (6052 of 12386) by region: below 0, in 815, above 4751, not scored 486

  Verity: recall 0.5114, TP 6334, FP 0
  Oracle ceiling (band decided perfectly, outside kept): recall 0.5772 (+6.58 points), TP 7149, FP 0, precision 1.0000

  band_pairs.csv (117 MB) is gitignored.

Run 9 vs run 8, per pair (scratchpad script, exit 1: one expected-count miss, explained below)
--------------------------------------------------------------------------------------------
  FAIL  match_type counts equal the expected table  ({'below_threshold': 33299, 'full_signal': 4681, 'insufficient_evidence': 176825, 'weak_signal': 1653})
  PASS  same set of pairs as run 8  (216458 vs 216458)
  PASS  run_info embedding_only_blocked_same_file == 1898  (1898)
  PASS  run_info git_src_uncommitted_changes is false  (git_commit 38cd6dcea3e634c9a959c6b69e49595cef3c5faf)
  PASS  evidence_passes changed exactly on run 8's embedding_only rows (true -> false)  (1898 changed, 1898 embedding_only in run 8)
  PASS  run 9 score_uncapped == run 8 score_uncapped - diversity_bonus (1e-5)  (216458 rows, max abs error 1.00e-06)
  PASS  cosine, thresholds, evidence and field columns identical to run 8  (0 mismatches)

  RESULT: 1 check(s) failed

  The miss is in the offline prediction, not in Verity. Two pairs land exactly on the 0.82
  threshold in 6-decimal values: run 8 wrote 1.062250 - 0.242250 = 0.820000 for one, and
  0.820000 for the other (which run 8 itself did not mark below_threshold). Verity decides
  at full precision: both are just above 0.82, so both are insufficient_evidence, not
  below_threshold. Matches, the band and the evaluator are unaffected.

compare_runs.py run8 vs run9 (exit 1, expected)
-----------------------------------------------
  A: C:\verity-hackathon\results\dev_2026_run8
  B: C:\verity-hackathon\results\dev_2026_run9

  NOT_COMPARED   README.txt  (run notes / log)
  ONLY_IN_B      band_pairs.csv
  DIFF           evaluator_output.txt  (line 1: 'True pairs: 12386   Matched pairs: 8234' vs 'True pairs: 12386   Matched pairs: 6334')
  SAME           orgs_raw.csv
  SAME           orgs_raw_truth.csv
  SAME           verity_data/.verity_data_dir
  SAME_IGNORING  verity_data/attribute_weights.json  (differs only in column_classification_path, plot_path, source_csv)
  DIFF           verity_data/column_classification.json  (differing keys: reasoning)
  DIFF           verity_data/column_classification_global.json  (differing keys: reasoning)
  SAME           verity_data/embeddings_global.pkl
  DIFF           verity_data/golden_records.csv  (5165 vs 5668 rows)
  SAME_IGNORING  verity_data/learned_rules.md  (1 line(s) ignored)
  SAME_IGNORING  verity_data/learned_threshold.json  (differs only in column_classification_path, plot_path, source_csv)
  DIFF           verity_data/match_candidates.csv  (8234 vs 6334 rows)
  DIFF           verity_data/match_explorer.html  (line 123: '<span>Total candidate pairs (unique): <strong>8234</strong></span>' vs '<span>Total candidate pairs (unique): <strong>6334</strong></span>')
  DIFF           verity_data/match_summary.csv  (9990 of 10000 rows differ)
  SAME           verity_data/orgs_raw/embeddings.pkl
  DIFF           verity_data/run_info.json  (differing keys: embedding_only_blocked_same_file, match_candidates_rows)
  DIFF           verity_data/scored_pairs.csv  (headers differ)
  DIFF           verity_data/scored_pairs_columns.json  (differing keys: score_formula)
  SAME           verity_data/threshold_similarity_distribution.png
  DIFF           verity_data/verity_store.sqlite  (tables with differing rows: golden, records)
  NOT_COMPARED   verity_run.log  (run notes / log)

  INFO  run_info.json run 0 git_commit: A='54d5cee5fc0b370335fd31ff5120556611e92f03'  B='38cd6dcea3e634c9a959c6b69e49595cef3c5faf'
  INFO  run_info.json run 0 git_src_uncommitted_changes: A=False  B=False

  DIFF 11, NOT_COMPARED 2, ONLY_IN_B 1, SAME 6, SAME_IGNORING 3
  RESULT: 12 file(s) differ

  Expected: the 1,898 embedding_only false matches are gone, so match_candidates.csv,
  golden_records.csv (5,165 -> 5,668 golden records), match_summary.csv, match_explorer.html,
  the store and evaluator_output.txt change; scored_pairs.csv loses the diversity_bonus
  column; the sidecar rule drops the bonus term; the classification reasoning for
  source_system loses the removed name rule; run_info.json gains
  embedding_only_blocked_same_file; band_pairs.csv is new.

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder, including run_info.json and
                                     scored_pairs_columns.json
  verity_run.log, evaluator_output.txt
  Hand check of generator labels (98/100 blind agreement): hand_check_score.txt (the
  eval/hand_check.py --score output) and hand_check_review.md (result and reviewer notes);
  inputs hand_check_pairs.csv (blind labels) and hand_check_key.csv (answers)
  Not in git: band_pairs.csv, verity_data/scored_pairs.csv, verity_data/verity_store.sqlite,
  verity_data/embeddings_global.pkl, verity_data/orgs_raw/embeddings.pkl

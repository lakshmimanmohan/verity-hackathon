dev_2026_run11: learned threshold = low cluster mean + 2 sd; decision_status in scored_pairs.csv
==========================================================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-24 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run11/verity_data
Mode:            --nn-mode exact (resolved exact; Annoy installed: no), n = 10,000

Code
----
run_info.json records HEAD de373d6 (git_commit de373d6dd4115a267b4c114004ed15494cdc2f54,
git_src_uncommitted_changes false). HEAD is a docs commit; verity/ is identical to commit
c01c820: git diff c01c820 de373d6 -- verity/ is empty. Verity changes since run 9:
  8d9269b  per-stage timings and host block in run_info.json
  369870c  learned threshold = lower-cluster mean + 2 sd (threshold stability fix)
  c01c820  decision_status (match / non_match / review) in scored_pairs.csv, REVIEW_MARGIN
369870c and c01c820 were each checked with a full seed-2026 run in a scratch folder
(deleted afterwards):
  369870c: learned threshold 0.447917; match_candidates.csv byte-identical to run 9; in
           scored_pairs.csv only applied_threshold changed (0.411965 -> 0.447917), on exactly
           the 142,399 rows with no shared keys
  c01c820: match_candidates.csv byte-identical to run 9; decision_status match 6,276,
           non_match 9,328, review 200,854; band.py decision_status checks pass

How it was run (from the repo root)
-----------------------------------
  export PYTHONHASHSEED=0 PYTHONUTF8=1
  export VERITY_DATA_DIR=results/dev_2026_run11/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run11/orgs_raw.csv
  python verity/src/main.py --nn-mode exact --input "$PWD/results/dev_2026_run11/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run11/orgs_raw_truth.csv \
      --matches results/dev_2026_run11/verity_data/match_candidates.csv
  python eval/check_scored_pairs.py results/dev_2026_run11 --reference results/dev_2026_run9
  python eval/band.py results/dev_2026_run11 --truth results/dev_2026_run11/orgs_raw_truth.csv

orgs_raw.csv is byte-identical to dev_2026_run9's.

Phase 1 learns without embeddings
---------------------------------
Phase 1 (rule_learner) reads embeddings_global.pkl before phase 3 (embed) writes it.
In a fresh data folder that file does not exist, so the weights and threshold are
learned without embeddings: NAME and DESCRIPTIVE fields are scored by Jaccard only.
This is Verity's current behaviour and was not changed.

Threshold stability fix
-----------------------
The learned threshold is now the lower KMeans cluster's mean + 2 standard deviations
(population sd) instead of the midpoint of the two cluster centres. It is applied because
protocol.md specifies it ("The threshold stability fix (low cluster mean + 2 std dev) is
applied before the band is computed"), not because it measured better.

Measured on this run's data before the change, replaying phase 1 in memory for 5 sampling
seeds (the seed drives both the 1,000-pair sample and KMeans; seed 42 is Verity's and
reproduces run 9's 0.411965):

  seed   current (midpoint)   low mean + 2 sd   low-cluster pairs   low mean / sd
    42             0.411965          0.447917                 424   0.3543 / 0.0468
     1             0.419532          0.457831                 512   0.3595 / 0.0491
     2             0.408342          0.445890                 446   0.3482 / 0.0488
     3             0.422770          0.461570                 522   0.3660 / 0.0478
     4             0.409116          0.453618                 422   0.3492 / 0.0522

  method               mean       std        min        max        reaches 0.82?
  midpoint             0.414345   0.005778   0.408342   0.422770   no
  low mean + 2 sd      0.453365   0.005880   0.445890   0.461570   no

  The new rule is not more stable than the midpoint (std 0.0059 vs 0.0058 across seeds);
  it moves the threshold up by about 0.039.

This run: learned threshold 0.447917 (kmeans_midpoint 0.411965, lower cluster 424 pairs,
mean 0.3543, sd 0.0468). The effective threshold, max(learned, 0.82), stays at the 0.82
floor, so the band stays 0.77 to 0.87 and no match decision changes. The learned value
still applies to the 142,399 pairs with no shared keys; none of them scores between
0.411965 and 0.447917, so no match_type changes.

decision_status and the review band
-----------------------------------
scored_pairs.csv now has decision_status: review when the score is within REVIEW_MARGIN =
0.05 of the effective threshold, inclusive (the protocol's borderline band); otherwise
Verity's own decision, match for the rows of match_candidates.csv and non_match for the
rest. review covers 200,854 of the 216,458 scored pairs (93%); match 6,276, non_match 9,328.

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
  Run: C:\verity-hackathon\results\dev_2026_run11

  PASS  one row per unordered pair  (216458 rows, 0 duplicates)
  PASS  record_a < record_b in every row
  PASS  run_info.json has the scored-pair and skip counts  (skipped_missing_store_context=0, skipped_missing_embeddings=0, skipped_no_semantic_field=0, skipped_single_token_semantic_fields=0)
  PASS  row count equals run_info scored_pairs_rows  (216458 vs 216458)
  PASS  row count equals scored_neighbor_pairs + exact_only_pairs  (216458 vs 216458 + 0)
  PASS  written-type rows equal match_candidates.csv (pairs, score, match_type)  (6334 vs 6334 rows; pairs in only one: 0; value mismatches: 0)
  PASS  rebuilt score_uncapped within 1e-05  (216458 rows, max abs error 1.26e-06)
  PASS  min(rebuilt, 1) equals score within 1e-05 (pair_source neighbor)  (max abs error 1.26e-06)
  PASS  match_candidates.csv byte-identical to dev_2026_run9
  PASS  evaluator_output.txt identical to dev_2026_run9

  Information
    rows by pair_source: {'both': 729, 'neighbor': 215729}
    rows by match_type:  {'below_threshold': 33299, 'full_signal': 4681, 'insufficient_evidence': 176825, 'weak_signal': 1653}
    rows with score_uncapped > 1 (score capped): 0 of 216458

  RESULT: all checks passed

band.py (exit 0)
----------------
  Run: C:\verity-hackathon\results\dev_2026_run11
  Band: effective_threshold +/- 0.05 per row, inclusive
  Wrote 200854 band rows to results\dev_2026_run11\band_pairs.csv

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

  decision_status by region
    region    match  non_match   review
    below         0       2806        0
    in            0          0   200854
    above      6276       6522        0
    total  match 6276, non_match 9328, review 200854
  CHECK decision_status == review selects exactly the band pairs: PASS
  CHECK outside the band, decision_status equals Verity's decision: PASS

  band_pairs.csv (about 120 MB) is gitignored.

compare_runs.py run9 vs run11 (exit 1, expected)
------------------------------------------------
  A: C:\verity-hackathon\results\dev_2026_run9
  B: C:\verity-hackathon\results\dev_2026_run11

  NOT_COMPARED   README.txt  (run notes / log)
  DIFF           band_pairs.csv  (headers differ)
  SAME           evaluator_output.txt
  ONLY_IN_A      hand_check_key.csv
  ONLY_IN_A      hand_check_pairs.csv
  ONLY_IN_A      hand_check_review.md
  ONLY_IN_A      hand_check_score.txt
  SAME           orgs_raw.csv
  SAME           orgs_raw_truth.csv
  ONLY_IN_A      split_records.csv
  ONLY_IN_A      split_summary.txt
  SAME           verity_data/.verity_data_dir
  SAME_IGNORING  verity_data/attribute_weights.json  (differs only in column_classification_path, plot_path, source_csv)
  SAME           verity_data/column_classification.json
  SAME           verity_data/column_classification_global.json
  SAME           verity_data/embeddings_global.pkl
  SAME_IGNORING  verity_data/golden_records.csv  (golden_id ignored)
  DIFF           verity_data/learned_rules.md  (line 2: 'Source file: `C:\\verity-hackathon\\results\\dev_2026_run9\\orgs_raw.csv`' vs 'Source file: `C:\\verity-hackathon\\results\\dev_2026_run11\\orgs_raw.csv`')
  DIFF           verity_data/learned_threshold.json  (differing keys: kmeans_midpoint, low_cluster_mean, low_cluster_n, low_cluster_std, method, sd_multiplier, threshold, threshold_method)
  SAME           verity_data/match_candidates.csv
  SAME_IGNORING  verity_data/match_explorer.html  (1 line(s) ignored)
  SAME           verity_data/match_summary.csv
  SAME           verity_data/orgs_raw/embeddings.pkl
  DIFF           verity_data/run_info.json  (differing keys: decision_status_counts, host, model_cache_hit, review_margin, timing_summary, timings)
  DIFF           verity_data/scored_pairs.csv  (headers differ)
  DIFF           verity_data/scored_pairs_columns.json  (differing keys: decision_status_rule, learned_threshold, review_margin)
  DIFF           verity_data/threshold_similarity_distribution.png  (binary)
  SAME_IGNORING  verity_data/verity_store.sqlite  (rows compared table by table, golden_id masked, order ignored)
  NOT_COMPARED   verity_run.log  (run notes / log)

  INFO  run_info.json run 0 git_commit: A='38cd6dcea3e634c9a959c6b69e49595cef3c5faf'  B='de373d6dd4115a267b4c114004ed15494cdc2f54'
  INFO  run_info.json run 0 git_src_uncommitted_changes: A=False  B=False

  DIFF 7, NOT_COMPARED 2, ONLY_IN_A 6, SAME 10, SAME_IGNORING 4
  RESULT: 13 file(s) differ

  Expected differences: learned_threshold.json, the threshold section of learned_rules.md
  (compare_runs shows its first differing line, the ignored source path; the change is the
  threshold section) and the threshold plot; scored_pairs.csv gains decision_status (header
  differs) and band_pairs.csv with it; the sidecar records review_margin, the
  decision_status rule and the new learned threshold; run_info.json gains
  decision_status_counts, review_margin and the timing records (host, timings,
  timing_summary, model_cache_hit, added in 8d9269b). The hand-check and split files exist
  only in run 9. match_candidates.csv, golden_records.csv (golden_id ignored), the store,
  match_summary.csv and evaluator_output.txt are the same.

  Timings in run_info.json were recorded on an uncontrolled laptop state and are not for
  quoting.

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder, including run_info.json and
                                     scored_pairs_columns.json
  verity_run.log, evaluator_output.txt
  Not in git: band_pairs.csv, verity_data/scored_pairs.csv, verity_data/verity_store.sqlite,
  verity_data/embeddings_global.pkl, verity_data/orgs_raw/embeddings.pkl

dev_2026_run8: scored_pairs.csv (every scored pair), exact neighbor search
==========================================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-24 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run8/verity_data
Purpose:         First committed run with scored_pairs.csv and scored_pairs_columns.json.
                 Replaces dev_2026_run7 (same code, run before committing; deleted).

Code
----
run_info.json records HEAD 54d5cee (git_commit 54d5cee5fc0b370335fd31ff5120556611e92f03,
git_src_uncommitted_changes false). HEAD is a docs commit; verity/ is identical to commit
3e82e35 (Verity: scored_pairs.csv and scored_pairs_columns.json, pre-scoring skip counts
in run_info.json): git diff 3e82e35 54d5cee -- verity/ is empty.

Run settings (from verity_data/run_info.json)
---------------------------------------------
  Neighbor search mode:   requested exact (from --nn-mode), resolved exact; Annoy installed: no
  Records in index (n):   10,000
  Python 3.13.16, sentence-transformers 6.3.0, embedding model all-MiniLM-L6-v2, PYTHONHASHSEED 0
  Pair counts:            neighbor_pairs 216,458; scored_neighbor_pairs 216,458;
                          exact_duplicate_pairs 729; exact_only_pairs 0;
                          scored_pairs_rows 216,458; match_candidates_rows 8,234
  Dropped before scoring: 0 (missing store context 0, missing embeddings 0,
                          no NAME/DESCRIPTIVE/STREET_ADDRESS value 0, single-token values 0)

How it was run (from the repo root)
-----------------------------------
  export PYTHONHASHSEED=0 PYTHONUTF8=1
  export VERITY_DATA_DIR=results/dev_2026_run8/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run8/orgs_raw.csv
  python verity/src/main.py --nn-mode exact --input "$PWD/results/dev_2026_run8/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run8/orgs_raw_truth.csv       --matches results/dev_2026_run8/verity_data/match_candidates.csv

orgs_raw.csv is byte-identical to dev_2026_run6's.

Phase 1 learns without embeddings
---------------------------------
Phase 1 (rule_learner) reads embeddings_global.pkl before phase 3 (embed) writes it.
In a fresh data folder that file does not exist, so the weights and threshold are
learned without embeddings: NAME and DESCRIPTIVE fields are scored by Jaccard only.
This is Verity's current behaviour and was not changed.

Check: eval/check_scored_pairs.py
---------------------------------
  python eval/check_scored_pairs.py results/dev_2026_run8 --reference results/dev_2026_run6   (exit 0)

  Run: C:\verity-hackathon\results\dev_2026_run8

  PASS  one row per unordered pair  (216458 rows, 0 duplicates)
  PASS  record_a < record_b in every row
  PASS  run_info.json has the scored-pair and skip counts  (skipped_missing_store_context=0, skipped_missing_embeddings=0, skipped_no_semantic_field=0, skipped_single_token_semantic_fields=0)
  PASS  row count equals run_info scored_pairs_rows  (216458 vs 216458)
  PASS  row count equals scored_neighbor_pairs + exact_only_pairs  (216458 vs 216458 + 0)
  PASS  written-type rows equal match_candidates.csv (pairs, score, match_type)  (8234 vs 8234 rows; pairs in only one: 0; value mismatches: 0)
  PASS  rebuilt score_uncapped within 1e-05  (216458 rows, max abs error 1.42e-06)
  PASS  min(rebuilt, 1) equals score within 1e-05 (pair_source neighbor)  (max abs error 1.22e-06)
  PASS  match_candidates.csv byte-identical to dev_2026_run6
  PASS  evaluator_output.txt identical to dev_2026_run6

  Information
    rows by pair_source: {'both': 729, 'neighbor': 215729}
    rows by match_type:  {'below_threshold': 8181, 'embedding_only': 1898, 'full_signal': 4681, 'insufficient_evidence': 200043, 'weak_signal': 1655}
    rows with diversity_bonus > 0 (value 0.242250), by match_type:
      below_threshold              0 of    8181
      embedding_only            1470 of    1898
      full_signal               3519 of    4681
      insufficient_evidence   155907 of  200043
      weak_signal               1238 of    1655
      total                   162134 of  216458
    rows with score_uncapped > 1 (score capped): 161692 of 216458
    embedding_only score_uncapped: n=1898 min=0.811821 median=1.087185 max=1.147852

  RESULT: all checks passed

Diversity bonus and embedding_only scores
-----------------------------------------
  The diversity bonus is learned_threshold * (1 - learned_threshold) = 0.242250, added when
  the two records come from different source systems. Rows with diversity_bonus > 0:

    match_type              bonus > 0       of
    below_threshold                 0     8,181
    embedding_only              1,470     1,898
    full_signal                 3,519     4,681
    insufficient_evidence     155,907   200,043
    weak_signal                 1,238     1,655
    total                     162,134   216,458

  No below_threshold row has the bonus: every cross-source pair scores above 0.82.
  161,692 of 216,458 rows (75%) have score_uncapped > 1, so their score is capped at 1.0.

  embedding_only score_uncapped: n=1,898, min 0.811821, median 1.087185, max 1.147852.

Comparison with dev_2026_run6 (before scored_pairs.csv)
-------------------------------------------------------
  python eval/compare_runs.py results/dev_2026_run6 results/dev_2026_run8   (exit 1, expected)

  A: C:\verity-hackathon\results\dev_2026_run6
  B: C:\verity-hackathon\results\dev_2026_run8

  NOT_COMPARED   README.txt  (run notes / log)
  SAME           evaluator_output.txt
  SAME           orgs_raw.csv
  SAME           orgs_raw_truth.csv
  SAME           verity_data/.verity_data_dir
  SAME_IGNORING  verity_data/attribute_weights.json  (differs only in column_classification_path, plot_path, source_csv)
  SAME           verity_data/column_classification.json
  SAME           verity_data/column_classification_global.json
  SAME           verity_data/embeddings_global.pkl
  SAME_IGNORING  verity_data/golden_records.csv  (golden_id ignored)
  SAME_IGNORING  verity_data/learned_rules.md  (1 line(s) ignored)
  SAME_IGNORING  verity_data/learned_threshold.json  (differs only in column_classification_path, plot_path, source_csv)
  SAME           verity_data/match_candidates.csv
  SAME_IGNORING  verity_data/match_explorer.html  (1 line(s) ignored)
  SAME           verity_data/match_summary.csv
  SAME           verity_data/orgs_raw/embeddings.pkl
  DIFF           verity_data/run_info.json  (differing keys: exact_duplicate_pairs, exact_only_pairs, match_candidates_rows, neighbor_pairs, scored_neighbor_pairs, scored_pairs_rows, skipped_missing_embeddings, skipped_missing_store_context, skipped_no_semantic_field, skipped_single_token_semantic_fields)
  ONLY_IN_B      verity_data/scored_pairs.csv
  ONLY_IN_B      verity_data/scored_pairs_columns.json
  SAME           verity_data/threshold_similarity_distribution.png
  SAME_IGNORING  verity_data/verity_store.sqlite  (rows compared table by table, golden_id masked, order ignored)
  NOT_COMPARED   verity_run.log  (run notes / log)

  INFO  run_info.json run 0 git_commit: A='35b6a1adcd47443f2507b0ef510452c4e098717d'  B='54d5cee5fc0b370335fd31ff5120556611e92f03'
  INFO  run_info.json run 0 git_src_uncommitted_changes: A=False  B=False

  DIFF 1, NOT_COMPARED 2, ONLY_IN_B 2, SAME 11, SAME_IGNORING 6
  RESULT: 3 file(s) differ

  Expected differences only: run_info.json gains the ten count keys listed above (no
  existing key changed apart from git_commit, shown as INFO, and the ignored per-run
  started_utc and input_csv); scored_pairs.csv and scored_pairs_columns.json are new.

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder for this run, including run_info.json
                                     and scored_pairs_columns.json
  verity_run.log, evaluator_output.txt
  Not in git: verity_data/scored_pairs.csv (127 MB), verity_data/verity_store.sqlite,
  verity_data/embeddings_global.pkl, verity_data/orgs_raw/embeddings.pkl

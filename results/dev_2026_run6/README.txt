dev_2026_run6: exact neighbor search, one row per pair (clean commit)
=====================================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-23 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run6/verity_data
Purpose:         Same run as dev_2026_run5 (now deleted), made on committed code so that
                 run_info.json records a clean commit.

Code
----
run_info.json records HEAD 35b6a1a (git_commit 35b6a1adcd47443f2507b0ef510452c4e098717d,
git_src_uncommitted_changes false). That is a docs-only commit; verity/ is identical to
commit 9c435aa (Verity: neighbor-search mode, run_info.json, requirements-ann.txt, one row
per pair): git diff 9c435aa 35b6a1a -- verity/ is empty.

Run settings (from verity_data/run_info.json)
---------------------------------------------
  Neighbor search mode:   requested exact (from --nn-mode), resolved exact
  Annoy installed:        no
  Records in index (n):   10,000
  Python:                 3.13.16
  sentence-transformers:  6.3.0, embedding model all-MiniLM-L6-v2
  PYTHONHASHSEED:         0

How it was run (from the repo root)
-----------------------------------
  export PYTHONHASHSEED=0 PYTHONUTF8=1
  export VERITY_DATA_DIR=results/dev_2026_run6/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run6/orgs_raw.csv
  python verity/src/main.py --nn-mode exact --input "$PWD/results/dev_2026_run6/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run6/orgs_raw_truth.csv \
      --matches results/dev_2026_run6/verity_data/match_candidates.csv

Phase 1 learns without embeddings
---------------------------------
Phase 1 (rule_learner) reads embeddings_global.pkl before phase 3 (embed) writes it.
In a fresh data folder that file does not exist, so the weights and threshold are
learned without embeddings: NAME and DESCRIPTIVE fields are scored by Jaccard only.
This is Verity's current behaviour and was not changed.

Results
-------
  match_candidates.csv        8,234 rows, 0 duplicate unordered pairs ("Wrote 8234 row(s)")
  match_type counts           full_signal 4,681, weak_signal 1,655, embedding_only 1,898
  Exact-duplicate pairs       729, of which 670 were also found by neighbor search; one row kept
  Effective score threshold   0.8200 (max(learned=0.4120, floor=0.82))
  evaluator_output.txt        identical to dev_2026_run2 and dev_2026_run5
    Precision 0.7695  Recall 0.5115  F1 0.6145
  Against dev_2026_run2 (8,904 rows for the same 8,234 pairs, before the dedupe fix):
    same unique pair set, golden_records.csv SAME_IGNORING, evaluator output identical.

Comparison with dev_2026_run5 (same code, run before committing)
----------------------------------------------------------------
  python eval/compare_runs.py results/dev_2026_run5 results/dev_2026_run6   (exit 0)

  A: C:\verity-hackathon\results\dev_2026_run5
  B: C:\verity-hackathon\results\dev_2026_run6

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
  SAME_IGNORING  verity_data/run_info.json  (ignored input_csv, started_utc; git_commit, git_src_uncommitted_changes reported as INFO)
  SAME           verity_data/threshold_similarity_distribution.png
  SAME_IGNORING  verity_data/verity_store.sqlite  (rows compared table by table, golden_id masked, order ignored)
  NOT_COMPARED   verity_run.log  (run notes / log)

  INFO  run_info.json run 0 git_commit: A='a1c44c85ef725305c22ce214a7babe4b069e7a7d'  B='35b6a1adcd47443f2507b0ef510452c4e098717d'
  INFO  run_info.json run 0 git_src_uncommitted_changes: A=True  B=False

  NOT_COMPARED 2, SAME 11, SAME_IGNORING 7
  RESULT: all compared files match

  run_info.json differs only in the INFO fields: run 5 ran on uncommitted changes on top of
  a1c44c8; run 6 ran on committed code. dev_2026_run5 was deleted after this comparison.

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder for this run, including run_info.json
  verity_run.log, evaluator_output.txt
  Not in git (*.sqlite, *.pkl): verity_data/verity_store.sqlite, verity_data/embeddings_global.pkl,
  verity_data/orgs_raw/embeddings.pkl

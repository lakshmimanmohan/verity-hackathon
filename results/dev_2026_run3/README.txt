dev_2026_run3: second clean run with a per-run data folder
===========================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-23 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run3/verity_data
Python:          3.13 on Windows 11; sentence-transformers 6.3.0, torch 2.15.0+cpu, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1
Purpose:         Check that a run reading and writing only its own data folder is reproducible.
                 Paired with dev_2026_run2 (same command, separate folder).

Code
----
Commit cd30990 (Verity: per-run data folder (VERITY_DATA_DIR), full-run guard and --fresh).
The run was made on that code before it was committed; the committed files are the ones it used.

How it was run (from the repo root)
-----------------------------------
  export PYTHONHASHSEED=0 PYTHONUTF8=1
  export VERITY_DATA_DIR=results/dev_2026_run3/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run3/orgs_raw.csv
  python verity/src/main.py --input "$PWD/results/dev_2026_run3/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run3/orgs_raw_truth.csv \
      --matches results/dev_2026_run3/verity_data/match_candidates.csv

The data folder did not exist before the run; Verity created it (with the
.verity_data_dir marker). Annoy is not installed, so matching used exact search
(sklearn NearestNeighbors), as in run 1.

Phase 1 learns without embeddings
---------------------------------
Phase 1 (rule_learner) reads embeddings_global.pkl before phase 3 (embed) writes it.
In a fresh data folder that file does not exist, so the weights and threshold are
learned without embeddings: NAME and DESCRIPTIVE fields are scored by Jaccard only.
This is Verity's current behaviour and was not changed. A rerun into a folder that
already holds embeddings would learn differently, which is why a full run now
requires an empty folder.

Results
-------
  Effective score threshold 0.8200 (max(learned=0.4120, floor=0.82))
  Evaluator output (evaluator_output.txt): identical to dev_2026_run1
    Precision 0.7695  Recall 0.5115  F1 0.6145

Comparison with dev_2026_run2 (the pass check)
----------------------------------------------
  python eval/compare_runs.py results/dev_2026_run2 results/dev_2026_run3   (exit 0)

  A: C:\verity-hackathon\results\dev_2026_run2
  B: C:\verity-hackathon\results\dev_2026_run3

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
  SAME           verity_data/threshold_similarity_distribution.png
  SAME_IGNORING  verity_data/verity_store.sqlite  (rows compared table by table, golden_id masked, order ignored)
  NOT_COMPARED   verity_run.log  (run notes / log)

  NOT_COMPARED 2, SAME 11, SAME_IGNORING 6
  RESULT: all compared files match

  Ignored by design: plot_path, source_csv, column_classification_path in the JSON files;
  golden_id values (random UUIDs); the "Source file:" line in learned_rules.md; the
  "Generated:" line in match_explorer.html.

Comparison with dev_2026_run1 (information only)
------------------------------------------------
  orgs_raw.csv, orgs_raw_truth.csv   byte-identical
  match_candidates.csv               byte-identical
  learned_threshold.json, attribute_weights.json   identical apart from the path fields above
  evaluator_output.txt               identical
  So run 1, which used verity/data/, learned the same weights and threshold as a run with
  no prior embeddings.

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder for this run (all its outputs)
  verity_run.log, evaluator_output.txt
  Not in git (*.sqlite, *.pkl): verity_data/verity_store.sqlite, verity_data/embeddings_global.pkl,
  verity_data/orgs_raw/embeddings.pkl

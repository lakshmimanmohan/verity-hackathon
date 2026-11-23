dev_2026_run1: Verity baseline on development data
===================================================

Seed:            2026 (development seed since the Nov 23 protocol amendment; not the evaluation seed)
Date:            2026-11-23 (data generated 07:28, Verity run finished 07:38, US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026, default rows)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1
Python:          3.13 on Windows 11; sentence-transformers 6.3.0, torch 2.15.0, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1

Code
----
The run was made on uncommitted files, before the first commits of the code.
The code it used is byte-identical to what was then committed:
  verity/src/ and eval/   commit a30413f / 9816607 (unchanged in every later commit)
  data/generate_data_org.py   commit 9816607 (DEFAULT_SEED = 2026 at that commit;
                              later commits changed only the default, to 42)
Use commit 9816607 to reproduce this run.

How it was run
--------------
  python data/generate_data_org.py --output data/orgs_raw.csv          (default seed was 2026 then)
  cp data/orgs_raw.csv verity/data/orgs_raw.csv
  python verity/src/main.py --input "$PWD/verity/data/orgs_raw.csv"
  python eval/evaluate_matches.py --truth data/orgs_raw_truth.csv --matches verity/data/match_candidates.csv

Deviations from the documented commands:
  - The input CSV was copied into verity/data/ because match and merge read the learned files from
    the input CSV's folder. With --input data/orgs_raw.csv the match phase stops with
    "Missing attribute_weights.json".
  - Annoy is not installed (needs Microsoft C++ Build Tools). With exactly 10,000 records Verity
    would use Annoy if available; this run used exact search (sklearn NearestNeighbors, cosine, brute).
    A run with Annoy installed will produce different candidate pairs.

Thresholds (from verity_run.log)
--------------------------------
  Effective score threshold: 0.8200 (max(learned=0.4120, floor=0.82))
  Embedding-only score threshold: 0.4120 (learned threshold from learned_threshold.json, no floor)

Evaluator output (evaluator_output.txt, re-run on the files in this folder)
---------------------------------------------------------------------------
True pairs: 12386   Matched pairs: 8234
TP 6336  FP 1898  FN 6050
Precision 0.7695  Recall 0.5115  F1 0.6145

Recall by hard-case slice (pairs where either record carries the tag):
  acronym         878 / 1997   recall 0.4397
  truncation     1114 / 2141   recall 0.5203
  po_box          431 / 1895   recall 0.2274
  easy           4054 / 6798   recall 0.5964

Decoy pairs (same name, different entity): 482
  wrongly matched 1  (0.21%)

Summary: 77% precision, 51% recall. Raw Verity output, before any option or threshold fix.

Notes
-----
  - match_candidates.csv has 8,904 rows but 8,234 unique pairs (670 exact-duplicate pairs appear twice).
    The evaluator counts unique pairs.
  - All 1,898 false positives are match_type embedding_only (pairs sharing no field values).
    Every full_signal and weak_signal pair is a true match.
  - Pairs scoring below the effective threshold are not written, so the 0.77-0.82 part of the
    borderline band is not in this file.

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv    generated data (truth file is the sidecar answer key)
  match_candidates.csv                 Verity's written pairs
  golden_records.csv, match_summary.csv, match_explorer.html
  attribute_weights.json, learned_threshold.json, learned_rules.md, column_classification*.json,
  threshold_similarity_distribution.png
  verity_run.log                       full pipeline log (home folder path replaced with %USERPROFILE%)
  evaluator_output.txt
  Not in git (over 5 MB): verity_store.sqlite (15 MB), embeddings_global.pkl and orgs_raw/embeddings.pkl (33 MB each)

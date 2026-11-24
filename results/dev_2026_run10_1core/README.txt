dev_2026_run10_1core: timing instrumentation check, one thread (OMP_NUM_THREADS=1; torch used 1 thread)
=======================================================================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-24 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run10_1core/verity_data
Mode:            --nn-mode exact (resolved exact; Annoy installed: no), n = 10,000
Purpose:         Check the per-stage timing instrumentation: outputs unchanged, every stage
                 recorded, stages adding up to the total.

TIMINGS ARE NOT FOR QUOTING
---------------------------
Timings recorded on an uncontrolled laptop state; not for quoting. Final timings are
measured near the freeze: plugged in, Best performance power mode, idle machine,
3 repeats, median.

The log's "Total wall time (all phases)" (405.37s here) excludes startup (main.py imports,
argument parsing, data folder and store checks); verity.total in run_info.json includes it.

Code
----
run_info.json records HEAD 5220bec with git_src_uncommitted_changes true: the run was made
before the timing change was committed. The code it used is commit 8d9269b (Verity:
per-stage timings and host block in run_info.json); verity/ in the working tree at run time
is identical to that commit. eval/timings.py is commit 5652827.

How it was run (from the repo root)
-----------------------------------
  export PYTHONHASHSEED=0 PYTHONUTF8=1
  export OMP_NUM_THREADS=1
  export VERITY_DATA_DIR=results/dev_2026_run10_1core/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run10_1core/orgs_raw.csv
  python verity/src/main.py --nn-mode exact --input "$PWD/results/dev_2026_run10_1core/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run10_1core/orgs_raw_truth.csv       --matches results/dev_2026_run10_1core/verity_data/match_candidates.csv
  python eval/timings.py results/dev_2026_run10_1core

Phase 1 learns without embeddings
---------------------------------
Phase 1 (rule_learner) reads embeddings_global.pkl before phase 3 (embed) writes it.
In a fresh data folder that file does not exist, so the weights and threshold are
learned without embeddings: NAME and DESCRIPTIVE fields are scored by Jaccard only.
This is Verity's current behaviour and was not changed.

Checks
------
  match_candidates.csv             byte-identical to dev_2026_run9          PASS
  eval/timings.py                  prints every stage, exit 0               PASS
  top-level stages vs total        100.0% (within 5%)                       PASS

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

eval/timings.py
---------------
  == verity_data\run_info.json run 0 (full)
     host: cpu_model=Intel(R) Core(TM) Ultra 7 265U, logical_cores=14, ram_gb=31.5, os=Windows-11-10.0.26300-SP0, python=3.13.16, torch_threads=1, omp_num_threads=1
     stage                              seconds  % total            items   p50 ms   p95 ms    max ms  note
     verity.startup                      14.303     3.4%                                               main.py imports, argument parsing, data folder, store check
     verity.rule_learning                26.588     6.3%                                               rule_learner subprocess, including its Python start-up
     verity.build_graph                   3.549     0.8%                                               
     verity.embed                       318.783    76.0%                                               
       verity.embed.model_load            0.925     0.2%                                               model all-MiniLM-L6-v2; model_cache_hit=True
       verity.embed.encode              315.925    75.3%   34,001 segment                              sentence-transformers encode only, 10000 record(s) embedded
       verity.embed.other                 1.933     0.5%                                               untimed remainder of the parent stage
     verity.match                        54.817    13.1%                                               
       verity.neighbors.build             0.022     0.0%    10,000 record                              sklearn NearestNeighbors fit (cosine, brute)
       verity.neighbors.query             3.718     0.9%    10,000 record                              kneighbors top-30 + pair set
       verity.score                      42.167    10.0%     216,458 pair   0.1588   0.3057  298.6516  per pair: field similarity, score, shared keys, evidence; plus the decision pass
       verity.write_outputs               6.887     1.6%      222,792 row                              match_candidates.csv, scored_pairs.csv, scored_pairs_columns.json
       verity.match.other                 2.024     0.5%                                               untimed remainder of the parent stage
     verity.merge                         1.322     0.3%                                               
     verity.explore                       0.294     0.1%                                               match_explorer.html and match_summary.csv
     TOTAL                              419.675
     top-level stages sum to 419.655s, 100.0% of total (within 5%)
     children of verity.embed sum to 318.783s of 318.783s
     children of verity.match sum to 54.817s of 54.817s
     summary: cold_total_s=419.675, warm_total_s=418.749, model_load_s=0.925, definition=cold = total; warm = total minus model load

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder, including run_info.json (timings, host)
  verity_run.log, evaluator_output.txt
  Not in git: verity_data/scored_pairs.csv, verity_data/verity_store.sqlite,
  verity_data/embeddings_global.pkl, verity_data/orgs_raw/embeddings.pkl

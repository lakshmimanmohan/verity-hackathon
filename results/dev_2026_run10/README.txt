dev_2026_run10: timing instrumentation check, normal threads (OMP_NUM_THREADS not set; torch used 12 threads)
=============================================================================================================

Seed:            2026 (development seed; not the evaluation seed)
Date:            2026-11-24 (US Central time)
Records:         10,000 records, 4,105 entities (data/generate_data_org.py --seed 2026)
Environment:     PYTHONHASHSEED=0, PYTHONUTF8=1, VERITY_DATA_DIR=results/dev_2026_run10/verity_data
Mode:            --nn-mode exact (resolved exact; Annoy installed: no), n = 10,000
Purpose:         Check the per-stage timing instrumentation: outputs unchanged, every stage
                 recorded, stages adding up to the total.

TIMINGS ARE NOT FOR QUOTING
---------------------------
Timings recorded on an uncontrolled laptop state; not for quoting. Final timings are
measured near the freeze: plugged in, Best performance power mode, idle machine,
3 repeats, median.

The log's "Total wall time (all phases)" (237.61s here) excludes startup (main.py imports,
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
  export VERITY_DATA_DIR=results/dev_2026_run10/verity_data
  python data/generate_data_org.py --seed 2026 --output results/dev_2026_run10/orgs_raw.csv
  python verity/src/main.py --nn-mode exact --input "$PWD/results/dev_2026_run10/orgs_raw.csv"
  python eval/evaluate_matches.py --truth results/dev_2026_run10/orgs_raw_truth.csv       --matches results/dev_2026_run10/verity_data/match_candidates.csv
  python eval/timings.py results/dev_2026_run10

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
     host: cpu_model=Intel(R) Core(TM) Ultra 7 265U, logical_cores=14, ram_gb=31.5, os=Windows-11-10.0.26300-SP0, python=3.13.16, torch_threads=12, omp_num_threads=None
     stage                              seconds  % total            items   p50 ms   p95 ms    max ms  note
     verity.startup                      97.539    29.1%                                               main.py imports, argument parsing, data folder, store check
     verity.rule_learning                32.311     9.6%                                               rule_learner subprocess, including its Python start-up
     verity.build_graph                   3.279     1.0%                                               
     verity.embed                        54.162    16.2%                                               
       verity.embed.model_load            1.485     0.4%                                               model all-MiniLM-L6-v2; model_cache_hit=True
       verity.embed.encode               49.768    14.8%   34,001 segment                              sentence-transformers encode only, 10000 record(s) embedded
       verity.embed.other                 2.909     0.9%                                               untimed remainder of the parent stage
     verity.match                       142.885    42.6%                                               
       verity.neighbors.build             0.053     0.0%    10,000 record                              sklearn NearestNeighbors fit (cosine, brute)
       verity.neighbors.query             4.828     1.4%    10,000 record                              kneighbors top-30 + pair set
       verity.score                     112.061    33.4%     216,458 pair   0.5363   0.8335  766.6303  per pair: field similarity, score, shared keys, evidence; plus the decision pass
       verity.write_outputs              21.202     6.3%      222,792 row                              match_candidates.csv, scored_pairs.csv, scored_pairs_columns.json
       verity.match.other                 4.740     1.4%                                               untimed remainder of the parent stage
     verity.merge                         4.106     1.2%                                               
     verity.explore                       0.783     0.2%                                               match_explorer.html and match_summary.csv
     TOTAL                              335.144
     top-level stages sum to 335.065s, 100.0% of total (within 5%)
     children of verity.embed sum to 54.162s of 54.162s
     children of verity.match sum to 142.885s of 142.885s
     summary: cold_total_s=335.144, warm_total_s=333.659, model_load_s=1.485, definition=cold = total; warm = total minus model load

Files
-----
  orgs_raw.csv, orgs_raw_truth.csv   generated data (truth file is the sidecar answer key)
  verity_data/                       Verity's data folder, including run_info.json (timings, host)
  verity_run.log, evaluator_output.txt
  Not in git: verity_data/scored_pairs.csv, verity_data/verity_store.sqlite,
  verity_data/embeddings_global.pkl, verity_data/orgs_raw/embeddings.pkl

# Verity hackathon repo

Experiment: can Verity (schema-agnostic entity matching) resolve borderline record pairs
as well as a frontier LLM, at lower cost and latency? Demo is Dec 7, numbers freeze Dec 4.

See @README.md for layout and run steps, and @protocol.md for the quality bars,
borderline definition and volume assumptions. The protocol is fixed: never edit it
to fit results.

## Hard rules

- Verity uses the SQLite record store only (`verity/src/record_store.py`). Never add Neo4j code or dependencies.
- No column names hardcoded in Verity. Logic works from semantic types (NAME, STREET_ADDRESS, DESCRIPTIVE) assigned by `rule_learner.py`. Reject fixes that break this.
- Ground truth lives only in the sidecar `*_truth.csv`. Never put an answer-key column in a CSV that Verity reads: the classifier treats an integer label column as content, so it would leak into matching.
- Evaluation seed is 7919 (changed Nov 23, see the protocol's Amendments). Never tune thresholds or features on data generated with it, and don't generate it until the eval run. 42 (the generator default) and 2026 are development seeds.
- Every option scores the same candidate pairs on the same held-out test split.
- Run everything with `PYTHONHASHSEED=0`. Without it, scores shift up to 0.0075 between runs.
- Also set `PYTHONUTF8=1`. Without it, Verity crashes on Windows when printing its column report.
- Verify claims by running the code. Say plainly when something was not run.

## Where things go

- Core Verity changes (threshold fix, review band, per-field similarity export): in `verity/src/`, which is vendored source, not a submodule, so there is no submodule pin to update. Commit them here, one change per commit, separate from experiment code; each run's README.txt records the commit it used. Verity's dependencies go in `verity/requirements.txt`.
- Run outputs: each run gets its own data folder, set with `VERITY_DATA_DIR` in the shell (never in `.env`). A full run stops if that folder is not empty; `--fresh` empties it, but only if Verity created it (marker file `.verity_data_dir`).
- Experiment code (options A to E, cost model, eval): in this repo only.
- No company, product or people names in anything under `verity/`.

## Secrets and data

- LLM key goes in the repo-root `.env` only. Never print it, log it or commit it. Verity never reads that file; it loads only `verity/.env`, and stops if `VERITY_DATA_DIR` is set there.
- Synthetic data only. No customer, personal or confidential data.
- Open-weight models: pinned revision hash, safetensors only, `trust_remote_code=False`.

## Commands

```
export PYTHONHASHSEED=0
export PYTHONUTF8=1    # Windows: Verity prints characters the default console codepage can't encode
export VERITY_DATA_DIR=results/<run>/verity_data   # new folder per run; shell only, never .env
python data/generate_data_org.py --output data/orgs_raw.csv
python verity/src/main.py --input data/orgs_raw.csv   # relative paths resolve from the current directory
python eval/evaluate_matches.py --truth data/orgs_raw_truth.csv --matches "$VERITY_DATA_DIR/match_candidates.csv"
```

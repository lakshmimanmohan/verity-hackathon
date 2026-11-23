# Verity Hackathon: Entity Matching Without a Frontier LLM

Can a schema-agnostic matcher resolve borderline record pairs as well as a frontier LLM, at lower cost and latency?

This repo holds the experiment. The matcher itself, Verity, is vendored as source in `verity/src/` (a plain folder, not a submodule). Changes to it are committed here like any other file, and each run records the commit it used.

## What is compared

Every option judges the same candidate pairs, produced by Verity's nearest-neighbor blocking (exact search for evaluation runs), on the same held-out test split.

| Option | What decides a borderline pair | Code |
| --- | --- | --- |
| A. Threshold only | Verity's effective threshold | `options/a_threshold.py` |
| B. Frontier LLM (baseline) | LLM yes/no on the record pair | `options/b_llm_baseline.py` |
| C. Classical ML | Gradient-boosted trees on per-field similarities, calibrated confidence | `options/c_decision_model.py` |
| D. Hybrid | C, with low-confidence pairs sent to the LLM | `options/d_hybrid.py` |
| E. SLM / cross-encoder | Small open-weight model, self-hosted | `options/e_slm.py` |

Quality bars, the borderline definition and volume assumptions are fixed in [`protocol.md`](protocol.md). That file was committed before any option was run.

## Layout

```
verity/     vendored Verity source (src/), requirements.txt, and requirements-ann.txt (optional Annoy)
data/       generate_data_org.py   synthetic B2B records + sidecar truth file
eval/       evaluate_matches.py    precision, recall, hard-case slices, decoy false matches
options/    one script per option (A to E)
cost/       cost_model.py          cost per pair at 100K / 1M / 10M records per day, break-even
results/    one folder per run: metrics, timings, config used
```

## Setup

Python 3.12 or later (Verity uses `itertools.batched`).

```
git clone https://github.com/lakshmimanmohan/verity-hackathon.git
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # includes verity/requirements.txt
# Optional, only for approximate neighbor search (ANN timing for the cost model).
# On Windows it needs the Microsoft C++ Build Tools.
pip install -r verity/requirements-ann.txt
cp .env.example .env               # add the LLM key; never commit .env
```

Verity keeps its records in a local SQLite file (`verity_store.sqlite`) in its data folder. No database server is needed.

Every file Verity reads or writes, apart from the input CSV, lives in one data folder: `VERITY_DATA_DIR` if set in the shell, otherwise `verity/data/`. Give each run its own folder. A full run stops if the folder is not empty, so nothing from an earlier run can leak in. `--fresh` empties the folder first, but only one Verity created (it holds a `.verity_data_dir` marker).

Neighbor search mode is set with `--nn-mode exact|approximate|auto`, or `VERITY_NN_MODE` in the shell (never in `.env`); the flag wins. `auto`, the default, uses Annoy at 10,000 records or more when it is installed, otherwise exact search. `approximate` stops if Annoy is not installed. Each run writes `run_info.json` into its data folder: requested and resolved mode, Annoy installed, record count, git commit, Python and sentence-transformers versions, embedding model and `PYTHONHASHSEED`.

Two `.env` files are in play. The repo-root `.env` (copied from `.env.example`) holds the LLM key for the options that call the LLM; Verity never reads it. Verity loads only `verity/.env`, reads no setting from it today, and stops if `VERITY_DATA_DIR` is set there: set it in the shell.

## Reproduce the results

```
# Set PYTHONHASHSEED=0 for every run: without it, scores shift slightly between runs.
# Set PYTHONUTF8=1 too: without it, Verity crashes on Windows when printing its report.
export PYTHONHASHSEED=0
export PYTHONUTF8=1

# 1. Generate data with the development seed (default 42). Seed 7919 is reserved for
#    the freeze run below; never generate or tune on it before then.
python data/generate_data_org.py --output data/orgs_raw.csv

# 2. Run Verity to produce candidate pairs and borderline pairs, into a new folder per run.
#    Relative paths resolve from the current directory.
export VERITY_DATA_DIR=results/<run>/verity_data
#    Evaluation runs use exact neighbor search (see Amendments in protocol.md).
python verity/src/main.py --nn-mode exact --input data/orgs_raw.csv

# 3. Score each option on the test split
python options/a_threshold.py
python options/b_llm_baseline.py
python options/c_decision_model.py
python options/d_hybrid.py
python options/e_slm.py

# 4. Evaluate and price
python eval/evaluate_matches.py --truth data/orgs_raw_truth.csv --matches results/<run>/verity_data/match_candidates.csv
python cost/cost_model.py --results results/<run>
```

Each run writes its config, metrics and timings to `results/<run>/`, so any number in the results table traces back to a folder.

## Freeze run (Dec 4 only)

The evaluation set uses seed 7919 (see Amendments in `protocol.md`). Generate it once, on Dec 4, after every threshold, feature and prompt is fixed. Then run steps 2 to 4 above unchanged.

```
export PYTHONHASHSEED=0
export PYTHONUTF8=1
python data/generate_data_org.py --seed 7919 --output data/orgs_raw.csv
```

## Data and models

- All data is synthetic, generated from a fixed seed. No customer, personal or confidential data is used.
- Ground truth lives in a separate `_truth.csv` file, never in the records Verity reads, so the answer key cannot become a matching feature.
- Open-weight models are pinned to a revision hash, loaded from safetensors with `trust_remote_code=False`, and run only in the approved sandbox. License checks are in `protocol.md`.

## Results

To be filled in after the runs. One row per option: borderline accuracy, overall precision and recall, p95 latency, cost per pair at each volume, and whether it met every bar.

## Limitations

- Synthetic data may not reflect real customer noise.
- English B2B organization records only.
- Token counts and model prices are measured on the test date and will change.

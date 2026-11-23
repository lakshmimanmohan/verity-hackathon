# Verity Hackathon: Entity Matching Without a Frontier LLM

Can a schema-agnostic matcher resolve borderline record pairs as well as a frontier LLM, at lower cost and latency?

This repo holds the experiment. The matcher itself, Verity, is vendored as source in `verity/src/` (a plain folder, not a submodule). Changes to it are committed here like any other file, and each run records the commit it used.

## What is compared

Every option judges the same candidate pairs, produced by Verity's ANN blocking, on the same held-out test split.

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
verity/     vendored Verity source (src/) and its requirements.txt
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
cp .env.example .env               # add the LLM key; never commit .env
```

Verity keeps its records in a local SQLite file (`verity/data/verity_store.sqlite`). No database server is needed.

## Reproduce the results

```
# Set PYTHONHASHSEED=0 for every run: without it, scores shift slightly between runs.
# Set PYTHONUTF8=1 too: without it, Verity crashes on Windows when printing its report.
export PYTHONHASHSEED=0
export PYTHONUTF8=1

# 1. Generate data (seed 2026 is the evaluation seed; do not tune on it)
python data/generate_data_org.py --output data/orgs_raw.csv

# 2. Run Verity to produce candidate pairs and borderline pairs.
#    Verity writes its outputs to verity/data/; copy them into a new results/<run>/ folder.
python verity/src/main.py --input "$PWD/data/orgs_raw.csv"   # absolute path: Verity resolves relative paths from verity/

# 3. Score each option on the test split
python options/a_threshold.py
python options/b_llm_baseline.py
python options/c_decision_model.py
python options/d_hybrid.py
python options/e_slm.py

# 4. Evaluate and price
python eval/evaluate_matches.py --truth data/orgs_raw_truth.csv --matches results/<run>/match_candidates.csv
python cost/cost_model.py --results results/<run>
```

Each run writes its config, metrics and timings to `results/<run>/`, so any number in the results table traces back to a folder.

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

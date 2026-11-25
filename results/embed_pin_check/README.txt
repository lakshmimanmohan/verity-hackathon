Embedding model pin check (Part A, 2026-11-25). verity/src/embed.py loads sentence-transformers/all-MiniLM-L6-v2 at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 from model.safetensors only (model_kwargs use_safetensors=True), trust_remote_code=False.
Verity was rerun on results/dev_2026_run11/orgs_raw.csv (seed 2026) into a fresh data folder with --nn-mode exact, PYTHONHASHSEED=0,
PYTHONUTF8=1. match_candidates.csv is byte-identical to run11's (sha256 in match_candidates.sha256.txt). The data folder itself
(embeddings, SQLite store) was deleted after the check; run_info.json is kept.

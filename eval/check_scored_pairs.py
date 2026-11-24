"""
Check a run's scored_pairs.csv against its match_candidates.csv and its own columns.

Usage:
  python eval/check_scored_pairs.py results/<run> [--reference results/<earlier run>]

Checks (exit 1 if any fails):
  - scored_pairs.csv has one row per unordered pair, record_a < record_b
  - its row count equals run_info.json's scored_pairs_rows, and scored_neighbor_pairs plus
    exact_only_pairs; the pre-scoring skip counts are present in run_info.json
  - rows whose match_type is written to match_candidates.csv (full_signal, weak_signal,
    embedding_only) equal match_candidates.csv in pairs, score strings and match_type
  - score_uncapped rebuilt from each scored row's own columns plus scored_pairs_columns.json
    is within 1e-5 of the written value, and min(rebuilt, cap) is within 1e-5 of score
    (pair_source "neighbor"; exact-duplicate rows have score 1.0 by rule)
  - with --reference: match_candidates.csv is byte-identical to the reference run's, and
    evaluator_output.txt is identical when both runs have one

Also prints, for information: rows by pair_source and match_type, rows with a diversity
bonus by match_type, and score_uncapped min / median / max for embedding_only rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

WRITE_TYPES = {"full_signal", "weak_signal", "embedding_only"}
TOL = 1e-5
SKIP_KEYS = (
    "skipped_missing_store_context",
    "skipped_missing_embeddings",
    "skipped_no_semantic_field",
    "skipped_single_token_semantic_fields",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description="Check a run's scored_pairs.csv.")
    p.add_argument("run", type=Path)
    p.add_argument("--reference", type=Path, default=None)
    args = p.parse_args()
    csv.field_size_limit(10**8)

    data = args.run / "verity_data"
    sp = read_csv(data / "scored_pairs.csv")
    mc = read_csv(data / "match_candidates.csv")
    cols = json.loads((data / "scored_pairs_columns.json").read_text(encoding="utf-8"))
    info = json.loads((data / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    sem = cols["field_semantic_types"]
    fm = cols["score_formula"]

    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        failures += 0 if cond else 1
        print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    print(f"Run: {args.run.resolve()}\n")

    keys = [(r["record_a"], r["record_b"]) for r in sp]
    check("one row per unordered pair", len(set(keys)) == len(keys),
          f"{len(keys)} rows, {len(keys) - len(set(keys))} duplicates")
    check("record_a < record_b in every row", all(a < b for a, b in keys))

    missing = [k for k in ("scored_pairs_rows", "scored_neighbor_pairs", "exact_only_pairs", *SKIP_KEYS)
               if k not in info]
    check("run_info.json has the scored-pair and skip counts", not missing,
          f"missing: {', '.join(missing)}" if missing else ", ".join(f"{k}={info[k]}" for k in SKIP_KEYS))
    if not missing:
        check("row count equals run_info scored_pairs_rows", len(sp) == info["scored_pairs_rows"],
              f"{len(sp)} vs {info['scored_pairs_rows']}")
        want = info["scored_neighbor_pairs"] + info["exact_only_pairs"]
        check("row count equals scored_neighbor_pairs + exact_only_pairs", len(sp) == want,
              f"{len(sp)} vs {info['scored_neighbor_pairs']} + {info['exact_only_pairs']}")

    sp_w = {(r["record_a"], r["record_b"]): (r["score"], r["match_type"])
            for r in sp if r["match_type"] in WRITE_TYPES}
    mc_w = {tuple(sorted((r["record_a"], r["record_b"]))): (r["score"], r["match_type"]) for r in mc}
    only = len(set(sp_w) ^ set(mc_w))
    mism = sum(1 for k in set(sp_w) & set(mc_w) if sp_w[k] != mc_w[k])
    check("written-type rows equal match_candidates.csv (pairs, score, match_type)",
          sp_w == mc_w and len(mc_w) == len(mc),
          f"{len(sp_w)} vs {len(mc)} rows; pairs in only one: {only}; value mismatches: {mism}")

    hyb = set(fm["hybrid_semantic_types"])
    cap = fm["score_cap"]
    err_u = err_s = 0.0
    n_rebuilt = 0
    for r in sp:
        if r["pair_source"] == "exact":
            continue
        cos = float(r["cosine"])
        div = float(r["diversity_bonus"])
        fw = json.loads(r["field_weights"])
        fj = json.loads(r["field_similarities"])
        if fw:
            we, wf = fm["w_embed_blend"], fm["w_field_blend"]
        else:
            we = fm["w_when_field_weights_empty"]["embed"]
            wf = fm["w_when_field_weights_empty"]["field"]
        fsim = 0.0
        for k, wn in fw.items():
            if sem.get(k) in hyb:
                s = fm["hybrid_embed_weight"] * cos + fm["hybrid_char_weight"] * fj[k]
            else:
                s = fj[k]
            fsim += wn * s
        rebuilt = we * cos + wf * fsim + div
        err_u = max(err_u, abs(rebuilt - float(r["score_uncapped"])))
        if r["pair_source"] == "neighbor":
            err_s = max(err_s, abs(min(rebuilt, cap) - float(r["score"])))
        n_rebuilt += 1
    check(f"rebuilt score_uncapped within {TOL:g}", err_u <= TOL,
          f"{n_rebuilt} rows, max abs error {err_u:.2e}")
    check(f"min(rebuilt, {cap:g}) equals score within {TOL:g} (pair_source neighbor)", err_s <= TOL,
          f"max abs error {err_s:.2e}")

    if args.reference is not None:
        ref = args.reference / "verity_data" / "match_candidates.csv"
        check(f"match_candidates.csv byte-identical to {args.reference.name}",
              ref.read_bytes() == (data / "match_candidates.csv").read_bytes())
        ea, eb = args.reference / "evaluator_output.txt", args.run / "evaluator_output.txt"
        if ea.is_file() and eb.is_file():
            check(f"evaluator_output.txt identical to {args.reference.name}",
                  ea.read_bytes() == eb.read_bytes())

    print("\nInformation")
    print("  rows by pair_source:", dict(sorted(Counter(r["pair_source"] for r in sp).items())))
    print("  rows by match_type: ", dict(sorted(Counter(r["match_type"] for r in sp).items())))
    scored = [r for r in sp if r["diversity_bonus"]]
    pos = Counter(r["match_type"] for r in scored if float(r["diversity_bonus"]) > 0)
    tot = Counter(r["match_type"] for r in scored)
    bonus_vals = sorted({r["diversity_bonus"] for r in scored if float(r["diversity_bonus"]) > 0})
    print(f"  rows with diversity_bonus > 0 (value {', '.join(bonus_vals) or 'none'}), by match_type:")
    for t in sorted(tot):
        print(f"    {t:<22} {pos.get(t, 0):>7} of {tot[t]:>7}")
    print(f"    {'total':<22} {sum(pos.values()):>7} of {sum(tot.values()):>7}")
    capped = sum(1 for r in scored if float(r["score_uncapped"]) > cap)
    print(f"  rows with score_uncapped > {cap:g} (score capped): {capped} of {len(scored)}")
    eo = sorted(float(r["score_uncapped"]) for r in sp
                if r["match_type"] == "embedding_only" and r["score_uncapped"])
    if eo:
        print(f"  embedding_only score_uncapped: n={len(eo)} min={eo[0]:.6f} "
              f"median={statistics.median(eo):.6f} max={eo[-1]:.6f}")

    print(f"\nRESULT: {'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

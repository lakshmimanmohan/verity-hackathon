"""
Score match_candidates.csv against the generator's sidecar truth file.

Reports precision / recall / F1 overall, recall on each hard-case slice,
false matches on decoy pairs, and how many true pairs never became candidates.

Usage:
  python evaluate_matches.py --truth data/orgs_raw_truth.csv \
      --matches data/match_candidates.csv [--candidates data/all_candidate_pairs.csv]

--candidates is optional: a CSV of every pair that reached scoring (record_a, record_b).
With it, the report separates pairs lost at ANN blocking from pairs lost at scoring.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HARD_TAGS = ("acronym", "truncation", "po_box")


def read_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {tuple(sorted((r["record_a"], r["record_b"]))) for r in csv.DictReader(f)}


def pr(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--matches", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=None)
    args = ap.parse_args()

    entity, tags, decoy = {}, {}, {}
    with args.truth.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            entity[r["record_id"]] = r["true_entity_id"]
            tags[r["record_id"]] = set(r["variants"].split("+"))
            decoy[r["record_id"]] = r["decoy_group"]

    by_entity: dict[str, list[str]] = defaultdict(list)
    by_decoy: dict[str, list[str]] = defaultdict(list)
    for rid, eid in entity.items():
        by_entity[eid].append(rid)
        if decoy[rid]:
            by_decoy[decoy[rid]].append(rid)

    true_pairs = {tuple(sorted(p)) for ids in by_entity.values() for p in combinations(ids, 2)}
    decoy_pairs = {tuple(sorted(p)) for ids in by_decoy.values() for p in combinations(ids, 2)
                   if entity[p[0]] != entity[p[1]]}

    matched = read_pairs(args.matches)
    unknown = {p for p in matched if p[0] not in entity or p[1] not in entity}
    if unknown:
        print(f"WARNING: {len(unknown)} matched pairs have record ids missing from the truth file")
    matched -= unknown

    tp, fp, fn = len(matched & true_pairs), len(matched - true_pairs), len(true_pairs - matched)
    p, r, f1 = pr(tp, fp, fn)
    print(f"True pairs: {len(true_pairs)}   Matched pairs: {len(matched)}")
    print(f"TP {tp}  FP {fp}  FN {fn}")
    print(f"Precision {p:.4f}  Recall {r:.4f}  F1 {f1:.4f}\n")

    print("Recall by hard-case slice (pairs where either record carries the tag):")
    for tag in HARD_TAGS:
        sl = {q for q in true_pairs if tag in tags[q[0]] or tag in tags[q[1]]}
        hit = len(sl & matched)
        print(f"  {tag:<12} {hit:>6} / {len(sl):<6} recall {hit / len(sl) if sl else 0:.4f}")
    easy = {q for q in true_pairs if not any(t in tags[q[0]] | tags[q[1]] for t in HARD_TAGS)}
    hit = len(easy & matched)
    print(f"  {'easy':<12} {hit:>6} / {len(easy):<6} recall {hit / len(easy) if easy else 0:.4f}")

    wrong = len(decoy_pairs & matched)
    print(f"\nDecoy pairs (same name, different entity): {len(decoy_pairs)}")
    print(f"  wrongly matched {wrong}  ({wrong / len(decoy_pairs) if decoy_pairs else 0:.2%})")

    if args.candidates:
        cands = read_pairs(args.candidates)
        never = len(true_pairs - cands)
        print(f"\nTrue pairs never generated as candidates (lost at blocking): "
              f"{never} ({never / len(true_pairs):.2%})")
        print(f"True pairs that were candidates but not matched (lost at scoring): "
              f"{len((true_pairs & cands) - matched)}")


if __name__ == "__main__":
    main()

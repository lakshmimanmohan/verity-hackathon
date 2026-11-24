"""
Borderline band report for a run, from its scored_pairs.csv.

Band (protocol.md): effective_threshold +/- 0.05 per row, inclusive. Pairs above the band
are matches and pairs below it are non-matches for every option; inside the band each
option decides. Region is decided by the row's score alone (protocol Amendments, Nov 23).

Usage:
  python eval/band.py results/<run> --truth results/<run>/orgs_raw_truth.csv

Writes <run>/band_pairs.csv: the scored_pairs.csv rows inside the band, same columns, no
truth labels (the answer key stays in the sidecar truth file). Prints, using --truth:
  - pairs below / in / above the band, by truth and by match_type
  - missed true pairs (not matched by Verity) by region, including pairs never scored
  - oracle ceiling: recall if every band pair were decided perfectly and every pair outside
    the band kept Verity's decision; gain over Verity's recall in points, TP, FP
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HALF_WIDTH = 0.05
WRITE_TYPES = {"full_signal", "weak_signal", "embedding_only"}


def main() -> int:
    p = argparse.ArgumentParser(description="Borderline band report from scored_pairs.csv.")
    p.add_argument("run", type=Path)
    p.add_argument("--truth", type=Path, required=True)
    args = p.parse_args()
    csv.field_size_limit(10**8)

    src = args.run / "verity_data" / "scored_pairs.csv"
    out = args.run / "band_pairs.csv"

    entity: dict[str, str] = {}
    with args.truth.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            entity[r["record_id"]] = r["true_entity_id"]
    by_entity: dict[str, list[str]] = defaultdict(list)
    for rid, eid in entity.items():
        by_entity[eid].append(rid)
    true_pairs = {tuple(sorted(q)) for ids in by_entity.values() for q in combinations(ids, 2)}

    counts: Counter = Counter()
    region_of: dict[tuple[str, str], str] = {}
    matched: set[tuple[str, str]] = set()
    oracle_tp = oracle_fp = 0
    n_band = 0
    with src.open(newline="", encoding="utf-8") as f, out.open("w", newline="", encoding="utf-8") as g:
        rdr = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=rdr.fieldnames)
        w.writeheader()
        for r in rdr:
            key = (r["record_a"], r["record_b"])
            s = float(r["score"])
            t = float(r["effective_threshold"])
            region = "below" if s < t - HALF_WIDTH else ("in" if s <= t + HALF_WIDTH else "above")
            region_of[key] = region
            is_true = key in true_pairs
            is_match = r["match_type"] in WRITE_TYPES
            if is_match:
                matched.add(key)
            counts[(region, r["match_type"], is_true)] += 1
            if region == "in":
                w.writerow(r)
                n_band += 1
                decided = is_true
            else:
                decided = is_match
            if decided:
                oracle_tp += is_true
                oracle_fp += not is_true

    n_true = len(true_pairs)
    tp = len(matched & true_pairs)
    recall = tp / n_true
    print(f"Run: {args.run.resolve()}")
    print(f"Band: effective_threshold +/- {HALF_WIDTH} per row, inclusive")
    print(f"Wrote {n_band} band rows to {out}\n")

    types = sorted({k[1] for k in counts})
    print("Pairs by region, match_type and truth (true = same entity)")
    print(f"  {'region':<6} {'match_type':<22} {'true':>7} {'false':>8} {'total':>8}")
    for region in ("below", "in", "above"):
        rt = rf = 0
        for mt in types:
            a, b = counts[(region, mt, True)], counts[(region, mt, False)]
            if a or b:
                print(f"  {region:<6} {mt:<22} {a:>7} {b:>8} {a + b:>8}")
            rt += a
            rf += b
        print(f"  {region:<6} {'ALL':<22} {rt:>7} {rf:>8} {rt + rf:>8}")

    missed = true_pairs - matched
    mr = Counter(region_of.get(q, "not scored") for q in missed)
    print(f"\nMissed true pairs ({len(missed)} of {n_true}) by region: "
          + ", ".join(f"{k} {mr.get(k, 0)}" for k in ("below", "in", "above", "not scored")))

    o_recall = oracle_tp / n_true
    o_prec = oracle_tp / (oracle_tp + oracle_fp) if oracle_tp + oracle_fp else 0.0
    print(f"\nVerity: recall {recall:.4f}, TP {tp}, FP {len(matched) - tp}")
    print(f"Oracle ceiling (band decided perfectly, outside kept): recall {o_recall:.4f} "
          f"({(o_recall - recall) * 100:+.2f} points), TP {oracle_tp}, FP {oracle_fp}, "
          f"precision {o_prec:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

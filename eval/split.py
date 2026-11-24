"""
Train / test split of a run's records (protocol.md, Amendments, Nov 24, 2026).

  python eval/split.py results/<run>

Records are grouped by connected component of the graph whose edges are Verity's matched
pairs (match_candidates.csv) plus all pairs scoring above the band (scored_pairs.csv, score
above effective_threshold + 0.05). Every record of the run's input CSV is a node, so a record
with no such edge is its own component. Components are ordered by their smallest record id,
shuffled with seed 0, and assigned to train in that order until train holds 55% of records;
the rest go to test. A pair is kept only if both of its records are on the same side; pairs
that cross the split are excluded from training and scoring.

Uses no ground-truth labels. Writes into the run folder:
  split_records.csv   record_id, side (train / test)
  split_summary.txt   method, inputs, records and components per side, and pair counts per
                      side (band pairs, all scored pairs, matched pairs)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = 0
TRAIN_RECORD_SHARE = 0.55
HALF_WIDTH = 0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in nodes}


def main() -> int:
    p = argparse.ArgumentParser(description="Record-level train/test split (no truth labels).")
    p.add_argument("run", type=Path)
    args = p.parse_args()
    csv.field_size_limit(10**8)
    run, data = args.run, args.run / "verity_data"

    cls = json.loads((data / "column_classification.json").read_text(encoding="utf-8"))
    rid_col = str(cls["record_id_column"]).lower()
    input_csv = run / "orgs_raw.csv"
    raw = read_csv(input_csv)
    rid_key = next(k for k in raw[0] if k.lower() == rid_col)
    records = sorted({r[rid_key] for r in raw})

    sp_path, mc_path = data / "scored_pairs.csv", data / "match_candidates.csv"
    region: dict[tuple[str, str], str] = {}
    for r in read_csv(sp_path):
        s, t = float(r["score"]), float(r["effective_threshold"])
        region[(r["record_a"], r["record_b"])] = (
            "below" if s < t - HALF_WIDTH else ("in" if s <= t + HALF_WIDTH else "above")
        )
    matched = [tuple(sorted((r["record_a"], r["record_b"]))) for r in read_csv(mc_path)]
    above = [k for k, g in region.items() if g == "above"]
    edges = sorted(set(matched) | set(above))

    comp = components(records, edges)
    groups: dict[str, list[str]] = defaultdict(list)
    for n in records:
        groups[comp[n]].append(n)
    order = sorted(groups, key=lambda g: min(groups[g]))
    random.Random(SEED).shuffle(order)
    side: dict[str, str] = {}
    n_train = 0
    target = TRAIN_RECORD_SHARE * len(records)
    comp_side = Counter()
    for g in order:
        s = "train" if n_train < target else "test"
        for n in groups[g]:
            side[n] = s
        comp_side[s] += 1
        if s == "train":
            n_train += len(groups[g])

    with (run / "split_records.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "side"])
        for n in records:
            w.writerow([n, side[n]])

    def by_side(pairs) -> Counter:
        c = Counter()
        for a, b in pairs:
            c[side[a] if side[a] == side[b] else "crossing"] += 1
        return c

    band = [k for k, g in region.items() if g == "in"]
    cb, ca, cm = by_side(band), by_side(region), by_side(matched)
    used: dict[str, set[str]] = defaultdict(set)
    for a, b in region:
        if side[a] == side[b]:
            used[a].add(side[a])
            used[b].add(side[a])
    both = sum(1 for v in used.values() if len(v) > 1)
    sizes = Counter(comp.values())
    kept_band = cb["train"] + cb["test"]

    lines = [
        f"Split of {run.name} (protocol.md, Amendments, Nov 24, 2026). No ground-truth labels used.",
        "",
        "Method",
        "  Groups: connected components over all records of the input CSV; edges are Verity's",
        "  matched pairs plus all pairs scoring above the band (score > effective_threshold + 0.05).",
        f"  Components ordered by smallest record id, shuffled with seed {SEED}, assigned to train",
        f"  until train holds {TRAIN_RECORD_SHARE:.0%} of records. A pair is kept only if both records are on the",
        "  same side; crossing pairs are excluded from training and scoring.",
        "",
        "Inputs (sha256)",
        f"  {input_csv.name:<22} {sha256(input_csv)}",
        f"  {mc_path.name:<22} {sha256(mc_path)}",
        f"  {sp_path.name:<22} {sha256(sp_path)}",
        "",
        f"Graph: {len(records)} records, {len(edges)} edges ({len(set(matched))} matched, {len(above)} above-band),",
        f"  {len(sizes)} components, largest {max(sizes.values())} records",
        "",
        f"Records:    train {n_train}, test {len(records) - n_train} "
        f"({n_train / len(records):.2%} train)",
        f"Components: train {comp_side['train']}, test {comp_side['test']}",
        "",
        "Pairs by side (crossing = records on different sides; excluded)",
        f"  {'':<22} {'train':>8} {'test':>8} {'crossing':>9}",
        f"  {'band pairs':<22} {cb['train']:>8} {cb['test']:>8} {cb['crossing']:>9}",
        f"  {'all scored pairs':<22} {ca['train']:>8} {ca['test']:>8} {ca['crossing']:>9}",
        f"  {'matched pairs':<22} {cm['train']:>8} {cm['test']:>8} {cm['crossing']:>9}",
        f"  train share of kept band pairs: {cb['train'] / kept_band:.1%}" if kept_band else "",
        "",
        f"Records appearing in kept pairs on both sides: {both}",
    ]
    (run / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if both:
        print("ERROR: a record appears on both sides", file=sys.stderr)
        return 1
    if cm["crossing"]:
        print("ERROR: a matched pair crosses the split", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

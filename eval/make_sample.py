"""
Draw the fixed sample of test band pairs that the LLM options (B, D) are scored on
(protocol.md, Amendments, Nov 24, 2026: "LLM options scored on a fixed sample").

  python eval/make_sample.py results/<run> [--size 5000] [--seed 0] [--out <file>]

A test band pair is a row of verity_data/scored_pairs.csv with decision_status == review whose
two records are both in the test group of split_records.csv. The sample holds every same-entity
test band pair (weight 1) plus randomly drawn non-matches, up to --size pairs in all. Each
non-match carries the weight (non-matches in the band) / (non-matches drawn), the inverse of
its sampling rate, so the weights sum to the number of test band pairs and weighted precision,
recall and F1 estimate the full test band.

The truth file (<run>/orgs_raw_truth.csv) is read for test records only and only to choose the
pairs. Labels are not written to the output, which has columns record_a, record_b, weight.

--n-nonmatches K draws every same-entity pair plus K non-matches (size = same-entity pairs + K), used by the rehearsal.

Default output: <run>/options/b_llm/test_sample<size>.csv, for seed 0 only (another seed needs
--out, so the committed sample is never overwritten by accident). Other code imports
draw_sample() and test_band_pairs() from here; there is no second copy of the sampler.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

WEIGHT_TOLERANCE = 1e-6


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**8)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_band_pairs(run: Path) -> list[tuple[str, str]]:
    """Sorted (record_a, record_b) of the test band pairs, as written in scored_pairs.csv."""
    side = {r["record_id"]: r["side"] for r in read_csv(run / "split_records.csv")}
    pairs = []
    with (run / "verity_data" / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
        csv.field_size_limit(10**8)
        for r in csv.DictReader(f):
            if r["decision_status"] != "review":
                continue
            a, b = r["record_a"], r["record_b"]
            if side.get(a) == "test" and side.get(b) == "test":
                pairs.append((a, b))
    return sorted(pairs)


def test_entities(run: Path) -> dict[str, str]:
    """record_id -> true_entity_id for test-group records only."""
    side = {r["record_id"]: r["side"] for r in read_csv(run / "split_records.csv")}
    return {r["record_id"]: r["true_entity_id"] for r in read_csv(run / "orgs_raw_truth.csv")
            if side.get(r["record_id"]) == "test"}


def draw_sample(band: list[tuple[str, str]], entity: dict[str, str], size: int, seed: int = 0):
    """Return (pairs, weights, n_true, n_false_in_band, n_false_drawn, weight_of_a_non_match)."""
    band = sorted(band)
    true = [p for p in band if entity[p[0]] == entity[p[1]]]
    false = [p for p in band if entity[p[0]] != entity[p[1]]]
    n_false = size - len(true)
    if n_false < 0 or n_false > len(false):
        raise SystemExit(f"cannot draw {size}: {len(true)} true pairs and {len(false)} non-matches in the test band")
    picked = sorted(random.Random(seed).sample(false, n_false))
    w_false = len(false) / n_false
    weights = {p: 1.0 for p in true}
    weights.update({p: w_false for p in picked})
    pairs = sorted(true + picked)
    total = sum(weights.values())
    if abs(total - len(band)) > WEIGHT_TOLERANCE * len(band):
        raise SystemExit(f"weights sum to {total} but the test band has {len(band)} pairs")
    return pairs, weights, len(true), len(false), n_false, w_false


def write_sample(path: Path, pairs, weights) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_a", "record_b", "weight"])
        for p in pairs:
            w.writerow([p[0], p[1], f"{weights[p]:.6f}"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw the fixed weighted sample of test band pairs.")
    ap.add_argument("run", type=Path)
    ap.add_argument("--size", type=int, default=5000)
    ap.add_argument("--n-nonmatches", type=int, default=None,
                    help="instead of --size: every same-entity pair plus this many non-matches (the rehearsal sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        if args.seed != 0:
            raise SystemExit("--out is required for a seed other than 0")
        args.out = args.run / "options" / "b_llm" / f"test_sample{args.size}.csv"
    band = test_band_pairs(args.run)
    entity = test_entities(args.run)
    if args.n_nonmatches is not None:
        args.size = sum(entity[a] == entity[b] for a, b in band) + args.n_nonmatches
    pairs, weights, n_true, n_false_all, n_false, w_false = draw_sample(band, entity, args.size, args.seed)
    write_sample(args.out, pairs, weights)
    print(f"test band pairs {len(band):,}: {n_true} same-entity, {n_false_all:,} non-matches")
    print(f"sample (seed {args.seed}): {len(pairs):,} pairs = {n_true} same-entity (weight 1) + {n_false:,} non-matches "
          f"(weight {w_false:.6f} each)")
    print(f"weights sum to {sum(weights.values()):,.4f}; test band has {len(band):,} pairs")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

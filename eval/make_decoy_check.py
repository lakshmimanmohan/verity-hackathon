"""
List every test band pair whose two records share a decoy group but have different entities (the hard negatives),
for the supplementary decoy check of option B. Truth is read for test records and only to select the pairs; the
output has no label: record_a, record_b, weight (1.0 for every pair, so it can be passed to
options/b_llm_baseline.py --final-sample).

  python eval/make_decoy_check.py results/<run> [--out <file>]

Prints the pair count, how many are in the fixed 5,000-pair sample (their answers exist already) and how many are
already in the local answer cache, then the cost estimate for the rest at B's measured cost per call.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_sample as ms  # noqa: E402

COST_PER_CALL = 0.001527


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.run
    side = {r["record_id"]: r["side"] for r in ms.read_csv(run / "split_records.csv")}
    truth = {r["record_id"]: r for r in ms.read_csv(run / "orgs_raw_truth.csv") if side.get(r["record_id"]) == "test"}
    band = ms.test_band_pairs(run)
    pairs = []
    for a, b in band:
        ta, tb = truth[a], truth[b]
        if ta["decoy_group"] and ta["decoy_group"] == tb["decoy_group"] and ta["true_entity_id"] != tb["true_entity_id"]:
            pairs.append((a, b))
    out = args.out or run / "options" / "b_llm" / "decoy_check_pairs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_a", "record_b", "weight"])
        for a, b in pairs:
            w.writerow([a, b, "1.0"])
    sample = {tuple(sorted((r["record_a"], r["record_b"]))) for r in ms.read_csv(run / "options" / "b_llm" / "test_sample5000.csv")}
    in_sample = sum(p in sample for p in pairs)
    print(f"decoy test band pairs (same decoy group, different entities): {len(pairs)} of {len(band):,} band pairs; "
          f"{in_sample} are in the 5,000-pair sample (already answered in the cache)")
    print(f"written: {out}")
    print(f"upper-bound cost estimate if none were cached: {len(pairs)} x ${COST_PER_CALL} = ${len(pairs) * COST_PER_CALL:.3f}; "
          f"for the {len(pairs) - in_sample} not in the sample: ${(len(pairs) - in_sample) * COST_PER_CALL:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

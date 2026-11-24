"""
Option A: threshold only. Inside the band the decision is Verity's own: "same" when the
pair's match_type is one written to match_candidates.csv (full_signal, weak_signal,
embedding_only), else "different".

  python options/a_threshold.py results/<run>
  python eval/score_option.py results/<run> --decisions results/<run>/options/a_threshold/decisions.csv

Reads verity_data/scored_pairs.csv and split_records.csv; writes one row per test band pair
(decision_status == review, both records in the test group) to
<run>/options/a_threshold/decisions.csv: record_a, record_b, decision, confidence (empty:
option A has no confidence). Uses no truth labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

WRITE_TYPES = {"full_signal", "weak_signal", "embedding_only"}


def main() -> int:
    p = argparse.ArgumentParser(description="Option A: Verity's threshold decision in the band.")
    p.add_argument("run", type=Path)
    args = p.parse_args()
    csv.field_size_limit(10**8)
    with (args.run / "split_records.csv").open(newline="", encoding="utf-8") as f:
        side = {r["record_id"]: r["side"] for r in csv.DictReader(f)}
    out_dir = args.run / "options" / "a_threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "decisions.csv"
    n = n_same = 0
    with (args.run / "verity_data" / "scored_pairs.csv").open(newline="", encoding="utf-8") as f, \
            out.open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["record_a", "record_b", "decision", "confidence"])
        for r in csv.DictReader(f):
            a, b = r["record_a"], r["record_b"]
            if r["decision_status"] != "review" or side.get(a) != "test" or side.get(b) != "test":
                continue
            same = r["match_type"] in WRITE_TYPES
            w.writerow([a, b, "same" if same else "different", ""])
            n += 1
            n_same += same
    print(f"Wrote {n} test band decisions to {out} ({n_same} same, {n - n_same} different)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

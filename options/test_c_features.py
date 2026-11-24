"""
Rename unit test for options/c_features.py.

Renames every content column to col1..colN in scored_pairs_columns.json and in the per-field
JSON of every scored_pairs.csv row (also shuffling key order), then checks the features are
identical for every row. This tests Option C's feature code only: Verity itself embeds
"<column name>:<value>", so a renamed input CSV changes Verity's scores (see the planned
end-to-end rename test on dev seed 2026).

  python options/test_c_features.py results/<run> [--rows N]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_features import FEATURE_NAMES, FeatureBuilder, same_features  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--rows", type=int, default=0, help="Check only the first N rows (0 = all)")
    args = p.parse_args()
    csv.field_size_limit(10**8)
    data = args.run / "verity_data"
    doc = json.loads((data / "scored_pairs_columns.json").read_text(encoding="utf-8"))
    cols = sorted(doc["field_semantic_types"])
    rng = random.Random(0)
    new_names = [f"col{i}" for i in range(1, len(cols) + 1)]
    rng.shuffle(new_names)
    rename = dict(zip(cols, new_names))
    doc2 = dict(doc)
    doc2["field_semantic_types"] = {rename[k]: v for k, v in doc["field_semantic_types"].items()}
    fb, fb2 = FeatureBuilder(doc), FeatureBuilder(doc2)

    def renamed(js: str) -> str:
        if not js:
            return js
        d = json.loads(js)
        items = [(rename[k], v) for k, v in d.items()]
        rng.shuffle(items)
        return json.dumps(dict(items))

    n = bad = 0
    with (data / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row2 = dict(row)
            for c in ("field_similarities", "field_weights", "evidence_field_similarities"):
                row2[c] = renamed(row[c])
            if not same_features(fb.features(row), fb2.features(row2)):
                bad += 1
            n += 1
            if args.rows and n >= args.rows:
                break
    print(f"rename test: {n} rows, {len(FEATURE_NAMES)} features, rename {rename}")
    print(f"{'PASS' if bad == 0 else 'FAIL'}: {bad} row(s) with different features after renaming")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

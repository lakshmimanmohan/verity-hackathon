"""
Column names with spaces and mixed case ("Company Name", "Street Name", "Postal Code") must reach the scorer.

The record store keeps attributes under snake_case keys (company_name), so every column-keyed lookup in matching goes through
column_classification.col_key. This test runs Verity end to end on a small synthetic file (no real data) with such names and
three near-duplicate records, and checks that neighbor pairs were scored and none was skipped for "no NAME/DESCRIPTIVE/STREET_ADDRESS value".

  python eval/test_verity_key_normalization.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "verity" / "src"))
from column_classification import col_key  # noqa: E402

WORDS = ["Amber", "Birch", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor", "Indigo", "Juniper", "Kestrel", "Lumen"]
KINDS = ["Logistics", "Foods", "Partners", "Systems", "Textiles", "Marine", "Energy", "Dental"]
STREETS = ["Oak Avenue", "Mill Road", "River Street", "Hill Lane", "Park Way", "Lake Drive"]


def records() -> list[list[str]]:
    rows = []
    for i in range(48):
        w, k, s = WORDS[i % len(WORDS)], KINDS[(i * 3 + i // 12) % len(KINDS)], STREETS[(i * 5) % len(STREETS)]
        rows.append([f"R{i:03d}", f"{w} {k} Holdings {i}", f"{100 + i * 7} {s}", f"{10000 + i * 137}"])
    # three near-duplicates of one organization (not exact: spelling and number differ)
    rows.append(["R100", "Zephyr Optical Systems Incorporated", "4521 Cedar Boulevard", "55102"])
    rows.append(["R101", "Zephyr Optical Systems Inc", "4521 Cedar Blvd", "55102"])
    rows.append(["R102", "Zephyr Optical System Incorporated", "4521 Cedar Boulevard Suite 2", "55102"])
    return rows


class KeyTests(unittest.TestCase):
    def test_col_key_is_snake_lower_and_idempotent(self):
        self.assertEqual([col_key(c) for c in ("Company Name", "Street Name", "Postal Code")], ["company_name", "street_name", "postal_code"])
        self.assertEqual(col_key(col_key("House Number1")), col_key("House Number1"))
        self.assertEqual(col_key("city"), "city")

    def test_spaced_mixed_case_columns_reach_the_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            src = d / "spaced.csv"
            with src.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Id", "Company Name", "Street Name", "Postal Code"])
                w.writerows(records())
            data = d / "verity_data"
            env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUTF8="1", VERITY_DATA_DIR=str(data))
            r = subprocess.run([sys.executable, str(REPO / "verity" / "src" / "main.py"), "--nn-mode", "exact", "--input", str(src)],
                               env=env, cwd=str(d), capture_output=True, text=True, encoding="utf-8", timeout=900)
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            ri = json.loads((data / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
            self.assertGreater(ri["neighbor_pairs"], 0)
            self.assertEqual(ri["skipped_no_semantic_field"], 0)
            self.assertGreater(ri["scored_neighbor_pairs"], 0)
            with (data / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            near = {"R100", "R101", "R102"}
            hit = [x for x in rows if x["record_a"] in near and x["record_b"] in near and x["field_similarities"].strip()]
            self.assertTrue(hit, "no scored pair among the three near-duplicates")
            for x in hit:
                keys = set(json.loads(x["field_similarities"]))              # the fields the two records share exactly, in snake_case
                self.assertIn("postal_code", keys)
                self.assertLessEqual(keys, {"company_name", "street_name", "postal_code"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

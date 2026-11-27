"""
Tests for the code path in verity/src/match.py (pairs that fail the multi-word test are compared on code-like columns).

Synthetic data only, arbitrary column names ("Item Code", "GTIN", "Vendor"). The unit tests use hand-computed numbers; the end-to-end tests run Verity
on tiny files. The company-style test runs the source as it was before the change (git commit d51c5b9) and the current source on the same file and
requires byte-identical scored_pairs.csv and match_candidates.csv.

  python eval/test_verity_code_path.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "verity" / "src"))
os.environ.setdefault("VERITY_DATA_DIR", str(Path(tempfile.gettempdir()) / "verity_code_path_import"))
import match as M  # noqa: E402

BASELINE_COMMIT = "d51c5b9"      # the last change to verity/src before the code path


class UnitTests(unittest.TestCase):
    def test_normalize_drops_case_spaces_and_punctuation(self):
        self.assertEqual(M.code_normalize(" Ab-12_3.4 /x "), "ab1234x")
        self.assertEqual(M.code_normalize("---"), "")

    def test_code_like_columns_from_types_and_value_statistics(self):
        rows = {str(i): {"a": f"x{i}", "b": "two words" if i < 5 else f"y{i}", "c": "two words", "d": ""} for i in range(10)}
        sem = {"a": "DESCRIPTIVE", "b": "DESCRIPTIVE", "c": "NAME", "d": "DESCRIPTIVE", "e": "QUASI_IDENTIFIER"}
        got = M.code_like_columns(sem, rows, frozenset({"a", "b", "c", "d", "e"}))
        # a: 100% no space; b: 50% (not enough); c: 0%; d: no values; e: QUASI_IDENTIFIER by type
        self.assertEqual(got, frozenset({"a", "e"}))
        rows9 = {str(i): {"a": "x y" if i == 0 else f"x{i}"} for i in range(10)}
        self.assertEqual(M.code_like_columns({"a": "DESCRIPTIVE"}, rows9, frozenset({"a"})), frozenset({"a"}))   # exactly 90% passes

    def test_pair_scoring_by_hand(self):
        disc = {"id1": 0.6, "id2": 0.3, "cat": 0.1}
        uniq = {"id1": 0.9, "id2": 0.7, "cat": 0.1}
        cols = frozenset({"id1", "id2", "cat"})
        a = {"id1": "AB-12", "id2": "xy9", "cat": "Acme", "other": "zzz"}
        b = {"id1": "ab_12", "id2": "xy8", "cat": "ACME"}
        r = M.code_path_pair(a, b, cols, disc, uniq)
        self.assertAlmostEqual(r["weights"]["id1"], 0.6)
        self.assertAlmostEqual(r["evidence_mass"], 0.7)                    # id1 and cat equal: 0.6 + 0.1
        self.assertAlmostEqual(r["conflict_mass"], 0.3)                    # id2 differs and its uniqueness 0.7 >= 0.5
        sim_id2 = 2 / 4                                                    # {x,y,9} vs {x,y,8}: 2 common of 4
        self.assertAlmostEqual(r["sims"]["id2"], sim_id2)
        self.assertAlmostEqual(r["score_uncapped"], 0.6 + 0.3 * sim_id2 + 0.1)
        self.assertFalse(r["passes"])                                      # conflict 0.3 > 0.25
        self.assertIsNone(M.code_path_pair({"id1": "x"}, {"id2": "x"}, cols, disc, uniq))   # no shared code-like value
        self.assertIsNone(M.code_path_pair({"id1": "--"}, {"id1": "--"}, cols, disc, uniq))  # empty after normalizing

    def test_low_uniqueness_difference_is_not_a_conflict(self):
        r = M.code_path_pair({"id1": "k1", "cat": "red"}, {"id1": "k1", "cat": "blue"}, frozenset({"id1", "cat"}), {"id1": 0.5, "cat": 0.5}, {"id1": 0.9, "cat": 0.2})
        self.assertAlmostEqual(r["conflict_mass"], 0.0)
        self.assertAlmostEqual(r["evidence_mass"], 0.5)
        self.assertTrue(r["passes"])

    def test_equal_weights_when_all_learned_weights_are_zero(self):
        r = M.code_path_pair({"a": "x", "b": "y"}, {"a": "x", "b": "z"}, frozenset({"a", "b"}), {}, {})
        self.assertEqual(r["weights"], {"a": 0.5, "b": 0.5})


def run_verity(src_dir: Path, csv_path: Path, data: Path, cwd: Path) -> dict:
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUTF8="1", VERITY_DATA_DIR=str(data))
    r = subprocess.run([sys.executable, str(src_dir / "main.py"), "--nn-mode", "exact", "--input", str(csv_path)], env=env, cwd=str(cwd),
                       capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return json.loads((data / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def code_rows() -> list[list[str]]:
    vendors = ["Acme", "Borealis", "Cygnus", "Delta", "Eon", "Fulcrum"]
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    rows = []
    for i in range(50):
        item = f"{letters[i % 24]}{letters[(i * 7 + 3) % 24]}-{1000 + i * 37}"
        gtin = f"{4000000000000 + i * 1234567891}"
        if i in (1, 3):                                                  # two repeated codes keep the column below the primary-key cut (99% unique)
            item = rows[i - 1][1]
        rows.append([f"R{i:03d}", item, gtin, vendors[i % 6]])
    rows += [
        ["P1", "AB-9234", "4006381333931", "Orbit"], ["P2", "ab_9234", "4006-381-333931", "ORBIT"],        # same product, case and punctuation differ
        ["Q1", "KT-5521", "5012345678900", "Zenith"], ["Q2", "MX-9086", "7312345678123", "Zenith"],        # same vendor, different unique codes
        ["R1", "CD-2468", "8712345000017", "Vega"], ["R2", "CD-8642", "8712345000017", "Vega"],            # same gtin and vendor, conflicting unique code
    ]
    return rows


class EndToEnd(unittest.TestCase):
    def test_code_columns_match_blocks_and_counters(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            src = d / "codes.csv"
            with src.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Id", "Item Code", "GTIN", "Vendor"])
                w.writerows(code_rows())
            data = d / "verity_data"
            ri = run_verity(REPO / "verity" / "src", src, data, d)
            self.assertGreater(ri["code_path_pairs"], 0)
            self.assertEqual(ri["text_path_pairs"], 0)
            self.assertEqual(ri["skipped_single_token_semantic_fields"], ri["code_path_pairs"] + ri["skipped_no_code_field"])
            self.assertEqual(ri["scored_neighbor_pairs"], ri["text_path_pairs"] + ri["code_path_pairs"])
            self.assertTrue(ri["code_like_columns"])
            sp = {(r["record_a"], r["record_b"]): r for r in read_csv(data / "scored_pairs.csv")}
            mc = {(r["record_a"], r["record_b"]): r for r in read_csv(data / "match_candidates.csv")}
            p = sp[("P1", "P2")]                                                       # case and punctuation differences in a code match
            self.assertEqual((p["match_type"], p["decision_status"]), ("code_signal", "match"))
            self.assertIn(("P1", "P2"), mc)
            self.assertEqual(mc[("P1", "P2")]["match_type"], "code_signal")
            fs, fw, efs = json.loads(p["field_similarities"]), json.loads(p["field_weights"]), json.loads(p["evidence_field_similarities"])
            self.assertEqual(set(fs), set(fw))
            self.assertEqual(set(fs), set(efs))
            self.assertAlmostEqual(sum(fw.values()), 1.0, places=5)
            self.assertAlmostEqual(float(p["score"]), min(1.0, sum(fw[k] * fs[k] for k in fs)), places=5)   # field-only score, no cosine
            q = sp.get(("Q1", "Q2"))                                                    # same vendor, different unique codes: no match
            self.assertNotIn(("Q1", "Q2"), mc)
            if q:
                self.assertNotEqual(q["match_type"], "code_signal")
            r = sp[("R1", "R2")]                                                        # a conflicting unique code blocks the match
            self.assertGreater(float(r["score"]), 0.82)                                 # the score alone would pass the threshold
            self.assertEqual(r["match_type"], "insufficient_evidence")
            self.assertEqual(r["evidence_passes"], "false")
            self.assertGreater(float(r["conflict_mass"]), M.MAX_CONFLICT_MASS)
            self.assertNotIn(("R1", "R2"), mc)

    def test_company_style_file_is_identical_with_and_without_the_change(self):
        words = ["Amber", "Birch", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor", "Indigo", "Juniper", "Kestrel", "Lumen"]
        kinds = ["Logistics", "Foods", "Partners", "Systems", "Textiles", "Marine", "Energy", "Dental"]
        streets = ["Oak Avenue", "Mill Road", "River Street", "Hill Lane", "Park Way", "Lake Drive"]
        rows = [[f"R{i:03d}", f"{words[i % 12]} {kinds[(i * 3 + i // 12) % 8]} Holdings {i}", f"{100 + i * 7} {streets[(i * 5) % 6]}", f"{10000 + i * 137}"] for i in range(48)]
        rows += [["R100", "Zephyr Optical Systems Incorporated", "4521 Cedar Boulevard", "55102"], ["R101", "Zephyr Optical Systems Inc", "4521 Cedar Blvd", "55102"],
                 ["R102", "Zephyr Optical System Incorporated", "4521 Cedar Boulevard Suite 2", "55102"]]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            src = d / "company.csv"
            with src.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Id", "Company Name", "Street Name", "Postal Code"])
                w.writerows(rows)
            base = d / "baseline"
            base.mkdir()
            tar = subprocess.run(["git", "archive", BASELINE_COMMIT, "verity/src"], cwd=str(REPO), capture_output=True)
            self.assertEqual(tar.returncode, 0, tar.stderr)
            subprocess.run(["tar", "-x", "-C", str(base)], input=tar.stdout, check=True)
            ri_old = run_verity(base / "verity" / "src", src, d / "old_data", d)
            ri_new = run_verity(REPO / "verity" / "src", src, d / "new_data", d)
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ("scored_pairs.csv", "match_candidates.csv"):
                self.assertEqual(sha(d / "old_data" / name), sha(d / "new_data" / name), name)
            self.assertEqual(ri_new["code_path_pairs"], 0)
            self.assertEqual(ri_new["skipped_no_code_field"], 0)
            self.assertEqual(ri_new["text_path_pairs"], ri_new["scored_neighbor_pairs"])
            self.assertEqual(ri_old["scored_neighbor_pairs"], ri_new["scored_neighbor_pairs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

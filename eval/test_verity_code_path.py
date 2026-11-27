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
import inspect
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
        code = frozenset({"id1", "id2"})
        content = frozenset({"id1", "id2", "cat"})
        a = {"id1": "AB-12", "id2": "xy9", "cat": "Acme", "other": "zzz"}
        b = {"id1": "ab_12", "id2": "xy8", "cat": "ACME"}
        r = M.code_path_pair(a, b, code, content, disc, frozenset({"id1", "id2"}))
        self.assertAlmostEqual(r["weights"]["id1"], 0.6)
        self.assertAlmostEqual(r["evidence_mass"], 0.7)                    # id1 and cat equal after normalizing: 0.6 + 0.1
        self.assertAlmostEqual(r["conflict_mass"], 0.3)                    # id2 is code-like, present in both, and differs
        self.assertTrue(r["conflict"])
        self.assertAlmostEqual(r["sims"]["id2"], 2 * 2 / 6)                # SequenceMatcher: "xy" matches, 4 of 6 characters
        self.assertAlmostEqual(r["score_uncapped"], 0.6 + 0.3 * (2 * 2 / 6) + 0.1)
        self.assertTrue(r["passes"])                                       # evidence 0.7 >= MIN_EVIDENCE_MASS; the conflict is a separate rule
        r2 = M.code_path_pair(a, b, code, content, disc, frozenset({"id1"}))                      # id2 differs but is not identifying: no conflict
        self.assertFalse(r2["conflict"])
        self.assertAlmostEqual(r2["conflict_mass"], 0.0)
        self.assertIsNone(M.code_path_pair({"id1": "x"}, {"id2": "x"}, code, content, disc))      # no shared code-like value
        self.assertIsNone(M.code_path_pair({"id1": "--"}, {"id1": "--"}, code, content, disc))    # empty after normalizing
        self.assertIsNone(M.code_path_pair({"cat": "x"}, {"cat": "x"}, code, content, disc))      # shared, but not in a code-like column

    def test_order_aware_similarity_is_not_a_character_set(self):
        r = M.code_path_pair({"c": "AB1234"}, {"c": "AB4321"}, frozenset({"c"}), frozenset({"c"}), {"c": 1.0}, frozenset({"c"}))
        self.assertAlmostEqual(r["sims"]["c"], 0.5)                        # same characters, different order: not 1.0
        self.assertTrue(r["conflict"])

    def test_difference_in_a_field_that_is_not_code_like_is_not_a_conflict(self):
        r = M.code_path_pair({"id1": "k1", "cat": "red"}, {"id1": "k1", "cat": "blue"}, frozenset({"id1"}), frozenset({"id1", "cat"}), {"id1": 0.5, "cat": 0.5})
        self.assertFalse(r["conflict"])
        self.assertAlmostEqual(r["conflict_mass"], 0.0)
        self.assertAlmostEqual(r["evidence_mass"], 0.5)
        self.assertTrue(r["passes"])

    def test_identifying_columns_cut_is_one_over_two_n(self):
        w = {"a": 0.5, "b": 0.1, "c": 0.0999, "d": 0.3001}
        self.assertEqual(M.identifying_columns(frozenset({"a", "b", "c"}), w, 5), frozenset({"a", "b"}))     # cut 1/10: 0.1 counts, 0.0999 does not
        self.assertEqual(M.identifying_columns(frozenset({"a", "b", "c", "d"}), w, 2), frozenset({"a", "d"}))  # cut 1/4
        self.assertEqual(M.identifying_columns(frozenset({"a"}), w, 0), frozenset())

    def test_labels_conflict_only_when_it_blocks_a_pass_or_the_band(self):
        L = M.code_path_label
        self.assertEqual(L(0.95, 0.82, True, False), ("code_signal", True))
        self.assertEqual(L(0.95, 0.82, True, True), ("code_conflict", True))           # would have matched: conflict, evidence_passes marks it
        self.assertEqual(L(0.80, 0.82, False, True), ("code_conflict", False))         # in the band, below the threshold: conflict, review by band
        self.assertEqual(L(0.90, 0.82, False, True), ("insufficient_evidence", False))  # above the band, evidence fails: ordinary label
        self.assertEqual(L(0.50, 0.82, True, True), ("below_threshold", True))         # far below the band: ordinary label
        self.assertEqual(L(0.50, 0.82, False, False), ("below_threshold", False))

    def test_threshold_exact_score_does_not_match_on_either_path(self):
        self.assertEqual(M.code_path_label(0.82, 0.82, True, False)[0], "below_threshold")        # code path: strictly greater
        self.assertEqual(M.code_path_label(0.82, 0.82, True, True)[0], "code_conflict")           # a conflict in the band is still not a match
        self.assertNotEqual(M.code_path_label(0.82, 0.82, True, False)[0], "code_signal")
        self.assertEqual(M.code_path_label(0.820001, 0.82, True, False)[0], "code_signal")
        src = inspect.getsource(M.main)                                                           # text path: the unchanged comparison is also strict
        self.assertIn("elif score <= effective_score_threshold:", src)
        self.assertIn("score, effective_score_threshold, evidence_passes", src)                  # the code path passes the same threshold variable

    def test_equal_weights_when_all_learned_weights_are_zero(self):
        r = M.code_path_pair({"a": "x", "b": "y"}, {"a": "x", "b": "z"}, frozenset({"a", "b"}), frozenset({"a", "b"}), {})
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
        ["T1", "EF-7777", "6001234567895", "Lyra"], ["T2", "ef 7777", "6001234567895", "Altair"],          # same codes, conflicting vendor
        ["U1", "GH-3141", "9001234567893", "Rigel"], ["U2", "GH-3142", "9001234567893", "Rigel"],          # one character off in the unique code
        ["V1", "AB1234", "9501234567899", "Sirius"], ["V2", "AB4321", "9501234567899", "Sirius"],          # same characters, other order
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
            self.assertEqual(ri["identifying_columns"], ["gtin", "item_code"])         # vendor has a low learned weight: not identifying
            t = sp[("T1", "T2")]                                                        # same codes, conflicting low-weight vendor: still a match
            self.assertEqual((t["match_type"], t["decision_status"]), ("code_signal", "match"))
            self.assertIn(("T1", "T2"), mc)
            self.assertAlmostEqual(float(t["conflict_mass"]), 0.0)
            q = sp[("Q1", "Q2")]                                                        # same vendor, different unique codes, far below the band: ordinary label
            self.assertNotIn(("Q1", "Q2"), mc)
            self.assertEqual((q["match_type"], q["decision_status"]), ("below_threshold", "non_match"))
            u = sp[("U1", "U2")]                                                        # one character off in a high-weight code: blocks a match, review
            self.assertEqual((u["match_type"], u["decision_status"], u["evidence_passes"]), ("code_conflict", "review", "true"))
            self.assertGreater(float(u["score"]), 0.87)
            self.assertNotIn(("U1", "U2"), mc)
            for pair in (("R1", "R2"), ("V1", "V2")):                                     # the unique code differs a lot: score in the band, never a match
                r = sp[pair]
                self.assertEqual((r["match_type"], r["decision_status"], r["evidence_passes"]), ("code_conflict", "review", "false"), pair)
                self.assertLess(float(r["score"]), 0.82, pair)
                self.assertNotIn(pair, mc)
            v = json.loads(sp[("V1", "V2")]["field_similarities"])
            self.assertAlmostEqual(min(v.values()), 0.5, places=5)                    # AB1234 vs AB4321 scores 0.5, not 1.0

    def test_same_postal_code_and_house_number_with_a_different_name_does_not_match(self):
        first = ["Amber", "Birch", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor"]
        second = ["Holdings", "Partners", "Systems", "Logistics", "Textiles", "Marine"]
        rows = [[f"A{i:03d}", f"{first[i % 8]} {second[(i * 5) % 6]} {i}", str(10 + i * 3), f"{20000 + (i % 25) * 111}"] for i in range(60)]
        rows[1][1] = rows[0][1]                                                         # repeated names keep the name column below the primary-key cut (99% unique)
        rows[3][1] = rows[2][1]
        rows += [["S1", "Quill", "4521", "55102"], ["S2", "Rowan Cartage Limited", "4521", "55102"]]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            src = d / "addr.csv"
            with src.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Id", "Party Label", "House No", "Zip Area"])
                w.writerows(rows)
            ri = run_verity(REPO / "verity" / "src", src, d / "data", d)
            sp = {(r["record_a"], r["record_b"]): r for r in read_csv(d / "data" / "scored_pairs.csv")}
            mc = {(r["record_a"], r["record_b"]) for r in read_csv(d / "data" / "match_candidates.csv")}
            self.assertGreater(ri["code_path_pairs"], 0)
            r = sp[("S1", "S2")]
            self.assertNotEqual(r["decision_status"], "match")
            self.assertNotEqual(r["match_type"], "code_signal")
            self.assertNotIn(("S1", "S2"), mc)
            fs = json.loads(r["field_similarities"])
            self.assertEqual(sorted(fs.values())[-2:], [1.0, 1.0])                    # house number and postal code equal; the name is compared too and is not
            self.assertLess(min(fs.values()), 0.5)

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

"""
Tests for option D's routing and selection (options/d_hybrid.py): no API, no truth.

  python options/test_d_hybrid.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d_hybrid as D  # noqa: E402


class RouteTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(D.route_of(0.8999, 0.05, 0.9), "llm")
        self.assertEqual(D.route_of(0.9, 0.05, 0.9), "c_same")            # p >= hi: same without the LLM
        self.assertEqual(D.route_of(0.05, 0.05, 0.9), "llm")              # p >= lo: to the LLM
        self.assertEqual(D.route_of(0.0499999, 0.05, 0.9), "c_different")
        self.assertEqual(D.route_of(1.0, 0.05, None), "llm")              # hi none: nothing is decided same by C
        self.assertEqual(D.route_of(0.0, 0.005, None), "c_different")

    def test_grid_has_only_valid_settings(self):
        s = D.valid_settings()
        self.assertTrue(all(hi is None or hi > lo for lo, hi in s))
        self.assertEqual(len(s), 8 * 4)                                   # hi 0.9 > every lo, so all 32 are valid
        self.assertIn((0.5, None), s)
        self.assertIn((0.5, 0.9), s)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.c = {("a", "b"): ("different", 0.0), ("c", "d"): ("same", 0.3), ("e", "f"): ("same", 0.97),
                  ("g", "h"): ("same", 0.6)}
        self.b = {("a", "b"): ("different", 0.01), ("c", "d"): ("same", 0.9), ("e", "f"): ("different", 0.2),
                  ("g", "h"): ("different", 0.15)}
        self.sample = [(("a", "b"), 9.5), (("c", "d"), 1.0), (("e", "f"), 1.0), (("g", "h"), 1.0)]

    def test_routes_decisions_and_confidence(self):
        rows = {(r["record_a"], r["record_b"]): r for r in D.build(self.c, self.b, self.sample, 0.05, 0.9)}
        self.assertEqual(rows[("a", "b")]["route"], "c_different")
        self.assertEqual((rows[("a", "b")]["decision"], rows[("a", "b")]["confidence"]), ("different", 0.0))   # C's p
        self.assertEqual(rows[("c", "d")]["route"], "llm")
        self.assertEqual((rows[("c", "d")]["decision"], rows[("c", "d")]["confidence"]), ("same", 0.9))       # B's p_same
        self.assertEqual(rows[("e", "f")]["route"], "c_same")
        self.assertEqual((rows[("e", "f")]["decision"], rows[("e", "f")]["confidence"]), ("same", 0.97))      # C overrides B
        self.assertEqual(rows[("g", "h")]["route"], "llm")
        self.assertEqual(rows[("g", "h")]["decision"], "different")                                           # LLM takes its decision

    def test_fails_when_a_routed_pair_has_no_b_answer(self):
        del self.b[("c", "d")]
        with self.assertRaises(SystemExit) as cm:
            D.build(self.c, self.b, self.sample, 0.05, 0.9)
        self.assertIn("no recorded B answer", str(cm.exception))
        rows = D.build(self.c, self.b, self.sample, 0.5, 0.9)             # not routed at lo 0.5 (p 0.3): fine
        self.assertEqual([r["route"] for r in rows][1], "c_different")

    def test_fails_when_a_sampled_pair_has_no_c_decision(self):
        with self.assertRaises(SystemExit):
            D.build({k: v for k, v in self.c.items() if k != ("a", "b")}, self.b, self.sample, 0.05, 0.9)

    def test_llm_share_is_weighted(self):
        rows = D.build(self.c, self.b, self.sample, 0.05, 0.9)
        st = D.llm_stats(rows, 12.5)
        self.assertEqual(st["sampled_llm_calls"], 2)
        self.assertAlmostEqual(st["weighted_llm_share"], 2.0 / 12.5)
        self.assertAlmostEqual(st["projected_calls"], 2.0)

    def test_decisions_file_columns(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            D.write_decisions(tmp / "d.csv", D.build(self.c, self.b, self.sample, 0.05, None))
            with (tmp / "d.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), ["record_a", "record_b", "decision", "confidence", "route"])
            self.assertEqual(len(rows), 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class FullBandTests(unittest.TestCase):
    def test_routed_pairs_keep_the_files_orientation_and_use_lo_hi(self):
        rows = [{"record_a": "z", "record_b": "a", "confidence": "0.6"}, {"record_a": "b", "record_b": "c", "confidence": "0.9"},
                {"record_a": "d", "record_b": "e", "confidence": "0.4999"}, {"record_a": "f", "record_b": "g", "confidence": "0.5"}]
        got = D.routed_pairs(rows, 0.5, 0.9)
        self.assertEqual([(a, b) for a, b, _ in got], [("f", "g"), ("z", "a")])        # 0.9 is c_same, 0.4999 c_different

    def test_merge_b_unions_runs_and_rejects_conflicting_answers(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            for name, rows in (("one", [("a", "b", "same", "0.9"), ("c", "d", "different", "0.1")]),
                               ("two", [("b", "a", "same", "0.9"), ("e", "f", "different", "0.2")])):
                with (tmp / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["record_a", "record_b", "decision", "confidence"])
                    w.writerows(rows)
            merged = D.merge_b([tmp / "one.csv", tmp / "two.csv", tmp / "missing.csv"])
            self.assertEqual(set(merged), {("a", "b"), ("c", "d"), ("e", "f")})
            with (tmp / "three.csv").open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([["record_a", "record_b", "decision", "confidence"], ["a", "b", "different", "0.1"]])
            with self.assertRaises(SystemExit):
                D.merge_b([tmp / "one.csv", tmp / "three.csv"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_full_band_build_needs_a_b_answer_for_every_routed_pair(self):
        c = {("a", "b"): ("same", 0.6), ("c", "d"): ("same", 0.95), ("e", "f"): ("different", 0.0)}
        with self.assertRaises(SystemExit):
            D.build(c, {}, [(k, 1.0) for k in sorted(c)], 0.5, 0.9)
        rows = D.build(c, {("a", "b"): ("different", 0.2)}, [(k, 1.0) for k in sorted(c)], 0.5, 0.9)
        self.assertEqual([(r["route"], r["decision"]) for r in rows], [("llm", "different"), ("c_same", "same"), ("c_different", "different")])


class FallbackOffTests(unittest.TestCase):
    def test_routed_pairs_take_c_own_decision_and_the_rest_follow_the_route(self):
        c = {("a", "b"): ("different", 0.0), ("c", "d"): ("different", 0.7), ("e", "f"): ("same", 0.85), ("g", "h"): ("same", 0.95), ("i", "j"): ("different", 0.49)}
        rows = {(r["record_a"], r["record_b"]): r for r in D.build_fallback_off(c, 0.5, 0.9)}
        self.assertEqual([rows[k]["route"] for k in sorted(rows)], ["c_different", "llm", "llm", "c_same", "c_different"])
        self.assertEqual({k: rows[k]["decision"] for k in rows}, {("a", "b"): "different", ("c", "d"): "different", ("e", "f"): "same",
                                                                 ("g", "h"): "same", ("i", "j"): "different"})
        self.assertTrue(all(rows[k]["decision"] == c[k][0] for k in rows))     # identical to C when C's cutoff lies between lo and hi

    def test_it_differs_from_c_when_the_cutoff_is_outside_lo_and_hi(self):
        c = {("a", "b"): ("same", 0.45), ("c", "d"): ("different", 0.95)}          # a cutoff of 0.4 or 1.0: C's decision is not the route's
        rows = {(r["record_a"], r["record_b"]): r for r in D.build_fallback_off(c, 0.5, 0.9)}
        self.assertEqual(rows[("a", "b")]["decision"], "different")                  # route c_different overrides C's "same"
        self.assertEqual(rows[("c", "d")]["decision"], "same")                       # route c_same overrides C's "different"


class SelectionTests(unittest.TestCase):
    def r(self, lo, hi, share, f1):
        return {"lo": lo, "hi": hi, "weighted_llm_share": share, "band_f1": f1}

    def test_smallest_share_wins_among_those_that_qualify(self):
        res = [self.r(0.1, 0.9, 0.05, 0.99), self.r(0.2, 0.9, 0.02, 0.985), self.r(0.5, 0.9, 0.01, 0.80),     # too low F1
               self.r(0.005, None, 0.30, 0.999)]                                                             # too much LLM
        best, ok = D.select(res, b_f1=0.995)
        self.assertEqual((best["lo"], best["hi"]), (0.2, 0.9))
        self.assertEqual(len(ok), 2)

    def test_ties_go_to_higher_f1_then_higher_lo(self):
        res = [self.r(0.01, 0.9, 0.03, 0.990), self.r(0.05, 0.9, 0.03, 0.990), self.r(0.02, 0.9, 0.03, 0.992)]
        self.assertEqual(D.select(res, 0.995)[0]["lo"], 0.02)             # higher F1
        res = [self.r(0.01, 0.9, 0.03, 0.990), self.r(0.05, 0.9, 0.03, 0.990)]
        self.assertEqual(D.select(res, 0.995)[0]["lo"], 0.05)             # same F1: higher lo

    def test_the_f1_floor_and_share_cap_are_inclusive_and_nothing_qualifying_gives_none(self):
        self.assertIsNotNone(D.select([self.r(0.1, 0.9, 0.20, 0.985)], 0.995)[0])     # F1 = B - 0.01 (0.985), share 0.20
        self.assertIsNone(D.select([self.r(0.1, 0.9, 0.2001, 0.99)], 0.995)[0])
        self.assertIsNone(D.select([self.r(0.1, 0.9, 0.1, 0.9849)], 0.995)[0])

    def test_rule_hash_ignores_line_endings(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "a.txt").write_bytes(b"x\ny\n")
            (tmp / "b.txt").write_bytes(b"x\r\ny\r\n")
            self.assertEqual(D.rule_hash(tmp / "a.txt"), D.rule_hash(tmp / "b.txt"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SettingTests(unittest.TestCase):
    """Two frozen settings: v2 (plain file names) and the original rule (suffix _original)."""

    def make_run(self, root: Path) -> Path:
        run = root / "run"
        d = run / "options" / "d_hybrid"
        d.mkdir(parents=True)
        cdir = run / "options" / "c_model" / "c_v1_lgbm"
        cdir.mkdir(parents=True)
        # C: five pairs with calibrated p 0.001 (below both), 0.01 (v2 only), 0.3 (v2 only), 0.6 (both), 0.95 (neither)
        with (cdir / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["record_a", "record_b", "decision", "confidence"])
            for i, (dec, p) in enumerate([("different", 0.001), ("different", 0.01), ("different", 0.3), ("same", 0.6), ("same", 0.95)]):
                w.writerow([f"a{i}", f"b{i}", dec, p])
        b = run / "options" / "b_llm" / "openai" / "sample5000"
        b.mkdir(parents=True)
        with (b / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["record_a", "record_b", "decision", "confidence"])
            for i in (1, 2, 3):
                w.writerow([f"a{i}", f"b{i}", "same" if i == 3 else "different", 0.9 if i == 3 else 0.1])
        (d / "frozen_settings.json").write_text(json.dumps({"selected": True, "lo": 0.5, "hi": 0.9}), encoding="utf-8")
        (d / "frozen_settings_v2.json").write_text(json.dumps({"selected": True, "lo": 0.005, "hi": 0.9}), encoding="utf-8")
        return run

    def test_file_names(self):
        self.assertEqual(D.setting_file("decisions_fullband.csv", "v2"), "decisions_fullband.csv")
        self.assertEqual(D.setting_file("decisions_fullband.csv", "original"), "decisions_fullband_original.csv")
        self.assertEqual(D.setting_file("fallback_off_check.json", "original"), "fallback_off_check_original.json")

    def test_each_setting_reads_its_own_settings_file(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            P = D.paths(self.make_run(tmp))
            self.assertEqual(D.frozen_lo_hi(P), (0.005, 0.9))                 # the default is v2
            self.assertEqual(D.frozen_lo_hi(P, "v2"), (0.005, 0.9))
            self.assertEqual(D.frozen_lo_hi(P, "original"), (0.5, 0.9))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_routed_lists_and_full_band_decisions_per_setting(self):
        import argparse
        import contextlib
        import io
        tmp = Path(tempfile.mkdtemp())
        try:
            run = self.make_run(tmp)
            d = run / "options" / "d_hybrid"
            with contextlib.redirect_stdout(io.StringIO()):
                for setting in ("v2", "original"):
                    self.assertEqual(D.cmd_list_routed(argparse.Namespace(run=run, setting=setting)), 0)
                    self.assertEqual(D.cmd_full(argparse.Namespace(run=run, setting=setting)), 0)
            v2 = {(r["record_a"], r["record_b"]) for r in D.read_csv(d / "routed_fullband_pairs.csv")}
            orig = {(r["record_a"], r["record_b"]) for r in D.read_csv(d / "routed_fullband_pairs_original.csv")}
            self.assertEqual(v2, {("a1", "b1"), ("a2", "b2"), ("a3", "b3")})  # p 0.01, 0.3, 0.6 (0.001 is below lo 0.005)
            self.assertEqual(orig, {("a3", "b3")})                            # p 0.6 only
            self.assertTrue(orig <= v2)                                       # the original rule's pairs are inside v2's: no new B call for the secondary row
            full_v2 = {(r["record_a"], r["record_b"]): r["route"] for r in D.read_csv(d / "decisions_fullband.csv")}
            full_orig = {(r["record_a"], r["record_b"]): r["route"] for r in D.read_csv(d / "decisions_fullband_original.csv")}
            self.assertEqual(sorted(k for k, r in full_v2.items() if r == "llm"), sorted(v2))
            self.assertEqual(full_orig[("a1", "b1")], "c_different")
            self.assertEqual(full_v2[("a1", "b1")], "llm")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fallback_off_check_per_setting_uses_its_own_files(self):
        import argparse
        import contextlib
        import io
        tmp = Path(tempfile.mkdtemp())
        try:
            run = self.make_run(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(D.cmd_full_off(argparse.Namespace(run=run, setting="v2")), 0)
            d = run / "options" / "d_hybrid"
            self.assertTrue((d / "fallback_off_check.json").exists())
            self.assertFalse((d / "fallback_off_check_original.json").exists())
            chk = json.loads((d / "fallback_off_check.json").read_text(encoding="utf-8"))
            self.assertEqual((chk["lo"], chk["hi"]), (0.005, 0.9))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_committed_v2_settings_match_the_committed_rule_file(self):
        dev = HERE.parent / "results" / "dev_2026_run11" / "options" / "d_hybrid"
        fz = json.loads((dev / "frozen_settings_v2.json").read_text(encoding="utf-8"))
        self.assertEqual((fz["lo"], fz["hi"]), (0.005, 0.9))
        self.assertEqual(fz["rule_file"], "selection_rule_v2.txt")
        self.assertEqual(fz["rule_file_sha256_lf"], D.rule_hash(dev / "selection_rule_v2.txt"))
        orig = json.loads((dev / "frozen_settings.json").read_text(encoding="utf-8"))
        self.assertEqual((orig["lo"], orig["hi"]), (0.5, 0.9))
        self.assertEqual(orig["rule_file_sha256_lf"], D.rule_hash(dev / "selection_rule.txt"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

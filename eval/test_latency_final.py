"""
Tests for eval/latency_final.py on a small hand-computed case: percentiles, the per-record view of routed pairs, B's time per record,
the inventory of saved latency arrays, the reading rules and the single output file.

  python eval/test_latency_final.py
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import latency_final as LF  # noqa: E402


class MathTests(unittest.TestCase):
    def test_percentile_linear_interpolation_hand_computed(self):
        xs = [5.0, 1.0, 3.0, 2.0, 4.0]
        self.assertEqual(LF.percentile(xs, 50), 3.0)
        self.assertAlmostEqual(LF.percentile(xs, 95), 4.8)                 # position 4 x 0.95 = 3.8 -> 4 + 0.8 x (5 - 4)
        self.assertAlmostEqual(LF.percentile(xs, 99), 4.96)
        self.assertEqual(LF.percentile(xs, 100), 5.0)
        self.assertEqual(LF.percentile(xs, 0), 1.0)
        self.assertEqual(LF.percentile([7.0], 99.9), 7.0)
        with self.assertRaises(ValueError):
            LF.percentile([], 50)

    def test_per_record_hand_computed(self):
        # six test records T1..T6 (and two train records elsewhere); three routed pairs; 12 test band pairs; 2 band pairs per record
        test = {f"T{i}" for i in range(1, 7)}
        routed = [("T1", "T2"), ("T1", "T3"), ("T4", "T5")]
        pr = LF.per_record(routed, test, n_band_pairs=12, band_pairs_per_record=2.0)
        self.assertEqual((pr["records"], pr["routed_pairs"]), (6, 3))
        self.assertAlmostEqual(pr["pair_share"], 0.25)                                   # 3 / 12
        self.assertAlmostEqual(pr["endpoints_per_record"], 1.0)                          # 2 x 3 / 6
        self.assertAlmostEqual(pr["pairs_once_per_record"], 0.5)                         # 3 / 6
        self.assertEqual(pr["records_with_any"], 5)                                      # T1..T5
        self.assertAlmostEqual(pr["share_with_any"], 5 / 6)
        self.assertEqual(pr["max_per_record"], 2)                                        # T1 is in two routed pairs
        self.assertEqual(pr["histogram"], {1: 4, 2: 1})
        self.assertAlmostEqual(pr["expected_share_user"], 1 - 0.75 ** 2)                 # 0.4375
        self.assertAlmostEqual(pr["exponent_degree"], 4.0)                               # 2 x 12 / 6
        self.assertAlmostEqual(pr["expected_share_degree"], 1 - 0.75 ** 4)               # 0.68359375
        self.assertEqual(pr["records_outside_test"], 0)
        self.assertEqual(LF.per_record([("T1", "X9")], test, 12, 2.0)["records_outside_test"], 1)
        self.assertEqual(LF.per_record([], test, 12, 2.0)["records_with_any"], 0)

    def test_b_per_record_hand_computed(self):
        self.assertEqual(LF.b_per_record(20, 1.0), (2.0, 2.0))                           # 20 x 1 / 10 = 2; 2 rounds x 1 s
        mean_s, rounds_s = LF.b_per_record(20.05, 1.0)
        self.assertAlmostEqual(mean_s, 2.005)
        self.assertEqual(rounds_s, 3.0)                                                  # ceil(2.005) = 3 rounds
        self.assertEqual(LF.b_per_record(20, 2.0, workers=5), (8.0, 8.0))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "results" / "eval_7919"
        self.freeze = self.root / "results" / "freeze_2026-11-25"
        self.dev = self.root / "results" / "dev_run" / "concurrency.json"
        self.out = self.run / "latency_final"
        self.write_all()
        LF.REPO_BACKUP, LF.REPO = LF.REPO, self.root

    def tearDown(self):
        LF.REPO = LF.REPO_BACKUP
        self.tmp.cleanup()

    def w(self, p: Path, text: str) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def write_all(self):
        O = self.run / "options"
        rep = {"p50_ms": 0.08, "p95_ms": 0.28, "max_ms": 2.0, "mean_ms": 0.1, "pairs": 12, "total_s": 1.0}
        self.w(O / "latency" / "latency_summary.json", json.dumps({"pairs": 12, "C": {"repeats": [rep, rep, rep], "median_of_repeats": rep}}))
        calls = "\n".join(json.dumps({"latency_s": x, "error": None}) for x in (1.0, 2.0, 3.0, 4.0, 5.0)) + "\n"
        for lab in ("sample5000", "dfull", "decoycheck"):
            self.w(O / "b_llm" / "openai" / lab / "calls.jsonl", calls)
        (self.freeze / "b_archive").mkdir(parents=True)
        with gzip.open(self.freeze / "b_archive" / "calls.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(calls)
        self.w(self.run / "split_records.csv", "record_id,side\n" + "".join(f"T{i},test\n" for i in range(1, 7)) + "R1,train\nR2,train\n")
        self.w(O / "d_hybrid" / "routed_fullband_pairs.csv", "record_a,record_b,weight\nT1,T2,1.0\nT1,T3,1.0\nT4,T5,1.0\n")
        self.w(O / "c_model" / "c_v1_lgbm" / "decisions.csv", "record_a,record_b,decision,confidence\n" + "".join(f"T1,T{i % 5 + 2},different,0.1\n" for i in range(12)))
        self.w(self.run / "band_summary.json", json.dumps({"records": 10, "band_pairs": 20}))               # 2.0 band pairs per record
        self.w(O / "b_llm" / "openai" / "sample5000" / "timings.json", json.dumps({"timings": [{"stage": "optionB.llm_call", "per_item_ms": {"p50": 1000.0, "p95": 1500.0}}]}))
        self.w(self.dev, json.dumps({"levels": [{"workers": 2, "latency_p50_s": 9.0, "latency_p95_s": 9.9}, {"workers": 10, "latency_p50_s": 1.0, "latency_p95_s": 2.0}]}))

    def run_build(self):
        R = LF.Reader([self.run, self.freeze], [self.dev])
        return LF.build(self.run, self.freeze, self.dev, R), R


class BuildTests(Fixture):
    def test_inventory_says_no_array_and_lists_the_b_calls(self):
        res, _ = self.run_build()
        self.assertEqual(res["c_arrays"], [])
        self.assertFalse(res["d_sim_possible"])
        self.assertEqual(res["c_summary"]["pairs"], 12)
        sample = res["b_calls"]["sample5000"]
        self.assertEqual((sample["n"], sample["p50"], sample["max"]), (5, 3.0, 5.0))
        self.assertAlmostEqual(sample["p95"], 4.8)
        self.assertEqual(res["b_calls"]["sample5000 archive"]["n"], 5)

    def test_an_array_in_the_latency_folder_is_found(self):
        (self.run / "options" / "latency" / "c_per_pair_ms.npz").write_bytes(b"x")
        res, _ = self.run_build()
        self.assertEqual(res["c_arrays"], ["results/eval_7919/options/latency/c_per_pair_ms.npz"])
        self.assertTrue(res["d_sim_possible"])

    def test_per_record_and_b_from_the_files(self):
        res, _ = self.run_build()
        pr = res["per_record"]
        self.assertEqual((pr["records"], pr["routed_pairs"], pr["band_pairs"], pr["records_with_any"]), (6, 3, 12, 5))
        self.assertAlmostEqual(pr["exponent_user"], 2.0)                                  # 20 band pairs / 10 records
        self.assertAlmostEqual(pr["expected_share_user"], 0.4375)
        b = res["b_per_record"]
        self.assertEqual((b["freeze_p50"], b["freeze_p95"]), (1.0, 1.5))
        self.assertEqual(b["dev_level"]["latency_p50_s"], 1.0)                            # the 10-worker level, not the 2-worker one

    def test_markdown_says_not_run_and_labels_dev(self):
        res, _ = self.run_build()
        md = LF.render(res)
        self.assertIn("**No per-pair array was saved**", md)
        self.assertIn("**Not run.**", md)
        self.assertIn("not superseded", md)
        self.assertIn("DEV: call p50 at 10 workers", md)
        self.assertIn("| test group records | 6 |", md)
        self.assertIn("| records with at least one routed pair | 5 of 6 = 83.33% |", md)
        self.assertIn("43.75%", md)                                                       # the expected independent share, exponent 2
        self.assertNotIn("simulated p99", md)


class ReadingRulesTests(Fixture):
    def test_only_allowed_files_are_read(self):
        R = LF.Reader([self.run, self.freeze], [self.dev])
        with self.assertRaises(SystemExit):
            R.lines(self.root / "results" / "dev_run" / "other.json")                      # a dev file that is not the listed one
        for name in ("orgs_raw.csv", "orgs_raw_truth.csv"):
            (self.run / name).write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                R.lines(self.run / name)
        (self.run / "hand_check").mkdir()
        (self.run / "hand_check" / "hand_check_key.csv").write_text("x", encoding="utf-8")
        with self.assertRaises(SystemExit):
            R.lines(self.run / "hand_check" / "hand_check_key.csv")
        self.assertEqual(R.lines(self.dev)[0][:1], "{")                                    # the listed dev file is readable

    def test_a_run_reads_only_allowed_files_and_writes_only_latency_final_md(self):
        before = {p for p in self.root.rglob("*") if p.is_file()}
        res, R = self.run_build()
        for p in R.read:
            self.assertTrue(p == self.dev.resolve() or self.run.resolve() in p.parents or self.freeze.resolve() in p.parents, p)
        self.out.mkdir(parents=True)
        (self.out / "latency_final.md").write_text(LF.render(res), encoding="utf-8")
        new = {p for p in self.root.rglob("*") if p.is_file()} - before
        self.assertEqual({p.name for p in new}, {"latency_final.md"})
        self.assertTrue(all(self.out.resolve() in p.resolve().parents for p in new))


if __name__ == "__main__":
    unittest.main(verbosity=2)

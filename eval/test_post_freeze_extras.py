"""
Tests for eval/post_freeze_extras.py on a small hand-computed case: exact and bootstrap intervals, the oracle steward, the daily queue, the rule that the truth
file is read only through truth_rows(), the header line, and a whole build on a tiny run folder.

  python eval/test_post_freeze_extras.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import post_freeze_extras as PF  # noqa: E402


class StatTests(unittest.TestCase):
    def test_clopper_pearson_known_values(self):
        lo, hi = PF.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** (1 / 10), places=6)                 # closed form for x = 0
        lo, hi = PF.clopper_pearson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.025 ** (1 / 10), places=6)                     # closed form for x = n
        lo, hi = PF.clopper_pearson(1, 2)
        self.assertAlmostEqual(lo, 0.0126, places=4)
        self.assertAlmostEqual(hi, 0.9874, places=4)

    def test_bootstrap_with_one_sampled_pair_is_the_point_estimate(self):
        # one band pair, weight 2, predicted same, a false merge; outside the band 6 true matches and no false merge: FP 2, predicted 2 + 6 = 8, rate 0.25
        lo, hi = PF.bootstrap_false_merge_rate(np.array([2.0]), np.array([1.0]), np.array([1.0]), outside_tp=6.0, outside_fp=0.0, n_boot=50)
        self.assertEqual((lo, hi), (0.25, 0.25))

    def test_bootstrap_without_false_merges_is_zero_and_is_reproducible(self):
        w, pred, fm = np.array([1.0, 1.0, 5.0, 5.0]), np.array([1.0, 1.0, 0.0, 0.0]), np.zeros(4)
        self.assertEqual(PF.bootstrap_false_merge_rate(w, pred, fm, 10.0, 0.0, n_boot=100), (0.0, 0.0))
        w2, pred2, fm2 = np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])
        a = PF.bootstrap_false_merge_rate(w2, pred2, fm2, 2.0, 0.0, n_boot=200, seed=0)
        b = PF.bootstrap_false_merge_rate(w2, pred2, fm2, 2.0, 0.0, n_boot=200, seed=0)
        self.assertEqual(a, b)
        self.assertTrue(0.0 <= a[0] <= a[1] <= 4 / 6 + 1e-9)                         # the extreme resample: four copies of the false-merge pair, 4 / (4 + 2 outside)

    def test_oracle_steward_hand_computed(self):
        rep = {"band_tp": 8.0, "band_fp": 2.0, "band_fn": 2.0, "tp": 18.0, "fp": 2.0, "fn": 12.0}
        o = PF.oracle_steward(rep, routed_fp=1, routed_fn=2)
        self.assertEqual((o["band_tp"], o["band_fp"], o["band_fn"]), (10.0, 1.0, 0.0))
        self.assertAlmostEqual(o["band_f1"], 20 / 21)
        self.assertAlmostEqual(o["band_precision"], 10 / 11)
        self.assertEqual(o["band_recall"], 1.0)
        self.assertAlmostEqual(o["overall_precision"], 20 / 21)                          # overall TP 18 + 2, FP 2 - 1
        self.assertAlmostEqual(o["overall_recall"], 20 / 30)                              # FN 12 - 2

    def test_daily_queue(self):
        self.assertEqual(PF.daily_queue(1000, 5.0, 0.1), 500.0)
        self.assertAlmostEqual(PF.daily_queue(100_000, 20.0499, 426 / 44022), 100_000 * 20.0499 * 426 / 44022)


def rep_text(tp_b, fp_b, fn_b, tn_b, tp, fp, fn, f1="0.9000", decoy=("20", "1", "5.00")):
    return (f"Option: x\n\nBand (test band pairs, same-entity = positive)\n  pairs 10  (true 1, 10.00%)\n  F1 {f1}  95% CI [0.5000, 1.0000]  (bootstrap)\n"
            f"  accuracy 0.9000  precision 0.9000  recall 0.9000\n  TP {tp_b}  FP {fp_b}  FN {fn_b}  TN {tn_b}\n\n"
            f"Overall (test group; outside the band Verity's decision_status)\n  precision 0.9500  recall 0.5000  F1 0.6000\n  TP {tp}  FP {fp}  FN {fn}\n\n"
            f"Decoy pairs in the test group: {decoy[0]}, wrongly matched {decoy[1]} ({decoy[2]}%)\n")


class ParseTests(unittest.TestCase):
    def test_parse_report(self):
        r = PF.parse_report(rep_text("1.0", "2.0", "0.0", "4.0", "11.0", "2.0", "5.0"))
        self.assertEqual((r["band_tp"], r["band_fp"], r["band_fn"], r["band_tn"]), (1.0, 2.0, 0.0, 4.0))
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (11.0, 2.0, 5.0))
        self.assertEqual((r["decoy_pairs"], r["decoy_false"]), (20.0, 1.0))
        self.assertEqual(r["band_f1"], 0.9)


def w_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class BuildFixture(unittest.TestCase):
    """Entities: R1,R2 = E1; R3,R4 = E2; R5 = E3; R6 = E4; R7 = E5; R8 = E6.
    Sample (weights): (R1,R2, 1) true; (R3,R5, 2) false; (R6,R7, 2) false; (R4,R8, 2) false. D predicts same for (R1,R2) and (R3,R5) -> band predicted weight 1 + 2,
    band FP weight 2; outside the band 10 true matches decided right by Verity. D overall: TP 11, FP 2 -> predicted 13, rate 2/13."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "results" / "eval_7919"
        self.dstudy = self.root / "results" / "dstudy_summary"
        self.lc = self.root / "results" / "dev" / "lc_v2"
        r, O = self.run, self.run / "options"
        outside = {"tp": 10.0, "fp": 0.0}
        # reports: A and C predict nothing in the band (TP 0 FP 0), D predicts the two pairs; sample basis weighted, full band exact (here the same pairs, weight 1)
        none = rep_text("0.0", "0.0", "1.0", "6.0", "10.0", "0.0", "6.0")
        d_rep = rep_text("1.0", "2.0", "0.0", "4.0", "11.0", "2.0", "5.0")
        for p, t in ((O / "a_threshold" / "score_report.txt", none), (O / "c_model" / "c_v1_lgbm" / "score_report.txt", none), (O / "d_hybrid" / "score_report_fullband.txt", d_rep),
                     (O / "d_hybrid" / "score_report_fullband_original.txt", none), (O / "a_threshold" / "score_report_sample5000.txt", none),
                     (O / "b_llm" / "openai" / "sample5000" / "score_report_sample5000.txt", d_rep), (O / "c_model" / "c_v1_lgbm" / "score_report_sample5000.txt", none),
                     (O / "d_hybrid" / "score_report_sample5000.txt", d_rep)):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t, encoding="utf-8")
        w_csv(O / "b_llm" / "test_sample5000.csv", ["record_a", "record_b", "weight"], [["R1", "R2", "1.0"], ["R3", "R5", "2.0"], ["R6", "R7", "2.0"], ["R4", "R8", "2.0"]])
        dec_d = [["R1", "R2", "same", "0.9"], ["R3", "R5", "same", "0.9"], ["R6", "R7", "different", "0.1"], ["R4", "R8", "different", "0.1"]]
        dec_none = [[a, b, "different", "0.1"] for a, b, _, _ in dec_d]
        h = ["record_a", "record_b", "decision", "confidence"]
        w_csv(O / "a_threshold" / "decisions.csv", h, dec_none)
        w_csv(O / "c_model" / "c_v1_lgbm" / "decisions.csv", h, dec_none + [[f"X{i}", f"Y{i}", "different", "0.0"] for i in range(16)])      # 20 band pairs
        w_csv(O / "b_llm" / "openai" / "sample5000" / "decisions.csv", h, dec_d)
        w_csv(O / "d_hybrid" / "decisions.csv", h, dec_d)
        w_csv(O / "d_hybrid" / "decisions_fullband.csv", h + ["route"], [[a, b, d, c, "llm" if (a, b) in (("R3", "R5"), ("R6", "R7")) else "c_different"] for a, b, d, c in dec_d])
        w_csv(O / "d_hybrid" / "decisions_fullband_original.csv", h + ["route"], [[a, b, "different", c, "c_different"] for a, b, d, c in dec_d])
        w_csv(O / "d_hybrid" / "routed_fullband_pairs.csv", ["record_a", "record_b", "weight"], [["R3", "R5", "1.0"], ["R6", "R7", "1.0"]])
        w_csv(O / "d_hybrid" / "routed_fullband_pairs_original.csv", ["record_a", "record_b", "weight"], [])
        w_csv(r / "cost_final" / "assumptions.csv", ["name", "value", "unit", "status", "source", "note"],
              [["band pairs per record", "5.0", "pairs", "measured", "x", ""], ["labeling hours per 1,000 pairs", "5.0", "hours", "measured", "effort log row", "a caveat"]])
        w_csv(r / "demo_pairs" / "demo_pairs.csv", ["pair_no", "record_a", "record_b", "route", "c_p"], [["1", "R1", "R2", "Verity decides (outside the band)", ""]])
        sims = {"company_name": 1.0, "phone": 0.5}
        w_csv(r / "verity_data" / "scored_pairs.csv", ["record_a", "record_b", "score", "score_uncapped", "match_type", "decision_status", "applied_threshold", "effective_threshold",
                                                      "evidence_passes", "evidence_mass", "conflict_mass", "field_similarities", "field_weights", "evidence_field_similarities"],
              [["R2", "R1", "0.97", "0.97", "weak_signal", "match", "0.82", "0.82", "true", "0.65", "0.0", json.dumps(sims), json.dumps({"company_name": 0.5, "phone": 0.5}), json.dumps({"company_name": 1.0})]])
        w_csv(self.dstudy / "d_select_lo.csv", ["lo", "pooled_f1", "precision", "recall", "fp", "fn", "routed", "llm_share", "eligible"],
              [["0.005", "0.98", "0.99", "0.97", "15", "21", "576", "0.0026", "True"], ["share_dstudy_11", "238", "45676", "0.005", "0.007"]])
        w_csv(self.lc / "learning_curve.csv", ["n", "n_true", "seeds", "f1_mean", "f1_min", "f1_max"], [["500", "125", "5", "0.77706", "0.55", "0.91"], ["2000", "500", "5", "0.9", "0.85", "0.95"]])
        w_csv(r / "orgs_raw_truth.csv", ["record_id", "true_entity_id", "variants", "decoy_group"],
              [["R1", "E1", "", ""], ["R2", "E1", "", ""], ["R3", "E2", "", ""], ["R4", "E2", "", ""], ["R5", "E3", "", ""], ["R6", "E4", "", ""], ["R7", "E5", "", ""], ["R8", "E6", "", ""]])
        PF.REPO_BACKUP, PF.REPO = PF.REPO, self.root

    def tearDown(self):
        PF.REPO = PF.REPO_BACKUP
        self.tmp.cleanup()

    def build(self):
        R = PF.Reader([self.run, self.dstudy, self.lc], self.run / "orgs_raw_truth.csv")
        return PF.build(self.run, self.dstudy, self.lc, R), R


class BuildTests(BuildFixture):
    def test_false_merges_on_both_bases(self):
        res, _ = self.build()
        d = res["full"]["D"]
        self.assertEqual((d["predicted"], d["false"]), (13, 2))
        self.assertAlmostEqual(d["rate"], 2 / 13)
        self.assertEqual((res["full"]["A"]["predicted"], res["full"]["A"]["false"]), (10, 0))
        self.assertEqual(res["full"]["A"]["lo"], 0.0)
        s = res["sample"]["D"]
        self.assertEqual((s["predicted"], s["false"]), (13.0, 2.0))
        self.assertTrue(s["reproduced"])                                                 # the decisions and the truth give FP weight 2 for (R3,R5)
        self.assertTrue(res["sample"]["B"]["reproduced"] and res["sample"]["A"]["reproduced"])
        self.assertEqual((res["sample"]["D"]["decoy_pairs"], res["sample"]["D"]["decoy_false"]), (20.0, 1.0))
        self.assertEqual(res["outside"], {"tp": 10.0, "fp": 0.0})

    def test_review_queue_and_oracle(self):
        res, _ = self.build()
        q = res["queue"]["D"]
        self.assertEqual((q["routed"], q["band_pairs"]), (2, 20))
        self.assertAlmostEqual(q["per_1000"], 100.0)                                      # 2 of 20
        self.assertAlmostEqual(q["daily"][100_000], 100_000 * 5.0 * 0.1)                   # 50,000 a day
        self.assertEqual((q["routed_fp"], q["routed_fn"]), (1, 0))                        # (R3,R5) was a false merge; (R6,R7) correct
        self.assertEqual(q["oracle"]["band_fp"], 1.0)                                     # 2 - 1
        self.assertEqual(q["oracle"]["band_tp"], 1.0)
        self.assertEqual(res["queue"]["D (orig)"]["routed"], 0)
        self.assertEqual(res["dev_tradeoff"][0]["lo"], "0.005")
        self.assertEqual(len(res["dev_tradeoff"]), 1)                                     # the share_ rows are skipped

    def test_explanation_and_effort(self):
        res, _ = self.build()
        e = res["explain"][0]
        self.assertEqual(e["pair"], ("R1", "R2"))
        self.assertEqual(e["row"]["evidence_passes"], "true")
        self.assertEqual(res["hours_per_1000"], 5.0)
        self.assertEqual([(r["n"], r["hours"]) for r in res["lc"]], [(500, 2.5), (2000, 10.0)])
        self.assertEqual(res["lc"][0]["f1_mean"], 0.77706)

    def test_the_output_starts_with_the_exploratory_line_and_lists_sources(self):
        res, _ = self.build()
        md = PF.render(res)
        self.assertEqual(md.splitlines()[0], "EXPLORATORY: not a protocol result; no setting was chosen from these numbers.")
        self.assertIn("## 1. False merges", md)
        self.assertIn("## 4. Effort to quality (development seed 2026)", md)
        self.assertIn("dstudy_summary/d_select_lo.csv", md)
        self.assertIn("upper bound", md.lower())
        self.assertIn("### Pair 1: R1 / R2", md)
        self.assertIn("not applicable (the pair is outside the band; Verity decides)", md)
        self.assertIn("15.385%", md)                                                      # D's false-merge rate 2 / 13


class ReaderRuleTests(BuildFixture):
    def test_the_truth_file_is_read_only_through_truth_rows_and_only_for_the_ids_asked(self):
        R = PF.Reader([self.run], self.run / "orgs_raw_truth.csv")
        with self.assertRaises(SystemExit):
            R.text(self.run / "orgs_raw_truth.csv")
        with self.assertRaises(SystemExit):
            list(R.csv_rows(self.run / "orgs_raw_truth.csv"))
        self.assertEqual(R.truth_rows({"R1", "R5"}), {"R1": "E1", "R5": "E3"})
        for name in ("orgs_raw.csv", "hand_check_key.csv"):
            with self.assertRaises(SystemExit):
                R.text(self.run / name)
        with self.assertRaises(SystemExit):
            R.text(self.root / "elsewhere.txt")

    def test_a_build_writes_nothing_and_reads_only_allowed_files(self):
        before = {p for p in self.root.rglob("*") if p.is_file()}
        res, R = self.build()
        self.assertEqual({p for p in self.root.rglob("*") if p.is_file()}, before)
        for p in R.read:
            self.assertTrue(any(root.resolve() in p.parents for root in (self.run, self.dstudy, self.lc)), p)


if __name__ == "__main__":
    unittest.main(verbosity=2)

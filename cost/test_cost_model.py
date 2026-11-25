"""
Tests for cost/cost_model.py on a small case worked out by hand (no repo files needed).

  python cost/test_cost_model.py

The case: 1,000 records per day, 10 band pairs per record, 10% routed to the LLM, 1,000 input and 10 output tokens per
call, hit ratio 0.5, prices $2.0 / $0.2 / $10.0 per million tokens (input / cached / output), hourly rate $100.

  cost per call            (1000 x 0.5 x 2.0 + 1000 x 0.5 x 0.2 + 10 x 10) / 1e6 = 0.0012
  B                        10,000 calls a day x 30.4 days x 0.0012 = $364.80 a month; D: 1,000 calls = $36.48
  C hosting                10,000 pairs x 0.001 s = 10 core-seconds a day; always-on 1 core x 24 x 30.4 x 0.05 = $36.48,
                           serverless 10 x 30.4 x 0.0001 = $0.0304 (used)
  Verity hosting           1,000 records x 0.1 s = 100 core-seconds a day; always-on $36.48, serverless $0.304 (used)
  operations hours         A 1, B 2 + 4 = 6, C 2 + 4 x 1 = 6, D 2 + 4 + 4 = 10
  one-time (development)   (80 + 100) h x $100 = $18,000 for every option (the Verity build and productization)
  one-time (labeling)      C and D: 2,000 labels x 5 h per 1,000 x $100 = $1,000; so A and B 18,000, C and D 19,000
                           (the PoC's effort-log hours and API spend are not charged: the separate evidence block)
  per month (amortized)    development 18,000 / 12 = 1,500; labeling 1,000 / 12 = 83.3333

Who pays, per month, at 1,000 records/day (components: llm, hosting, operations, development, labeling)
  A   0, 0.304, 100, 1,500, 0                  B   364.80, 0.304, 600, 1,500, 0
  C   0, 0.3344, 600, 1,500, 83.3333           D   36.48, 0.3344, 1,000, 1,500, 83.3333
  S1 everything is Precisely's; S2 the LLM line is the customer's; S3 the LLM, hosting, operations and labeling lines are the
  customer's and Precisely's development is not modeled (left out of the total).
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_model as M  # noqa: E402


def small() -> M.Params:
    return M.Params(band_pairs_per_record=10, routed_share=0.1, in_tokens=1000, out_tokens=10, hit_ratio=0.5, price_in=2.0,
                    price_cached=0.2, price_out=10.0, batch_in=1.0, batch_cached=0.1, batch_out=5.0, c_cpu_s_per_pair=0.001,
                    d_cpu_s_per_pair=0.001, verity_cpu_s_per_record=0.1, core_hour_always_on=0.05, core_second_serverless=0.0001,
                    utilization=0.5, monitor_hours=2, retrain_hours_per_run=4, retrains_per_month=1, llm_upkeep_hours=4,
                    a_ops_hours=1, label_hours_per_1000=5, effort_hours={"verity": 6.5, "A": 1, "B": 0.5, "C": 2},
                    verity_preexisting_hours=80, verity_productization_hours=100, b_dev_api_usd=10, horizon_months=12)


class CostTests(unittest.TestCase):
    def setUp(self):
        self.p = small()

    def test_cost_per_call_and_modes(self):
        self.assertAlmostEqual(M.llm_cost_per_call(self.p), 0.0012, places=12)
        self.assertAlmostEqual(M.llm_cost_per_call(self.p, "standard_nocache"), (1000 * 2.0 + 10 * 10) / 1e6, places=12)
        self.assertAlmostEqual(M.llm_cost_per_call(self.p, "batch"), (500 * 1.0 + 500 * 0.1 + 10 * 5) / 1e6, places=12)
        self.assertAlmostEqual(M.llm_cost_per_call(self.p, "standard", hit_ratio=0.0), 0.0021, places=12)
        self.assertAlmostEqual(M.llm_cost_per_call(replace_factor(self.p, 1.5)), 0.0018, places=12)

    def test_calls_and_llm_cost(self):
        self.assertEqual(M.calls_per_day(self.p, "B", 1000), 10000)
        self.assertAlmostEqual(M.calls_per_day(self.p, "D", 1000), 1000)
        self.assertEqual(M.calls_per_day(self.p, "C", 1000), 0)
        self.assertAlmostEqual(M.llm_cost_month(self.p, "B", 1000), 364.80, places=6)
        self.assertAlmostEqual(M.llm_cost_month(self.p, "D", 1000), 36.48, places=6)

    def test_hosting_uses_the_cheaper_of_always_on_and_serverless(self):
        h = M.hosting_month(self.p, "C", 1000)
        self.assertEqual(h["model"]["cores"], 1)
        self.assertAlmostEqual(h["model"]["always_on"], 36.48, places=6)
        self.assertAlmostEqual(h["model"]["serverless"], 0.0304, places=6)
        self.assertAlmostEqual(h["verity"]["serverless"], 0.304, places=6)
        self.assertAlmostEqual(h["used"], 0.0304 + 0.304, places=6)
        self.assertAlmostEqual(M.hosting_month(self.p, "B", 1000)["used"], 0.304, places=6)         # Verity only, B included
        self.assertAlmostEqual(M.hosting_month(self.p, "A", 1000)["used"], 0.304, places=6)

    def test_always_on_cores_grow_with_volume(self):
        # 1e9 core-seconds a day at 50% utilisation needs ceil(1e9 / 86400 / 0.5) cores
        c = M.hosting_component(1e9, self.p)
        self.assertEqual(c["cores"], 23149)

    def test_operations_hours(self):
        self.assertEqual([M.ops_hours_month(self.p, o) for o in "ABCD"], [1, 6, 6, 10])
        self.assertEqual(M.ops_hours_month(self.p, "C", retrains_per_month=4.33), 2 + 4 * 4.33)

    def test_one_time_costs_are_development_and_labeling_only(self):
        for o, expect in (("A", 18000), ("B", 18000), ("C", 19000), ("D", 19000)):
            self.assertAlmostEqual(M.one_time_total(self.p, o, 100, 2000), expect, msg=o)
        self.assertEqual(M.label_hours(self.p, 63665), 63665 / 1000 * 5)
        self.assertAlmostEqual(M.one_time_total(self.p, "C", 100, 63665), 18000 + 63665 / 1000 * 5 * 100)
        self.assertAlmostEqual(M.one_time_total(self.p, "B", 100, 63665), 18000)                    # B: zero labels
        self.assertEqual(M.one_time_split(self.p, "C", 100, 2000), {"development": 18000.0, "labeling": 1000.0})
        # the PoC's effort-log hours and API spend never enter an option's one-time cost
        richer = replace(self.p, effort_hours={"verity": 99, "A": 99, "B": 99, "C": 99, "D": 99}, b_dev_api_usd=1e6)
        self.assertAlmostEqual(M.one_time_total(richer, "D", 100, 2000), 19000)

    def test_monthly_totals(self):
        c = M.monthly(self.p, "C", 1000, 100, 2000)
        self.assertAlmostEqual(c["running"], 0.0304 + 0.304 + 600, places=6)
        self.assertAlmostEqual(c["one_time_amortized"], 19000 / 12, places=6)
        self.assertAlmostEqual(c["development"], 1500, places=6)
        self.assertAlmostEqual(c["labeling"], 1000 / 12, places=6)
        self.assertAlmostEqual(c["total_month"], 0.3344 + 600 + 19000 / 12, places=6)
        self.assertAlmostEqual(c["total_horizon"], 19000 + 12 * 600.3344, places=4)
        b = M.monthly(self.p, "B", 1000, 100, 2000)
        self.assertAlmostEqual(b["running"], 364.80 + 0.304 + 600, places=6)
        self.assertAlmostEqual(b["total_horizon"], 18000 + 12 * 965.104, places=4)

    def rows(self, option):
        return {(r["scenario"], r["label"]): r for r in M.scenario_rows(self.p, option, 1000, 100, 2000)}

    def test_s1_everything_is_precisely(self):
        for o, total in (("A", 100 + 0.304 + 1500), ("B", 364.80 + 0.304 + 600 + 1500), ("C", 0.3344 + 600 + 1500 + 1000 / 12),
                         ("D", 36.48 + 0.3344 + 1000 + 1500 + 1000 / 12)):
            r = self.rows(o)[("S1", o)]
            self.assertAlmostEqual(r["precisely"], total, places=6, msg=o)
            self.assertEqual(r["customer"], 0.0)
            self.assertAlmostEqual(r["total"], total, places=6)

    def test_s2_only_the_llm_line_moves_to_the_customer(self):
        b = self.rows("B")[("S2", "B")]
        self.assertAlmostEqual(b["customer"], 364.80, places=6)
        self.assertAlmostEqual(b["precisely"], 0.304 + 600 + 1500, places=6)
        self.assertAlmostEqual(b["total"], self.rows("B")[("S1", "B")]["total"], places=9)         # the split changes, not the total
        d = self.rows("D")[("S2", "D")]
        self.assertAlmostEqual(d["customer"], 36.48, places=6)
        self.assertEqual(self.rows("C")[("S2", "C")]["customer"], 0.0)                              # C has no LLM line

    def test_s3_customer_pays_hosting_operations_labeling_and_the_llm_if_connected(self):
        c = self.rows("C")[("S3", "C")]
        self.assertAlmostEqual(c["customer"], 0.3344 + 600 + 1000 / 12, places=6)
        self.assertEqual(c["precisely"], 0.0)                                                       # development and support not modeled
        self.assertAlmostEqual(c["total"], c["customer"], places=9)
        self.assertEqual(c["feasible_data_stays"], "yes")
        self.assertEqual(self.rows("A")[("S3", "A")]["feasible_data_stays"], "yes")
        self.assertAlmostEqual(self.rows("A")[("S3", "A")]["customer"], 0.304 + 100, places=6)
        b = self.rows("B")[("S3", "B (own key connected)")]
        self.assertAlmostEqual(b["customer"], 364.80 + 0.304 + 600, places=6)
        self.assertEqual(b["feasible_data_stays"], "no")

    def test_s3_d_has_two_rows_and_the_fallback_off_row_has_no_llm_cost(self):
        rows = self.rows("D")
        off, on = rows[("S3", "D (LLM fallback off)")], rows[("S3", "D (own key connected)")]
        self.assertAlmostEqual(off["customer"], 0.3344 + 1000 + 1000 / 12, places=6)
        self.assertAlmostEqual(on["customer"] - off["customer"], 36.48, places=6)
        self.assertEqual((off["quality"], on["quality"]), ("not measured", "as evaluated"))
        self.assertEqual(off["feasible_data_stays"], "yes, with the LLM fallback off")
        self.assertEqual(off["calls_per_day"], 0.0)

    def test_every_option_has_every_scenario_and_e_is_not_run(self):
        self.assertEqual(sorted({r["scenario"] for r in M.scenario_rows(self.p, "B", 1000, 100, 2000)}), ["S1", "S2", "S3"])
        e = M.scenario_rows(self.p, "E", 1000, 100, 2000)
        self.assertEqual([(r["option"], r["status"]) for r in e], [("E", "not run")])

    def test_party_table_covers_every_component(self):
        for sc in M.SCENARIOS:
            self.assertEqual(set(M.PARTY[sc]), set(M.COMPONENTS))
        self.assertEqual(M.PARTY["S1"]["llm"], "Precisely")
        self.assertEqual(M.PARTY["S2"]["llm"], "customer")
        self.assertEqual(M.PARTY["S3"]["hosting"], "customer")
        self.assertTrue(M.PARTY["S3"]["development"].startswith("Precisely"))

    def test_scenario_total_horizon_is_twelve_months_of_the_total(self):
        self.assertAlmostEqual(M.scenario_total_horizon(self.p, "C", 1000, 100, 2000, "S1"), M.total_horizon(self.p, "C", 1000, 100, 2000), places=6)
        self.assertAlmostEqual(M.scenario_total_horizon(self.p, "C", 1000, 100, 2000, "S3"), 12 * (0.3344 + 600 + 1000 / 12), places=6)

    def test_option_e_is_not_run(self):
        e = M.monthly(self.p, "E", 1000, 100, 2000)
        self.assertEqual(e, {"option": "E", "status": "not run"})


class VariantAndBarTests(unittest.TestCase):
    def setUp(self):
        self.p = small()

    def test_b_price_variant_scales_only_the_llm_line(self):
        base = M.monthly(self.p, "B", 1000, 100, 0)
        v = M.monthly_b_variant(self.p, "price-70", 1000, 100, 0)
        self.assertAlmostEqual(v["llm"], 364.80 * 0.3, places=6)                 # 109.44
        self.assertAlmostEqual(v["running"] - v["llm"], base["running"] - base["llm"], places=9)
        self.assertAlmostEqual(v["one_time"], base["one_time"], places=9)

    def test_b_without_the_shared_verity_build(self):
        v = M.monthly_b_variant(self.p, "no-verity-build", 1000, 100, 0)
        self.assertAlmostEqual(v["one_time"], 0.0, places=9)                      # B has no labeling and no shared build
        self.assertAlmostEqual(M.one_time_total(self.p, "B", 100, 0), 18000)      # with the Verity lines
        self.assertAlmostEqual(v["running"], M.monthly(self.p, "B", 1000, 100, 0)["running"], places=9)
        with self.assertRaises(ValueError):
            M.monthly_b_variant(self.p, "nope", 1000, 100, 0)

    def test_f1_bar_verdicts(self):
        self.assertEqual(M.f1_bar_verdict(-0.0154, 0.0), "pass")                  # lower end above -0.02
        self.assertEqual(M.f1_bar_verdict(-0.0936, -0.0045), "inconclusive")      # straddles -0.02
        self.assertEqual(M.f1_bar_verdict(-0.8739, -0.7722), "fail")              # upper end below -0.02
        self.assertEqual(M.f1_bar_verdict(-0.02, 0.0), "inconclusive")            # lower end must be strictly above

    def test_bar_rows(self):
        base = {"band_f1": 0.9, "overall": (0.99, 0.58), "ece": 0.01, "latency_p95_ms": 0.3, "llm_share": 0.001}
        ok = M.bars_row("D", {**base, "f1_diff": (-0.0075, -0.0154, 0.0)}, a_recall=0.5248)
        self.assertEqual((ok["f1"], ok["precision"], ok["recall"], ok["ece"], ok["llm_share"], ok["every"]), ("pass", "pass", "pass", "pass", "pass", "yes"))
        inc = M.bars_row("C", {**base, "f1_diff": (-0.0464, -0.0936, -0.0045)}, a_recall=0.5248)
        self.assertEqual((inc["f1"], inc["latency"], inc["every"]), ("inconclusive", "pass", "not shown (inconclusive)"))
        a = M.bars_row("A", {**base, "overall": (1.0, 0.5248), "ece": None, "f1_diff": (-0.825, -0.8739, -0.7722)}, a_recall=0.5248)
        self.assertEqual((a["f1"], a["recall"], a["every"]), ("fail", "reference (A is the baseline)", "no"))
        low = M.bars_row("D", {**base, "overall": (0.97, 0.5700), "f1_diff": (0.0, -0.01, 0.01)}, a_recall=0.5248)
        self.assertEqual((low["precision"], low["recall"], low["every"]), ("fail", "fail", "no"))     # 0.57 is 4.5 points above A: fail
        b = M.bars_row("B", {**base, "f1_diff": None}, a_recall=0.5248)
        self.assertEqual((b["f1"], b["every"]), ("reference (B is the baseline)", "yes"))

    def test_report_parsers(self):
        text = ("Band (test band pairs, same-entity = positive) (weighted estimate)\n  pairs 44882.0  (true 340.0, 0.76%)\n"
                "  F1 0.9971  95% CI [0.99, 1.0]\n\n"
                "Overall (test group; outside the band Verity's decision_status) (weighted estimate)\n"
                "  precision 1.0000  recall 0.5800  F1 0.7342\n  ECE 0.0142   (bar: <= 0.05)\n")
        self.assertEqual(M.parse_band_f1(text), 0.9971)
        self.assertEqual(M.parse_overall(text), (1.0, 0.58))
        self.assertEqual(M.parse_ece(text), 0.0142)
        cmp_ = "Difference (first minus second) = D minus B\n  band F1         -0.0075  95% CI [-0.0154, +0.0000]  (width 0.0154)\n"
        self.assertEqual(M.parse_f1_diff(cmp_), (-0.0075, -0.0154, 0.0))


class PipelineFactorTests(unittest.TestCase):
    def test_the_shared_pipeline_cost_scales_and_is_the_same_line_for_every_option(self):
        p = small()
        base = M.hosting_month(p, "B", 1000)["verity"]["used"]                       # 100 core-s a day, serverless 0.304
        self.assertAlmostEqual(base, 0.304, places=6)
        p10 = replace(p, verity_cost_factor=10.0)
        self.assertAlmostEqual(M.hosting_month(p10, "B", 1000)["verity"]["used"], 3.04, places=6)      # 1,000 core-s a day x 30.4 x 0.0001
        for o in "ABCD":                                                                     # the same added cost for every option
            self.assertAlmostEqual(M.monthly(p10, o, 1000, 100, 2000)["hosting_verity"] - M.monthly(p, o, 1000, 100, 2000)["hosting_verity"], 3.04 - 0.304, places=6)
        self.assertEqual(M.hosting_month(replace(p, verity_cost_factor=1e7), "A", 1000)["verity"]["cores"], 23149)   # 1e7 x 100 = 1e9 core-s / 86,400 / 0.5 rounded up to whole cores

    def test_the_gap_between_options_does_not_depend_on_the_pipeline_factor(self):
        p, p100 = small(), replace(small(), verity_cost_factor=100.0)
        gap = lambda q: M.monthly(q, "B", 1000, 100, 2000)["total_month"] - M.monthly(q, "C", 1000, 100, 2000)["total_month"]
        # C's model hosting differs from B's (none), so the gap moves only by that hosting line, never by the pipeline
        self.assertAlmostEqual(gap(p), gap(p100), places=6)


class BreakEvenTests(unittest.TestCase):
    def test_linear_crossing(self):
        r = M.break_even(lambda v: 1000 + 2 * v, lambda v: 100 + 5 * v, lo=1, hi=1e4)
        self.assertAlmostEqual(r["volume"], 300, delta=1e-6)
        self.assertEqual(r["kind"], "option cheaper above")

    def test_no_crossing_cases(self):
        self.assertEqual(M.break_even(lambda v: v, lambda v: 2 * v + 1, lo=1, hi=100)["kind"], "option cheaper everywhere in range")
        self.assertEqual(M.break_even(lambda v: 3 * v + 5, lambda v: 2 * v, lo=1, hi=100)["kind"], "B cheaper everywhere in range")
        self.assertIsNone(M.break_even(lambda v: v, lambda v: 2 * v + 1, lo=1, hi=100)["volume"])

    def test_option_cheaper_below(self):
        r = M.break_even(lambda v: 10 + 5 * v, lambda v: 100 + 2 * v, lo=1, hi=1e4)
        self.assertEqual(r["kind"], "option cheaper below")
        self.assertAlmostEqual(r["volume"], 30, delta=1e-6)


class InputTests(unittest.TestCase):
    def test_date_formats(self):
        self.assertEqual(M.parse_date("2026-11-24"), datetime(2026, 11, 24))
        self.assertEqual(M.parse_date("11/24/2026"), datetime(2026, 11, 24))
        self.assertEqual(M.parse_date("1/5/2026"), datetime(2026, 1, 5))
        with self.assertRaises(ValueError):
            M.parse_date("24.11.2026")

    def write_log(self, tmp: Path, rows: list[str]) -> Path:
        path = tmp / "effort_log.csv"
        path.write_text("date,hours,activity,bucket,production_needed\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_effort_log_buckets_labels_and_unfinished_rows(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = self.write_log(tmp, ["11/23/2026,1,Eval protocol,eval,no", "2026-11-24,2,Option C,C,yes", "2026-11-24,1,Option A,A,yes",
                                        "11/24/2026,0.5,Hand check of 100 pairs,eval,no", "11/24/2026,,Option D (in progress),D,yes",
                                        "11/24/2026,2,Learning curve,C,no"])
            rows = M.read_effort_log(path)
            self.assertEqual(len(rows), 5)                                       # the row without hours is skipped
            b = M.hours_by_bucket(rows)
            self.assertEqual(b["C"], {"total": 4.0, "production": 2.0, "other": 2.0})
            self.assertEqual(b["eval"]["total"], 1.5)
            self.assertEqual(M.label_hours_per_1000_from_log(rows), 5.0)         # 0.5 h per 100 pairs
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ledger_classes(self):
        self.assertEqual(M.classify_ledger_label("tune1"), "B development")
        self.assertEqual(M.classify_ledger_label("tune2"), "B development")
        self.assertEqual(M.classify_ledger_label("cache_diagnostic_r2"), "B development")
        self.assertEqual(M.classify_ledger_label("smoke5"), "B development")
        self.assertEqual(M.classify_ledger_label("concurrency"), "B development")
        for lab in ("sample5000", "decoycheck", "dfull", "final-20261204", "dec4-b"):
            self.assertEqual(M.classify_ledger_label(lab), "evaluation", lab)
        self.assertEqual(M.classify_ledger_label("mystery"), "unclassified")

    def test_ledger_spend_sums_settled_amounts_by_label(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "ledger.jsonl"
            path.write_text("\n".join([
                '{"type":"reserve","run":"r/b_llm/openai/tune1","usd":0.5}',
                '{"type":"settle","run":"r/b_llm/openai/tune1","usd":0.25}',
                '{"type":"settle","run":"r/b_llm/openai/tune1","usd":0.5}',
                '{"type":"settle","run":"r/b_llm/openai/sample5000","usd":2}',
                '{"type":"cache_hits","run":"r/b_llm/openai/sample5000","usd":0.0}',
                '{"type":"settle","run":"r/b_llm/openai/mystery","usd":1}', '{"torn']), encoding="utf-8")
            s = M.ledger_spend(path)
            self.assertAlmostEqual(s["by_label"]["tune1"], 0.75)
            self.assertEqual(s["by_class"], {"B development": 0.75, "evaluation": 2.0, "unclassified": 1.0})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_draft_banner_is_the_first_line_of_a_csv_unless_the_log_is_complete(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            M.write_csv(tmp / "a.csv", ["x"], [[1]], draft=True)
            M.write_csv(tmp / "b.csv", ["x"], [[1]], draft=False)
            self.assertEqual((tmp / "a.csv").read_text(encoding="utf-8").splitlines()[0],
                             "# DRAFT: effort log incomplete, one-time costs for B and D undercounted")
            self.assertEqual((tmp / "b.csv").read_text(encoding="utf-8").splitlines()[0], "x")
            self.assertEqual(list(csv.reader((x for x in (tmp / "a.csv").read_text(encoding="utf-8").splitlines() if not x.startswith("#")))), [["x"], ["1"]])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def replace_factor(p: M.Params, f: float) -> M.Params:
    from dataclasses import replace
    return replace(p, llm_price_factor=f)


if __name__ == "__main__":
    unittest.main(verbosity=2)

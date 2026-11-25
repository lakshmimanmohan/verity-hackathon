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
  one-time (verity)        (6.5 + 80 + 100) h x $100 = $18,650 for every option
  one-time C               18,650 + 2 h x 100 (option C hours) + 2,000 labels x 5 h per 1,000 x 100 = 19,850
  one-time B               18,650 + 0.5 h x 100 + 10 (API spend) = 18,710; D = 18,650 + 50 + 10 + 200 + 1,000 = 19,910
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
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

    def test_one_time_costs(self):
        self.assertAlmostEqual(M.one_time_total(self.p, "A", 100, 2000), 18650 + 100)
        self.assertAlmostEqual(M.one_time_total(self.p, "B", 100, 2000), 18710)
        self.assertAlmostEqual(M.one_time_total(self.p, "C", 100, 2000), 19850)
        self.assertAlmostEqual(M.one_time_total(self.p, "D", 100, 2000), 19910)
        self.assertEqual(M.label_hours(self.p, 63665), 63665 / 1000 * 5)
        self.assertAlmostEqual(M.one_time_total(self.p, "C", 100, 63665), 18650 + 200 + 63665 / 1000 * 5 * 100)
        self.assertAlmostEqual(M.one_time_total(self.p, "B", 100, 63665), 18710)                    # B: zero labels

    def test_monthly_totals(self):
        c = M.monthly(self.p, "C", 1000, 100, 2000)
        self.assertAlmostEqual(c["running"], 0.0304 + 0.304 + 600, places=6)
        self.assertAlmostEqual(c["one_time_amortized"], 19850 / 12, places=6)
        self.assertAlmostEqual(c["total_month"], 0.3344 + 600 + 19850 / 12, places=6)
        self.assertAlmostEqual(c["total_horizon"], 19850 + 12 * 600.3344, places=4)
        b = M.monthly(self.p, "B", 1000, 100, 2000)
        self.assertAlmostEqual(b["running"], 364.80 + 0.304 + 600, places=6)
        self.assertAlmostEqual(b["total_horizon"], 18710 + 12 * 965.104, places=4)

    def test_who_pays(self):
        b = M.monthly(self.p, "B", 1000, 100, 2000)
        self.assertAlmostEqual(b["cost_customer_byok"], 364.80, places=6)
        self.assertAlmostEqual(b["cost_precisely_byok"], b["total_month"] - 364.80, places=6)
        self.assertAlmostEqual(b["cost_precisely_hosted_llm"], b["total_month"], places=6)
        c = M.monthly(self.p, "C", 1000, 100, 2000)
        self.assertEqual(c["cost_customer_byok"], 0.0)                                              # C has no LLM cost

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
        self.assertAlmostEqual(v["one_time"], 0.5 * 100 + 10, places=9)           # B hours + development API spend only
        self.assertAlmostEqual(M.one_time_total(self.p, "B", 100, 0), 18710)      # with the Verity lines (18,650)
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

"""
Tests for the bars table and the results table with D's two settings (eval/bars_table.py, eval/results_table.py):
D is routing rule v2 and "D (orig)" is the original rule, a secondary row scored the same way. No API, no truth.

  python eval/test_bars_results.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bars_table as BT  # noqa: E402
import results_table as RT  # noqa: E402

DEV = HERE.parent / "results" / "dev_2026_run11"


class BarsTableTests(unittest.TestCase):
    def test_rows_have_v2_and_the_original_rule_in_order(self):
        rows = BT.build(DEV)
        self.assertEqual([r["option"] for r in rows], ["A", "B", "C", "D", "D (orig)", "E"])
        d = {r["option"]: r for r in rows}
        self.assertAlmostEqual(d["D"]["llm_share"], 51 / 44882, places=9)
        self.assertAlmostEqual(d["D (orig)"]["llm_share"], 38 / 44882, places=9)
        self.assertNotEqual(d["D"]["band_f1"], d["D (orig)"]["band_f1"])
        for r in (d["D"], d["D (orig)"]):
            self.assertEqual(r["llm_share_bar"], "pass")                  # at most 20% of band pairs
            self.assertIn(r["f1_bar"].split(" (")[0], ("pass", "fail", "inconclusive"))
        self.assertEqual(d["D (orig)"]["basis"].split(";")[-1].strip(), "original routing rule")

    def test_the_two_d_rows_are_scored_from_their_own_reports(self):
        self.assertEqual(BT.REPORTS["D"], "options/d_hybrid/score_report_fullband.txt")
        self.assertEqual(BT.REPORTS["D_original"], "options/d_hybrid/score_report_fullband_original.txt")
        self.assertTrue((DEV / BT.REPORTS["D"]).exists() and (DEV / BT.REPORTS["D_original"]).exists())


class ResultsTableTests(unittest.TestCase):
    def fake_run(self, with_original: bool) -> Path:
        run = Path(tempfile.mkdtemp())
        fields = BT.HEADER
        rows = []
        for opt in ("A", "B", "C", "D") + (("D (orig)",) if with_original else ()) + ("E",):
            r = {k: "" for k in fields}
            r.update({"option": opt, "band_f1": "0.9", "band_f1_ci_lo": "0.8", "band_f1_ci_hi": "0.95", "band_precision": "0.9", "band_recall": "0.9",
                      "overall_precision": "0.99", "overall_recall": "0.95", "meets_every_bar": "yes", "f1_bar": "pass (x)"})
            if opt.startswith("D"):
                r["llm_share"] = "0.002" if opt == "D" else "0.001"
            rows.append(r)
        with (run / "bars_table.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        (run / "cost").mkdir()
        with (run / "cost" / "options_vs_bars.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["option", "total_usd_month_at_1M_per_day"])
            for opt, v in (("A", 1), ("B", 2), ("C", 3), ("D", 4), ("D_original", 5), ("E", "")):
                w.writerow([opt, v])
        meta = run / "options" / "b_llm" / "openai" / "sample5000"
        meta.mkdir(parents=True)
        (meta / "run_meta.json").write_text(json.dumps({"spent_usd": 1.0, "calls": 1000}), encoding="utf-8")
        return run

    def test_secondary_row_reads_its_own_cost_and_share(self):
        run = self.fake_run(True)
        rows = {r["option"]: r for r in RT.build(run)}
        self.assertEqual(list(rows), ["A", "B", "C", "D", "D (orig)", "E"])
        self.assertEqual(rows["D"]["total_usd_month_at_1M_records_per_day"], "4")
        self.assertEqual(rows["D (orig)"]["total_usd_month_at_1M_records_per_day"], "5")
        self.assertEqual(rows["D"]["llm_usd_per_1000_band_pairs"], f"{0.002 * 0.001 * 1000:.4f}")
        self.assertEqual(rows["D (orig)"]["llm_usd_per_1000_band_pairs"], f"{0.001 * 0.001 * 1000:.4f}")

    def test_a_run_without_the_secondary_row_still_builds(self):
        rows = RT.build(self.fake_run(False))
        self.assertEqual([r["option"] for r in rows], ["A", "B", "C", "D", "E"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for eval/head_to_head.py on a small hand-built run folder: every cell is read from a file or derived by stated arithmetic, a figure
the files do not hold prints "missing" with its expected source, files outside the allowed roots and data or truth files are refused,
and the script writes only into its output folder.

  python eval/test_head_to_head.py
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
import head_to_head as H  # noqa: E402

REPORT = """Option: {name}
Run: x
Decisions: x

Band (test band pairs, same-entity = positive)
  pairs 100  (true 10, 10.00%)
  F1 {f1}  95% CI [{lo}, {hi}]  (bootstrap, 1000 resamples, seed 0)
  accuracy 0.9000  precision {bp}  recall {br}
  TP 5  FP 1  FN 1  TN 93

Overall (test group; outside the band Verity's decision_status)
  precision {op}  recall {orc}  F1 0.5000

Decoy pairs in the test group: {dp}, wrongly matched {dw} ({dpct}%)
{ece}"""


def report(name, f1, lo, hi, bp, br, op, orc, ece=None, dp="20", dw="1", dpct="5.00"):
    return REPORT.format(name=name, f1=f1, lo=lo, hi=hi, bp=bp, br=br, op=op, orc=orc, dp=dp, dw=dw, dpct=dpct,
                         ece="" if ece is None else f"\nCalibration\n  ECE {ece}   (bar: <= 0.05)\n")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.repo_rel = lambda p: p
        self.run = self.root / "results" / "eval_7919"
        self.freeze = self.root / "results" / "freeze_2026-11-25"
        self.out = self.run / "head_to_head"
        self.write_all()

    def tearDown(self):
        self.tmp.cleanup()

    def w(self, rel: str, text: str, base=None) -> None:
        p = (base or self.run) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def wcsv(self, rel: str, header: list[str], rows: list[list], banner=True) -> None:
        import io
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        if banner:
            buf.write("# DRAFT: test banner\n")
        writer.writerow(header)
        writer.writerows(rows)
        self.w(rel, buf.getvalue())

    def write_all(self):
        o = "options/"
        self.w(o + "a_threshold/score_report_sample5000.txt", report("A", "0.1000", "0.0500", "0.1500", "1.0000", "0.0500", "0.9900", "0.5000"))
        self.w(o + "b_llm/openai/sample5000/score_report_sample5000.txt", report("B", "0.9000", "0.8000", "1.0000", "0.9500", "0.9500", "0.9800", "0.6000", ece="0.0100"))
        self.w(o + "c_model/c_v1_lgbm/score_report_sample5000.txt", report("C", "0.8000", "0.7000", "0.9000", "0.9000", "0.7500", "0.9700", "0.5500", ece="0.0200"))
        self.w(o + "d_hybrid/score_report_sample5000.txt", report("D", "0.8800", "0.8000", "0.9500", "0.9100", "0.8700", "0.9800", "0.5900", ece="0.0300"))
        self.w(o + "d_hybrid/score_report_sample5000_original.txt", report("Do", "0.8200", "0.7500", "0.9000", "0.9000", "0.8000", "0.9700", "0.5600", ece="0.0250"))
        self.w(o + "a_threshold/score_report.txt", report("A", "0.1100", "0.0600", "0.1600", "1.0000", "0.0600", "0.9900", "0.5000"))
        self.w(o + "c_model/c_v1_lgbm/score_report.txt", report("C", "0.8100", "0.7100", "0.9100", "0.9100", "0.7600", "0.9700", "0.5500", ece="0.0200", dp="30", dw="2", dpct="6.67"))
        self.w(o + "d_hybrid/score_report_fullband.txt", report("D", "0.8900", "0.8100", "0.9600", "0.9200", "0.8800", "0.9800", "0.5900", ece="0.0300"))
        self.w(o + "d_hybrid/score_report_fullband_original.txt", report("Do", "0.8300", "0.7600", "0.9100", "0.9100", "0.8100", "0.9700", "0.5600", ece="0.0250"))
        bh = ["option", "basis", "band_f1", "f1_diff_vs_B", "f1_diff_lo", "f1_diff_hi", "f1_bar", "llm_share", "meets_every_bar"]
        self.wcsv("bars_table.csv", bh, [["A", "x", 0.11, -0.8, -0.9, -0.7, "fail", "", "no"], ["B", "x", 0.9, "", "", "", "reference (B is the baseline)", "", "yes"],
                                          ["C", "x", 0.81, -0.1, -0.2, 0.01, "inconclusive (straddles)", "", "not shown (inconclusive)"],
                                          ["D", "x", 0.89, -0.02, -0.05, 0.01, "pass (lower end above)", 0.01, "yes"],
                                          ["D (orig)", "x", 0.83, -0.07, -0.1, 0.0, "inconclusive (straddles)", 0.001, "not shown (inconclusive)"]], banner=False)
        rh = ["option", "llm_usd_per_1000_band_pairs"]
        self.wcsv("results_table.csv", rh, [["A", "0.0000"], ["B", "2.0000"], ["C", "0.0000"], ["D", "0.0200"], ["D (orig)", "0.0010"]], banner=False)
        ch = ["labels_for_C_and_D", "hourly_rate_usd", "records_per_day", "option", "option_row", "scenario", "precisely_usd_month", "customer_usd_month", "total_usd_month",
              "llm_paid_by", "hosting_paid_by", "operations_paid_by", "development_paid_by", "labeling_paid_by"]
        rows = []
        for opt, base in (("A", 10), ("B", 1000), ("C", 20), ("D", 100)):
            for vol, mult in ((100000, 1), (1000000, 2), (10000000, 3)):
                rows.append([500, 100.0, vol, opt, opt, "S1", base * mult, 5, base * mult + 5, "Precisely", "Precisely", "Precisely", "Precisely", "customer"])
                rows.append([500, 50.0, vol, opt, opt, "S1", 1, 1, 2, "Precisely", "Precisely", "Precisely", "Precisely", "customer"])        # other rate: must not be picked
                rows.append([77, 100.0, vol, opt, opt, "S1", 9, 9, 18, "Precisely", "Precisely", "Precisely", "Precisely", "customer"])      # other labels
        rows.append([500, 100.0, 1000000, "D", "D (LLM fallback off)", "S3", "n/m", 7, 7, "customer", "customer", "customer", "Precisely (not modeled)", "customer"])
        self.wcsv("cost/cost_model.csv", ch, rows)
        self.wcsv("cost/options_vs_bars.csv", ["option", "total_usd_month_at_1M_per_day"], [["A", 1], ["D_original", 555.5]])
        bh = ["scenario", "labels_for_C_and_D", "hourly_rate_usd", "option", "break_even_records_per_day", "result"]
        brows = []
        for opt in "ACD":
            for rate in ("50.0", "100.0", "200.0"):
                for lab in ("500", "2000"):
                    brows.append(["S1 and S2 (same total)", lab, rate, opt, "", "option cheaper everywhere in range"])
        brows[brows.index(next(r for r in brows if r[3] == "D" and r[2] == "100.0" and r[1] == "500"))] = ["S1 and S2 (same total)", "500", "100.0", "D", 1234.4, "option cheaper above"]
        self.wcsv("cost/break_even.csv", bh, brows)
        self.wcsv("cost/assumptions.csv", ["name", "value", "unit", "status", "source", "note"],
                  [["band pairs per record", "10.0", "pairs", "measured", "band_summary.json", ""], ["labels as evaluated (C and D)", "500", "labels", "measured", "x", ""],
                   ["D per-pair latency with the LLM fallback (simulation)", "p50 0.05 ms, p95 0.1 ms", "ms", "measured-derived", "x", ""]])
        self.w("options/latency/latency_summary.json", json.dumps({"C": {"median_of_repeats": {"p50_ms": 0.08, "p95_ms": 0.28}}, "D": {"median_of_repeats": {"p50_ms": 0.09, "p95_ms": 0.29}}}))
        self.w("options/b_llm/openai/sample5000/timings.json", json.dumps({"timings": [{"stage": "optionB.llm_call", "per_item_ms": {"p50": 1000.0, "p95": 1500.0}}]}))
        self.w("options/b_llm/openai/decoycheck/decoy_report.txt", "every pair is a different-organization pair: false matches 0 of 66\n  false-match rate <= 0.04438 (up to 3)\n")
        self.w("options/b_llm/openai/decoycheck/run_meta.json", json.dumps({"calls": 52, "cache_hits": 14, "spent_usd": 0.0785}))
        self.wcsv("cost/one_time_cost.csv", ["option", "hourly_rate_usd", "build_before_usd", "build_after_usd", "labeling_2000_labels_at_100_per_hour_usd",
                                             "labeling_as_evaluated_labels_at_100_per_hour_usd"], [["C", 100.0, 18000, 18450, 1000, 33015.5]])
        self.w("cost/summary.txt", "DRAFT: test banner\n\nLabel rows\n  C     as evaluated     500       2.5          125/250/500              0.9 (full band)        1\n")
        self.w("freeze_report.txt", "x\n  effort log read as it is (not frozen): sha256 " + "ab" * 32 + "\n", base=self.freeze)

    def build(self):
        R = H.Reader([self.run, self.freeze])
        return H.build(self.run, self.freeze, R), R


class CellTests(Fixture):
    def setUp(self):
        super().setUp()
        H.REPO_BACKUP = H.REPO
        H.REPO = self.root                                   # the script reports paths relative to the repo root
        self.res, self.R = self.build()
        self.T = self.res["T"]

    def tearDown(self):
        H.REPO = H.REPO_BACKUP
        super().tearDown()

    def test_quality_cells_are_read_from_the_sample_reports_and_the_bars_table(self):
        self.assertEqual(self.T.get("B", "band F1 [95% CI]"), "0.9000 [0.8000, 1.0000]")
        self.assertEqual(self.T.get("D", "band precision"), "0.9100")
        self.assertEqual(self.T.get("D", "overall recall"), "0.5900")
        self.assertEqual(self.T.get("C", "ECE"), "0.0200")
        self.assertEqual(self.T.get("D", "F1 vs B [95% CI]"), "-0.0200 [-0.0500, +0.0100]")
        self.assertEqual(self.T.get("D", "F1 verdict"), "pass")
        self.assertEqual(self.T.get("C", "F1 verdict"), "inconclusive")
        self.assertEqual(self.T.get("B", "F1 vs B [95% CI]"), "baseline")
        self.assertEqual(self.T.get("D", H.EVERY_SAMPLE), "yes")
        self.assertEqual(self.T.cells[("D", "band F1 [95% CI]")][1], "results/eval_7919/options/d_hybrid/score_report_sample5000.txt")
        self.assertEqual(self.T.cells[("D (orig)", "band F1 [95% CI]")][1], "results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt")

    def test_cost_per_record_is_the_hand_computed_arithmetic(self):
        self.assertEqual(self.T.get("B", "LLM $ per 1,000 band pairs"), "2.0000")
        self.assertEqual(self.T.get("B", "LLM $ per record"), f"{2.0 * 10 / 1000:.5f}")            # 2.0 per 1,000 pairs x 10 pairs per record / 1,000 = 0.02
        self.assertEqual(self.T.get("B", "LLM $ per record"), "0.02000")
        self.assertEqual(self.T.get("D", "LLM $ per record"), "0.00020")
        self.assertEqual(self.T.cells[("B", "LLM $ per record")][2].split(":")[0], "derived")

    def test_monthly_cost_takes_s1_100_dollars_and_the_labels_as_evaluated(self):
        self.assertEqual(self.T.get("D", "monthly $ at 1M records/day (Precisely / customer / total)"), "200.00 / 5.00 / 205.00")   # base 100 x 2
        self.assertEqual(self.T.get("A", "monthly $ at 100K records/day (Precisely / customer / total)"), "10.00 / 5.00 / 15.00")
        self.assertEqual(self.T.get("B", "monthly $ at 10M records/day (Precisely / customer / total)"), "3,000.00 / 5.00 / 3,005.00")

    def test_missing_figures_say_missing_and_name_the_expected_source(self):
        for col in ("monthly $ at 100K records/day (Precisely / customer / total)", "monthly $ at 10M records/day (Precisely / customer / total)"):
            txt = self.T.get("D (orig)", col)
            self.assertTrue(txt.startswith("missing (expected source: cost/cost_model.csv"), txt)
        self.assertEqual(self.T.get("D (orig)", "monthly $ at 1M records/day (Precisely / customer / total)"), "missing / missing / 555.50 (total only)")
        self.assertEqual(self.T.get("A", "latency per pair (p50, p95)"), "not measured (threshold comparison)")
        self.assertTrue(self.T.get("D (orig)", "break-even vs B at $100/h, labels as evaluated").startswith("missing (expected source: cost/break_even.csv"))

    def test_latency_and_break_even(self):
        self.assertEqual(self.T.get("C", "latency per pair (p50, p95)"), "p50 0.08 ms, p95 0.28 ms (one core, features included)")
        self.assertEqual(self.T.get("B", "latency per pair (p50, p95)"), "p50 1.000 s, p95 1.500 s (per LLM call)")
        self.assertIn("p95 0.1 ms", self.T.get("D", "D fallback tail"))
        self.assertEqual(self.T.get("D", "break-even vs B at $100/h, labels as evaluated"), "option cheaper above 1,234 records/day")
        self.assertEqual(self.T.get("D", "break-even vs B at $100/h, 2,000 labels"), "option cheaper everywhere in range")
        self.assertEqual(self.T.get("B", "break-even vs B at $50/h, 2,000 labels"), "baseline")

    def test_full_band_table_uses_the_full_band_reports_and_has_no_b_row(self):
        full = self.res["full"]
        self.assertEqual(H.full_rows(full), ["A", "C", "D"])                               # D (orig) is in the secondary table
        self.assertEqual(H.full_rows(full, H.OPTS), ["A", "C", "D", "D (orig)"])
        self.assertEqual(full.get("D (orig)", "band F1 [95% CI]"), "0.8300 [0.7600, 0.9100]")
        self.assertEqual(full.get("D", "band F1 [95% CI]"), "0.8900 [0.8100, 0.9600]")
        self.assertEqual(full.cells[("C", "band recall")][1], "results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt")

    def test_decoys_and_the_decoy_check(self):
        rows = {(o, b.split(" (")[0]): (n, w) for o, b, n, w, _ in self.res["decoys"]}
        self.assertEqual(rows[("C", "full band")], ("30", "2 (6.67%)"))
        self.assertEqual(rows[("B", "5,000-pair sample")], ("20", "1 (5.00%)"))
        dc = self.res["decoy_check"]
        self.assertEqual((dc["pairs"], dc["false_matches"], dc["live_calls"], dc["cache_hits"], dc["bound"]), (66, 0, 52, 14, "0.04438"))

    def test_slides_numbers(self):
        S = self.res["slides"]
        self.assertEqual(S["effort_log_sha256_frozen_run"], "ab" * 32)
        self.assertEqual(S["draft_banner"], "DRAFT: test banner")
        self.assertEqual(S["label_cost"][("C", "as evaluated")], ("500", "2.5", "125/250/500"))
        self.assertEqual(S["build"], [("C", "100.0", "18000", "18450", "1000", "33015.5")])
        self.assertEqual([r for r in S["who_pays"] if r[1] == "S3"][0][2], "n/m")


class BarsAndBasesTests(Fixture):
    """The two 'meets every bar' columns, the bases text, the secondary table and --cost-dir."""

    def setUp(self):
        super().setUp()
        H.REPO_BACKUP, H.REPO = H.REPO, self.root

    def tearDown(self):
        H.REPO = H.REPO_BACKUP
        super().tearDown()

    def build(self, cost_dir=None):
        R = H.Reader([self.run, self.freeze])
        return H.build(self.run, self.freeze, R, cost_dir), R

    def test_both_columns_by_the_protocol_rules_hand_computed(self):
        res, _ = self.build()
        T = res["T"]
        # sample basis (fixture): A recall 0.5000. B p 0.98 r 0.60 -> yes. C p 0.9700 < 0.98 -> fail -> no. D p 0.98, r 0.59 >= 0.55, ECE 0.03, share 0.01, F1 pass -> yes
        self.assertEqual([T.get(o, H.EVERY_SAMPLE) for o in H.MAIN], ["no", "yes", "no", "yes"])
        self.assertEqual([T.get(o, H.EVERY_FULL) for o in H.MAIN], ["no", "n/a (B exists only on the sample)", "no", "yes"])
        self.assertEqual(T.get("D (orig)", H.EVERY_FULL), "not computed (column is for A, C and D only)")
        self.assertEqual(res["bar_notes"], [])                                            # the two columns agree

    def test_a_note_names_the_bar_and_both_values_when_the_columns_differ(self):
        self.w("options/d_hybrid/score_report_fullband.txt", report("D", "0.8900", "0.8100", "0.9600", "0.9200", "0.8800", "0.9700", "0.5900", ece="0.0300"))
        res, _ = self.build()
        T = res["T"]
        self.assertEqual(T.get("D", H.EVERY_SAMPLE), "yes")
        self.assertEqual(T.get("D", H.EVERY_FULL), "no")
        self.assertEqual(len(res["bar_notes"]), 1)
        note = res["bar_notes"][0]
        self.assertIn("D: bar 'overall precision >= 0.98'", note)
        self.assertIn("sample basis 0.9800 (pass)", note)
        self.assertIn("full-band basis 0.9700 (fail)", note)
        H.write_outputs(res, self.out)
        self.assertIn("Where the two columns differ:", (self.out / "head_to_head.md").read_text(encoding="utf-8"))

    def test_the_rules_function_directly(self):
        self.assertEqual(H.bar_verdicts("D", 0.99, 0.60, 0.50, 0.04, "pass", None, 0.30)[0], "no")             # LLM share 30% > 20%
        self.assertEqual(H.bar_verdicts("D", 0.99, 0.549, 0.50, 0.04, "pass", None, 0.01)[0], "no")           # recall 0.549 < 0.50 + 0.05
        self.assertEqual(H.bar_verdicts("D", 0.99, 0.55, 0.50, 0.04, "pass", None, 0.01)[0], "yes")           # exactly A + 0.05 passes
        self.assertEqual(H.bar_verdicts("C", 0.99, 0.60, 0.50, 0.04, "pass", 25.0, None)[0], "no")           # latency p95 25 ms > 20 ms
        self.assertEqual(H.bar_verdicts("C", 0.99, 0.60, 0.50, 0.04, "inconclusive", 0.3, None)[0], "not shown (inconclusive)")
        self.assertEqual(H.bar_verdicts("A", 1.0, 0.5, 0.5, None, "pass", None, None)[0], "no")

    def test_bases_text_and_secondary_table_in_the_markdown(self):
        res, _ = self.build()
        H.write_outputs(res, self.out)
        md = (self.out / "head_to_head.md").read_text(encoding="utf-8")
        self.assertIn(H.BARS_TEXT, md)
        self.assertEqual(H.BARS_TEXT, "Quality columns (band F1 and its interval, band precision and recall, overall precision and recall, ECE, F1 versus B) are on the fixed "
                                      "5,000-pair sample. Cost, break-even and latency columns come from the cost model and benchmarks and are not sample figures.")
        main, secondary = md.split("## Secondary row: D (orig)")
        self.assertNotIn("| D (orig) |", main.split("## Table 2")[0])                      # not a row of Table 1
        self.assertIn("missing (expected source: cost/break_even.csv (no row for D (orig)))", secondary)
        self.assertIn("meets every bar (sample basis)", md)
        self.assertIn("meets every bar (full-band basis, A, C and D only)", md)
        self.assertNotIn("**DRAFT", md.split("## Cost numbers")[0])

    def test_cost_dir_selects_the_cost_outputs_and_the_default_is_the_stage_12_folder(self):
        final = self.run / "cost_final"
        for name in ("cost_model.csv", "options_vs_bars.csv", "break_even.csv", "assumptions.csv", "one_time_cost.csv", "summary.txt"):
            (final / name).parent.mkdir(exist_ok=True)
            text = (self.run / "cost" / name).read_text(encoding="utf-8")
            if name == "one_time_cost.csv":
                text = text.replace("18450", "19999")
            if name == "summary.txt":
                text = text.replace("DRAFT: test banner", "Effort log: 36 rows with hours")
            (final / name).write_text(text, encoding="utf-8")
        res_default, _ = self.build()
        res_final, _ = self.build(final)
        self.assertEqual(res_default["slides"]["build"][0][3], "18450")
        self.assertEqual(res_final["slides"]["build"][0][3], "19999")
        self.assertEqual(res_final["slides"]["cost_dir"], "results/eval_7919/cost_final")
        self.assertEqual(res_default["slides"]["cost_dir"], "results/eval_7919/cost")
        self.assertTrue(res_default["slides"]["draft_banner"].startswith("DRAFT"))
        self.assertFalse(res_final["slides"]["draft_banner"].startswith("DRAFT"))
        H.write_outputs(res_final, self.out)
        self.assertIn("No DRAFT banner", (self.out / "head_to_head.md").read_text(encoding="utf-8"))
        # the cost dir must lie under the run: anything else is refused
        with self.assertRaises(SystemExit):
            H.Reader([self.run, self.freeze]).path(self.root / "elsewhere")

    def test_hand_check_files_are_never_read(self):
        (self.run / "hand_check").mkdir()
        (self.run / "hand_check" / "hand_check_key.csv").write_text("secret", encoding="utf-8")
        R = H.Reader([self.run, self.freeze])
        with self.assertRaises(SystemExit):
            R.text(self.run / "hand_check" / "hand_check_key.csv")


class ReadingRulesAndOutputs(Fixture):
    def test_files_outside_the_roots_and_data_or_truth_files_are_refused(self):
        R = H.Reader([self.run, self.freeze])
        with self.assertRaises(SystemExit):
            R.text(self.root / "elsewhere.txt")
        (self.run / "orgs_raw_truth.csv").write_text("secret", encoding="utf-8")
        (self.run / "orgs_raw.csv").write_text("data", encoding="utf-8")
        for name in ("orgs_raw_truth.csv", "orgs_raw.csv"):
            with self.assertRaises(SystemExit):
                R.text(self.run / name)
        self.assertEqual(R.read, [])

    def test_a_run_reads_only_allowed_files_and_never_data_or_truth(self):
        H.REPO_BACKUP, H.REPO = H.REPO, self.root
        try:
            (self.run / "orgs_raw_truth.csv").write_text("secret", encoding="utf-8")
            res, R = self.build()
        finally:
            H.REPO = H.REPO_BACKUP
        roots = [self.run.resolve(), self.freeze.resolve()]
        self.assertTrue(R.read)
        for p in R.read:
            self.assertTrue(any(r in p.parents for r in roots), p)
            self.assertNotIn("truth", p.name)
            self.assertFalse(p.name.startswith("orgs_raw"))

    def test_outputs_are_written_only_into_the_output_folder_and_hold_the_bases_and_sources(self):
        H.REPO_BACKUP, H.REPO = H.REPO, self.root
        try:
            res, R = self.build()
            before = {p for p in self.root.rglob("*") if p.is_file()}
            H.write_outputs(res, self.out)
        finally:
            H.REPO = H.REPO_BACKUP
        after = {p for p in self.root.rglob("*") if p.is_file()}
        new = after - before
        self.assertEqual({p.name for p in new}, {"head_to_head.csv", "head_to_head.md", "head_to_head_sources.csv"})
        self.assertTrue(all(self.out.resolve() in p.resolve().parents for p in new))
        md = (self.out / "head_to_head.md").read_text(encoding="utf-8")
        self.assertIn("sample basis", md.lower())
        self.assertIn("full band, not comparable with B's sample figures", md.lower().replace("full band, not comparable with b's", "full band, not comparable with B's"))
        self.assertIn("## Sources", md)
        with (self.out / "head_to_head_sources.csv").open(newline="", encoding="utf-8") as f:
            src = list(csv.DictReader(f))
        cells = [r for r in src if r["table"] == "sample basis"]
        self.assertTrue(all(r["source_file"] or r["how"] != "read" or r["column"] == "D fallback tail" for r in cells))
        row = next(r for r in cells if r["row"] == "D" and r["column"] == "band F1 [95% CI]")
        self.assertEqual(row["source_file"], "results/eval_7919/options/d_hybrid/score_report_sample5000.txt")
        with (self.out / "head_to_head.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0][0], "option")
        self.assertEqual([r[0] for r in rows[1:5]], ["A", "B", "C", "D"])                  # D (orig) is a secondary row below
        self.assertIn(["# secondary row"], rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)

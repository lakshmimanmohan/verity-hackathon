"""
Tests for eval/demo_pairs.py on a small hand-computed run folder: the selection rule (test-group pairs only, sorted ids, first by the rule), that selection
needs no truth, the correct / not correct marks, the time and cost per pair, a missing live call, and the single pair set printed.

  python eval/test_demo_pairs.py
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
import demo_pairs as DP  # noqa: E402


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class Fixture(unittest.TestCase):
    """Eight records: R1..R6 in the test group, R7 and R8 in train.
    scored_pairs: (R7,R1) match [train: skipped], (R4,R3) match -> (R3,R4), (R2,R1) match -> (R1,R2)  => pair 1 = (R1,R2).
    D: (R5,R4) c_same same -> (R4,R5); (R2,R3) c_same but different [C decides different: skipped]; (R1,R5) c_same same -> (R1,R5); (R7,R1) c_same [train];
       (R6,R3) llm -> (R3,R6); (R2,R5) llm -> (R2,R5)  => pair 2 = (R1,R5), pair 3 = (R2,R5).
    Truth: R1 and R2 are entity E1, R5 is E2. Pair 1 (R1, R2) is the same entity and the final answer is same: correct. Pair 2 (R1, R5) is different entities
    and the final answer is same: wrong. Pair 3 (R2, R5) is different entities, B says different and the final answer is different: correct. N_CORRECT = 2."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "results" / "eval_7919"
        r = self.run
        write_csv(r / "split_records.csv", ["record_id", "side"], [[f"R{i}", "test"] for i in range(1, 7)] + [["R7", "train"], ["R8", "train"]])
        write_csv(r / "verity_data" / "scored_pairs.csv", ["record_a", "record_b", "score", "match_type", "decision_status"],
                  [["R7", "R1", "1.0", "full_signal", "match"], ["R4", "R3", "0.95", "full_signal", "match"], ["R2", "R1", "0.97", "weak_signal", "match"],
                   ["R1", "R5", "0.84", "insufficient_evidence", "review"], ["R2", "R5", "0.82", "below_threshold", "review"], ["R5", "R4", "0.86", "full_signal", "review"]])
        write_csv(r / "options" / "d_hybrid" / "decisions_fullband.csv", ["record_a", "record_b", "decision", "confidence", "route"],
                  [["R5", "R4", "same", "1.0", "c_same"], ["R2", "R3", "different", "0.95", "c_same"], ["R1", "R5", "same", "0.99", "c_same"], ["R7", "R1", "same", "1.0", "c_same"],
                   ["R6", "R3", "same", "0.9", "llm"], ["R2", "R5", "different", "0.1", "llm"]])
        write_csv(r / "options" / "c_model" / "c_v1_lgbm" / "decisions.csv", ["record_a", "record_b", "decision", "confidence", "model"],
                  [["R1", "R5", "same", "0.99", "c"], ["R5", "R2", "different", "0.3", "c"], ["R5", "R4", "same", "1.0", "c"]])
        write_csv(r / "options" / "b_llm" / "openai" / "dfull" / "decisions.csv", ["record_a", "record_b", "decision", "confidence", "model", "settings"],
                  [["R2", "R5", "different", "0.1000", "m", "s"]])
        header = ["record_id", "company_name", "address_line", "city", "state", "postal_code", "phone", "website", "source_system"]
        write_csv(r / "orgs_raw.csv", header, [[f"R{i}", f"Name {i}", f"{i} Main St", f"City{i}", "NY", f"1000{i}", f"555000{i}", f"site{i}.com", "x"] for i in range(1, 9)])
        write_csv(r / "orgs_raw_truth.csv", ["record_id", "true_entity_id", "variants", "decoy_group"],
                  [["R1", "E1", "", ""], ["R2", "E1", "", ""], ["R3", "E3", "", ""], ["R4", "E4", "", ""], ["R5", "E2", "", ""], ["R6", "E6", "", ""], ["R7", "E7", "", ""], ["R8", "E8", "", ""]])
        (r / "verity_data" / "column_classification.json").write_text(json.dumps({"record_id_column": "record_id"}), encoding="utf-8")
        (r / "verity_data" / "run_info.json").write_text(json.dumps({"runs": [{"timings": [{"stage": "verity.score", "items": 100, "per_item_ms": {"p50": 0.2, "p95": 0.6, "max": 5}}]}]}), encoding="utf-8")
        rep = {"p50_ms": 0.08, "p95_ms": 0.28, "max_ms": 2.0, "mean_ms": 0.1}
        (r / "options" / "latency").mkdir(parents=True)
        (r / "options" / "latency" / "latency_summary.json").write_text(json.dumps({"C": {"median_of_repeats": rep}}), encoding="utf-8")
        calls = [{"a": "R5", "b": "R2", "mode": "live", "usd": 0.002, "latency_s": 0.5, "error": None}, {"a": "R9", "b": "R9", "mode": "live", "usd": 0.5, "latency_s": 9.0, "error": None}]
        for lab in ("sample5000", "dfull", "decoycheck"):
            (r / "options" / "b_llm" / "openai" / lab).mkdir(parents=True, exist_ok=True)
            (r / "options" / "b_llm" / "openai" / lab / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in (calls if lab == "dfull" else [])) + "\n", encoding="utf-8")
        DP.REPO_BACKUP, DP.REPO = DP.REPO, self.root

    def tearDown(self):
        DP.REPO = DP.REPO_BACKUP
        self.tmp.cleanup()


class SelectionTests(Fixture):
    def test_the_rule_picks_the_first_pair_of_each_kind_among_test_group_pairs(self):
        test = {f"R{i}" for i in range(1, 7)}
        picked = DP.select_pairs(DP.read_csv_rows(self.run / "verity_data" / "scored_pairs.csv"), list(DP.read_csv_rows(self.run / "options" / "d_hybrid" / "decisions_fullband.csv")), test)
        self.assertEqual(picked, {1: ("R1", "R2"), 2: ("R1", "R5"), 3: ("R2", "R5")})

    def test_selection_needs_no_truth(self):
        import inspect
        self.assertEqual(list(inspect.signature(DP.select_pairs).parameters), ["scored_rows", "d_rows", "test_records"])
        (self.run / "orgs_raw_truth.csv").unlink()
        test = {f"R{i}" for i in range(1, 7)}
        picked = DP.select_pairs(DP.read_csv_rows(self.run / "verity_data" / "scored_pairs.csv"), list(DP.read_csv_rows(self.run / "options" / "d_hybrid" / "decisions_fullband.csv")), test)
        self.assertEqual(picked[1], ("R1", "R2"))

    def test_train_pairs_and_c_different_pairs_are_skipped_and_an_empty_rule_gives_none(self):
        test = {"R1"}                                                                           # only R1 in the test group: no pair has both records in it
        picked = DP.select_pairs(DP.read_csv_rows(self.run / "verity_data" / "scored_pairs.csv"), list(DP.read_csv_rows(self.run / "options" / "d_hybrid" / "decisions_fullband.csv")), test)
        self.assertEqual(picked, {1: None, 2: None, 3: None})

    def test_live_call_lookup_either_order_and_missing(self):
        files = [self.run / "options" / "b_llm" / "openai" / lab / "calls.jsonl" for lab in ("sample5000", "dfull", "decoycheck")]
        self.assertEqual(DP.find_live_call(files, ("R2", "R5"))[:2], (0.5, 0.002))              # stored as (R5, R2)
        self.assertIsNone(DP.find_live_call(files, ("R1", "R2")))
        self.assertIsNone(DP.find_live_call([self.run / "nope.jsonl"], ("R2", "R5")))


class BuildTests(Fixture):
    def test_fields_marks_times_and_costs(self):
        res = DP.build(self.run)
        p1, p2, p3 = res["pairs"]
        self.assertEqual([p["pair"] for p in res["pairs"]], [("R1", "R2"), ("R1", "R5"), ("R2", "R5")])
        self.assertEqual(p1["records"]["A"]["company_name"], "Name 1")
        self.assertEqual((p1["verity_score"], p1["verity_status"], p1["final"]), ("0.97", "match", "same"))
        self.assertTrue(p1["correct"])                                                          # R1 and R2 are both E1
        self.assertFalse(p2["correct"])                                                         # final same, but E1 vs E2
        self.assertEqual((p2["c_p"], p2["c_decision"], p2["final"]), ("0.99", "same", "same"))
        self.assertEqual((p3["b_p"], p3["b_decision"], p3["final"]), ("0.1000", "different", "different"))
        self.assertTrue(p3["correct"])                                                          # R2 and R5 are different entities, final different
        self.assertEqual(res["n_correct"], 2)
        self.assertTrue(p1["time"].startswith("0.2 ms") and "scored pairs" in p1["time"])
        self.assertTrue(p2["time"].startswith("0.08 ms") and "median over all test band pairs" in p2["time"])
        self.assertEqual((p3["time"], p3["cost"]), ("0.500 s (this pair's live call)", "$0.002000 (this pair's live call)"))
        self.assertEqual((p1["cost"], p2["cost"]), ("$0", "$0"))

    def test_a_pair_without_a_live_call_prints_missing(self):
        (self.run / "options" / "b_llm" / "openai" / "dfull" / "calls.jsonl").write_text("\n", encoding="utf-8")
        p3 = DP.build(self.run)["pairs"][2]
        self.assertEqual((p3["time"], p3["cost"], p3["time_source"]), ("missing", "missing", "missing"))

    def test_the_markdown_and_csv_show_only_these_three_pairs(self):
        res = DP.build(self.run)
        md = DP.render(res)
        self.assertEqual(md.count("## Pair "), 3)
        self.assertIn("N_CORRECT = 2 of 3", md)
        for rid in ("R7", "R8", "R3", "R4", "R6"):                                              # records of pairs not picked never appear
            self.assertNotIn(f"({rid})", md)
            self.assertNotIn(f"record B ({rid})", md)
        out = self.run / "demo_pairs"
        out.mkdir()
        DP.write_csv(res, out / "demo_pairs.csv")
        with (out / "demo_pairs.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["record_a"], r["record_b"]) for r in rows], [("R1", "R2"), ("R1", "R5"), ("R2", "R5")])
        self.assertEqual([r["correct"] for r in rows], ["yes", "no", "yes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

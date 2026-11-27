"""
Tests for eval/blind_review.py on a synthetic run folder (no Verity run, arbitrary column names): which pairs count as code-path, the fill order of the sample, that
review_pairs.csv carries no Verity decision, determinism with the seed, and the agreement arithmetic of --score.

  python eval/test_blind_review.py
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
import blind_review as BR  # noqa: E402

SP_COLS = ["record_a", "record_b", "score", "match_type", "decision_status", "pair_source"]


def build(root: Path):
    """Ten records: R0-R5 have single-token text fields (code path), R6-R9 have multi-word text fields (text path)."""
    run = root / "run"
    V = run / "verity_data"
    V.mkdir(parents=True)
    src = root / "in.csv"
    rows = [[f"R{i}", f"K{i}", f"G{i}", "two words here" if i >= 6 else f"tok{i}"] for i in range(10)]
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Rec Key", "Item Ref", "Barcode", "Label Text"])
        w.writerows(rows)
    (V / "column_classification.json").write_text(json.dumps({
        "record_id_column": "rec key", "content_columns": ["Item Ref", "Barcode", "Label Text"], "metadata_columns": ["Rec Key"],
        "semantic_types": {"Item Ref": "QUASI_IDENTIFIER", "Barcode": "QUASI_IDENTIFIER", "Label Text": "DESCRIPTIVE"}}), encoding="utf-8")
    (V / "run_info.json").write_text(json.dumps({"runs": [{"input_csv": str(src)}]}), encoding="utf-8")
    sp = []
    # code-path pairs among R0-R5: 6 code_signal matches, 4 near-threshold non-matches (review), 5 plain non-matches
    ids = [f"R{i}" for i in range(6)]
    code_pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]            # 15 pairs
    for n, (a, b) in enumerate(code_pairs):
        if n < 6:
            sp.append([a, b, "0.95", "code_signal", "match", "neighbor"])
        elif n < 10:
            sp.append([a, b, "0.80", "insufficient_evidence", "review", "neighbor"])
        else:
            sp.append([a, b, "0.30", "below_threshold", "non_match", "neighbor"])
    sp.append(["R6", "R7", "0.99", "weak_signal", "match", "neighbor"])              # text path: never sampled
    sp.append(["R0", "R1x", "1.0", "full_signal", "match", "exact"])                 # not a neighbor pair: never sampled
    with (V / "scored_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SP_COLS)
        w.writerows(sp)
    return run, src


class Tests(unittest.TestCase):
    def test_code_path_test_matches_the_guard(self):
        self.assertTrue(BR.is_code_path({"t": "x"}, {"t": "y z"}, ["t"]))            # one record without a space
        self.assertFalse(BR.is_code_path({"t": "x w"}, {"t": "y z"}, ["t"]))         # both have a space: text path
        self.assertFalse(BR.is_code_path({"t": "x"}, {"t": "y"}, []))                # no guard column: the guard does not apply

    def test_fill_order_matches_then_review_then_random(self):
        with tempfile.TemporaryDirectory() as d:
            run, src = build(Path(d))
            info = BR.write_review([run], [src], run / "review", 0, 12, 4)
            self.assertEqual(info["code_path_pairs_available"], 15)
            self.assertEqual(info["by_group"], {"code_signal_match": 4, "in_review": 4, "random_code_path": 4})      # 4 matches (cap), 4 review pairs exist, 4 random fill the rest
            info = BR.write_review([run], [src], run / "review2", 0, 9, 20)
            self.assertEqual(info["by_group"], {"code_signal_match": 6, "in_review": 3})                              # all 6 matches, then review pairs, then nothing is left to fill
            info = BR.write_review([run], [src], run / "review3", 0, 30, 20)
            self.assertEqual(info["written"], 15)                                                                       # never more than the code-path pairs available

    def test_conflict_cap_then_random(self):
        with tempfile.TemporaryDirectory() as d:
            run, src = build(Path(d))
            kr = BR.read_rows(run / "verity_data" / "scored_pairs.csv")
            for r in kr[6:10]:                                                      # make the four review pairs code_conflict pairs
                r["match_type"] = "code_conflict"
            with (run / "verity_data" / "scored_pairs.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(kr[0]))
                w.writeheader()
                w.writerows(kr)
            info = BR.write_review([run], [src], run / "o", 0, 12, 3, 2)
            self.assertEqual(info["by_group"], {"code_signal_match": 3, "code_conflict_review": 2, "random_code_path": 7})
            info = BR.write_review([run], [src], run / "o2", 0, 12, 3, 10)
            self.assertEqual(info["by_group"], {"code_signal_match": 3, "code_conflict_review": 4, "random_code_path": 5})

    def test_no_decision_in_the_review_file_and_two_runs_pool(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run1, src1 = build(root / "one")
            run2, src2 = build(root / "two")
            info = BR.write_review([run1, run2], [src1, src2], root / "out", 0, 20, 5)
            self.assertEqual(info["code_path_pairs_available"], 30)
            self.assertEqual(info["written"], 20)
            self.assertEqual(set(info["by_run"]), {"run"})                                                               # both run folders are named "run"
            pairs = BR.read_rows(root / "out" / "review_pairs.csv")
            self.assertEqual(list(pairs[0])[:4], ["pair_id", "run", "record_a", "record_b"])
            self.assertIn("Item Ref_a", pairs[0])
            self.assertTrue(all(r["your_label"] == "" for r in pairs))
            text = (root / "out" / "review_pairs.csv").read_text(encoding="utf-8")
            for word in ("code_signal", "review", "match_type", "decision", "score", "verity", "in_review", "code_conflict", "random_code_path"):
                self.assertNotIn(word, text)
            self.assertFalse(any(r["record_a"] in ("R6", "R7") for r in pairs))      # text-path and exact rows are never sampled
            key = BR.read_rows(root / "out" / "review_key.csv")
            self.assertEqual(sorted(r["pair_id"] for r in key), sorted(r["pair_id"] for r in pairs))
            self.assertEqual(sum(int(r["verity_match"]) for r in key if r["group"] == "code_signal_match"), 5)

    def test_same_seed_same_sample_and_other_seed_differs(self):
        with tempfile.TemporaryDirectory() as d:
            run, src = build(Path(d))
            BR.write_review([run], [src], run / "a", 0, 7, 3)
            BR.write_review([run], [src], run / "b", 0, 7, 3)
            BR.write_review([run], [src], run / "c", 1, 7, 3)
            read = lambda n: (run / n / "review_pairs.csv").read_text(encoding="utf-8")
            self.assertEqual(read("a"), read("b"))
            self.assertNotEqual(read("a"), read("c"))

    def test_score_agreement_by_hand(self):
        with tempfile.TemporaryDirectory() as d:
            run, src = build(Path(d))
            out = run / "review"
            BR.write_review([run], [src], out, 0, 8, 4)
            key = {r["pair_id"]: r for r in BR.read_rows(out / "review_key.csv")}
            rows = BR.read_rows(out / "review_pairs.csv")
            # label every pair as Verity decided, except two pairs flipped and one left blank
            labeled = []
            flips = 0
            for i, r in enumerate(rows):
                v = int(key[r["pair_id"]]["verity_match"])
                if i == 0:
                    r["your_label"] = ""
                elif i in (1, 2):
                    r["your_label"] = "no" if v else "yes"
                    flips += 1
                else:
                    r["your_label"] = "yes" if v else "no"
                labeled.append(r)
            with (out / "review_pairs.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(labeled[0]))
                w.writeheader()
                w.writerows(labeled)
            res = BR.score(out)
            self.assertEqual((res["pairs"], res["labeled"], res["skipped"]), (8, 7, 1))
            self.assertEqual(res["agree"], 7 - flips)
            self.assertAlmostEqual(res["agreement"], round((7 - flips) / 7, 4))
            self.assertEqual(sum(res["confusion"].values()), 7)

    def test_label_parsing(self):
        self.assertEqual([BR.parse_label(x) for x in ("1", "Yes", " same ", "0", "NO", "different", "", "?", "maybe")], [1, 1, 1, 0, 0, 0, None, None, None])


if __name__ == "__main__":
    unittest.main(verbosity=2)

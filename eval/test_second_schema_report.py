"""
Tests for eval/second_schema_report.py with synthetic data only: path redaction, the weight comparison, the cell scan (column names masked, vocabulary words
separated, digit cells separated from stated counts), the explanation card with and without per-field data, and the finding when nothing was scored.

  python eval/test_second_schema_report.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import second_schema_report as SS  # noqa: E402


class RedactionTests(unittest.TestCase):
    def test_paths_are_replaced_in_every_written_form(self):
        p = r"C:\Users\someone\Docs\data\file.csv"
        text = f"Source: `{p}` and json {p.replace(chr(92), chr(92) * 2)} and slashes {p.replace(chr(92), '/')}"
        out = SS.redact_text(text, [p])
        self.assertEqual(out.count(SS.PATH_REMOVED), 3)
        self.assertFalse(SS.local_path_left(out))

    def test_attribute_weights_keys_are_replaced_and_nothing_else_changes(self):
        doc = {"source_csv": r"C:\x\a.csv", "column_classification_path": r"C:\x\c.json", "n_rows": 5, "columns": {"a": {"x": 1}}}
        out = SS.redact_attribute_weights(doc)
        self.assertEqual((out["source_csv"], out["column_classification_path"]), (SS.PATH_REMOVED, SS.PATH_REMOVED))
        self.assertEqual((out["n_rows"], out["columns"]), (5, {"a": {"x": 1}}))
        self.assertEqual(doc["source_csv"], r"C:\x\a.csv")                                 # the original is not modified

    def test_local_path_detection(self):
        self.assertTrue(SS.local_path_left(r"see C:\Users\x"))
        self.assertTrue(SS.local_path_left("OneDrive - Org"))
        self.assertFalse(SS.local_path_left("results/second_schema/summary"))


class ComparisonTests(unittest.TestCase):
    def test_weight_comparison_hand_computed(self):
        cols = {"A": {"normalized_discriminative_weight": 0.362878}, "B": {"normalized_discriminative_weight": 0.140000}}
        rows = SS.weight_comparison(cols, {"A": 0.362878, "B": 0.138752, "C": 0.5})
        self.assertEqual(rows[0], ["A", "0.362878", "0.362878", "+0.000000", "no (within 6-decimal rounding)"])
        self.assertEqual(rows[1][3:], ["+0.001248", "yes"])                                  # 0.140000 - 0.138752
        self.assertEqual(rows[2], ["C", "not in this run", "0.500000", "n/a", "n/a"])


class ScanTests(unittest.TestCase):
    def test_scan_masks_column_names_separates_vocabulary_and_stated_counts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "r.md").write_text("columns: Street Name, City Name. Pairs: 12345. Vocab word Source here. A leaked value Springfield appears. Token 98765 too.", encoding="utf-8")
            words = {"Street", "City", "Source", "Springfield", "Absent", "Dorchester"}
            nums = {"12345", "98765", "55555"}
            res = SS.scan_for_cells(d, words, nums, {"12345"}, ["Street Name", "City Name"], "the Source of the report")
            self.assertEqual(res["word_cells_found_as_values"], 1)                           # Springfield
            self.assertEqual(res["word_cells_found_only_as_vocabulary"], 1)                   # Source is a vocabulary word; Street and City sit inside masked column names
            self.assertEqual(res["number_cells_found_not_counts"], 1)                         # 98765 is a cell and not a stated count
            self.assertEqual(res["number_cells_found_that_are_stated_counts"], 1)             # 12345 is a stated count

    def test_input_cells_split_words_and_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.csv"
            p.write_text("Id,Name,Zip\r\n1,Acme Corp,12345\r\n2,Ab,99\r\n3,,100000\r\n", encoding="utf-8-sig")
            words, nums = SS.input_cells(p)
            self.assertEqual(words, {"Acme Corp"})                                            # 'Ab' is too short
            self.assertEqual(nums, {"12345", "100000"})                                       # '99' and the ids are too short


class CardAndFindingTests(unittest.TestCase):
    def test_card_without_per_field_data_says_so(self):
        r = {"field_similarities": "", "field_weights": "", "evidence_field_similarities": "", "score": "1.000000", "match_type": "full_signal",
             "decision_status": "match", "pair_source": "exact", "evidence_mass": "", "conflict_mass": "", "evidence_passes": ""}
        rows, extra = SS.explanation_card((("a", "b"), r))
        self.assertEqual(rows, [])
        self.assertEqual(extra["pair_source"], "exact")
        self.assertIn("not stored", extra["per-field similarities, weights and evidence"])

    def test_card_with_per_field_data_lists_names_numbers_only(self):
        r = {"field_similarities": json.dumps({"f1": 1.0, "f2": 0.5}), "field_weights": json.dumps({"f1": 0.6, "f2": 0.4}), "evidence_field_similarities": json.dumps({"f1": 1.0}),
             "score": "0.9", "match_type": "full_signal", "decision_status": "match", "evidence_mass": "0.5", "conflict_mass": "0.0", "evidence_passes": "true"}
        rows, extra = SS.explanation_card((("a", "b"), r))
        self.assertEqual(rows, [["f1", "1.0000", "0.6000", "1.0000"], ["f2", "0.5000", "0.4000", "n/a"]])
        self.assertEqual(extra["evidence_passes"], "true")

    def test_finding_only_when_nothing_was_scored(self):
        F = {"ri": {"neighbor_pairs": 100, "scored_neighbor_pairs": 0, "skipped_no_semantic_field": 100, "exact_duplicate_pairs": 7}}
        text = "\n".join(SS.finding(F))
        self.assertIn("Of 100 neighbor pairs, 100 were skipped", text)
        self.assertIn("0 were scored", text)
        note = "\n".join(SS.finding({"ri": {"neighbor_pairs": 100, "scored_neighbor_pairs": 5, "skipped_single_token_semantic_fields": 3, "exact_duplicate_pairs": 7}}))
        self.assertIn("**5 were scored** and 3 were skipped", note)                         # a note on what was scored, not the no-scoring finding
        self.assertNotIn("no neighbor pair was scored", note)
        self.assertEqual(SS.finding({"ri": {"neighbor_pairs": 0, "scored_neighbor_pairs": 0}}), [])

    def test_the_header_line(self):
        self.assertEqual(SS.HEADER_LINE, "EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

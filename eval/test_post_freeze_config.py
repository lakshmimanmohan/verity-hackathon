"""
Tests for eval/post_freeze_config.py on a small hand-computed run folder: which learned weight the score used, the setup-time arithmetic, the learned-or-fixed
marks, the header line, the command line taken from the stage log, and the reading rules.

  python eval/test_post_freeze_config.py
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
import post_freeze_config as PC  # noqa: E402

COLS = {
    "name": {"semantic_type": "NAME", "physical_kind": "STRING", "uniqueness_ratio": 0.8, "null_rate": 0.0, "discriminative_score": 0.8,
             "normalized_discriminative_weight": 0.8, "reliability_score": 0.1, "composite_weight": 0.7},
    "city": {"semantic_type": "DESCRIPTIVE", "physical_kind": "STRING", "uniqueness_ratio": 0.2, "null_rate": 0.0, "discriminative_score": 0.2,
             "normalized_discriminative_weight": 0.2, "reliability_score": 0.9, "composite_weight": 0.3},
}


class HandTests(unittest.TestCase):
    def test_pick_weights_used(self):
        self.assertEqual(PC.pick_weights_used({"field_weights": json.dumps({"name": 0.8, "city": 0.2})}, COLS), "normalized_discriminative_weight")
        self.assertEqual(PC.pick_weights_used({"field_weights": json.dumps({"name": 0.7, "city": 0.3})}, COLS), "composite_weight")
        self.assertEqual(PC.pick_weights_used({"field_weights": json.dumps({"name": 0.5, "city": 0.5})}, COLS), "neither (see the file)")
        self.assertEqual(PC.pick_weights_used({"field_weights": json.dumps({})}, COLS), "neither (see the file)")

    def test_setup_times_use_top_level_stages_only(self):
        ri = {"timings": [{"stage": "verity.startup", "parent": None, "seconds": 1.0}, {"stage": "verity.rule_learning", "parent": None, "seconds": 2.0},
                          {"stage": "verity.build_graph", "parent": None, "seconds": 0.5}, {"stage": "verity.embed.encode", "parent": "verity.embed", "seconds": 2.5},
                          {"stage": "verity.embed", "parent": None, "seconds": 3.0}, {"stage": "verity.match", "parent": None, "seconds": 4.0},
                          {"stage": "verity.score", "parent": "verity.match", "seconds": 3.0}, {"stage": "verity.merge", "parent": None, "seconds": 0.25},
                          {"stage": "verity.explore", "parent": None, "seconds": 0.25}, {"stage": "verity.total", "parent": None, "seconds": 11.5}]}
        t = PC.setup_times(ri)
        self.assertEqual(t["sum"], 11.0)                                                   # 1 + 2 + 0.5 + 3 + 4 + 0.25 + 0.25, the children not counted
        self.assertEqual(t["total"], 11.5)
        self.assertEqual([s for _, s in t["rows"]], [1.0, 2.0, 0.5, 3.0, 4.0, 0.25, 0.25])


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "results" / "eval_7919"
        self.freeze = self.root / "results" / "freeze_2026-11-25"
        V = self.run / "verity_data"
        V.mkdir(parents=True)
        (V / "attribute_weights.json").write_text(json.dumps({"n_rows": 100, "n_pairs_sampled": 50, "columns": COLS}), encoding="utf-8")
        (V / "column_classification.json").write_text(json.dumps({"record_id_column": "id", "metadata_columns": ["id"], "reasoning": {"id": "excluded: primary key"}}), encoding="utf-8")
        (V / "learned_threshold.json").write_text(json.dumps({"threshold": 0.4, "n_pairs": 50, "low_cluster_n": 20, "low_cluster_mean": 0.3, "low_cluster_std": 0.05, "sd_multiplier": 2.0}), encoding="utf-8")
        (V / "learned_rules.md").write_text("# Learned matching rules\n", encoding="utf-8")
        (V / "scored_pairs_columns.json").write_text(json.dumps({"effective_threshold": 0.82, "learned_threshold": 0.4, "review_margin": 0.05,
                                                                "score_formula": {"w_embed_blend": 0.65, "w_field_blend": 0.35, "hybrid_embed_weight": 0.65, "hybrid_char_weight": 0.35}}), encoding="utf-8")
        timings = [{"stage": "verity.startup", "parent": None, "seconds": 1.0}, {"stage": "verity.rule_learning", "parent": None, "seconds": 2.0},
                   {"stage": "verity.build_graph", "parent": None, "seconds": 0.5}, {"stage": "verity.embed", "parent": None, "seconds": 3.0},
                   {"stage": "verity.match", "parent": None, "seconds": 4.0}, {"stage": "verity.merge", "parent": None, "seconds": 0.25},
                   {"stage": "verity.explore", "parent": None, "seconds": 0.25}, {"stage": "verity.total", "parent": None, "seconds": 11.0}]
        (V / "run_info.json").write_text(json.dumps({"runs": [{"timings": timings, "scored_pairs_rows": 1234, "n_records_in_index": 100, "nn_mode_resolved": "exact"}]}), encoding="utf-8")
        with (V / "scored_pairs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["record_a", "record_b", "field_weights"])
            w.writerow(["a", "b", json.dumps({"name": 0.8, "city": 0.2})])
            w.writerow(["c", "d", json.dumps({"name": 0.0, "city": 0.0})])
        (self.run / "options" / "d_hybrid").mkdir(parents=True)
        (self.run / "options" / "d_hybrid" / "frozen_settings_v2.json").write_text(json.dumps({"lo": 0.005, "hi": 0.9, "rule_file": "selection_rule_v2.txt", "development_seeds": [11, 12]}), encoding="utf-8")
        (self.freeze / "logs").mkdir(parents=True)
        (self.freeze / "logs" / "stage_02.log").write_text("$ python verity/src/main.py --nn-mode exact --input x.csv   [truth sealed; 2026-11-25T17:16:06Z]\nmore\n", encoding="utf-8")
        PC.REPO_BACKUP, PC.REPO = PC.REPO, self.root

    def tearDown(self):
        PC.REPO = PC.REPO_BACKUP
        self.tmp.cleanup()

    def build(self):
        R = PC.Reader([self.run, self.freeze])
        return PC.build(self.run, self.freeze, R), R


class BuildTests(Fixture):
    def test_build_reads_the_learned_values_the_weights_used_and_the_command(self):
        res, _ = self.build()
        self.assertEqual(res["weights_used"], "normalized_discriminative_weight")            # the first scored pair carries the normalized weights
        self.assertEqual(res["command"], "python verity/src/main.py --nn-mode exact --input x.csv")      # the bracketed note is cut off
        self.assertEqual(res["threshold"]["threshold"], 0.4)
        self.assertEqual(res["settings_v2"]["lo"], 0.005)

    def test_the_markdown_marks_learned_and_fixed_and_starts_with_the_header(self):
        res, _ = self.build()
        md = PC.render(res)
        self.assertEqual(md.splitlines()[0], "EXPLORATORY: not a protocol result; no setting was chosen from these numbers.")
        self.assertEqual(md.count("| learned from the data |"), 3)                           # two learned columns and the learned threshold
        self.assertIn("| name | NAME | 0.8000 | 0.0000 | 0.8000 | 0.1000 | 0.800000 | 0.700000 | learned from the data |", md)
        self.assertIn("| learned threshold | 0.400000 | learned from the data |", md)
        self.assertIn("| effective threshold | 0.82 | fixed floor applied to the learned value |", md)
        self.assertIn("| review margin | 0.05 | fixed setting |", md)
        self.assertIn("| embedding blend weight (record cosine) | 0.65 | fixed setting |", md)
        self.assertIn("| D routing, lo | 0.005 | fixed setting, chosen before the freeze |", md)
        self.assertIn("**No column mapping or parameter was set by hand.**", md)
        self.assertIn("`python verity/src/main.py --nn-mode exact --input x.csv`", md)

    def test_setup_time_rows_and_the_untimed_remainder(self):
        res, _ = self.build()
        md = PC.render(res)
        self.assertIn("| rule learning (column types, weights, threshold) | 2.0 |", md)
        self.assertIn("| **total (verity.total)** | **11.0** |", md)
        self.assertIn("add up to 11.00 s of the 11.00 s total (0.00 s untimed)", md)
        self.assertIn("ready after start-up and rule learning (3.0 s)", md)                  # 1.0 + 2.0

    def test_reading_rules_and_no_writes(self):
        before = {p for p in self.root.rglob("*") if p.is_file()}
        res, R = self.build()
        self.assertEqual({p for p in self.root.rglob("*") if p.is_file()}, before)
        for p in R.read:
            self.assertTrue(self.run.resolve() in p.parents or self.freeze.resolve() in p.parents, p)
        for name in ("orgs_raw.csv", "orgs_raw_truth.csv"):
            with self.assertRaises(SystemExit):
                R.text(self.run / name)
        with self.assertRaises(SystemExit):
            R.text(self.root / "elsewhere.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)

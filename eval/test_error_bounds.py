"""
Tests for eval/error_bounds.py (exact one-sided Clopper-Pearson bounds and the implied worst case).

  python eval/test_error_bounds.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_bounds as E  # noqa: E402


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


class BoundTests(unittest.TestCase):
    def test_zero_errors_matches_the_closed_form(self):
        for n in (10, 250, 750, 44542):
            self.assertAlmostEqual(E.cp_upper(0, n), 1 - 0.05 ** (1 / n), places=10)

    def test_rule_of_three_is_close_for_large_n(self):
        self.assertAlmostEqual(E.cp_upper(0, 750) * 750, 3.0, delta=0.02)     # -ln(0.05) = 2.9957
        self.assertAlmostEqual(E.cp_upper(0, 750), 0.0039863, places=6)
        self.assertAlmostEqual(E.cp_upper(0, 250), 0.0119114, places=6)

    def test_the_bound_solves_its_defining_equation(self):
        for k, n in ((1, 50), (3, 1000), (5, 200)):
            p = E.cp_upper(k, n)
            self.assertAlmostEqual(binom_cdf(k, n, p), 0.05, places=8)

    def test_more_errors_or_fewer_trials_widen_the_bound(self):
        self.assertLess(E.cp_upper(0, 1000), E.cp_upper(1, 1000))
        self.assertLess(E.cp_upper(2, 1000), E.cp_upper(2, 500))
        self.assertEqual(E.cp_upper(5, 5), 1.0)
        self.assertEqual(E.cp_upper(0, 0), 1.0)

    def test_worst_case_on_the_run11_test_band(self):
        w = E.worst_case(fp=0, n_false=750, fn=0, n_true=250, recall_point=1.0)
        self.assertAlmostEqual(w["fp_max"], E.cp_upper(0, 750) * 44542, places=6)
        self.assertAlmostEqual(w["recall"], 1 - E.cp_upper(0, 250), places=10)
        tp = w["recall"] * 340
        self.assertAlmostEqual(w["precision"], tp / (tp + w["fp_max"]), places=10)
        self.assertTrue(w["recall_is_bound"])

    def test_observed_recall_is_kept_when_there_are_false_non_matches(self):
        w = E.worst_case(fp=0, n_false=4660, fn=308, n_true=340, recall_point=32 / 340)
        self.assertAlmostEqual(w["recall"], 32 / 340)
        self.assertFalse(w["recall_is_bound"])

    def test_lines_only_when_a_count_is_zero(self):
        self.assertEqual(E.bound_lines(fp=3, n_false=750, fn=2, n_true=250, recall_point=0.99), [])
        both = E.bound_lines(fp=0, n_false=750, fn=0, n_true=250, recall_point=1.0)
        self.assertTrue(any("false matches" in x for x in both) and any("false non-matches" in x for x in both))
        only_fp = E.bound_lines(fp=0, n_false=750, fn=2, n_true=250, recall_point=0.99)
        self.assertTrue(any("false matches" in x for x in only_fp))
        self.assertFalse(any("false non-matches" in x for x in only_fp))


if __name__ == "__main__":
    unittest.main(verbosity=2)

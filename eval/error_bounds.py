"""
Exact one-sided upper bounds for error rates when an error count is zero or small, and the worst-case
precision and recall they imply on a band of known size.

A bootstrap interval is degenerate when a sample has no errors of one kind (every resample has none
either), so it says nothing about how many errors a larger set could hold. The Clopper-Pearson
(exact binomial) one-sided upper bound does: with k errors in n trials, the upper bound at confidence
c is the p for which P(X <= k | n, p) = 1 - c, which is the c quantile of Beta(k + 1, n - k). For
k = 0 it is 1 - (1 - c)^(1/n) (about 3 / n at 95%).

Used by eval/score_option.py, eval/compare_options.py and options/b_tune_report.py. The test band of
run dev_2026_run11 has 340 true and 44,542 false pairs (BAND_TRUE, BAND_FALSE below), the default
target of the "implied worst case".

The bounds treat the sampled pairs as independent draws from the population the rate is quoted for.
A sample that over-represents hard pairs (for example decoys) gives a bound for that mix, which is
conservative for the population when the over-represented pairs are the harder ones. The false-match
and false-non-match bounds are each one-sided 95%; both holding together is at least 90%.
"""

from __future__ import annotations

from scipy.stats import beta

CONFIDENCE = 0.95
BAND_TRUE = 340
BAND_FALSE = 44542


def cp_upper(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """Exact one-sided upper confidence bound for a binomial rate with k events in n trials."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def worst_case(fp: int, n_false: int, fn: int, n_true: int, recall_point: float,
               band_true: float = BAND_TRUE, band_false: float = BAND_FALSE) -> dict:
    """Bounds from fp errors in n_false false pairs and fn in n_true true pairs, and the precision and
    recall they imply on a band of band_true true and band_false false pairs. A zero false-non-match
    count is replaced by its upper bound; otherwise the observed recall_point is used."""
    fmr_ub = cp_upper(fp, n_false)
    fnmr_ub = cp_upper(fn, n_true)
    recall = (1.0 - fnmr_ub) if fn == 0 else recall_point
    tp = recall * band_true
    fp_max = fmr_ub * band_false
    precision = tp / (tp + fp_max) if tp + fp_max else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"fmr_ub": fmr_ub, "fnmr_ub": fnmr_ub, "fp_max": fp_max, "recall": recall, "precision": precision,
            "f1": f1, "recall_is_bound": fn == 0}


def bound_lines(fp: int, n_false: int, fn: int, n_true: int, recall_point: float,
                band_true: float = BAND_TRUE, band_false: float = BAND_FALSE, indent: str = "  ") -> list[str]:
    """Report lines for every error type whose count is zero (empty list when neither is)."""
    if fp != 0 and fn != 0:
        return []
    w = worst_case(fp, n_false, fn, n_true, recall_point, band_true, band_false)
    out = [f"{indent}zero-error bounds (exact one-sided 95% Clopper-Pearson; the bootstrap interval is degenerate "
           f"when an error count is 0):"]
    if fp == 0:
        out.append(f"{indent}  false matches: 0 of {n_false:,} false pairs; false-match rate <= {w['fmr_ub']:.5f} "
                   f"(up to {w['fp_max']:.0f} false matches on {band_false:,.0f} false pairs)")
    if fn == 0:
        out.append(f"{indent}  false non-matches: 0 of {n_true:,} true pairs; false-non-match rate <= {w['fnmr_ub']:.5f} "
                   f"(recall >= {w['recall']:.4f})")
    what = "precision" if fp == 0 else "recall"
    if fp == 0:
        out.append(f"{indent}  implied worst case on a band of {band_true:,.0f} true and {band_false:,.0f} false pairs: "
                   f"precision >= {w['precision']:.4f}, recall {'>=' if w['recall_is_bound'] else '='} {w['recall']:.4f}, "
                   f"F1 >= {w['f1']:.4f}")
    else:
        out.append(f"{indent}  implied worst case on that band: recall >= {w['recall']:.4f}")
    return out

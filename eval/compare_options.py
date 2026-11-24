"""
Paired comparison of two options on the fixed sample of test band pairs (protocol.md, Amendments,
Nov 24, 2026: LLM options scored on a fixed sample).

  python eval/compare_options.py results/<run> --sample <test_sample.csv> \
      --first <decisions.csv> --second <decisions.csv> [--first-name X] [--second-name Y]

Both decisions files need record_a, record_b, decision (same / different) for every pair of the
sample; extra rows (a full-band file) are ignored. The script fails if either file lacks a sample
pair, repeats one, or holds a decision other than same / different, so both options are always
scored on the same sample pairs.

Scoring is the band scoring of eval/score_option.py --sample: same-entity is the positive class,
each sample pair counts with its weight (the inverse of its sampling rate). Truth is read for test
records only.

Paired weighted bootstrap: 1,000 resamples of the sample rows, seed 0, and the SAME resample
(same row indices) is applied to both options, so what the two options share cancels out of the
difference. Percentile 95% intervals. Resampling is the same as score_option.py's band F1
interval, so each option's own F1 interval equals the one in its score report.

Difference = first minus second, for band F1, precision and recall (point estimate on the sample,
interval from the resamples). Bar line (protocol.md: borderline F1 within 2 points of B), with the
first option as the candidate and the second as B:
  pass          interval lower end above -0.02
  fail          interval upper end below -0.02
  inconclusive  otherwise
When an option has no errors of one kind its own bootstrap interval is degenerate, so the exact one-sided 95%
Clopper-Pearson upper bounds on its false-match and false-non-match rates and the worst-case precision and
recall they imply on the band are printed under it (eval/error_bounds.py).
For reference the script also reports the unpaired interval of the F1 difference: each option
resampled independently (seeds 0 and 1) and the two resampled F1 values subtracted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sample as ms  # noqa: E402
import error_bounds as EB  # noqa: E402

RESAMPLES = 1000
SEED = 0
CHUNK = 50
MARGIN = -0.02


def load_decisions(path: Path, keys: list[tuple[str, str]], label: str) -> np.ndarray:
    seen: dict[tuple[str, str], bool] = {}
    want = set(keys)
    for i, r in enumerate(ms.read_csv(path), start=2):
        k = tuple(sorted((r["record_a"].strip(), r["record_b"].strip())))
        if k not in want:
            continue
        d = (r.get("decision") or "").strip().lower()
        if d not in ("same", "different"):
            raise SystemExit(f"{label} ({path}) line {i}: decision {r.get('decision')!r} is not same / different")
        if k in seen:
            raise SystemExit(f"{label} ({path}) line {i}: sample pair {k} repeated")
        seen[k] = d == "same"
    missing = [k for k in keys if k not in seen]
    if missing:
        raise SystemExit(f"{label} ({path}) has no decision for {len(missing)} sample pair(s), e.g. {missing[:3]}")
    return np.array([seen[k] for k in keys], dtype=bool)


def prf_vec(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return p, r, f1


def resample_stats(pred: np.ndarray, truth: np.ndarray, w: np.ndarray, idx: np.ndarray):
    tp_v, fp_v, fn_v = w * (pred & truth), w * (pred & ~truth), w * (~pred & truth)
    return prf_vec(tp_v[idx].sum(1), fp_v[idx].sum(1), fn_v[idx].sum(1))


def point(pred, truth, w):
    return tuple(float(x[0]) for x in prf_vec(*(np.array([float((w * m).sum())]) for m in
                                              (pred & truth, pred & ~truth, ~pred & truth))))


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired weighted bootstrap comparison of two options on the sample.")
    ap.add_argument("run", type=Path)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--first-name", default=None)
    ap.add_argument("--second-name", default=None)
    args = ap.parse_args()
    n1, n2 = args.first_name or args.first.parent.name, args.second_name or args.second.parent.name

    sample = [(tuple(sorted((r["record_a"].strip(), r["record_b"].strip()))), float(r["weight"]))
              for r in ms.read_csv(args.sample)]
    keys = [k for k, _ in sample]
    if len(set(keys)) != len(keys):
        raise SystemExit("sample file repeats a pair")
    w = np.array([x for _, x in sample], dtype=float)
    if not (w > 0).all():
        raise SystemExit("sample file has a weight that is not > 0")
    band = set(ms.test_band_pairs(args.run))
    outside = [k for k in keys if k not in band]
    if outside:
        raise SystemExit(f"{len(outside)} sample pair(s) are not test band pairs, e.g. {outside[:3]}")
    if abs(w.sum() - len(band)) > 1e-6 * len(band):
        raise SystemExit(f"sample weights sum to {w.sum():.4f}, the test band has {len(band)} pairs")
    ent = ms.test_entities(args.run)
    truth = np.array([ent[a] == ent[b] for a, b in keys], dtype=bool)
    p1 = load_decisions(args.first, keys, n1)
    p2 = load_decisions(args.second, keys, n2)

    n = len(keys)
    rng = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED + 1)
    res1, res2, un1, un2 = [[] for _ in range(3)], [[] for _ in range(3)], [], []
    for start in range(0, RESAMPLES, CHUNK):
        m = min(CHUNK, RESAMPLES - start)
        idx = rng.integers(0, n, size=(m, n))                      # one resample, used for both options
        for j, v in enumerate(resample_stats(p1, truth, w, idx)):
            res1[j].append(v)
        for j, v in enumerate(resample_stats(p2, truth, w, idx)):
            res2[j].append(v)
        un1.append(resample_stats(p1, truth, w, idx)[2])           # unpaired: option 1 uses the same stream as above,
        un2.append(resample_stats(p2, truth, w, rng_b.integers(0, n, size=(m, n)))[2])   # option 2 its own
    res1 = [np.concatenate(x) for x in res1]
    res2 = [np.concatenate(x) for x in res2]
    un1, un2 = np.concatenate(un1), np.concatenate(un2)

    pt1, pt2 = point(p1, truth, w), point(p2, truth, w)
    print(f"Paired comparison on the sample: {args.sample}")
    print(f"  run {args.run.resolve()}")
    print(f"  {n:,} sample pairs ({int(truth.sum())} same-entity), weights sum {w.sum():,.1f} = test band pairs; "
          f"{RESAMPLES} resamples, seed {SEED}, same resample for both options; percentile 95% intervals\n")
    names = ("band precision", "band recall", "band F1")
    order = (2, 0, 1)                                              # report F1, precision, recall
    band_true, band_false = float((w * truth).sum()), float((w * ~truth).sum())
    for nm, pt, res, pr in ((n1, pt1, res1, p1), (n2, pt2, res2, p2)):
        print(f"{nm}")
        for j in order:
            lo, hi = ci(res[j])
            print(f"  {names[j]:<15} {pt[j]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]" + ("  (degenerate)" if lo == hi else ""))
        for line in EB.bound_lines(int((pr & ~truth).sum()), int((~truth).sum()), int((~pr & truth).sum()),
                                   int(truth.sum()), pt[1], band_true=band_true, band_false=band_false):
            print(line)
    print(f"\nDifference (first minus second) = {n1} minus {n2}")
    diff_f1 = None
    for j in order:
        d = res1[j] - res2[j]
        lo, hi = ci(d)
        print(f"  {names[j]:<15} {pt1[j] - pt2[j]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (width {hi - lo:.4f})")
        if j == 2:
            diff_f1 = (pt1[j] - pt2[j], lo, hi)
    d, lo, hi = diff_f1
    verdict = "pass" if lo > MARGIN else ("fail" if hi < MARGIN else "inconclusive")
    print(f"\nBar: band F1 difference vs {MARGIN:+.2f} ({n1} against {n2} as the baseline): "
          f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict.upper()}  "
          f"(pass: lower end > {MARGIN:+.2f}; fail: upper end < {MARGIN:+.2f}; else inconclusive)")
    ulo, uhi = ci(un1 - un2)
    print(f"\nFor reference, unpaired (independent resamples): band F1 difference interval [{ulo:+.4f}, {uhi:+.4f}] "
          f"(width {uhi - ulo:.4f}) against paired width {hi - lo:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

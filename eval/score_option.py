"""
Score one option on a run's test group (protocol.md, Amendments, Nov 24, 2026).

  python eval/score_option.py results/<run> --decisions <decisions.csv>
      [--weighted-sample | --sample <test_sample.csv>] [--truth <truth.csv>] [--split <split_records.csv>] [--name <option>]

The decisions file has one row per test band pair: record_a, record_b, decision
(same / different), an optional confidence (0 to 1) and an optional weight. A test band pair
is a row of scored_pairs.csv with decision_status == review whose two records are both in the
test group of split_records.csv.

Full mode (default): the file must cover every test band pair exactly once; a weight column,
if present, is ignored (every pair counts once).
--weighted-sample: the file covers a sample of the test band pairs; every row needs a weight
> 0 (its inverse inclusion probability). Band and overall figures become weighted estimates:
each sampled band pair counts with its weight, unsampled band pairs are left out, and every
pair outside the band counts once.
--sample <file>: the decisions file covers the full test band (or more than the sample); the
file is a sample from eval/make_sample.py (record_a, record_b, weight). Only the pairs in the
sample are scored, with the sample's weights (any weight column in the decisions file is
ignored), exactly as --weighted-sample would score a decisions file cut down to those pairs.
Every sample pair must have a decision; decisions for other test band pairs are ignored and
counted. Option decisions are never changed by sampling, so this scores an option that ran on
the full band as if it had run on the sample only.
Either way the script fails (exit 1) on a pair that is not a test band pair, a repeated pair,
a decision other than same / different, a bad confidence or weight, or (full mode) a missing
test band pair.

Final decision for every pair in the test group (both records in test):
  inside the band   the option's decision
  outside the band  Verity's decision_status (match / non_match)
A true pair in the test group that was never scored counts as a miss (weight 1).

Reports (same-entity = positive)
  band: F1 with a bootstrap 95% interval (1,000 resamples of the band rows, seed 0,
  percentile interval; weighted mode resamples sampled rows and sums their weights),
  accuracy, precision, recall, confusion counts; when an error count is 0 the bootstrap interval is
  degenerate, so the exact one-sided 95% Clopper-Pearson upper bounds on the false-match and
  false-non-match rates and the worst-case precision and recall they imply on the same band are
  printed next to it (eval/error_bounds.py)
  overall (test group): precision, recall, F1, TP / FP / FN
  recall by hard-case slice (acronym, truncation, po_box, easy) on the test group
  decoy pairs (same decoy group, different entity) in the test group wrongly matched
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_bounds as EB  # noqa: E402

HARD_TAGS = ("acronym", "truncation", "po_box")
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
DECISIONS = {"same": True, "different": False}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap_f1(pred: np.ndarray, truth: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(pred)
    tp_v = w * (pred & truth)
    fp_v = w * (pred & ~truth)
    fn_v = w * (~pred & truth)
    f1s = np.empty(BOOTSTRAP_RESAMPLES)
    chunk = 50
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        m = min(chunk, BOOTSTRAP_RESAMPLES - start)
        idx = rng.integers(0, n, size=(m, n))
        tp, fp, fn = tp_v[idx].sum(1), fp_v[idx].sum(1), fn_v[idx].sum(1)
        denom = 2 * tp + fp + fn
        f1s[start:start + m] = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(x: float, weighted: bool) -> str:
    return f"{x:.1f}" if weighted else f"{int(round(x))}"


def main() -> int:
    p = argparse.ArgumentParser(description="Score an option's decisions on the test group.")
    p.add_argument("run", type=Path)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--weighted-sample", action="store_true",
                   help="Decisions cover a weighted sample of the test band pairs")
    p.add_argument("--sample", type=Path, default=None,
                   help="Score only the pairs of this sample file (eval/make_sample.py), with its weights")
    p.add_argument("--truth", type=Path, default=None)
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--name", default=None, help="Option name for the report header")
    p.add_argument("--calibration-plot", type=Path, default=None,
                   help="Write a reliability diagram (test band) to this PNG when confidences are given")
    p.add_argument("--oof-reliability", type=Path, default=None,
                   help="JSON reliability table from training (out-of-fold) to draw on the same plot")
    args = p.parse_args()
    csv.field_size_limit(10**8)
    if args.sample and args.weighted_sample:
        p.error("--sample and --weighted-sample are alternatives")
    run, weighted = args.run, bool(args.weighted_sample or args.sample)
    truth_path = args.truth or run / "orgs_raw_truth.csv"
    split_path = args.split or run / "split_records.csv"

    side = {r["record_id"]: r["side"] for r in read_csv(split_path)}
    truth = read_csv(truth_path)
    entity = {r["record_id"]: r["true_entity_id"] for r in truth}
    tags = {r["record_id"]: set(r["variants"].split("+")) for r in truth}
    decoy = {r["record_id"]: r["decoy_group"] for r in truth}
    by_entity: dict[str, list[str]] = defaultdict(list)
    by_decoy: dict[str, list[str]] = defaultdict(list)
    for rid, e in entity.items():
        by_entity[e].append(rid)
        if decoy[rid]:
            by_decoy[decoy[rid]].append(rid)

    def in_test(a: str, b: str) -> bool:
        return side.get(a) == "test" and side.get(b) == "test"

    true_test = {tuple(sorted(q)) for ids in by_entity.values() for q in combinations(ids, 2)
                 if in_test(*q)}
    decoy_test = {tuple(sorted(q)) for ids in by_decoy.values() for q in combinations(ids, 2)
                  if entity[q[0]] != entity[q[1]] and in_test(*q)}

    status: dict[tuple[str, str], str] = {}
    for r in read_csv(run / "verity_data" / "scored_pairs.csv"):
        if "decision_status" not in r:
            raise SystemExit("scored_pairs.csv has no decision_status column (run predates it).")
        k = (r["record_a"], r["record_b"])
        if in_test(*k):
            status[k] = r["decision_status"]
    band_keys = {k for k, s in status.items() if s == "review"}

    sample_w: dict[tuple[str, str], float] = {}
    if args.sample:
        for r in read_csv(args.sample):
            k = tuple(sorted((r["record_a"].strip(), r["record_b"].strip())))
            if k in sample_w:
                raise SystemExit(f"sample file repeats pair {k}")
            sample_w[k] = float(r["weight"])
        if not all(v > 0 for v in sample_w.values()):
            raise SystemExit("sample file has a weight that is not > 0")
    n_ignored = 0

    # Validate the decisions file.
    decisions: dict[tuple[str, str], bool] = {}
    weights: dict[tuple[str, str], float] = {}
    confidence: dict[tuple[str, str], float] = {}
    problems: list[str] = []
    n_conf = 0
    for i, r in enumerate(read_csv(args.decisions), start=2):
        k = tuple(sorted((r["record_a"].strip(), r["record_b"].strip())))
        if args.sample and k not in sample_w:
            n_ignored += 1
            continue
        d = (r.get("decision") or "").strip().lower()
        if d not in DECISIONS:
            problems.append(f"line {i}: decision {r.get('decision')!r} is not same / different")
            continue
        if k in decisions:
            problems.append(f"line {i}: pair {k} repeated")
            continue
        c = (r.get("confidence") or "").strip()
        if c:
            try:
                cv = float(c)
                if not 0.0 <= cv <= 1.0:
                    raise ValueError
                n_conf += 1
                confidence[k] = cv
            except ValueError:
                problems.append(f"line {i}: confidence {c!r} is not a number from 0 to 1")
        wv = 1.0
        if args.sample:
            wv = sample_w[k]
        elif weighted:
            ws = (r.get("weight") or "").strip()
            try:
                wv = float(ws)
                if not wv > 0:
                    raise ValueError
            except ValueError:
                problems.append(f"line {i}: weight {ws!r} must be a number > 0 in --weighted-sample mode")
                continue
        decisions[k] = DECISIONS[d]
        weights[k] = wv
    extra = sorted(set(decisions) - band_keys)
    missing = sorted(band_keys - set(decisions))
    if extra:
        problems.append(f"{len(extra)} pair(s) that are not test band pairs, e.g. {extra[:3]}")
    if args.sample:
        no_decision = sorted(set(sample_w) - set(decisions))
        if no_decision:
            problems.append(f"{len(no_decision)} sample pair(s) have no decision, e.g. {no_decision[:3]}")
        outside = sorted(set(sample_w) - band_keys)
        if outside:
            problems.append(f"{len(outside)} sample pair(s) are not test band pairs, e.g. {outside[:3]}")
        if abs(sum(sample_w.values()) - len(band_keys)) > 1e-6 * len(band_keys):
            problems.append(f"sample weights sum to {sum(sample_w.values()):.4f}, the test band has {len(band_keys)} pairs")
    if missing and not weighted:
        problems.append(f"{len(missing)} test band pair(s) missing, e.g. {missing[:3]}")
    if problems:
        print("Decisions file rejected:")
        for m in problems[:20]:
            print("  " + m)
        return 1

    # Population: every scored test-group pair (band pairs only if sampled), plus true and
    # decoy test-group pairs that were never scored. Each carries a weight and a final decision.
    pop_w: dict[tuple[str, str], float] = {}
    pop_match: dict[tuple[str, str], bool] = {}
    for k, s in status.items():
        if s == "review":
            if k in decisions:
                pop_w[k], pop_match[k] = weights[k], decisions[k]
        else:
            pop_w[k], pop_match[k] = 1.0, s == "match"
    for k in true_test | decoy_test:
        if k not in status:
            pop_w[k], pop_match[k] = 1.0, False

    def wsum(keys) -> float:
        return float(sum(pop_w[k] for k in keys))

    sampled = sorted(decisions)
    pred = np.array([decisions[k] for k in sampled], dtype=bool)
    tru = np.array([k in true_test for k in sampled], dtype=bool)
    w = np.array([weights[k] for k in sampled], dtype=float)
    btp, bfp = float((w * (pred & tru)).sum()), float((w * (pred & ~tru)).sum())
    bfn, btn = float((w * (~pred & tru)).sum()), float((w * (~pred & ~tru)).sum())
    bp, br, bf1 = prf(btp, bfp, bfn)
    btot = btp + bfp + bfn + btn
    acc = (btp + btn) / btot if btot else 0.0
    lo, hi = bootstrap_f1(pred, tru, w) if sampled else (0.0, 0.0)

    matched = {k for k, m in pop_match.items() if m}
    trues = {k for k in pop_w if k in true_test}
    otp = wsum(matched & trues)
    ofp = wsum(matched - trues)
    ofn = wsum(trues - matched)
    op, orc, of1 = prf(otp, ofp, ofn)

    name = args.name or args.decisions.stem
    est = " (weighted estimate)" if weighted else ""
    print(f"Option: {name}")
    print(f"Run: {run.resolve()}")
    print(f"Decisions: {args.decisions}  ({len(decisions)} test band pairs"
          f"{f' sampled of {len(band_keys)}, weights sum {w.sum():.1f}' if weighted else ''}"
          f"{f' [--sample {args.sample.name}; {n_ignored} other decisions ignored]' if args.sample else ''}; "
          f"confidence given for {n_conf})")
    print(f"Test group: {sum(1 for v in side.values() if v == 'test')} records, "
          f"{len(status)} scored pairs, {len(true_test)} true pairs\n")
    print(f"Band (test band pairs, same-entity = positive){est}")
    print(f"  pairs {fmt(btot, weighted)}  (true {fmt(btp + bfn, weighted)}, "
          f"{100 * (btp + bfn) / btot if btot else 0:.2f}%)")
    print(f"  F1 {bf1:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (bootstrap, {BOOTSTRAP_RESAMPLES} "
          f"resamples, seed {BOOTSTRAP_SEED})")
    for line in EB.bound_lines(int((pred & ~tru).sum()), int((~tru).sum()), int((~pred & tru).sum()), int(tru.sum()),
                               br, band_true=btp + bfn, band_false=bfp + btn):
        print(line)
    print(f"  accuracy {acc:.4f}  precision {bp:.4f}  recall {br:.4f}")
    print(f"  TP {fmt(btp, weighted)}  FP {fmt(bfp, weighted)}  FN {fmt(bfn, weighted)}  "
          f"TN {fmt(btn, weighted)}\n")
    print(f"Overall (test group; outside the band Verity's decision_status){est}")
    print(f"  precision {op:.4f}  recall {orc:.4f}  F1 {of1:.4f}")
    print(f"  TP {fmt(otp, weighted)}  FP {fmt(ofp, weighted)}  FN {fmt(ofn, weighted)}\n")
    print(f"Recall by hard-case slice (test group; pairs where either record carries the tag){est}")
    for tag in (*HARD_TAGS, "easy"):
        if tag == "easy":
            sl = {q for q in trues if not any(t in tags[q[0]] | tags[q[1]] for t in HARD_TAGS)}
        else:
            sl = {q for q in trues if tag in tags[q[0]] or tag in tags[q[1]]}
        hit, den = wsum(sl & matched), wsum(sl)
        print(f"  {tag:<12} {fmt(hit, weighted):>8} / {fmt(den, weighted):<8} "
              f"recall {hit / den if den else 0:.4f}")
    dk = {k for k in pop_w if k in decoy_test}
    wrong, dden = wsum(dk & matched), wsum(dk)
    print(f"\nDecoy pairs in the test group: {fmt(dden, weighted)}, wrongly matched "
          f"{fmt(wrong, weighted)} ({wrong / dden if dden else 0:.2%}){est}")

    # Calibration, when every decision carries a confidence (probability of "same").
    if sampled and len(confidence) == len(decisions):
        pc = np.array([confidence[k] for k in sampled], dtype=float)
        bins = np.minimum((pc * 10).astype(int), 9)
        table = []
        for i in range(10):
            m = bins == i
            W = float(w[m].sum())
            table.append((i, int(m.sum()), W,
                          float((w[m] * pc[m]).sum() / W) if W else None,
                          float((w[m] * tru[m]).sum() / W) if W else None))
        wt = float(w.sum())
        ece = sum(W / wt * abs(ft - mp) for _, _, W, mp, ft in table if W)
        hi_m = pc >= 0.05
        whi = float(w[hi_m].sum())
        ece_hi = sum(abs(float((w[hi_m & (bins == i)] * tru[hi_m & (bins == i)]).sum())
                         - float((w[hi_m & (bins == i)] * pc[hi_m & (bins == i)]).sum()))
                     for i in range(10)) / whi if whi else 0.0
        from sklearn.metrics import average_precision_score
        ap_band = float(average_precision_score(tru, pc, sample_weight=w)) if tru.any() else 0.0
        print(f"\nBand average precision (cutoff-free, ranked by confidence){est}: {ap_band:.4f}")
        print(f"\nCalibration (test band, 10 equal-width bins){est}")
        print(f"  ECE {ece:.4f}   (bar: <= 0.05)")
        print(f"  ECE among pairs with confidence >= 0.05: {ece_hi:.4f}  ({fmt(whi, weighted)} pairs)")
        print("  bin           pairs     mean conf   frac true")
        for i, n, W, mp, ft in table:
            if n:
                hi_edge = "]" if i == 9 else ")"
                print(f"  [{i / 10:.1f}, {(i + 1) / 10:.1f}{hi_edge}  {fmt(W, weighted):>9}   {mp:.4f}      {ft:.4f}")
        if args.calibration_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="perfect")
            pts = [(mp, ft, n) for _, n, _, mp, ft in table if n]
            ax.plot([q[0] for q in pts], [q[1] for q in pts], marker="o", label="test band")
            for mp, ft, n in pts:
                ax.annotate(str(n), (mp, ft), textcoords="offset points", xytext=(4, -10), fontsize=7)
            if args.oof_reliability and args.oof_reliability.is_file():
                import json
                oof = [r for r in json.loads(args.oof_reliability.read_text(encoding="utf-8")) if r["n"]]
                ax.plot([r["mean_p"] for r in oof], [r["frac_true"] for r in oof], marker="s",
                        linestyle=":", label="train out-of-fold (calibration fit)")
            ax.set_xlabel("mean predicted probability (confidence)")
            ax.set_ylabel("fraction same-entity")
            ax.set_title(f"{name}: reliability, ECE {ece:.4f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper left", fontsize=8)
            fig.text(0.01, 0.005, "10 equal-width bins; numbers = test band pairs per bin", fontsize=7)
            fig.tight_layout()
            fig.savefig(args.calibration_plot, dpi=120)
            plt.close(fig)
            print(f"  calibration plot: {args.calibration_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Post-freeze exploratory tables from the frozen outputs of the freeze run. Every output starts with the line
"EXPLORATORY: not a protocol result; no setting was chosen from these numbers." Read-only on results; writes only post_freeze_extras.md into
--out-dir (default results/eval_7919/post_freeze/). No call, no new model run.

  python eval/post_freeze_extras.py [--run results/eval_7919] [--out-dir results/eval_7919/post_freeze]

1. False merges: predicted matches, false merges, false-merge rate (overall, test group) with a 95% interval, and decoy false merges, for A, C, D, D (orig)
   on the full band (exact, Clopper-Pearson) and for A, B, C, D on the 5,000-pair sample (weighted; bootstrap, 1,000 resamples, seed 0). The counts come from
   the score reports; the truth file is read only to flag each sampled pair's false merges for the bootstrap and to count the errors among D's routed pairs.
2. Review queue: routed pairs per 1,000 band pairs, the daily queue at 100K, 1M and 10M records a day (the cost model's band pairs per record), the full-band
   quality if every routed pair were decided correctly (oracle steward, an upper bound), and the development-seed trade-off from results/dstudy_summary.
3. Explanation exhibit for the three demo pairs: the evidence stored in scored_pairs.csv, and C's probability.
4. Effort to quality: the development learning curve (lc_v2, development seed 2026) with 5 hours per 1,000 labeled pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parent.parent
HEADER_LINE = "EXPLORATORY: not a protocol result; no setting was chosen from these numbers."
FULL_OPTS = ("A", "C", "D", "D (orig)")
SAMPLE_OPTS = ("A", "B", "C", "D")
N_BOOT, BOOT_SEED = 1000, 0


# ---------------------------------------------------------------------------------------------- reading

class Reader:
    """Reads only under the allowed roots; the truth file is read only through truth_rows(), for the ids asked for."""

    def __init__(self, roots: list[Path], truth_path: Path):
        self.roots = [Path(r).resolve() for r in roots]
        self.truth_path = Path(truth_path).resolve()
        self.read: list[Path] = []

    def check(self, p: Path) -> Path:
        q = Path(p).resolve()
        if q == self.truth_path:
            raise SystemExit("the truth file is read only through truth_rows()")
        if q.name.startswith("orgs_raw") or "truth" in q.name or q.name.startswith("hand_check") or "hand_check" in q.parts:
            raise SystemExit(f"refusing to read {q.name}")
        if not any(q == r or r in q.parents for r in self.roots):
            raise SystemExit(f"refusing to read {p}: outside the allowed folders")
        return q

    def text(self, p: Path) -> str:
        q = self.check(p)
        self.read.append(q)
        return q.read_text(encoding="utf-8")

    def csv_rows(self, p: Path):
        q = self.check(p)
        self.read.append(q)
        csv.field_size_limit(10**9)
        with q.open(newline="", encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)

    def csv(self, p: Path) -> list[dict]:
        return list(self.csv_rows(p))

    def truth_rows(self, ids: set[str]) -> dict[str, str]:
        """record_id -> true_entity_id for the ids asked for (the only use of the truth file)."""
        self.read.append(self.truth_path)
        out = {}
        with self.truth_path.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["record_id"] in ids:
                    out[r["record_id"]] = r["true_entity_id"]
        return out


def parse_report(text: str) -> dict:
    """Band and overall counts, decoy line and band F1 of a score report."""
    num = r"([\d.]+)"
    out: dict = {}
    m = re.search(rf"\n  TP {num}  FP {num}  FN {num}  TN {num}", text)
    out["band_tp"], out["band_fp"], out["band_fn"], out["band_tn"] = (float(x) for x in m.groups())
    m = re.search(rf"\nOverall \([^\n]*\n  precision {num}  recall {num}  F1 {num}\n  TP {num}  FP {num}  FN {num}", text)
    out["overall_precision"], out["overall_recall"], out["overall_f1"], out["tp"], out["fp"], out["fn"] = (float(x) for x in m.groups())
    m = re.search(rf"\n  F1 {num}  95% CI \[{num}, {num}\]", text)
    out["band_f1"] = float(m.group(1))
    m = re.search(r"Decoy pairs in the test group: ([\d.,]+), wrongly matched ([\d.,]+) \(([\d.]+)%\)", text)
    out["decoy_pairs"], out["decoy_false"] = (float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))) if m else (None, None)
    return out


# ---------------------------------------------------------------------------------------------- statistics

def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval for x successes in n trials."""
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def bootstrap_false_merge_rate(weights: np.ndarray, pred: np.ndarray, false_merge: np.ndarray, outside_tp: float, outside_fp: float,
                               n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% bootstrap interval of the false-merge rate = FP / (TP + FP) overall: resample the sampled band pairs with replacement (weights kept), add the exact
    counts outside the band. pred: 1 if the option predicts same; false_merge: 1 if it predicts same and the pair is a different entity."""
    rng = np.random.default_rng(seed)
    n = len(weights)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w = weights[idx]
        fp = float((w * false_merge[idx]).sum()) + outside_fp
        pm = float((w * pred[idx]).sum()) + outside_tp + outside_fp
        if pm > 0:
            rates.append(fp / pm)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def oracle_steward(rep: dict, routed_fp: int, routed_fn: int) -> dict:
    """Quality if every routed pair were decided correctly: the routed pairs' errors are removed (false merges and misses among them)."""
    tp, fp, fn = rep["band_tp"] + routed_fn, rep["band_fp"] - routed_fp, rep["band_fn"] - routed_fn
    otp, ofp, ofn = rep["tp"] + routed_fn, rep["fp"] - routed_fp, rep["fn"] - routed_fn
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if (2 * a + b + c) else float("nan")
    return {"band_tp": tp, "band_fp": fp, "band_fn": fn, "band_f1": f1(tp, fp, fn), "band_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "band_recall": tp / (tp + fn) if tp + fn else float("nan"), "overall_precision": otp / (otp + ofp) if otp + ofp else float("nan"),
            "overall_recall": otp / (otp + ofn) if otp + ofn else float("nan")}


def daily_queue(records_per_day: float, band_pairs_per_record: float, routed_share: float) -> float:
    return records_per_day * band_pairs_per_record * routed_share


# ---------------------------------------------------------------------------------------------- build

def build(run: Path, dstudy: Path, lc_dir: Path, R: Reader) -> dict:
    rel = lambda p: Path(p).resolve().relative_to(REPO).as_posix() if REPO in Path(p).resolve().parents else Path(p).resolve().as_posix()
    O = run / "options"
    full_reports = {"A": O / "a_threshold" / "score_report.txt", "C": O / "c_model" / "c_v1_lgbm" / "score_report.txt",
                    "D": O / "d_hybrid" / "score_report_fullband.txt", "D (orig)": O / "d_hybrid" / "score_report_fullband_original.txt"}
    sample_reports = {"A": O / "a_threshold" / "score_report_sample5000.txt", "B": O / "b_llm" / "openai" / "sample5000" / "score_report_sample5000.txt",
                      "C": O / "c_model" / "c_v1_lgbm" / "score_report_sample5000.txt", "D": O / "d_hybrid" / "score_report_sample5000.txt"}
    sample_decisions = {"A": O / "a_threshold" / "decisions.csv", "B": O / "b_llm" / "openai" / "sample5000" / "decisions.csv",
                        "C": O / "c_model" / "c_v1_lgbm" / "decisions.csv", "D": O / "d_hybrid" / "decisions.csv"}
    res: dict = {"sources": {}}
    # 1a. full band, exact
    full = {}
    for o, p in full_reports.items():
        rep = parse_report(R.text(p))
        n = int(round(rep["tp"] + rep["fp"]))
        x = int(round(rep["fp"]))
        lo, hi = clopper_pearson(x, n)
        full[o] = {"predicted": n, "false": x, "rate": x / n if n else float("nan"), "lo": lo, "hi": hi, "decoy_pairs": rep["decoy_pairs"], "decoy_false": rep["decoy_false"], "rep": rep, "source": rel(p)}
    res["full"] = full
    out_tp = full["A"]["rep"]["tp"] - full["A"]["rep"]["band_tp"]                 # outside the band: Verity's decision, the same for every option
    out_fp = full["A"]["rep"]["fp"] - full["A"]["rep"]["band_fp"]
    res["outside"] = {"tp": out_tp, "fp": out_fp}
    # 1b. sample basis, weighted, bootstrap
    sample_rows = R.csv(O / "b_llm" / "test_sample5000.csv")
    keys = [(r["record_a"].strip(), r["record_b"].strip()) for r in sample_rows]
    weights = np.array([float(r["weight"]) for r in sample_rows])
    ids = {x for k in keys for x in k}
    truth = R.truth_rows(ids)
    is_same = np.array([truth[a] == truth[b] for a, b in keys])
    sample = {}
    for o, p in sample_reports.items():
        rep = parse_report(R.text(p))
        dec = {}
        for r in R.csv_rows(sample_decisions[o]):
            k = tuple(sorted((r["record_a"].strip(), r["record_b"].strip())))
            dec[k] = r["decision"]
        pred = np.array([1.0 if dec[tuple(sorted(k))] == "same" else 0.0 for k in keys])
        fm = pred * (~is_same)
        lo, hi = bootstrap_false_merge_rate(weights, pred, fm, rep["tp"] - rep["band_tp"], rep["fp"] - rep["band_fp"])
        recomputed_fp = float((weights * fm).sum()) + (rep["fp"] - rep["band_fp"])
        n = rep["tp"] + rep["fp"]
        sample[o] = {"predicted": n, "false": rep["fp"], "rate": rep["fp"] / n if n else float("nan"), "lo": lo, "hi": hi, "decoy_pairs": rep["decoy_pairs"], "decoy_false": rep["decoy_false"],
                     "reproduced": abs(recomputed_fp - rep["fp"]) <= 0.05, "source": rel(p), "decisions": rel(sample_decisions[o])}
    res["sample"] = sample
    # 2. review queue
    band_pairs = sum(1 for _ in R.csv_rows(O / "c_model" / "c_v1_lgbm" / "decisions.csv"))
    assumptions = {r["name"]: r for r in R.csv(run / "cost_final" / "assumptions.csv")}
    bppr = float(assumptions["band pairs per record"]["value"])
    queue = {}
    for o, f in (("D", "routed_fullband_pairs.csv"), ("D (orig)", "routed_fullband_pairs_original.csv")):
        routed = [(r["record_a"].strip(), r["record_b"].strip()) for r in R.csv_rows(O / "d_hybrid" / f)]
        share = len(routed) / band_pairs
        rt = R.truth_rows({x for k in routed for x in k})
        dec = {tuple(sorted((r["record_a"].strip(), r["record_b"].strip()))): r["decision"] for r in R.csv_rows(O / "d_hybrid" / ("decisions_fullband.csv" if o == "D" else "decisions_fullband_original.csv"))}
        fp_r = sum(1 for a, b in routed if dec[tuple(sorted((a, b)))] == "same" and rt[a] != rt[b])
        fn_r = sum(1 for a, b in routed if dec[tuple(sorted((a, b)))] == "different" and rt[a] == rt[b])
        queue[o] = {"routed": len(routed), "band_pairs": band_pairs, "share": share, "per_1000": 1000 * share,
                    "daily": {v: daily_queue(v, bppr, share) for v in (100_000, 1_000_000, 10_000_000)}, "routed_fp": fp_r, "routed_fn": fn_r,
                    "oracle": oracle_steward(full[o]["rep"], fp_r, fn_r), "current": full[o]["rep"], "source": rel(O / "d_hybrid" / f)}
    res["queue"], res["bppr"], res["bppr_source"] = queue, bppr, rel(run / "cost_final" / "assumptions.csv")
    res["dev_tradeoff"] = [r for r in R.csv(dstudy / "d_select_lo.csv") if not r["lo"].startswith("share_")]
    res["dev_tradeoff_source"] = rel(dstudy / "d_select_lo.csv")
    # 3. explanation exhibit for the three demo pairs
    demo = R.csv(run / "demo_pairs" / "demo_pairs.csv")
    pairs = {tuple(sorted((r["record_a"], r["record_b"]))): r for r in demo if r.get("record_a") and r.get("record_b")}
    ex = {}
    for r in R.csv_rows(run / "verity_data" / "scored_pairs.csv"):
        k = tuple(sorted((r["record_a"], r["record_b"])))
        if k in pairs:
            ex[k] = r
            if len(ex) == len(pairs):
                break
    res["explain"] = [{"pair_no": pairs[k]["pair_no"], "pair": k, "row": ex[k], "c_p": pairs[k]["c_p"], "route": pairs[k]["route"]} for k in sorted(pairs, key=lambda k: int(pairs[k]["pair_no"]))]
    res["explain_sources"] = [rel(run / "demo_pairs" / "demo_pairs.csv"), rel(run / "verity_data" / "scored_pairs.csv")]
    # 4. effort to quality
    hours_per_1000 = float(assumptions["labeling hours per 1,000 pairs"]["value"])
    res["hours_per_1000"], res["hours_source"] = hours_per_1000, rel(run / "cost_final" / "assumptions.csv")
    res["hours_note"] = assumptions["labeling hours per 1,000 pairs"]["source"] + "; " + assumptions["labeling hours per 1,000 pairs"]["note"]
    lc = R.csv(lc_dir / "learning_curve.csv")
    res["lc"] = [{"n": int(r["n"]), "hours": int(r["n"]) / 1000 * hours_per_1000, "seeds": r["seeds"], "f1_mean": float(r["f1_mean"]), "f1_min": float(r["f1_min"]), "f1_max": float(r["f1_max"])} for r in lc]
    res["lc_source"] = rel(lc_dir / "learning_curve.csv")
    return res


# ---------------------------------------------------------------------------------------------- render

def md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def fmt4(x: float) -> str:
    return "n/a" if x != x else f"{x:.4f}"


def pct(x: float, d: int = 3) -> str:
    return f"{100 * x:.{d}f}%"


def render(res: dict) -> str:
    L = [HEADER_LINE, "", "# Post-freeze extras (seed 7919 freeze run)", "",
         "Everything below is read from the frozen outputs (files under `results/eval_7919`, plus `results/dstudy_summary` and the development learning curve for sections 2 and 4, "
         "both marked development). Nothing was rerun, no call was made, and no setting was chosen from these numbers.", "",
         "## 1. False merges", "",
         "False-merge rate = false merges / predicted matches, overall in the test group (Verity's own decisions outside the band, the option's decisions inside it). "
         "Counts are read from the score reports. Intervals: exact (Clopper-Pearson) on the full band; bootstrap (1,000 resamples, seed 0, sample weights kept, the exact "
         "counts outside the band added to every resample) on the sample basis. The truth file was read only to flag each sampled pair's false merges and, in section 2, "
         "to count the errors among D's routed pairs.", "", "### Full band (exact)", "",
         md_table(["option", "predicted matches", "false merges", "false-merge rate", "95% interval (exact binomial)", "decoy pairs", "decoy false merges", "source"],
                  [[o, f"{v['predicted']:,}", f"{v['false']:,}", pct(v["rate"]), f"[{pct(v['lo'])}, {pct(v['hi'])}]", f"{v['decoy_pairs']:,.0f}", f"{v['decoy_false']:,.0f}", v["source"]]
                   for o, v in res["full"].items()]), "",
         "### 5,000-pair sample (weighted estimates)", "",
         md_table(["option", "predicted matches", "false merges", "false-merge rate", "95% interval (bootstrap)", "decoy pairs", "decoy false merges", "report", "decisions"],
                  [[o, f"{v['predicted']:,.1f}", f"{v['false']:,.1f}", pct(v["rate"]), f"[{pct(v['lo'])}, {pct(v['hi'])}]", f"{v['decoy_pairs']:,.1f}", f"{v['decoy_false']:,.1f}", v["source"], v["decisions"]]
                   for o, v in res["sample"].items()]), "",
         "The sample-basis false merges recomputed from the decisions and the truth reproduce the report's figures for every option: "
         + ", ".join(f"{o} {'yes' if v['reproduced'] else 'NO'}" for o, v in res["sample"].items()) + " (the reports print weighted counts to one decimal, so the match is to within 0.05). "
         "The two bases are not comparable with each other.", ""]
    q = res["queue"]
    L += ["## 2. Review queue (if D's routed pairs went to a human steward instead of the LLM)", "",
          f"Band pairs per record: {res['bppr']:.4f} (`{res['bppr_source']}`). Daily queue = records a day x band pairs per record x routed share.", "",
          md_table(["setting", "routed pairs", "band pairs", "routed per 1,000 band pairs", "queue per day at 100K records", "at 1M records", "at 10M records", "source"],
                   [[o, f"{v['routed']:,}", f"{v['band_pairs']:,}", f"{v['per_1000']:.3f}", f"{v['daily'][100_000]:,.0f}", f"{v['daily'][1_000_000]:,.0f}", f"{v['daily'][10_000_000]:,.0f}", v["source"]]
                    for o, v in q.items()]), "",
          "### Oracle steward (upper bound): full-band quality if every routed pair were decided correctly", "",
          "An upper bound, not an estimate of what stewards would achieve: the routed pairs' errors are removed as if every steward decision were right.", "",
          md_table(["setting", "routed errors (false merges / misses)", "band F1 now", "band F1 oracle", "band precision now", "band precision oracle", "band recall now", "band recall oracle",
                    "overall precision now", "overall precision oracle", "overall recall now", "overall recall oracle"],
                   [[o, f"{v['routed_fp']} / {v['routed_fn']}", fmt4(v['current']['band_f1']), fmt4(v['oracle']['band_f1']),
                     fmt4(ratio(v['current']['band_tp'], v['current']['band_tp'] + v['current']['band_fp'])), fmt4(v['oracle']['band_precision']),
                     fmt4(ratio(v['current']['band_tp'], v['current']['band_tp'] + v['current']['band_fn'])), fmt4(v['oracle']['band_recall']),
                     f"{v['current']['overall_precision']:.4f}", f"{v['oracle']['overall_precision']:.4f}", f"{v['current']['overall_recall']:.4f}", f"{v['oracle']['overall_recall']:.4f}"]
                    for o, v in q.items()]), "",
          "Sources: the full-band score reports (counts) and D's full-band decisions with the routed lists (the routed pairs' errors; truth read only for those pairs).", "",
          "### Development-seed trade-off (seeds 11, 12, 13, 42 and 2026 pooled; existing file, not recomputed)", "",
          md_table(["lo (hi 0.9)", "pooled LLM share of the test band", "pooled band F1", "precision", "recall", "false merges", "misses", "routed pairs"],
                   [[r["lo"], pct(float(r["llm_share"])), f"{float(r['pooled_f1']):.4f}", f"{float(r['precision']):.4f}", f"{float(r['recall']):.4f}", r["fp"], r["fn"], r["routed"]] for r in res["dev_tradeoff"]]), "",
          f"Source: `{res['dev_tradeoff_source']}`. Development seeds: these rows were the basis of the lo choice made before the freeze; they are shown for the trade-off only.", ""]
    L += ["## 3. Explanation exhibit for the three demo pairs", "",
          "Evidence stored in `scored_pairs.csv` for each demo pair (per-field similarity and weight, the evidence mass and conflict mass, the evidence gate result) and C's probability. "
          "The weighted column is similarity x weight (derived). No new model run.", ""]
    for e in res["explain"]:
        row = e["row"]
        sims, wts, esims = json.loads(row["field_similarities"]), json.loads(row["field_weights"]), json.loads(row["evidence_field_similarities"])
        L += [f"### Pair {e['pair_no']}: {e['pair'][0]} / {e['pair'][1]}", "",
              md_table(["field", "similarity", "weight", "similarity x weight (derived)", "evidence similarity"],
                       [[f, f"{sims[f]:.4f}", f"{wts.get(f, float('nan')):.4f}", f"{sims[f] * wts.get(f, 0.0):.4f}", f"{esims[f]:.4f}" if f in esims else "n/a"] for f in sims]), "",
              md_table(["item", "value"],
                       [["score (final) / score uncapped", f"{row['score']} / {row['score_uncapped']}"], ["match type", row["match_type"]], ["decision status", row["decision_status"]],
                        ["evidence mass", row["evidence_mass"]], ["conflict mass", row["conflict_mass"]], ["evidence gate result (evidence_passes)", row["evidence_passes"]],
                        ["applied / effective threshold", f"{row['applied_threshold']} / {row['effective_threshold']}"],
                        ["C's calibrated probability", e["c_p"] if e["c_p"] else "not applicable (the pair is outside the band; Verity decides)"], ["route", e["route"]]]), ""]
    L += [f"Sources: {', '.join('`' + s + '`' for s in res['explain_sources'])}.", ""]
    L += ["## 4. Effort to quality (development seed 2026)", "",
          f"Labeling effort at {res['hours_per_1000']:g} hours per 1,000 labeled pairs (`{res['hours_source']}`). Caveat from that file: {res['hours_note']}. "
          "Mean band F1 is the mean over the sampling seeds of the development learning curve (lc_v2), **development seed 2026, not the freeze run**.", "",
          md_table(["labels", "hours", "sampling seeds", "mean band F1", "min", "max"],
                   [[f"{r['n']:,}", f"{r['hours']:.2f}", r["seeds"], f"{r['f1_mean']:.4f}", f"{r['f1_min']:.4f}", f"{r['f1_max']:.4f}"] for r in res["lc"]]), "",
          "The last row is the point of the same file at all train pairs (one run, so one seed): the whole labeled train band of the development run.", "",
          f"Source: `{res['lc_source']}`.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-freeze exploratory tables.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "eval_7919")
    ap.add_argument("--dstudy", type=Path, default=REPO / "results" / "dstudy_summary")
    ap.add_argument("--lc-dir", type=Path, default=REPO / "results" / "dev_2026_run11" / "options" / "c_model" / "lc_v2")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()
    R = Reader([run, args.dstudy, args.lc_dir], run / "orgs_raw_truth.csv")
    res = build(run, args.dstudy.resolve(), args.lc_dir.resolve(), R)
    out = args.out_dir or run / "post_freeze"
    out.mkdir(parents=True, exist_ok=True)
    (out / "post_freeze_extras.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'post_freeze_extras.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

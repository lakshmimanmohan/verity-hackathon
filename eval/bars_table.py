"""
The protocol bars table for a run: one row per option with the measured numbers and an automatic verdict per bar,
computed by the same rules as the cost model (cost/cost_model.py bars_row, f1_bar_verdict). No verdict is typed.

  python eval/bars_table.py results/<run>

Reads the run's score and comparison reports (the layout eval/freeze_run.py writes, the same as the dev run):
  A  options/a_threshold/score_report.txt (full band)
  B  options/b_llm/openai/sample5000/score_report_sample5000.txt (the fixed sample; B is the F1 baseline)
  C  options/c_model/c_v1_lgbm/score_report.txt (full band)
  D  options/d_hybrid/score_report_fullband.txt (exact, full band)
  F1 differences against B from the paired comparisons on the sample (compare/b_vs_a.txt, compare/c_v1_lgbm_vs_b.txt,
  d_hybrid/compare_d_vs_b.txt); C's latency p95 from options/latency/latency_summary.json; D's LLM share from
  d_hybrid/routed_fullband_pairs.csv over the test band. Option E: not run.
Writes <run>/bars_table.csv and <run>/bars_table.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cost"))
import cost_model as CM  # noqa: E402

REPORTS = {"A": "options/a_threshold/score_report.txt", "B": "options/b_llm/openai/sample5000/score_report_sample5000.txt",
           "C": "options/c_model/c_v1_lgbm/score_report.txt", "D": "options/d_hybrid/score_report_fullband.txt"}
HEADER = ["option", "basis", "band_f1", "band_f1_ci_lo", "band_f1_ci_hi", "band_precision", "band_recall", "f1_diff_vs_B", "f1_diff_lo", "f1_diff_hi",
          "f1_bar", "overall_precision", "precision_bar", "overall_recall", "recall_bar", "ece", "ece_bar", "latency_p95_ms", "latency_bar",
          "llm_share", "llm_share_bar", "meets_every_bar"]


def band_ci(text: str) -> tuple[float, float]:
    m = re.search(r"F1 [\d.]+\s+95% CI \[([\d.]+), ([\d.]+)\]", text)
    return float(m.group(1)), float(m.group(2))


def routed_share(run: Path) -> float:
    routed = sum(1 for _ in (run / "options" / "d_hybrid" / "routed_fullband_pairs.csv").open(encoding="utf-8")) - 1
    n_band = sum(1 for _ in (run / "options" / "c_model" / "c_v1_lgbm" / "decisions.csv").open(encoding="utf-8")) - 1
    return routed / n_band


def build(run: Path) -> list[dict]:
    lat_path = run / "options" / "latency" / "latency_summary.json"
    latency = json.loads(lat_path.read_text(encoding="utf-8")) if lat_path.exists() else None
    p = CM.Params(routed_share=routed_share(run))
    inp = CM.load_bar_inputs(run, p, latency)
    a_recall = inp["A"]["overall"][1]
    rows = []
    for opt in "ABCD":
        d = inp[opt]
        v = CM.bars_row(opt, d, a_recall)
        text = (run / REPORTS[opt]).read_text(encoding="utf-8")
        lo, hi = band_ci(text)
        bp, br = CM.parse_band_pr(text)
        fd = d["f1_diff"]
        rows.append({"option": opt, "basis": d["basis"], "band_f1": d["band_f1"], "band_f1_ci_lo": lo, "band_f1_ci_hi": hi, "band_precision": bp,
                     "band_recall": br, "f1_diff_vs_B": "" if fd is None else fd[0], "f1_diff_lo": "" if fd is None else fd[1],
                     "f1_diff_hi": "" if fd is None else fd[2], "f1_bar": v["f1"], "overall_precision": d["overall"][0], "precision_bar": v["precision"],
                     "overall_recall": d["overall"][1], "recall_bar": v["recall"], "ece": "" if d["ece"] is None else d["ece"], "ece_bar": v["ece"],
                     "latency_p95_ms": d.get("latency_p95_ms", ""), "latency_bar": v["latency"], "llm_share": d.get("llm_share", ""),
                     "llm_share_bar": v["llm_share"], "meets_every_bar": v["every"]})
    rows.append({"option": "E", "basis": "not run", **{k: "" for k in HEADER[2:]}, "meets_every_bar": "not run"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    rows = build(args.run)
    with (args.run / "bars_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    L = [f"Protocol bars, run {args.run.name} (verdicts computed by the protocol rules, cost/cost_model.py bars_row)",
         "  F1: band F1 within 2 points of B on the paired sample (pass: interval lower end > -0.02; fail: upper end < -0.02; else inconclusive).",
         "  Precision >= 0.98; recall at least 5 points above A; ECE <= 0.05 (C, D); C latency p95 <= 20 ms; D LLM share <= 20%.",
         f"  {'option':<7}{'band F1 [95% CI]':<26}{'F1 vs B [95% CI]':<30}{'F1 bar':<14}{'prec':>7} {'bar':<6}{'recall':>7} {'bar':<7}{'ECE':>7} {'bar':<6}"
         f"{'p95 ms':>8} {'bar':<6}{'LLM share':>10} {'bar':<6} every bar?"]
    for r in rows:
        if r["option"] == "E":
            L.append(f"  {'E':<7}not run")
            continue
        fd = "baseline" if r["f1_diff_vs_B"] == "" else f"{r['f1_diff_vs_B']:+.4f} [{r['f1_diff_lo']:+.4f}, {r['f1_diff_hi']:+.4f}]"
        lat = "" if r["latency_p95_ms"] == "" else f"{r['latency_p95_ms']:.3f}"
        sh = "" if r["llm_share"] == "" else f"{r['llm_share']:.5f}"
        ece = "" if r["ece"] == "" else f"{r['ece']:.4f}"
        short = lambda x: x.split(" (")[0]
        L.append(f"  {r['option']:<7}{r['band_f1']:.4f} [{r['band_f1_ci_lo']:.4f}, {r['band_f1_ci_hi']:.4f}]   {fd:<30}{short(r['f1_bar']):<14}{r['overall_precision']:>7.4f} {short(r['precision_bar']):<6}"
                 f"{r['overall_recall']:>7.4f} {short(r['recall_bar']):<7}{ece:>7} {short(r['ece_bar']):<6}{lat:>8} {short(r['latency_bar']):<6}{sh:>10} {short(r['llm_share_bar']):<6} {r['meets_every_bar']}")
    L.append("  Bases: " + "; ".join(f"{r['option']} {r['basis']}" for r in rows if r["option"] != "E"))
    text = "\n".join(L) + "\n"
    (args.run / "bars_table.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

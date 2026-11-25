"""
The results table for a run: one row per option, from the bars table (eval/bars_table.py) and the run's cost model
outputs (cost/cost_model.py --run <run> --out <run>/cost).

  python eval/results_table.py results/<run>

Columns: option; band F1 and its 95% interval; band precision and recall; overall precision and recall; ECE; F1
difference against B on the paired sample and the F1 bar verdict; C latency p95; D LLM share; LLM cost per 1,000 band
pairs (calls per band pair x B's measured cost per call in this run); total monthly cost at 1M records per day (S1 total,
hourly rate $100, C and D with the labels as evaluated); meets every bar. Option E: not run.
Writes <run>/results_table.csv and <run>/results_table.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HEADER = ["option", "band_f1", "band_f1_ci_lo", "band_f1_ci_hi", "band_precision", "band_recall", "overall_precision", "overall_recall", "ece",
          "f1_diff_vs_B", "f1_bar", "latency_p95_ms", "llm_share", "llm_usd_per_1000_band_pairs", "total_usd_month_at_1M_records_per_day", "meets_every_bar"]


def read_csv_skip_comments(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def build(run: Path) -> list[dict]:
    bars = {r["option"]: r for r in read_csv_skip_comments(run / "bars_table.csv")}
    cost = {r["option"]: r for r in read_csv_skip_comments(run / "cost" / "options_vs_bars.csv")}
    orig = "D (orig)" in bars                                   # the secondary row: D with the original routing rule
    meta = json.loads((run / "options" / "b_llm" / "openai" / "sample5000" / "run_meta.json").read_text(encoding="utf-8"))
    per_call = meta["spent_usd"] / max(meta["calls"], 1)
    rows = []
    for opt in ("A", "B", "C", "D") + (("D (orig)",) if orig else ()) + ("E",):
        b = bars[opt]
        if opt == "E":
            rows.append({"option": "E", **{k: "" for k in HEADER[1:]}, "meets_every_bar": "not run"})
            continue
        share = 1.0 if opt == "B" else (float(b["llm_share"]) if opt.startswith("D") else 0.0)
        rows.append({"option": opt, "band_f1": b["band_f1"], "band_f1_ci_lo": b["band_f1_ci_lo"], "band_f1_ci_hi": b["band_f1_ci_hi"],
                     "band_precision": b["band_precision"], "band_recall": b["band_recall"], "overall_precision": b["overall_precision"],
                     "overall_recall": b["overall_recall"], "ece": b["ece"], "f1_diff_vs_B": b["f1_diff_vs_B"], "f1_bar": b["f1_bar"],
                     "latency_p95_ms": b["latency_p95_ms"], "llm_share": b["llm_share"], "llm_usd_per_1000_band_pairs": f"{share * per_call * 1000:.4f}",
                     "total_usd_month_at_1M_records_per_day": cost["D_original" if opt == "D (orig)" else opt]["total_usd_month_at_1M_per_day"], "meets_every_bar": b["meets_every_bar"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    rows = build(args.run)
    with (args.run / "results_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    L = [f"Results, run {args.run.name}", f"  {'option':<9}{'band F1 [95% CI]':<26}{'band P':>8}{'band R':>8}{'overall P':>11}{'overall R':>11}{'ECE':>8}"
         f"{'F1 vs B':>10}  {'F1 bar':<14}{'p95 ms':>8}{'LLM share':>11}{'LLM $/1,000 pairs':>19}{'$/month at 1M/day':>19}  every bar?"]
    for r in rows:
        if r["option"] == "E":
            L.append(f"  {'E':<9}not run")
            continue
        f = lambda x, fmt: "" if x in ("", None) else format(float(x), fmt)
        L.append(f"  {r['option']:<9}{f(r['band_f1'], '.4f')} [{f(r['band_f1_ci_lo'], '.4f')}, {f(r['band_f1_ci_hi'], '.4f')}]  {f(r['band_precision'], '.4f'):>8}"
                 f"{f(r['band_recall'], '.4f'):>8}{f(r['overall_precision'], '.4f'):>11}{f(r['overall_recall'], '.4f'):>11}{f(r['ece'], '.4f'):>8}"
                 f"{f(r['f1_diff_vs_B'], '+.4f') or 'baseline':>10}  {r['f1_bar'].split(' (')[0]:<14}{f(r['latency_p95_ms'], '.3f'):>8}{f(r['llm_share'], '.5f'):>11}"
                 f"{f(r['llm_usd_per_1000_band_pairs'], ',.4f'):>19}{f(r['total_usd_month_at_1M_records_per_day'], ',.0f'):>19}  {r['meets_every_bar']}")
    text = "\n".join(L) + "\n"
    (args.run / "results_table.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

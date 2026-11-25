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

DEV_RUN = Path(__file__).resolve().parent.parent / "results" / "dev_2026_run11"
HEADER = ["option", "band_f1", "band_f1_ci_lo", "band_f1_ci_hi", "band_precision", "band_recall", "overall_precision", "overall_recall", "ece",
          "f1_diff_vs_B", "f1_bar", "latency_p95_ms", "llm_share", "llm_usd_per_1000_band_pairs", "llm_cost_per_call_source", "total_usd_month_at_1M_records_per_day", "meets_every_bar"]


def read_csv_skip_comments(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def b_per_call(run: Path, dev_run: Path | None) -> tuple[float | None, str]:
    """B's cost per call: the run's own sample5000 stage; when that stage made no live call, the dev run's, marked "from dev run".
    (None, reason) when neither has a figure: the table prints n/a, never 0."""
    def read(r: Path):
        f = r / "options" / "b_llm" / "openai" / "sample5000" / "run_meta.json"
        if not f.exists():
            return None
        meta = json.loads(f.read_text(encoding="utf-8"))
        return meta["spent_usd"] / meta["calls"] if meta.get("calls") else None
    own = read(run)
    if own is not None:
        return own, "measured in this run"
    if dev_run is not None and Path(dev_run).resolve() != Path(run).resolve():
        dev = read(Path(dev_run))
        if dev is not None:
            return dev, "from dev run"
    return None, "n/a (no live B calls in this run or the dev run)"


def build(run: Path, dev_run: Path | None = DEV_RUN) -> list[dict]:
    bars = {r["option"]: r for r in read_csv_skip_comments(run / "bars_table.csv")}
    cost = {r["option"]: r for r in read_csv_skip_comments(run / "cost" / "options_vs_bars.csv")}
    extra = tuple(o for o in ("D (orig)", "D (forced)") if o in bars)       # secondary row (original rule) and diagnostic row (forced reading)
    per_call, per_call_src = b_per_call(run, dev_run)
    rows = []
    for opt in ("A", "B", "C", "D") + extra + ("E",):
        b = bars[opt]
        if opt == "E":
            rows.append({"option": "E", **{k: "" for k in HEADER[1:]}, "meets_every_bar": "not run"})
            continue
        share = 1.0 if opt == "B" else (float(b["llm_share"]) if opt.startswith("D") else 0.0)
        rows.append({"option": opt, "band_f1": b["band_f1"], "band_f1_ci_lo": b["band_f1_ci_lo"], "band_f1_ci_hi": b["band_f1_ci_hi"],
                     "band_precision": b["band_precision"], "band_recall": b["band_recall"], "overall_precision": b["overall_precision"],
                     "overall_recall": b["overall_recall"], "ece": b["ece"], "f1_diff_vs_B": b["f1_diff_vs_B"], "f1_bar": b["f1_bar"],
                     "latency_p95_ms": b["latency_p95_ms"], "llm_share": b["llm_share"], "llm_usd_per_1000_band_pairs": "n/a" if per_call is None else f"{share * per_call * 1000:.4f}",
                     "llm_cost_per_call_source": per_call_src,
                     "total_usd_month_at_1M_records_per_day": cost[{"D (orig)": "D_original", "D (forced)": "D_forced"}.get(opt, opt)]["total_usd_month_at_1M_per_day"], "meets_every_bar": b["meets_every_bar"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--dev-run", type=Path, default=DEV_RUN, help="where B's cost per call comes from when this run made no live B call")
    args = ap.parse_args()
    rows = build(args.run, args.dev_run)
    with (args.run / "results_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    L = [f"Results, run {args.run.name}", f"  {'option':<11}{'band F1 [95% CI]':<26}{'band P':>8}{'band R':>8}{'overall P':>11}{'overall R':>11}{'ECE':>8}"
         f"{'F1 vs B':>10}  {'F1 bar':<14}{'p95 ms':>8}{'LLM share':>11}{'LLM $/1,000 pairs':>19}{'$/month at 1M/day':>19}  every bar?"]
    for r in rows:
        if r["option"] == "E":
            L.append(f"  {'E':<11}not run")
            continue
        f = lambda x, fmt: "" if x in ("", None) else ("n/a" if x == "n/a" else format(float(x), fmt))
        L.append(f"  {r['option']:<11}{f(r['band_f1'], '.4f')} [{f(r['band_f1_ci_lo'], '.4f')}, {f(r['band_f1_ci_hi'], '.4f')}]  {f(r['band_precision'], '.4f'):>8}"
                 f"{f(r['band_recall'], '.4f'):>8}{f(r['overall_precision'], '.4f'):>11}{f(r['overall_recall'], '.4f'):>11}{f(r['ece'], '.4f'):>8}"
                 f"{f(r['f1_diff_vs_B'], '+.4f') or 'baseline':>10}  {r['f1_bar'].split(' (')[0]:<14}{f(r['latency_p95_ms'], '.3f'):>8}{f(r['llm_share'], '.5f'):>11}"
                 f"{f(r['llm_usd_per_1000_band_pairs'], ',.4f'):>19}{f(r['total_usd_month_at_1M_records_per_day'], ',.0f'):>19}  {r['meets_every_bar']}")
    L.append(f"  LLM $/1,000 pairs = LLM share x B's cost per call x 1,000; B's cost per call: {rows[0]['llm_cost_per_call_source']}.")
    L.append("  D uses routing rule v2 and C's own decision outside [lo, hi). D (orig) is the original rule (secondary row); D (forced) is v2 with the forced reading (diagnostic).")
    text = "\n".join(L) + "\n"
    (args.run / "results_table.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

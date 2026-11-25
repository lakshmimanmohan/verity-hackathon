"""
Head-to-head table of options A, B, C, D and the secondary row D (orig) from the frozen outputs of the freeze run. Read-only: it reads
only files under results/eval_7919 and results/freeze_<date> (anything else is refused) and writes head_to_head.csv, head_to_head.md
and head_to_head_sources.csv into --out-dir (default results/eval_7919/head_to_head/). Every cell is read from a file (or derived by plain
arithmetic from cells read from files, and marked as derived); a figure the frozen outputs do not hold prints "missing" with its expected
source. Nothing is invented, nothing is rerun, no truth file and no data file is opened.

  python eval/head_to_head.py [--run results/eval_7919] [--freeze results/freeze_2026-11-25]
                              [--cost-dir results/eval_7919/cost] [--latency-json results/eval_7919/latency_final/latency_final.json]
                              [--out-dir results/eval_7919/head_to_head]

--cost-dir is the folder of the cost model outputs to read (default: the freeze run's stage 12 outputs, <run>/cost; for example
results/eval_7919/cost_final for the rerun with the complete effort log). It must lie under the run folder.
--latency-json (a result file of the run, e.g. results/eval_7919/latency_final/latency_final.json) replaces the dev-simulation "D fallback tail" cell with
the freeze run's own figures and adds a per-record latency line for B and D; without it the cells are as before.

Bases: the quality columns are on the fixed 5,000-pair sample; cost, break-even and latency columns come from the cost model and
benchmarks and are not sample figures. The full-band table is exact for A, C and D and is not comparable with B's sample figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OPTS = ("A", "B", "C", "D", "D (orig)")
MAIN = ("A", "B", "C", "D")
SECONDARY = "D (orig)"
RATES = ("50.0", "100.0", "200.0")
VOLUMES = (("100000", "100K"), ("1000000", "1M"), ("10000000", "10M"))
BASE_RATE = "100.0"
BARS_TEXT = ("Quality columns (band F1 and its interval, band precision and recall, overall precision and recall, ECE, F1 versus B) are on the "
             "fixed 5,000-pair sample. Cost, break-even and latency columns come from the cost model and benchmarks and are not sample figures.")
EVERY_SAMPLE = "meets every bar (sample basis)"
EVERY_FULL = "meets every bar (full-band basis, A, C and D only)"

# the protocol's bars (protocol.md), applied as in cost/cost_model.py bars_row
BAR_PRECISION, BAR_RECALL_OVER_A, BAR_ECE, BAR_LATENCY_C_P95_MS, BAR_LLM_SHARE = 0.98, 0.05, 0.05, 20.0, 0.20


class Reader:
    """Reads files only under the allowed roots and remembers every file it read."""

    def __init__(self, roots: list[Path]):
        self.roots = [r.resolve() for r in roots]
        self.read: list[Path] = []

    def path(self, p: Path) -> Path:
        q = Path(p).resolve()
        if not any(q == r or r in q.parents for r in self.roots):
            raise SystemExit(f"refusing to read {p}: outside {[str(r) for r in self.roots]}")
        return q

    def text(self, p: Path) -> str:
        q = self.path(p)
        if q.name.startswith("orgs_raw") or "truth" in q.name or q.name.startswith("hand_check") or "hand_check" in q.parts:
            raise SystemExit(f"refusing to read {q.name}: data, truth and hand-check files are never read here")
        self.read.append(q)
        return q.read_text(encoding="utf-8")

    def csv(self, p: Path) -> list[dict]:
        lines = [ln for ln in self.text(p).splitlines() if not ln.startswith("#")]
        return list(csv.DictReader(lines))

    def json(self, p: Path):
        return json.loads(self.text(p))


class Table:
    """Cells with their sources: cell(row, column) -> (text, source, how)."""

    def __init__(self):
        self.cells: dict[tuple[str, str], tuple[str, str, str]] = {}
        self.columns: list[str] = []

    def put(self, row: str, col: str, text: str, source: str, how: str = "read") -> None:
        if col not in self.columns:
            self.columns.append(col)
        self.cells[(row, col)] = (text, source, how)

    def missing(self, row: str, col: str, expected: str) -> None:
        self.put(row, col, f"missing (expected source: {expected})", expected, "missing")

    def get(self, row: str, col: str) -> str:
        return self.cells.get((row, col), ("", "", ""))[0]


# ---------------------------------------------------------------------------------------------- parsing of the score reports

def parse_report(text: str) -> dict:
    """Band F1 and its interval, band precision and recall, overall precision and recall, ECE and the decoy line of a score report."""
    out: dict = {}
    m = re.search(r"\n  F1 ([\d.]+)  95% CI \[([\d.]+), ([\d.]+)\]", text)
    out["f1"], out["f1_lo"], out["f1_hi"] = m.groups()
    m = re.search(r"\n  accuracy [\d.]+  precision ([\d.]+)  recall ([\d.]+)", text)
    out["band_precision"], out["band_recall"] = m.groups()
    m = re.search(r"\nOverall \([^\n]*\n  precision ([\d.]+)  recall ([\d.]+)", text)
    out["overall_precision"], out["overall_recall"] = m.groups()
    m = re.search(r"\n  ECE ([\d.]+)   \(bar", text)
    out["ece"] = m.group(1) if m else None
    m = re.search(r"Decoy pairs in the test group: ([\d.,]+), wrongly matched ([\d.,]+) \(([\d.]+)%\)", text)
    out["decoy"] = m.groups() if m else None
    return out


def f4(x: str) -> str:
    return f"{float(x):.4f}"


def fmt_money(x: str) -> str:
    """A money cell as it is in the file: a number with thousands separators, or the file's own text ('n/m' = not modeled)."""
    try:
        return f"{float(x):,.2f}"
    except ValueError:
        return str(x)


# ---------------------------------------------------------------------------------------------- the protocol bars

def bar_verdicts(opt: str, overall_p: float, overall_r: float, a_recall: float, ece: float | None, f1_verdict: str,
                 latency_p95_ms: float | None, llm_share: float | None) -> tuple[str, dict]:
    """The verdict per bar and whether every bar is met, by the protocol rules (the same rules as cost/cost_model.py bars_row).
    Returns (meets every bar, {bar: (value text, verdict)})."""
    v: dict[str, tuple[str, str]] = {}
    if opt == "B":
        v["F1 within 2 points of B"] = ("baseline", "reference")
    else:
        v["F1 within 2 points of B"] = (f1_verdict, f1_verdict)
    v["overall precision >= 0.98"] = (f"{overall_p:.4f}", "pass" if overall_p >= BAR_PRECISION else "fail")
    if opt == "A":
        v["overall recall >= A + 0.05"] = (f"{overall_r:.4f}", "reference")
    else:
        v["overall recall >= A + 0.05"] = (f"{overall_r:.4f} (A {a_recall:.4f})", "pass" if overall_r >= a_recall + BAR_RECALL_OVER_A - 1e-12 else "fail")
    if opt in ("C", "D"):
        v["ECE <= 0.05"] = (f"{ece:.4f}", "pass" if ece <= BAR_ECE else "fail")
    if opt == "C":
        v["C latency p95 <= 20 ms"] = (f"{latency_p95_ms} ms", "pass" if latency_p95_ms <= BAR_LATENCY_C_P95_MS else "fail")
    if opt == "D":
        v["D LLM share <= 20%"] = (f"{llm_share:.5f}", "pass" if llm_share <= BAR_LLM_SHARE else "fail")
    checked = [x[1] for x in v.values() if x[1] in ("pass", "fail", "inconclusive")]
    if "fail" in checked or opt == "A":
        every = "no"
    elif "inconclusive" in checked:
        every = "not shown (inconclusive)"
    else:
        every = "yes"
    return every, v


# ---------------------------------------------------------------------------------------------- the tables

def build(run: Path, freeze: Path, R: Reader, cost_dir: Path | None = None, latency_json: Path | None = None) -> dict:
    cost_dir = Path(cost_dir) if cost_dir else run / "cost"
    rel = lambda p: Path(p).resolve().relative_to(REPO).as_posix()
    O = run / "options"
    sample_reports = {"A": O / "a_threshold" / "score_report_sample5000.txt", "B": O / "b_llm" / "openai" / "sample5000" / "score_report_sample5000.txt",
                      "C": O / "c_model" / "c_v1_lgbm" / "score_report_sample5000.txt", "D": O / "d_hybrid" / "score_report_sample5000.txt",
                      "D (orig)": O / "d_hybrid" / "score_report_sample5000_original.txt"}
    full_reports = {"A": O / "a_threshold" / "score_report.txt", "C": O / "c_model" / "c_v1_lgbm" / "score_report.txt",
                    "D": O / "d_hybrid" / "score_report_fullband.txt", "D (orig)": O / "d_hybrid" / "score_report_fullband_original.txt"}
    bars = {r["option"]: r for r in R.csv(run / "bars_table.csv")}
    bars_src = rel(run / "bars_table.csv")
    results = {r["option"]: r for r in R.csv(run / "results_table.csv")}
    results_src = rel(run / "results_table.csv")
    cost = R.csv(cost_dir / "cost_model.csv")
    cost_src = rel(cost_dir / "cost_model.csv")
    ovb = {r["option"]: r for r in R.csv(cost_dir / "options_vs_bars.csv")}
    ovb_src = rel(cost_dir / "options_vs_bars.csv")
    be = R.csv(cost_dir / "break_even.csv")
    be_src = rel(cost_dir / "break_even.csv")
    assumptions = {r["name"]: r for r in R.csv(cost_dir / "assumptions.csv")}
    assumptions_src = rel(cost_dir / "assumptions.csv")
    lat = R.json(O / "latency" / "latency_summary.json")
    lat_src = rel(O / "latency" / "latency_summary.json")
    btim = R.json(O / "b_llm" / "openai" / "sample5000" / "timings.json")
    btim_src = rel(O / "b_llm" / "openai" / "sample5000" / "timings.json")
    T = Table()
    cost_opt = {"A": "A", "B": "B", "C": "C", "D": "D"}
    bpr = assumptions["band pairs per record"]["value"]                     # band pairs per record, from the run
    parsed = {o: parse_report(R.text(sample_reports[o])) for o in OPTS}
    parsed_full = {o: parse_report(R.text(p)) for o, p in full_reports.items()}
    c_p95 = lat["C"]["median_of_repeats"]["p95_ms"]
    d_share = {o: float(bars[o]["llm_share"]) if bars[o]["llm_share"] != "" else None for o in ("D", "D (orig)")}

    for o in OPTS:
        src = rel(sample_reports[o])
        rep = parsed[o]
        T.put(o, "band F1 [95% CI]", f"{f4(rep['f1'])} [{f4(rep['f1_lo'])}, {f4(rep['f1_hi'])}]", src)
        b = bars[o]
        if o == "B":
            T.put(o, "F1 vs B [95% CI]", "baseline", bars_src)
            T.put(o, "F1 verdict", "reference (B is the baseline)", bars_src)
        else:
            T.put(o, "F1 vs B [95% CI]", f"{float(b['f1_diff_vs_B']):+.4f} [{float(b['f1_diff_lo']):+.4f}, {float(b['f1_diff_hi']):+.4f}]", bars_src)
            T.put(o, "F1 verdict", b["f1_bar"].split(" (")[0], bars_src)
        T.put(o, "band precision", f4(rep["band_precision"]), src)
        T.put(o, "band recall", f4(rep["band_recall"]), src)
        T.put(o, "overall precision", f4(rep["overall_precision"]), src)
        T.put(o, "overall recall", f4(rep["overall_recall"]), src)
        if rep["ece"] is None:
            T.put(o, "ECE", "not reported (bar is for C and D)", src, "read")
        else:
            T.put(o, "ECE", f4(rep["ece"]), src)
        # the protocol bars, computed from the frozen reports, on the two bases
        f1v = T.get(o, "F1 verdict").split(" (")[0]
        a_rec_s = float(parsed["A"]["overall_recall"])
        every_s, _ = bar_verdicts(o, float(rep["overall_precision"]), float(rep["overall_recall"]), a_rec_s,
                                  None if rep["ece"] is None else float(rep["ece"]), f1v, c_p95, d_share.get(o))
        T.put(o, EVERY_SAMPLE, every_s, f"{src}; {parsed_src(sample_reports['A'], rel)}; {bars_src}; {lat_src}",
              "derived: the protocol's bars applied to the sample-basis report" + ("; LLM share from bars_table.csv" if o in d_share else ""))
        if o in ("A", "C", "D"):
            fr = parsed_full[o]
            every_f, _ = bar_verdicts(o, float(fr["overall_precision"]), float(fr["overall_recall"]), float(parsed_full["A"]["overall_recall"]),
                                      None if fr["ece"] is None else float(fr["ece"]), f1v, c_p95, d_share.get(o))
            T.put(o, EVERY_FULL, every_f, f"{rel(full_reports[o])}; {rel(full_reports['A'])}; {bars_src}; {lat_src}",
                  "derived: the protocol's bars applied to the full-band report" + ("; LLM share from bars_table.csv" if o in d_share else ""))
        elif o == "B":
            T.put(o, EVERY_FULL, "n/a (B exists only on the sample)", "", "constant text")
        else:
            T.put(o, EVERY_FULL, "not computed (column is for A, C and D only)", "", "constant text")
        # LLM cost per 1,000 band pairs and per record
        per1000 = results[o]["llm_usd_per_1000_band_pairs"]
        T.put(o, "LLM $ per 1,000 band pairs", f"{float(per1000):.4f}", results_src)
        T.put(o, "LLM $ per record", f"{float(per1000) * float(bpr) / 1000:.5f}", f"{results_src}; {assumptions_src}",
              f"derived: per 1,000 band pairs x {bpr} band pairs per record / 1,000")
        T.put(o, "band pairs per record", bpr, assumptions_src)

    # monthly cost: S1, $100/h, labels as evaluated; Precisely / customer / total
    labels_eval = assumptions["labels as evaluated (C and D)"]["value"]
    for o in OPTS:
        for vol, vname in VOLUMES:
            col = f"monthly $ at {vname} records/day (Precisely / customer / total)"
            if o in cost_opt:
                rows = [r for r in cost if r["option"] == cost_opt[o] and r["scenario"] == "S1" and r["hourly_rate_usd"] == BASE_RATE
                        and r["records_per_day"] == vol and r["labels_for_C_and_D"] == labels_eval]
                if len(rows) == 1:
                    r = rows[0]
                    T.put(o, col, f"{fmt_money(r['precisely_usd_month'])} / {fmt_money(r['customer_usd_month'])} / {fmt_money(r['total_usd_month'])}", cost_src)
                    continue
            if o == "D (orig)" and vol == "1000000" and "D_original" in ovb:
                T.put(o, col, f"missing / missing / {fmt_money(ovb['D_original']['total_usd_month_at_1M_per_day'])} (total only)", ovb_src,
                      "read (total only: the cost model has no Precisely/customer split for D (orig))")
                continue
            T.missing(o, col, "cost/cost_model.csv (it has no D (orig) rows; cost/options_vs_bars.csv holds only the 1M total)" if o == "D (orig)" else "cost/cost_model.csv")

    # latency
    def lat_text(k):
        m = lat[k]["median_of_repeats"]
        return f"p50 {m['p50_ms']} ms, p95 {m['p95_ms']} ms"
    llm = [t for t in btim["timings"] if t["stage"].endswith("llm_call")][0]["per_item_ms"]
    T.put("A", "latency per pair (p50, p95)", "not measured (threshold comparison)", "", "constant text")
    T.put("B", "latency per pair (p50, p95)", f"p50 {llm['p50'] / 1000:.3f} s, p95 {llm['p95'] / 1000:.3f} s (per LLM call)", btim_src)
    T.put("C", "latency per pair (p50, p95)", lat_text("C") + " (one core, features included)", lat_src)
    T.put("D", "latency per pair (p50, p95)", lat_text("D") + " (one core, features included, plus routing)", lat_src)
    T.missing("D (orig)", "latency per pair (p50, p95)", "options/latency/latency_summary.json (measures D with the v2 setting only)")
    fb = assumptions.get("D per-pair latency with the LLM fallback (simulation)")
    lj = R.json(latency_json) if latency_json else None
    lj_src = rel(latency_json) if latency_json else ""
    for o in OPTS:
        if lj is not None and o == "D":
            b50, b95 = lj["b_call_latency_sample5000"]["p50_s"], lj["b_call_latency_sample5000"]["p95_s"]
            T.put(o, "D fallback tail",
                  f"{100 * lj['routed_share']:.3f}% of pairs ({lj['routed_pairs']:,} of {lj['band_pairs']:,}) call the LLM, each call p50 {b50:.3f} s, p95 {b95:.3f} s; "
                  f"{100 * lj['records_with_routed_pair_share']:.2f}% of test records include at least one; the dev simulation (0.085% share) is withdrawn", lj_src)
        elif lj is not None and o == "D (orig)":
            T.missing(o, "D fallback tail", "results/eval_7919/latency_final/latency_final.json (it describes the v2 setting only)")
        elif lj is None and o == "D" and fb:
            T.put(o, "D fallback tail", fb["value"] + " [simulation from measured distributions]", assumptions_src)
        elif o in ("D", "D (orig)"):
            T.missing(o, "D fallback tail", "cost/assumptions.csv row 'D per-pair latency with the LLM fallback (simulation)'")
        else:
            T.put(o, "D fallback tail", "n/a (no LLM fallback)", "", "constant text")
    if lj is not None:                                         # a per-record latency line for B and D
        col = "per-record latency (LLM calls)"
        rows = {r["latency_used"]: r for r in lj["b_per_record"]}
        fr = rows["freeze run, call p50 (2 workers)"]
        T.put("A", col, "no LLM call", "", "constant text")
        T.put("C", col, "no LLM call", "", "constant text")
        T.put("B", col, f"{lj['b_calls_per_record']:.2f} calls per record; at {fr['workers']} workers {fr['mean_time_per_record_s']:.2f} s on average, "
              f"{fr['whole_rounds_time_per_record_s']:.2f} s counting whole rounds of {fr['workers']} calls (freeze run call p50 {fr['latency_s']:.3f} s; queueing not included)", lj_src)
        T.put("D", col, f"{lj['routed_pairs_per_record_both_ends']:.4f} routed pairs (LLM calls) per record on average; {100 * lj['records_with_routed_pair_share']:.2f}% of test records "
              f"({lj['records_with_routed_pair']:,} of {lj['test_group_records']:,}) have at least one, at most {lj['max_routed_pairs_on_one_record']}; each call p50 "
              f"{lj['b_call_latency_sample5000']['p50_s']:.3f} s", lj_src)
        T.missing("D (orig)", col, "results/eval_7919/latency_final/latency_final.json (it describes the v2 setting only)")

    # break-even against B
    for rate in RATES:
        for lab_key, lab in (("labels as evaluated", labels_eval), ("2,000 labels", "2000")):
            col = f"break-even vs B at ${float(rate):.0f}/h, {lab_key}"
            for o in OPTS:
                if o == "B":
                    T.put(o, col, "baseline", be_src)
                    continue
                rows = [r for r in be if r["scenario"].startswith("S1 and S2") and r["option"] == o and r["hourly_rate_usd"] == rate and r["labels_for_C_and_D"] == lab]
                if len(rows) != 1:
                    T.missing(o, col, "cost/break_even.csv (no row for D (orig))" if o == "D (orig)" else "cost/break_even.csv")
                    continue
                r = rows[0]
                txt = r["result"] if not r["break_even_records_per_day"] else f"{r['result']} {int(float(r['break_even_records_per_day'])):,} records/day"
                T.put(o, col, txt, be_src)
    # full-band table
    full = Table()
    for o, p in full_reports.items():
        rep = parsed_full[o]
        src = rel(p)
        full.put(o, "band F1 [95% CI]", f"{f4(rep['f1'])} [{f4(rep['f1_lo'])}, {f4(rep['f1_hi'])}]", src)
        full.put(o, "band precision", f4(rep["band_precision"]), src)
        full.put(o, "band recall", f4(rep["band_recall"]), src)

    # the two "meets every bar" columns: a note when they differ for any option
    bar_notes = []
    for o in ("A", "C", "D"):
        f1v = T.get(o, "F1 verdict").split(" (")[0]
        every_s, vs = bar_verdicts(o, float(parsed[o]["overall_precision"]), float(parsed[o]["overall_recall"]), float(parsed["A"]["overall_recall"]),
                                   None if parsed[o]["ece"] is None else float(parsed[o]["ece"]), f1v, c_p95, d_share.get(o))
        every_f, vf = bar_verdicts(o, float(parsed_full[o]["overall_precision"]), float(parsed_full[o]["overall_recall"]), float(parsed_full["A"]["overall_recall"]),
                                   None if parsed_full[o]["ece"] is None else float(parsed_full[o]["ece"]), f1v, c_p95, d_share.get(o))
        if every_s != every_f:
            before = len(bar_notes)
            for bar in vs:
                if vs[bar][1] != vf[bar][1]:
                    bar_notes.append(f"{o}: bar '{bar}': sample basis {vs[bar][0]} ({vs[bar][1]}); full-band basis {vf[bar][0]} ({vf[bar][1]}); "
                                     f"meets every bar: sample basis '{every_s}', full-band basis '{every_f}'.")
            if len(bar_notes) == before:
                bar_notes.append(f"{o}: meets every bar differs: sample basis '{every_s}', full-band basis '{every_f}'.")

    # decoys
    decoys = []
    for o in OPTS:
        for basis, rp in (("5,000-pair sample (weighted estimate)", sample_reports.get(o)), ("full band (exact)", full_reports.get(o))):
            if rp is None:
                continue
            rep = parse_report(R.text(rp))
            if rep["decoy"] is None:
                decoys.append((o, basis, "missing", "missing", rel(rp)))
            else:
                decoys.append((o, basis, rep["decoy"][0], f"{rep['decoy'][1]} ({rep['decoy'][2]}%)", rel(rp)))
    dc = O / "b_llm" / "openai" / "decoycheck"
    dtext = R.text(dc / "decoy_report.txt")
    dmeta = R.json(dc / "run_meta.json")
    mm = re.search(r"every pair is a different-organization pair: false matches (\d+) of (\d+)", dtext)
    bound = re.search(r"false-match rate <= ([\d.]+)", dtext)
    decoy_check = {"pairs": int(mm.group(2)), "false_matches": int(mm.group(1)), "bound": bound.group(1) if bound else "missing",
                   "live_calls": dmeta["calls"], "cache_hits": dmeta["cache_hits"], "spent_usd": dmeta["spent_usd"],
                   "sources": [rel(dc / "decoy_report.txt"), rel(dc / "run_meta.json")]}

    # cost numbers for the slides
    slides: dict = {}
    ot = R.csv(cost_dir / "one_time_cost.csv")
    slides["build"] = [(r["option"], r["hourly_rate_usd"], r["build_before_usd"], r["build_after_usd"], r["labeling_2000_labels_at_100_per_hour_usd"],
                        r["labeling_as_evaluated_labels_at_100_per_hour_usd"]) for r in ot]
    slides["build_src"] = rel(cost_dir / "one_time_cost.csv")
    summ = R.text(cost_dir / "summary.txt")
    slides["draft_banner"] = summ.splitlines()[0]
    slides["summary_src"] = rel(cost_dir / "summary.txt")
    lab = {}
    for m in re.finditer(r"\n  ([CD])\s+(as evaluated|at 2,000)\s+([\d,]+)\s+([\d.]+)\s+([\d,]+/[\d,]+/[\d,]+)", summ):
        lab[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
    slides["label_cost"] = lab
    m = re.search(r"Effort log: (\d+) rows with hours", summ)
    slides["effort_log_rows"] = m.group(1) if m else "missing"
    pays = []
    for r in cost:
        if r["labels_for_C_and_D"] == labels_eval and r["hourly_rate_usd"] == BASE_RATE and r["records_per_day"] == "1000000" and r["option"] in "ABCD" and r["scenario"]:
            pays.append((r["option_row"], r["scenario"], r["precisely_usd_month"], r["customer_usd_month"], r["total_usd_month"], r["llm_paid_by"], r["hosting_paid_by"],
                         r["operations_paid_by"], r["development_paid_by"], r["labeling_paid_by"]))
    slides["who_pays"] = pays
    slides["who_pays_src"] = cost_src
    below = []
    for r in be:
        if r["scenario"].startswith("S1 and S2") and r["labels_for_C_and_D"] in (labels_eval, "2000") and r["option"] in ("A", "C", "D"):
            below.append((r["option"], r["hourly_rate_usd"], r["labels_for_C_and_D"], r["result"], r["break_even_records_per_day"]))
    slides["b_cheaper_below"] = below
    slides["break_even_src"] = be_src
    fr = R.text(freeze / "freeze_report.txt")
    m = re.search(r"effort log read as it is \(not frozen\): sha256 ([0-9a-f]{64})", fr)
    slides["effort_log_sha256_frozen_run"] = m.group(1) if m else "missing (expected source: freeze_report.txt stage 12 note)"
    slides["effort_log_src"] = rel(freeze / "freeze_report.txt")
    slides["labels_eval"] = labels_eval
    slides["cost_dir"] = rel(cost_dir)
    return {"T": T, "full": full, "decoys": decoys, "decoy_check": decoy_check, "slides": slides, "bpr": bpr, "labels_eval": labels_eval, "bar_notes": bar_notes}


def parsed_src(path: Path, rel) -> str:
    """The source path of A's sample report (the recall bar's reference)."""
    return rel(path)


# ---------------------------------------------------------------------------------------------- output

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def full_rows(full: Table, opts=MAIN) -> list[str]:
    return [o for o in opts if (o, "band F1 [95% CI]") in full.cells]


def write_outputs(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    T, full, S = res["T"], res["full"], res["slides"]
    with (out / "head_to_head.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["option"] + T.columns)
        for o in MAIN:
            w.writerow([o] + [T.get(o, c) for c in T.columns])
        w.writerow([])
        w.writerow(["# secondary row"])
        w.writerow(["option"] + T.columns)
        w.writerow([SECONDARY] + [T.get(SECONDARY, c) for c in T.columns])
        w.writerow([])
        w.writerow(["# full band, not comparable with B's sample figures"])
        w.writerow(["option"] + full.columns)
        for o in full_rows(full, OPTS):
            w.writerow([o] + [full.get(o, c) for c in full.columns])
    with (out / "head_to_head_sources.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["table", "row", "column", "source_file", "how"])
        for tname, tab in (("sample basis", T), ("full band", full)):
            for (row, col), (text, src, how) in sorted(tab.cells.items()):
                w.writerow([tname, row, col, src, how])

    L = ["# Head to head: options A to D (freeze run, seed 7919)", "",
         "Everything below is read from files under `results/eval_7919` and `results/freeze_2026-11-25` (see the sources section). "
         "`missing` means the frozen outputs do not hold the figure; the expected source is named. Nothing was rerun or invented.", "",
         "## Bases", "", BARS_TEXT, "",
         "Notes:", "",
         f"- Cost outputs read from `{S['cost_dir']}` (effort log: {S['effort_log_rows']} rows).",
         f"- Labels as evaluated for C and D in this run: {int(res['labels_eval']):,} (`assumptions.csv` of the cost outputs). Hourly rate $100 and scenario S1 unless a column says otherwise.",
         "- 'LLM $' columns are the LLM cost only (LLM share x B's measured cost per call); the total cost is in the monthly table.",
         "- Table 2 (full band) is exact for A, C and D and is **not comparable with B's sample figures**. The secondary row D (orig) is in its own table.", ""]
    groups = [("Quality, sample basis", ["band F1 [95% CI]", "F1 vs B [95% CI]", "F1 verdict", "band precision", "band recall", "overall precision", "overall recall", "ECE"]),
              ("Protocol bars (computed from the frozen reports with the protocol rules)", [EVERY_SAMPLE, EVERY_FULL]),
              ("Cost per pair and per record (not sample figures)", ["LLM $ per 1,000 band pairs", "LLM $ per record", "band pairs per record"]),
              ("Monthly cost (S1, $100/h, labels as evaluated; cost model, not sample figures)", [c for c in T.columns if c.startswith("monthly")]),
              ("Latency (benchmarks, not sample figures)", ["latency per pair (p50, p95)", "D fallback tail"] + (["per-record latency (LLM calls)"] if "per-record latency (LLM calls)" in T.columns else [])),
              ("Break-even against B (S1 and S2 total; cost model, not sample figures)", [c for c in T.columns if c.startswith("break-even")])]
    n = 1
    for title, cols in groups:
        L += [f"### Table 1.{n}: {title}", "", md_table(["option"] + cols, [[o] + [T.get(o, c) for c in cols] for o in MAIN]), ""]
        if n == 2:
            L += ["Bars (protocol.md): F1 within 2 points of B on the paired sample (from `bars_table.csv`); overall precision >= 0.98; overall recall at least 5 points above A; "
                  "ECE <= 0.05 (C, D); C latency p95 <= 20 ms; D LLM share <= 20% (from `bars_table.csv`). The two columns apply the same rules to the sample-basis reports and to the "
                  "full-band reports; the F1 verdict, the latency and the LLM share are the same in both.", ""]
            if res["bar_notes"]:
                L += ["Where the two columns differ:", ""] + [f"- {x}" for x in res["bar_notes"]] + [""]
            else:
                L += ["The two columns agree for A, C and D (B has no full-band value): no bar's verdict differs between the sample-basis and the full-band reports.", ""]
        n += 1
    L += ["## Table 2: full band, not comparable with B's sample figures", "",
          md_table(["option"] + full.columns, [[o] + [full.get(o, c) for c in full.columns] for o in full_rows(full)]), ""]

    L += ["## Secondary row: D (orig) (the original routing rule, lo 0.5, hi 0.9)", "",
          "Same figures as Table 1, one per line. `missing` marks a figure the frozen outputs do not hold, with its expected source.", "",
          md_table(["figure", "D (orig)"], [[c, T.get(SECONDARY, c)] for c in T.columns]), "",
          "Full band (exact, not comparable with B's sample figures): " + "; ".join(f"{c} {full.get(SECONDARY, c)}" for c in full.columns), ""]

    L += ["## Decoy outcomes (same name, different entity; test group)", "",
          md_table(["option", "basis", "decoy pairs", "wrongly matched", "source"], [list(d) for d in res["decoys"]]), ""]
    dc = res["decoy_check"]
    L += [f"B's decoy check: {dc['pairs']} test band decoy pairs, {dc['false_matches']} false matches; one-sided 95% bound on the false-match rate <= {dc['bound']}; "
          f"{dc['live_calls']} live calls and {dc['cache_hits']} cache hits, ${dc['spent_usd']:.4f} (sources: {', '.join(dc['sources'])}).", ""]

    banner = S["draft_banner"]
    L += ["## Cost numbers for the slides", "",
          f"- Effort log of the freeze run's stage 12: sha256 `{S['effort_log_sha256_frozen_run']}` (source: {S['effort_log_src']}). The cost outputs read here are from `{S['cost_dir']}`, "
          f"which used an effort log of {S['effort_log_rows']} rows.",
          f"- First line of {S['summary_src']}: " + (f"**{banner}** The DRAFT banner is on." if banner.startswith("DRAFT") else f"\"{banner}\". No DRAFT banner."), "",
          "### One-time build cost per option (USD; before = Verity pre-existing build and productization only; after = plus the option's own hours)", "",
          md_table(["option", "$/h", "build before", "build after", "labeling at 2,000 labels ($100/h)", "labeling as evaluated ($100/h)"],
                   [[o, f"{float(r):.0f}", fmt_money(b), fmt_money(a), fmt_money(l2), fmt_money(le)] for o, r, b, a, l2, le in S["build"]]), "",
          f"Source: {S['build_src']}. Labeling at $50 and $200 per hour (from {S['summary_src']}, 'label cost at $50/$100/$200'):", ""]
    L += [md_table(["option", "labels", "label hours", "cost at $50 / $100 / $200"],
                   [[k[0], f"{k[1]} ({v[0]})", v[1], v[2].replace("/", " / ")] for k, v in sorted(S["label_cost"].items())]), "",
          "### Who pays at 1M records/day (USD per month, $100/h, labels as evaluated)", "",
          md_table(["option row", "scenario", "Precisely", "customer", "total", "LLM paid by", "hosting paid by", "operations paid by", "development paid by", "labeling paid by"],
                   [[a, b, fmt_money(c), fmt_money(d), fmt_money(e), f, g, h, i, j] for a, b, c, d, e, f, g, h, i, j in S["who_pays"]]),
          "", f"Source: {S['who_pays_src']}.", "",
          "### Volume below which B is cheaper overall (S1 and S2 total over 12 months; break-even.csv)", "",
          md_table(["option vs B", "$/h", "labels", "file's result", "break-even records/day", "B is cheaper overall"],
                   [[o, f"{float(r):.0f}", f"{int(lab):,}", res_, f"{int(float(be)):,}" if be else "",
                     f"below {int(float(be)):,} records/day" if res_ == "option cheaper above" and be else "nowhere in 1,000 to 100,000,000 records/day"]
                    for o, r, lab, res_, be in S["b_cheaper_below"]]), "",
          f"Where the result is 'option cheaper above N', B is cheaper below N records/day. Where it is 'option cheaper everywhere in range', B is not cheaper anywhere in the range 1,000 to 100,000,000 records/day. Source: {S['break_even_src']}.", ""]

    L += ["## Sources", "", "Per column, the file each option's cell was read from (the full cell-by-cell list is `head_to_head_sources.csv`). "
          "'derived' cells are plain arithmetic or the protocol's rules applied to cells read from files.", ""]
    rows = []
    for tname, tab in (("Table 1 and secondary row", T), ("Table 2", full)):
        for col in tab.columns:
            by_src: dict[str, list[str]] = {}
            for o in OPTS:
                if (o, col) in tab.cells:
                    text, src, how = tab.cells[(o, col)]
                    by_src.setdefault(src + ("" if how == "read" else f" [{how}]"), []).append(o)
            rows.append([tname, col, "; ".join(f"{', '.join(os_)}: {s or '(constant text)'}" for s, os_ in by_src.items())])
    L += [md_table(["table", "column", "source file per option"], rows), ""]
    (out / "head_to_head.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Head-to-head table from the frozen outputs of the freeze run.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "eval_7919")
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--cost-dir", type=Path, default=None, help="folder of the cost model outputs to read (default: <run>/cost, the freeze run's stage 12 outputs)")
    ap.add_argument("--latency-json", type=Path, default=None, help="latency_final.json of the freeze run: replaces the dev-simulation 'D fallback tail' cell and adds a per-record latency line")
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write (default: <run>/head_to_head)")
    args = ap.parse_args()
    run = args.run.resolve()
    freeze = args.freeze or next(iter(sorted((REPO / "results").glob("freeze_????-??-??"))), None)
    if freeze is None:
        raise SystemExit("no results/freeze_<date> folder")
    R = Reader([run, Path(freeze)])
    cost_dir = (args.cost_dir or run / "cost").resolve()
    R.path(cost_dir)                                              # must lie under the allowed roots
    latency_json = args.latency_json.resolve() if args.latency_json else None
    if latency_json:
        R.path(latency_json)                                      # must lie under the run folder
    res = build(run, Path(freeze).resolve(), R, cost_dir, latency_json)
    out = args.out_dir or run / "head_to_head"
    write_outputs(res, out)
    print(f"wrote {out / 'head_to_head.csv'}, head_to_head.md and head_to_head_sources.csv; read {len(set(R.read))} files, all under {[str(r.relative_to(REPO)) for r in R.roots]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

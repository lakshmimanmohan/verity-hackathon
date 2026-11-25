"""
Head-to-head table of options A, B, C, D and D (orig) from the frozen outputs of the freeze run. Read-only: it reads only files under
results/eval_7919 and results/freeze_<date> (anything else is refused) and writes head_to_head.csv, head_to_head.md and
head_to_head_sources.csv into results/eval_7919/head_to_head/. Every cell is read from a file (or derived by plain arithmetic from cells
read from files, and marked as derived); a figure the frozen outputs do not hold prints "missing" with its expected source. Nothing is
invented, nothing is rerun, no truth file and no data file is opened.

  python eval/head_to_head.py [--run results/eval_7919] [--freeze results/freeze_2026-11-25]

Bases (printed in the output): the main table is on the 5,000-pair sample basis (weighted estimates; B exists only there). The second
table is the full band (exact for A, C, D, D (orig)) and is not comparable with B's sample figures.
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
RATES = ("50.0", "100.0", "200.0")
VOLUMES = (("100000", "100K"), ("1000000", "1M"), ("10000000", "10M"))
BASE_RATE, BASE_LABELS_KEY = "100.0", "as evaluated"


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
        if q.name.startswith("orgs_raw") or "truth" in q.name:
            raise SystemExit(f"refusing to read {q.name}: data and truth files are never read here")
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


# ---------------------------------------------------------------------------------------------- the tables

def build(run: Path, freeze: Path, R: Reader) -> dict:
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
    cost = R.csv(run / "cost" / "cost_model.csv")
    cost_src = rel(run / "cost" / "cost_model.csv")
    ovb = {r["option"]: r for r in R.csv(run / "cost" / "options_vs_bars.csv")}
    ovb_src = rel(run / "cost" / "options_vs_bars.csv")
    be = R.csv(run / "cost" / "break_even.csv")
    be_src = rel(run / "cost" / "break_even.csv")
    assumptions = {r["name"]: r for r in R.csv(run / "cost" / "assumptions.csv")}
    assumptions_src = rel(run / "cost" / "assumptions.csv")
    lat = R.json(O / "latency" / "latency_summary.json")
    lat_src = rel(O / "latency" / "latency_summary.json")
    btim = R.json(O / "b_llm" / "openai" / "sample5000" / "timings.json")
    btim_src = rel(O / "b_llm" / "openai" / "sample5000" / "timings.json")
    T = Table()
    bar_name = {"A": "A", "B": "B", "C": "C", "D": "D", "D (orig)": "D (orig)"}
    cost_opt = {"A": "A", "B": "B", "C": "C", "D": "D"}

    bpr = assumptions["band pairs per record"]["value"]                     # band pairs per record, from the run
    for o in OPTS:
        src = rel(sample_reports[o])
        rep = parse_report(R.text(sample_reports[o]))
        T.put(o, "band F1 [95% CI]", f"{f4(rep['f1'])} [{f4(rep['f1_lo'])}, {f4(rep['f1_hi'])}]", src)
        b = bars[bar_name[o]]
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
        # LLM cost per 1,000 band pairs and per record
        per1000 = results[o]["llm_usd_per_1000_band_pairs"]
        T.put(o, "LLM $ per 1,000 band pairs", f"{float(per1000):.4f}", results_src)
        T.put(o, "LLM $ per record", f"{float(per1000) * float(bpr) / 1000:.5f}", f"{results_src}; {assumptions_src}",
              f"derived: per 1,000 band pairs x {bpr} band pairs per record / 1,000")
        T.put(o, "band pairs per record", bpr, assumptions_src)
        T.put(o, "meets every bar", b["meets_every_bar"], bars_src)

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
    T.missing("A", "latency per pair (p50, p95)", "options/latency/latency_summary.json (no entry for A)")
    T.put("B", "latency per pair (p50, p95)", f"p50 {llm['p50'] / 1000:.3f} s, p95 {llm['p95'] / 1000:.3f} s (per LLM call)", btim_src)
    T.put("C", "latency per pair (p50, p95)", lat_text("C") + " (one core, features included)", lat_src)
    T.put("D", "latency per pair (p50, p95)", lat_text("D") + " (one core, features included, plus routing)", lat_src)
    T.missing("D (orig)", "latency per pair (p50, p95)", "options/latency/latency_summary.json (measures D with the v2 setting only)")
    fb = assumptions.get("D per-pair latency with the LLM fallback (simulation)")
    for o in OPTS:
        if o == "D" and fb:
            T.put(o, "D fallback tail", fb["value"] + " [simulation from measured distributions]", assumptions_src)
        elif o in ("D", "D (orig)"):
            T.missing(o, "D fallback tail", "cost/assumptions.csv row 'D per-pair latency with the LLM fallback (simulation)'")
        else:
            T.put(o, "D fallback tail", "n/a (no LLM fallback)", "")

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
        rep = parse_report(R.text(p))
        src = rel(p)
        full.put(o, "band F1 [95% CI]", f"{f4(rep['f1'])} [{f4(rep['f1_lo'])}, {f4(rep['f1_hi'])}]", src)
        full.put(o, "band precision", f4(rep["band_precision"]), src)
        full.put(o, "band recall", f4(rep["band_recall"]), src)

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
    ot = R.csv(run / "cost" / "one_time_cost.csv")
    slides["build"] = [(r["option"], r["hourly_rate_usd"], r["build_before_usd"], r["build_after_usd"], r["labeling_2000_labels_at_100_per_hour_usd"],
                        r["labeling_as_evaluated_labels_at_100_per_hour_usd"]) for r in ot]
    slides["build_src"] = rel(run / "cost" / "one_time_cost.csv")
    summ = R.text(run / "cost" / "summary.txt")
    slides["draft_banner"] = summ.splitlines()[0]
    slides["summary_src"] = rel(run / "cost" / "summary.txt")
    lab = {}
    for m in re.finditer(r"\n  ([CD])\s+(as evaluated|at 2,000)\s+([\d,]+)\s+([\d.]+)\s+([\d,]+/[\d,]+/[\d,]+)", summ):
        lab[(m.group(1), m.group(2))] = (m.group(3), m.group(4), m.group(5))
    slides["label_cost"] = lab
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
    slides["effort_log_sha256"] = m.group(1) if m else "missing (expected source: freeze_report.txt stage 12 note)"
    slides["effort_log_src"] = rel(freeze / "freeze_report.txt")
    slides["labels_eval"] = labels_eval
    return {"T": T, "full": full, "decoys": decoys, "decoy_check": decoy_check, "slides": slides, "bpr": bpr, "labels_eval": labels_eval}


# ---------------------------------------------------------------------------------------------- output

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def write_outputs(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    T, full, S = res["T"], res["full"], res["slides"]
    # csv: one row per option, one column per figure
    with (out / "head_to_head.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["option"] + T.columns)
        for o in OPTS:
            w.writerow([o] + [T.get(o, c) for c in T.columns])
        w.writerow([])
        w.writerow(["# full band, not comparable with B's sample figures"])
        w.writerow(["option"] + full.columns)
        for o in full_rows(full):
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
         "## Bases", "",
         "- **Table 1 (sample basis):** all columns of the main table are on the fixed 5,000-pair sample of the test band, weighted estimates "
         "(B exists only on this basis). Cost, latency and break-even cells are model or measurement figures, not sample estimates.",
         "- **Table 2 (full band):** exact figures on every test band pair for A, C, D and D (orig). **Full band, not comparable with B's sample figures.**",
         f"- Labels as evaluated for C and D in this run: {int(res['labels_eval']):,} (cost/assumptions.csv). Hourly rate $100 and scenario S1 unless a column says otherwise.",
         "- 'LLM $' columns are the LLM cost only (LLM share x B's measured cost per call); the total cost is in the monthly table.", ""]
    groups = [("Quality, sample basis", ["band F1 [95% CI]", "F1 vs B [95% CI]", "F1 verdict", "band precision", "band recall", "overall precision", "overall recall", "ECE", "meets every bar"]),
              ("Cost per pair and per record", ["LLM $ per 1,000 band pairs", "LLM $ per record", "band pairs per record"]),
              ("Monthly cost (S1, $100/h, labels as evaluated)", [c for c in T.columns if c.startswith("monthly")]),
              ("Latency", ["latency per pair (p50, p95)", "D fallback tail"]),
              ("Break-even against B (S1 and S2 total)", [c for c in T.columns if c.startswith("break-even")])]
    n = 1
    for title, cols in groups:
        L += [f"### Table 1.{n}: {title}", "", md_table(["option"] + cols, [[o] + [T.get(o, c) for c in cols] for o in OPTS]), ""]
        if n == 1:
            L += ["`F1 vs B` and its verdict, and `meets every bar`, are read from `bars_table.csv`: the protocol rules applied there use the paired sample comparison for the F1 bar "
                  "and the full band for A, C and D on the other bars, so `meets every bar` is not a sample-basis figure.", ""]
        n += 1
    L += ["## Table 2: full band, not comparable with B's sample figures", "",
          md_table(["option"] + full.columns, [[o] + [full.get(o, c) for c in full.columns] for o in full_rows(full)]), ""]

    L += ["## Decoy outcomes (same name, different entity; test group)", "",
          md_table(["option", "basis", "decoy pairs", "wrongly matched", "source"], [list(d) for d in res["decoys"]]), ""]
    dc = res["decoy_check"]
    L += [f"B's decoy check: {dc['pairs']} test band decoy pairs, {dc['false_matches']} false matches; one-sided 95% bound on the false-match rate <= {dc['bound']}; "
          f"{dc['live_calls']} live calls and {dc['cache_hits']} cache hits, ${dc['spent_usd']:.4f} (sources: {', '.join(dc['sources'])}).", ""]

    L += ["## Cost numbers for the slides", "",
          f"- Effort log used: sha256 `{S['effort_log_sha256']}` (source: {S['effort_log_src']}).",
          f"- Banner of the cost outputs (first line of {S['summary_src']}): **{S['draft_banner']}** The DRAFT banner is on.", "",
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

    # sources table (grouped)
    L += ["## Sources", "", "Per column, the file each option's cell was read from (the full cell-by-cell list is `head_to_head_sources.csv`). "
          "'derived' cells are plain arithmetic on cells read from files.", ""]
    rows = []
    for tname, tab in (("Table 1", T), ("Table 2", full)):
        for col in tab.columns:
            by_src: dict[str, list[str]] = {}
            for o in OPTS:
                if (o, col) in tab.cells:
                    text, src, how = tab.cells[(o, col)]
                    by_src.setdefault(src + ("" if how == "read" else f" [{how}]"), []).append(o)
            rows.append([tname, col, "; ".join(f"{', '.join(os_)}: {s or '(constant text)'}" for s, os_ in by_src.items())])
    L += [md_table(["table", "column", "source file per option"], rows), ""]
    (out / "head_to_head.md").write_text("\n".join(L), encoding="utf-8")


def full_rows(full: Table) -> list[str]:
    return [o for o in OPTS if (o, "band F1 [95% CI]") in full.cells]


def main() -> int:
    ap = argparse.ArgumentParser(description="Head-to-head table from the frozen outputs of the freeze run.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "eval_7919")
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()
    freeze = args.freeze or next(iter(sorted((REPO / "results").glob("freeze_????-??-??"))), None)
    if freeze is None:
        raise SystemExit("no results/freeze_<date> folder")
    R = Reader([run, Path(freeze)])
    res = build(run, Path(freeze).resolve(), R)
    out = args.out or run / "head_to_head"
    write_outputs(res, out)
    print(f"wrote {out / 'head_to_head.csv'}, head_to_head.md and head_to_head_sources.csv; read {len(set(R.read))} files, all under {[str(r.relative_to(REPO)) for r in R.roots]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

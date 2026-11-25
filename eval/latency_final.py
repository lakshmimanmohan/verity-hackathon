"""
Latency figures of the freeze run: what was saved, the per-record view of D's routed pairs, and B's per-record time. Read-only: it reads
only files under results/eval_7919 and results/freeze_<date>, plus one dev file (the concurrency measurement at 10 workers, labeled dev),
and writes only latency_final.md into --out-dir (default results/eval_7919/latency_final/). It makes no call and opens no data or truth file.

  python eval/latency_final.py [--run results/eval_7919] [--freeze results/freeze_2026-11-25]
                               [--dev-concurrency results/dev_2026_run11/options/b_llm/openai/concurrency/concurrency.json]
                               [--out-dir results/eval_7919/latency_final]

1. Inventory: which per-pair latency arrays the freeze run saved for C and which B call latencies exist.
2. D latency with the LLM fallback, simulated from the run's own measured distributions: it needs C's per-pair latency array. When the freeze
   run did not save one, this part is reported as not run; dev data is never substituted.
3. Per record: records of the test group, mean routed pairs per record, the share of records with at least one routed pair, and the share
   expected if the routed pairs were independent; B's time per record from the run's call latency and the dev concurrency measurement.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKERS = 10                                   # B per record is timed at 10 workers


class Reader:
    """Reads only under the allowed roots, plus explicitly listed extra files; remembers every file it read."""

    def __init__(self, roots: list[Path], extra_files: list[Path] = ()):
        self.roots = [Path(r).resolve() for r in roots]
        self.extra = {Path(p).resolve() for p in extra_files}
        self.read: list[Path] = []

    def check(self, p: Path) -> Path:
        q = Path(p).resolve()
        ok = q in self.extra or any(q == r or r in q.parents for r in self.roots)
        if not ok:
            raise SystemExit(f"refusing to read {p}: outside the allowed folders and files")
        if q.name.startswith("orgs_raw") or "truth" in q.name or q.name.startswith("hand_check") or "hand_check" in q.parts:
            raise SystemExit(f"refusing to read {q.name}: data, truth and hand-check files are never read here")
        return q

    def lines(self, p: Path) -> list[str]:
        q = self.check(p)
        self.read.append(q)
        if q.suffix == ".gz":
            with gzip.open(q, "rt", encoding="utf-8") as f:
                return f.read().splitlines()
        return q.read_text(encoding="utf-8").splitlines()

    def json(self, p: Path):
        return json.loads("\n".join(self.lines(p)))

    def csv(self, p: Path) -> list[dict]:
        return list(csv.DictReader([ln for ln in self.lines(p) if not ln.startswith("#")]))

    def exists(self, p: Path) -> bool:
        return Path(p).exists()


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default): q in [0, 100]."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def call_latencies(R: Reader, path: Path) -> list[float]:
    out = []
    for ln in R.lines(path):
        if ln.strip():
            rec = json.loads(ln)
            if rec.get("latency_s") is not None and not rec.get("error"):
                out.append(float(rec["latency_s"]))
    return out


def per_record(routed: list[tuple[str, str]], test_records: set[str], n_band_pairs: int, band_pairs_per_record: float) -> dict:
    """Per-record view of the routed pairs of the test group (no truth)."""
    deg = Counter()
    for a, b in routed:
        deg[a] += 1
        deg[b] += 1
    n_rec = len(test_records)
    outside = [r for r in deg if r not in test_records]
    n_routed = len(routed)
    share_pair = n_routed / n_band_pairs
    with_any = sum(1 for r in deg if r in test_records)
    degree_test = 2 * n_band_pairs / n_rec                     # test band pairs a test record takes part in, on average
    return {"records": n_rec, "routed_pairs": n_routed, "band_pairs": n_band_pairs, "pair_share": share_pair,
            "endpoints_per_record": 2 * n_routed / n_rec, "pairs_once_per_record": n_routed / n_rec,
            "records_with_any": with_any, "share_with_any": with_any / n_rec, "max_per_record": max(deg.values()) if deg else 0,
            "histogram": dict(sorted(Counter(deg.values()).items())), "records_outside_test": len(outside),
            "expected_share_user": 1 - (1 - share_pair) ** band_pairs_per_record, "exponent_user": band_pairs_per_record,
            "expected_share_degree": 1 - (1 - share_pair) ** degree_test, "exponent_degree": degree_test}


def b_per_record(calls: float, latency_s: float, workers: int = WORKERS) -> tuple[float, float]:
    """(mean time, time counting whole rounds of `workers` calls) for `calls` calls of `latency_s` each; queueing not included."""
    return calls * latency_s / workers, math.ceil(calls / workers) * latency_s


def build(run: Path, freeze: Path, dev_concurrency: Path, R: Reader) -> dict:
    rel = lambda p: Path(p).resolve().relative_to(REPO).as_posix() if REPO in Path(p).resolve().parents else Path(p).resolve().as_posix()
    O = run / "options"
    res: dict = {"sources": {}}
    # 1. inventory
    lat_dir = O / "latency"
    arrays = sorted(p for p in lat_dir.glob("*") if p.suffix in (".npz", ".npy")) if lat_dir.exists() else []
    other_arrays = [p for p in list((O / "c_model").rglob("*")) + list(run.glob("*.npz")) if p.suffix in (".npz", ".npy") or "per_pair" in p.name]
    summary_path = lat_dir / "latency_summary.json"
    summ = R.json(summary_path)
    res["c_arrays"] = [rel(p) for p in arrays + other_arrays]
    res["c_summary"] = {"source": rel(summary_path), "pairs": summ["pairs"], "repeats": summ["C"]["repeats"], "median": summ["C"]["median_of_repeats"]}
    calls = {}
    for label in ("sample5000", "dfull", "decoycheck"):
        p = O / "b_llm" / "openai" / label / "calls.jsonl"
        if R.exists(p):
            xs = call_latencies(R, p)
            calls[label] = {"source": rel(p), "n": len(xs), "p50": percentile(xs, 50), "p95": percentile(xs, 95), "max": max(xs)}
        else:
            calls[label] = {"source": rel(p), "n": 0, "missing": True}
    arch = freeze / "b_archive" / "calls.jsonl.gz"
    if R.exists(arch):
        xs = call_latencies(R, arch)
        calls["sample5000 archive"] = {"source": rel(arch), "n": len(xs), "p50": percentile(xs, 50), "p95": percentile(xs, 95), "max": max(xs)}
    res["b_calls"] = calls
    # 2. D with the fallback: needs C's per-pair array
    res["d_sim_possible"] = bool(res["c_arrays"])
    # 3. per record
    sides = {r["record_id"]: r["side"] for r in R.csv(run / "split_records.csv")}
    test_records = {r for r, s in sides.items() if s == "test"}
    routed = [(r["record_a"].strip(), r["record_b"].strip()) for r in R.csv(O / "d_hybrid" / "routed_fullband_pairs.csv")]
    n_band = len(R.csv(O / "c_model" / "c_v1_lgbm" / "decisions.csv"))
    band = R.json(run / "band_summary.json")
    bppr = band["band_pairs"] / band["records"]
    res["per_record"] = per_record(routed, test_records, n_band, bppr)
    res["per_record"]["band_pairs_per_record_source"] = rel(run / "band_summary.json")
    res["per_record"]["sources"] = [rel(run / "split_records.csv"), rel(O / "d_hybrid" / "routed_fullband_pairs.csv"), rel(O / "c_model" / "c_v1_lgbm" / "decisions.csv")]
    # B per record
    timings = R.json(O / "b_llm" / "openai" / "sample5000" / "timings.json")
    t = [x for x in timings["timings"] if x["stage"].endswith("llm_call")][0]["per_item_ms"]
    conc = R.json(dev_concurrency)
    lvl = [x for x in conc["levels"] if x["workers"] == WORKERS]
    res["b_per_record"] = {"calls": bppr, "freeze_p50": t["p50"] / 1000, "freeze_p95": t["p95"] / 1000, "freeze_source": rel(O / "b_llm" / "openai" / "sample5000" / "timings.json"),
                           "dev_level": lvl[0] if lvl else None, "dev_source": rel(dev_concurrency)}
    return res


def render(res: dict) -> str:
    L = ["# Latency, final figures from the freeze run (seed 7919)", "",
         "Read from files under `results/eval_7919` and `results/freeze_2026-11-25`; the one dev input is labeled dev. Nothing was rerun, no call was made, no truth or data file was opened.", "",
         "## 1. What the freeze run saved", "", "### C per-pair latency arrays", ""]
    cs = res["c_summary"]
    if res["c_arrays"]:
        L += ["Per-pair array files found: " + ", ".join(f"`{p}`" for p in res["c_arrays"]), ""]
    else:
        L += ["**No per-pair array was saved** for C in the freeze run: `options/latency/` holds only `latency_summary.json` and `latency_summary.txt`, and no `.npz`, `.npy` or per-pair file "
              "exists under the C model folder or the run folder. The stage 6 and stage 8 latency bench keeps summary statistics only.", ""]
    L += [f"What exists for C (source: `{cs['source']}`, {cs['pairs']:,} test band pairs, one core, features included, one pair at a time):", "",
          md_table(["repeat", "p50 ms", "p95 ms", "max ms", "mean ms", "pairs", "total s"],
                   [[i + 1, r["p50_ms"], r["p95_ms"], r["max_ms"], r["mean_ms"], r["pairs"], r["total_s"]] for i, r in enumerate(cs["repeats"])]
                   + [["median of repeats", cs["median"]["p50_ms"], cs["median"]["p95_ms"], cs["median"]["max_ms"], cs["median"]["mean_ms"], "", ""]]), "",
          "### B call latencies", "",
          md_table(["calls file", "calls", "p50 s", "p95 s", "max s", "source"],
                   [[k, v["n"], "missing" if v.get("missing") else f"{v['p50']:.3f}", "missing" if v.get("missing") else f"{v['p95']:.3f}",
                     "missing" if v.get("missing") else f"{v['max']:.3f}", v["source"]] for k, v in res["b_calls"].items()]), "",
          "Percentiles are computed here from the `latency_s` of each live call (linear interpolation). `sample5000/calls.jsonl` is a local file (not committed); the committed copy of the same calls "
          "is the freeze archive `calls.jsonl.gz`.", "",
          "## 2. D latency with the LLM fallback (simulation)", ""]
    if res["d_sim_possible"]:
        L += ["A C per-pair array exists, but this script does not simulate from it; see the code history. Not run."]
    else:
        L += ["**Not run.** The simulation draws a C latency per pair, and the freeze run did not save C's per-pair latency array (section 1). Dev data was not substituted: "
              "no dev array, no dev simulation values and no summary statistics were used in place of the missing array.", "",
              "Consequences:", "",
              "- There is no freeze-run value for D's p99, p99.5, p99.9, p99.95 or max with the fallback.",
              "- The \"D fallback tail\" row of `head_to_head_v2` is **not superseded**: it still shows the dev simulation recorded in `cost_final/assumptions.csv` "
              "(p99.9 0.5671 ms, p99.95 921.2648 ms, from a routed share of 0.085%), and it is a dev figure, not a freeze-run figure.",
              "- What would be needed: a rerun of the latency bench that also saves C's per-pair latencies (stage 8 of the runbook writes summary statistics only). That rerun was not made.", ""]
    pr = res["per_record"]
    L += ["## 3. Per record", "", "### D's routed pairs, per record of the test group (no truth)", "",
          md_table(["quantity", "value", "how"],
                   [["test group records", f"{pr['records']:,}", "`split_records.csv`, side = test"],
                    ["routed pairs (D, rule v2)", f"{pr['routed_pairs']:,} of {pr['band_pairs']:,} test band pairs ({100 * pr['pair_share']:.3f}%)", "`routed_fullband_pairs.csv`, rows of C's `decisions.csv`"],
                    ["mean routed pairs per record (each pair counted for both of its records)", f"{pr['endpoints_per_record']:.4f}", "2 x routed pairs / records"],
                    ["mean routed pairs per record (each pair counted once)", f"{pr['pairs_once_per_record']:.4f}", "routed pairs / records"],
                    ["records with at least one routed pair", f"{pr['records_with_any']:,} of {pr['records']:,} = {100 * pr['share_with_any']:.2f}%", "distinct records in the routed list"],
                    ["most routed pairs on one record", pr["max_per_record"], "routed list"],
                    ["records by number of routed pairs (n pairs: records)", ", ".join(f"{k}: {v}" for k, v in pr["histogram"].items()), "routed list"],
                    ["routed records outside the test group", pr["records_outside_test"], "must be 0"],
                    [f"share expected if routed pairs were independent, exponent {pr['exponent_user']:.2f} (band pairs per record, as specified)",
                     f"{100 * pr['expected_share_user']:.2f}%", f"1 - (1 - {pr['routed_pairs']}/{pr['band_pairs']:,})^{pr['exponent_user']:.2f}; band pairs per record from `{pr['band_pairs_per_record_source']}`"],
                    [f"same, exponent {pr['exponent_degree']:.2f} (test band pairs a test record takes part in: 2 x {pr['band_pairs']:,} / {pr['records']:,})",
                     f"{100 * pr['expected_share_degree']:.2f}%", "the per-record degree inside the test group"]]), "",
          "Sources: " + ", ".join(f"`{s}`" for s in pr["sources"]) + ".", "",
          "### B per record", "", ]
    b = res["b_per_record"]
    rows = []
    for label, lat in (("freeze run, call p50 (2 workers)", b["freeze_p50"]), ("freeze run, call p95 (2 workers)", b["freeze_p95"])):
        mean_s, rounds_s = b_per_record(b["calls"], lat)
        rows.append([label, f"{lat:.3f}", f"{mean_s:.2f}", f"{rounds_s:.2f}"])
    lv = b["dev_level"]
    if lv:
        for label, lat in (("DEV: call p50 at 10 workers", lv["latency_p50_s"]), ("DEV: call p95 at 10 workers", lv["latency_p95_s"])):
            mean_s, rounds_s = b_per_record(b["calls"], lat)
            rows.append([label, f"{lat:.3f}", f"{mean_s:.2f}", f"{rounds_s:.2f}"])
    else:
        rows.append(["DEV: call latency at 10 workers", "missing", "missing", "missing"])
    L += [f"B makes one call per band pair: {b['calls']:.2f} calls per record (`{pr['band_pairs_per_record_source']}`). At {WORKERS} workers, the time per record is calls x latency / {WORKERS} "
          f"(mean) or ceil(calls / {WORKERS}) x latency (whole rounds); queueing is not included.", "",
          md_table(["call latency used", "latency s", f"mean time per record at {WORKERS} workers, s", "whole rounds, s"], rows), "",
          f"Sources: freeze run call latency from `{b['freeze_source']}` (the run used 2 workers); DEV concurrency measurement from `{b['dev_source']}` (dev run, labeled dev: it was not measured in the freeze run).", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Latency figures of the freeze run.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "eval_7919")
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--dev-concurrency", type=Path, default=REPO / "results" / "dev_2026_run11" / "options" / "b_llm" / "openai" / "concurrency" / "concurrency.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()
    freeze = args.freeze or next(iter(sorted((REPO / "results").glob("freeze_????-??-??"))), None)
    if freeze is None:
        raise SystemExit("no results/freeze_<date> folder")
    R = Reader([run, Path(freeze)], [args.dev_concurrency])
    res = build(run, Path(freeze).resolve(), args.dev_concurrency, R)
    out = args.out_dir or run / "latency_final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency_final.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'latency_final.md'}; read {len(set(R.read))} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

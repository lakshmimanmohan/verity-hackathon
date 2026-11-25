"""
Three demo pairs from the freeze run, picked by a fixed rule on frozen outputs (test-group pairs only, sorted by (record_a, record_b) with the
two ids in sorted order inside each pair). Selection uses no truth; the truth file is read afterwards only to mark each pair correct or not.

  python eval/demo_pairs.py [--run results/eval_7919] [--out-dir results/eval_7919/demo_pairs]

Rule:
  1. the first pair whose decision_status in verity_data/scored_pairs.csv is "match" (Verity decides, outside the band);
  2. the first band pair that C decides "same" and that D does not route to the LLM (route c_same in options/d_hybrid/decisions_fullband.csv);
  3. the first pair D routes to the LLM (route llm).
For each pair: both records' fields (name, address, city, state, ZIP, phone, website), the route, Verity's score and status, C's calibrated p and
decision, for pair 3 B's p_same and decision, the final answer, correct or not, the time and the cost. Times: pair 1 = Verity's per-pair
scoring p50 from the run's run_info timings; pair 2 = C's median-of-repeats p50 over all test band pairs; pair 3 = the pair's live call
latency from the run's calls.jsonl files ("missing" if no live call for it is in them). Cost: $0 for pairs 1 and 2; pair 3 = that call's cost.
Writes demo_pairs.md and demo_pairs.csv. No other pair is printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIELDS = (("name", "company_name"), ("address", "address_line"), ("city", "city"), ("state", "state"), ("ZIP", "postal_code"), ("phone", "phone"), ("website", "website"))


def key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip(), b.strip())))


def read_csv_rows(path: Path):
    csv.field_size_limit(10**9)
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


# ---------------------------------------------------------------------------------------------- selection (no truth)

def select_pairs(scored_rows, d_rows, test_records: set[str]) -> dict:
    """The three pairs by the rule. scored_rows: dicts with record_a, record_b, decision_status (streamed); d_rows: D's full-band decisions
    (record_a, record_b, decision, route). Only pairs with both records in the test group are considered. No truth is involved."""
    in_test = lambda k: k[0] in test_records and k[1] in test_records
    first_match = None
    for r in scored_rows:
        if r["decision_status"] == "match":
            k = key(r["record_a"], r["record_b"])
            if in_test(k) and (first_match is None or k < first_match):
                first_match = k
    first_c_same = first_llm = None
    for r in d_rows:
        k = key(r["record_a"], r["record_b"])
        if not in_test(k):
            continue
        if r["route"] == "c_same" and r["decision"] == "same" and (first_c_same is None or k < first_c_same):
            first_c_same = k
        if r["route"] == "llm" and (first_llm is None or k < first_llm):
            first_llm = k
    return {1: first_match, 2: first_c_same, 3: first_llm}


def find_live_call(call_files: list[Path], k: tuple[str, str]):
    """The live call for a pair (either id order) in the calls.jsonl files: (latency_s, usd, file) or None."""
    for p in call_files:
        if not Path(p).exists():
            continue
        with Path(p).open(encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                c = json.loads(ln)
                if key(c["a"], c["b"]) == k and c.get("mode") == "live" and not c.get("error"):
                    return float(c["latency_s"]), float(c["usd"]), Path(p).as_posix()
    return None


# ---------------------------------------------------------------------------------------------- the build

def build(run: Path, truth_path: Path | None = None) -> dict:
    rel = lambda p: Path(p).resolve().relative_to(REPO).as_posix() if REPO in Path(p).resolve().parents else Path(p).resolve().as_posix()
    O = run / "options"
    sides = {r["record_id"]: r["side"] for r in read_csv_rows(run / "split_records.csv")}
    test = {r for r, s in sides.items() if s == "test"}
    d_rows = list(read_csv_rows(O / "d_hybrid" / "decisions_fullband.csv"))
    picked = select_pairs(read_csv_rows(run / "verity_data" / "scored_pairs.csv"), d_rows, test)
    wanted = {k for k in picked.values() if k}
    # Verity's row for each pair
    scored = {}
    for r in read_csv_rows(run / "verity_data" / "scored_pairs.csv"):
        k = key(r["record_a"], r["record_b"])
        if k in wanted:
            scored[k] = r
    d_by = {key(r["record_a"], r["record_b"]): r for r in d_rows if key(r["record_a"], r["record_b"]) in wanted}
    c_by = {key(r["record_a"], r["record_b"]): r for r in read_csv_rows(O / "c_model" / "c_v1_lgbm" / "decisions.csv") if key(r["record_a"], r["record_b"]) in wanted}
    b_by = {}
    for lab in ("dfull", "sample5000", "decoycheck"):
        p = O / "b_llm" / "openai" / lab / "decisions.csv"
        if p.exists():
            for r in read_csv_rows(p):
                k = key(r["record_a"], r["record_b"])
                if k in wanted:
                    b_by.setdefault(k, r)
    cls = json.loads((run / "verity_data" / "column_classification.json").read_text(encoding="utf-8"))
    recs = {}
    ids = {x for k in wanted for x in k}
    for r in read_csv_rows(run / "orgs_raw.csv"):
        if r[cls["record_id_column"]] in ids:
            recs[r[cls["record_id_column"]]] = r
    ri = json.loads((run / "verity_data" / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    score_t = [t for t in ri["timings"] if t["stage"] == "verity.score"][0]
    lat = json.loads((O / "latency" / "latency_summary.json").read_text(encoding="utf-8"))
    c_p50 = lat["C"]["median_of_repeats"]["p50_ms"]
    call_files = [O / "b_llm" / "openai" / lab / "calls.jsonl" for lab in ("sample5000", "dfull", "decoycheck")]
    out = {"pairs": [], "sources": [rel(run / "verity_data" / "scored_pairs.csv"), rel(O / "d_hybrid" / "decisions_fullband.csv"), rel(run / "split_records.csv")]}
    for n in (1, 2, 3):
        k = picked[n]
        if k is None:
            out["pairs"].append({"n": n, "pair": None})
            continue
        a, b = k
        sr, dr, cr, br = scored.get(k), d_by.get(k), c_by.get(k), b_by.get(k)
        item = {"n": n, "pair": k, "records": {"A": recs[a], "B": recs[b]}, "verity_score": sr["score"], "verity_status": sr["decision_status"], "match_type": sr["match_type"]}
        if n == 1:
            item.update({"route": "Verity decides (outside the band)", "c_p": None, "c_decision": None, "b_p": None, "b_decision": None, "final": "same",
                         "time": f"{score_t['per_item_ms']['p50']} ms (Verity's per-pair scoring p50, median over all {score_t['items']:,} scored pairs; stage verity.score of run_info.json)",
                         "cost": "$0", "time_source": rel(run / "verity_data" / "run_info.json")})
        elif n == 2:
            item.update({"route": f"{dr['route']} (band pair, C decides without the LLM)", "c_p": cr["confidence"], "c_decision": cr["decision"], "b_p": None, "b_decision": None,
                         "final": dr["decision"], "time": f"{c_p50} ms (C's median-of-repeats p50, median over all test band pairs; one core, features included)", "cost": "$0",
                         "time_source": rel(O / "latency" / "latency_summary.json")})
        else:
            call = find_live_call(call_files, k)
            item.update({"route": f"{dr['route']} (band pair routed to the LLM)", "c_p": cr["confidence"], "c_decision": cr["decision"],
                         "b_p": None if br is None else br["confidence"], "b_decision": None if br is None else br["decision"], "final": dr["decision"],
                         "time": "missing" if call is None else f"{call[0]:.3f} s (this pair's live call)",
                         "cost": "missing" if call is None else f"${call[1]:.6f} (this pair's live call)",
                         "time_source": "missing" if call is None else rel(call[2])})
        out["pairs"].append(item)
    # truth: read only now, only to mark each of the (up to) three pairs correct or not
    truth = {r["record_id"]: r["true_entity_id"] for r in read_csv_rows(truth_path or run / "orgs_raw_truth.csv") if r["record_id"] in ids}
    for item in out["pairs"]:
        if item.get("pair"):
            a, b = item["pair"]
            same = truth[a] == truth[b]
            item["correct"] = (item["final"] == "same") == same
    out["n_correct"] = sum(1 for i in out["pairs"] if i.get("correct"))
    return out


def md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def render(res: dict) -> str:
    L = ["# Three demo pairs from the freeze run (seed 7919)", "",
         "Rule (applied to the frozen outputs, test-group pairs only, sorted by (record_a, record_b), ids in sorted order inside each pair; selection uses no truth, the truth file is "
         "read afterwards only to mark each pair correct or not):", "",
         "1. the first pair whose `decision_status` in `scored_pairs.csv` is `match` (Verity decides, outside the band);",
         "2. the first band pair that C decides `same` and D does not route to the LLM (route `c_same` in D's decisions);",
         "3. the first pair D routes to the LLM (route `llm`).", ""]
    for it in res["pairs"]:
        L += [f"## Pair {it['n']}", ""]
        if not it.get("pair"):
            L += ["No pair satisfies the rule.", ""]
            continue
        a, b = it["pair"]
        L += [md_table(["field", f"record A ({a})", f"record B ({b})"],
                       [[label, it["records"]["A"].get(col, ""), it["records"]["B"].get(col, "")] for label, col in FIELDS]), "",
              md_table(["item", "value"],
                       [["route", it["route"]],
                        ["Verity's score and status", f"{it['verity_score']} ({it['verity_status']}, match type {it['match_type']})"],
                        ["C's calibrated p and decision", "not applicable (the pair is outside the band; Verity decides)" if it["c_p"] is None else f"{it['c_p']} ({it['c_decision']})"],
                        ["B's p_same and decision", "not called for this pair" if it["b_p"] is None and it["n"] != 3 else ("missing" if it["b_p"] is None else f"{it['b_p']} ({it['b_decision']})")],
                        ["final answer", it["final"]],
                        ["correct", "yes" if it["correct"] else "no"],
                        ["time", it["time"]],
                        ["cost", it["cost"]]]), ""]
    L += [f"N_CORRECT = {res['n_correct']} of {len(res['pairs'])}", "",
          "Sources: " + ", ".join(f"`{s}`" for s in res["sources"]) + ", `orgs_raw.csv` (record fields), `options/c_model/c_v1_lgbm/decisions.csv`, B's `decisions.csv` files, "
          "`options/b_llm/openai/*/calls.jsonl`, `verity_data/run_info.json`, `options/latency/latency_summary.json`; `orgs_raw_truth.csv` only for the correct / not correct mark.", ""]
    return "\n".join(L)


def write_csv(res: dict, path: Path) -> None:
    cols = ["pair_no", "record_a", "record_b"] + [f"{lab}_{side}" for lab, _ in FIELDS for side in ("A", "B")] + ["route", "verity_score", "verity_status", "c_p", "c_decision",
                                                                                                                 "b_p_same", "b_decision", "final_answer", "correct", "time", "cost"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for it in res["pairs"]:
            if not it.get("pair"):
                w.writerow([it["n"], "no pair satisfies the rule"])
                continue
            row = [it["n"], it["pair"][0], it["pair"][1]]
            for lab, col in FIELDS:
                row += [it["records"]["A"].get(col, ""), it["records"]["B"].get(col, "")]
            row += [it["route"], it["verity_score"], it["verity_status"], it["c_p"] or "", it["c_decision"] or "", it["b_p"] or "", it["b_decision"] or "", it["final"],
                    "yes" if it["correct"] else "no", it["time"], it["cost"]]
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Three demo pairs from the freeze run.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "eval_7919")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()
    out = args.out_dir or run / "demo_pairs"
    res = build(run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_pairs.md").write_text(render(res), encoding="utf-8")
    write_csv(res, out / "demo_pairs.csv")
    print(f"wrote {out / 'demo_pairs.md'} and demo_pairs.csv; N_CORRECT = {res['n_correct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

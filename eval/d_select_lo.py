"""
Apply the D routing rule v2 (results/dev_2026_run11/options/d_hybrid/selection_rule_v2.txt): hi = 0.9; lo is the setting of
{0.005, 0.01, 0.05, 0.1, 0.5} with the highest pooled band F1 over development seeds 11, 12, 13, 42, 2026, among settings whose pooled
LLM share of the test band is at most 1%; ties go to the higher lo. Exact full-band decisions: pairs routed to the LLM take B's answer,
all others take C's own decision. The rule file is read-only here; frozen_settings_v2.json is written by 'report'.

  python eval/d_select_lo.py list            # per seed and lo: routed pairs lacking a cached B answer; stop if the union exceeds --max-calls
  python eval/d_select_lo.py run-b --max-usd-total 1.0 --approve    # B on each seed's lo 0.005 pair list (it holds every smaller list)
  python eval/d_select_lo.py report          # the rule applied: table per lo, chosen lo, frozen_settings_v2.json, routed share and cost per seed

list and run-b never read truth. report reads the truth sidecar to score. Writes results/dstudy_summary/d_select_lo.{txt,csv}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d_lower_lo as L  # noqa: E402

D = L.D
REPO = L.REPO
RULE = REPO / "results" / "dev_2026_run11" / "options" / "d_hybrid" / "selection_rule_v2.txt"
SETTINGS_V2 = RULE.with_name("frozen_settings_v2.json")
LOS = (0.005, 0.01, 0.05, 0.1, 0.5)
HI = 0.9
SHARE_MAX = 0.01
LABEL = "lo0005"                                  # B run label of the union pair list (lo 0.005, hi 0.9)
COST_PER_CALL = 0.001527                          # measured on the dev run (spent / calls, results/dev_2026_run11 sample5000)


def pairs_file(run: Path, lo: float) -> Path:
    return run / "options" / "d_lo" / f"routed_lo{lo:g}_hi{HI:g}.csv"


def cmd_list(args) -> int:
    runs = L.run_dirs(args.runs)
    print(f"{'run':<16}{'test band':>10}" + "".join(f"{'lo ' + format(lo, 'g'):>16}" for lo in LOS) + f"{'new calls (union)':>20}")
    print(f"{'':<26}" + "".join(f"{'routed / uncached':>16}" for _ in LOS))
    union_total = 0
    for run in runs:
        rows = L.c_rows(run)
        L.install_frozen(run)
        cells = []
        for lo in LOS:
            pr = L.routed(rows, lo, HI)
            L.write_pairs(pairs_file(run, lo), pr)
            cells.append(f"{len(pr)} / {L.need_new_call(run, pr)}")
        n_union = L.need_new_call(run, L.routed(rows, LOS[0], HI))
        union_total += n_union
        print(f"{run.name:<16}{len(rows):>10,}" + "".join(f"{c:>16}" for c in cells) + f"{n_union:>20}")
    print(f"{'union total':<26}{'':>{16 * len(LOS)}}{union_total:>20}   cost estimate ${union_total * L.PILOT_COST:.4f} at the pilot ${L.PILOT_COST}/call "
          f"(gate x{L.GATE_MARGIN}: ${union_total * L.PILOT_COST * L.GATE_MARGIN:.4f}); at the measured ${COST_PER_CALL}: ${union_total * COST_PER_CALL:.4f}")
    if union_total > args.max_calls:
        print(f"STOP: {union_total} new calls exceed {args.max_calls}")
        return 2
    return 0


def cmd_run_b(args) -> int:
    runs = L.run_dirs(args.runs)
    todo = []
    for run in runs:
        f = pairs_file(run, LOS[0])
        if not f.exists():
            raise SystemExit(f"{f} missing: run 'list' first")
        pr = [(r["record_a"], r["record_b"]) for r in D.read_csv(f)]
        todo.append((run, f, L.need_new_call(run, pr)))
    est = sum(n for *_, n in todo) * L.PILOT_COST * L.GATE_MARGIN
    if est > args.max_usd_total:
        raise SystemExit(f"estimated ${est:.2f} (with the gate margin) exceeds the total cap ${args.max_usd_total}")
    import os
    import re
    import subprocess
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
    spent_cap = 0.0
    for run, f, n_new in todo:
        cap = max(0.05, round(n_new * L.PILOT_COST * L.GATE_MARGIN + 0.02, 3))
        spent_cap += cap
        cmd = [sys.executable, str(REPO / "options" / "b_llm_baseline.py"), str(run), "--provider", "openai", "--final-sample", str(f),
               "--label", LABEL, "--max-usd", str(cap), "--approve-final-run" if args.approve else "--estimate-only"]
        print(f"\n== {run.name}: {n_new} new calls, run cap ${cap}")
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for ln in r.stdout.splitlines():
            if re.match(r"(pairs:|cost gate|calls |decisions:|refusing)", ln):
                print("   " + ln[:230])
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise SystemExit(f"b_llm_baseline exited {r.returncode} for {run.name}")
    print(f"\nper-run caps sum to ${spent_cap:.2f}; total cap ${args.max_usd_total}")
    return 0


def literal_decisions(run: Path, lo: float, b: dict, c: dict) -> tuple[dict, int]:
    """The rule's decisions: routed to the LLM (lo <= p < hi) take B's answer, all others take C's own decision."""
    out, n_llm = {}, 0
    for k, (cd, p) in c.items():
        if D.route_of(p, lo, HI) == "llm":
            if k not in b:
                raise SystemExit(f"{run.name}: no B answer for routed pair {k} at lo {lo:g}")
            out[k] = b[k][0]
            n_llm += 1
        else:
            out[k] = cd
    return out, n_llm


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cmd_report(args) -> int:
    runs = L.run_dirs(args.runs)
    data = {}
    for run in runs:
        c = D.load_c(D.paths(run)["c"])
        b = D.merge_b(D.b_paths(run) + [run / "options" / "b_llm" / "openai" / n / "decisions.csv" for n in ("lo01", LABEL)])
        data[run.name] = {"c": c, "b": b, "ent": L.truth_entities(run), "run": run}
    band_total = sum(len(d["c"]) for d in data.values())
    L_ = ["D routing rule v2 applied (selection_rule_v2.txt): hi 0.9; lo from {0.005, 0.01, 0.05, 0.1, 0.5}; highest pooled band F1 among settings",
          "with pooled LLM share of the test band at most 1%; ties to the higher lo. Routed pairs take B's answer, all others take C's own decision.",
          f"Pooled over {len(runs)} development seeds ({', '.join(r.name for r in runs)}); {band_total:,} test band pairs. Pooled F1 is from the pooled TP, FP, FN.", ""]
    L_.append(f"  {'lo':>6}{'pooled F1':>10}{'prec':>8}{'recall':>8}{'FP':>5}{'FN':>5}{'routed':>8}{'LLM share':>11}  eligible   per-seed band F1 (" + ", ".join(r.name for r in runs) + ")")
    res, csv_rows = {}, [["lo", "pooled_f1", "precision", "recall", "fp", "fn", "routed", "llm_share", "eligible"] + [f"f1_{r.name}" for r in runs]]
    per_seed_cache: dict = {}
    for lo in LOS:
        tp = fp = fn = routed = 0
        f1s = []
        for name, d in data.items():
            dec, n = literal_decisions(d["run"], lo, d["b"], d["c"])
            m = L.counts(dec, d["ent"])
            tp, fp, fn, routed = tp + m["tp"], fp + m["fp"], fn + m["fn"], routed + n
            f1s.append(m["f1"])
            per_seed_cache[(lo, name)] = (m, n, len(d["c"]))
        p, r = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        share = routed / band_total
        res[lo] = {"f1": f1, "precision": p, "recall": r, "fp": fp, "fn": fn, "routed": routed, "share": share, "eligible": share <= SHARE_MAX, "f1s": f1s}
        L_.append(f"  {lo:>6g}{f1:>10.4f}{p:>8.4f}{r:>8.4f}{fp:>5}{fn:>5}{routed:>8,}{share:>11.3%}  {'yes' if share <= SHARE_MAX else 'no':<9}  " + "  ".join(f"{x:.4f}" for x in f1s))
        csv_rows.append([lo, f"{f1:.6f}", f"{p:.6f}", f"{r:.6f}", fp, fn, routed, f"{share:.6f}", share <= SHARE_MAX] + [f"{x:.6f}" for x in f1s])
    # reference rows (not part of the rule): C alone
    tp = fp = fn = 0
    for name, d in data.items():
        m = L.counts({k: v[0] for k, v in d["c"].items()}, d["ent"])
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
    p, r = tp / (tp + fp), tp / (tp + fn)
    L_.append(f"  {'C alone':>6}{2 * p * r / (p + r):>10.4f}{p:>8.4f}{r:>8.4f}{fp:>5}{fn:>5}{0:>8}{0.0:>11.3%}  (reference)")
    eligible = [lo for lo in LOS if res[lo]["eligible"]]
    best = max(res[lo]["f1"] for lo in eligible)
    tied = [lo for lo in eligible if abs(res[lo]["f1"] - best) < 1e-12]
    chosen = max(tied)
    L_ += ["", f"  Eligible settings: {', '.join(format(x, 'g') for x in eligible)}. Highest pooled band F1 {best:.6f}; settings at that value: {', '.join(format(x, 'g') for x in tied)}.",
           f"  CHOSEN lo = {chosen:g} (hi {HI:g})."]
    runner = sorted((lo for lo in eligible if lo != chosen), key=lambda lo: -res[lo]["f1"])[:1]
    if runner:
        L_.append(f"  Next best: lo {runner[0]:g} with pooled F1 {res[runner[0]]['f1']:.6f} ({res[runner[0]]['f1'] - res[chosen]['f1']:+.6f} against the chosen).")
    # step 6: routed share per seed and projected LLM cost
    L_ += ["", f"  D(v2) at lo {chosen:g}: routed pairs and share of the test band per seed; projected LLM cost per 1,000 band pairs at ${COST_PER_CALL} per call"]
    L_.append(f"  {'run':<16}{'test band':>10}{'routed':>8}{'share':>10}{'LLM $ per 1,000 band pairs':>28}")
    for name in data:
        m, n, tot = per_seed_cache[(chosen, name)]
        L_.append(f"  {name:<16}{tot:>10,}{n:>8}{n / tot:>10.3%}{n / tot * 1000 * COST_PER_CALL:>28.4f}")
        csv_rows.append([f"share_{name}", n, tot, f"{n / tot:.6f}", f"{n / tot * 1000 * COST_PER_CALL:.6f}"])
    pooled = res[chosen]["share"]
    L_.append(f"  {'pooled':<16}{band_total:>10,}{res[chosen]['routed']:>8}{pooled:>10.3%}{pooled * 1000 * COST_PER_CALL:>28.4f}")
    text = "\n".join(L_) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "d_select_lo.txt").write_text(text, encoding="utf-8")
    with (args.out / "d_select_lo.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    doc = {"date": time.strftime("%Y-%m-%d"), "rule_file": RULE.name, "rule_file_sha256_lf": sha256_lf(RULE), "selected": True, "lo": chosen, "hi": HI,
           "lo_candidates": list(LOS), "share_max": SHARE_MAX, "development_seeds": [11, 12, 13, 42, 2026],
           "chosen_metrics": {k: res[chosen][k] for k in ("f1", "precision", "recall", "fp", "fn", "routed", "share")},
           "per_seed_band_f1": dict(zip([r.name for r in runs], res[chosen]["f1s"])),
           "note": "chosen by the rule in the rule file; applied unchanged to seed 7919 with C retrained on that seed's train split"}
    if not args.no_write:
        SETTINGS_V2.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        text += f"\nwritten: {SETTINGS_V2.relative_to(REPO).as_posix()}\n"
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "run-b", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--runs", nargs="+", default=list(L.DEFAULT_RUNS))
    sub.choices["list"].add_argument("--max-calls", type=int, default=500)
    sub.choices["run-b"].add_argument("--max-usd-total", type=float, required=True)
    sub.choices["run-b"].add_argument("--approve", action="store_true")
    sub.choices["report"].add_argument("--out", type=Path, default=REPO / "results" / "dstudy_summary")
    sub.choices["report"].add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    return {"list": cmd_list, "run-b": cmd_run_b, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

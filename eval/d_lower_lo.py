"""
D with a lower lo on development seeds: the frozen rule (lo 0.5, hi 0.9) against lo 0.1, hi 0.9, on seeds 11, 12, 13, 42, 2026.
No frozen setting and no selection rule file is changed: both settings are given on the command line here only.

  python eval/d_lower_lo.py list   [--runs ...] [--max-calls 400]   # route lists + the number of pairs that need a new B call and the cost
  python eval/d_lower_lo.py run-b  [--runs ...] --max-usd-total 1.5 [--approve]    # B on each seed's pair list (frozen prompt, ledger, cache, keys as before)
  python eval/d_lower_lo.py report [--runs ...] [--out results/dstudy_summary]      # exact full-band D for both settings, scored; per seed and totals

list and run-b never read a truth file: the pair lists come from C's calibrated confidence on the test band, and the B step
is the ordinary --final-sample mode with a pair-list file (record_a, record_b, weight 1.0). report reads the truth sidecar to score.
Runs default to dstudy_11, dstudy_12, dstudy_13, rehearsal_42, rehearsal_2026 under results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "options"))
sys.path.insert(0, str(REPO / "eval"))
import d_hybrid as D  # noqa: E402

DEV = REPO / "results" / "dev_2026_run11"
DEFAULT_RUNS = ("dstudy_11", "dstudy_12", "dstudy_13", "rehearsal_42", "rehearsal_2026")
SETTINGS = {"frozen": (0.5, 0.9), "lo0.1": (0.1, 0.9)}
LABEL = "lo01"                                   # B run label of the pair list (lo 0.1, hi 0.9; it contains the frozen rule's pairs)
PILOT_COST = 0.001545
GATE_MARGIN = 1.3
FROZEN_ASSETS = ("fewshot.json", "frozen_prompt.json")      # copied unchanged from the dev run, as the freeze run installs them


def run_dirs(names) -> list[Path]:
    return [REPO / "results" / n for n in names]


def c_rows(run: Path) -> list[dict]:
    return D.read_csv(run / "options" / "c_model" / "c_v1_lgbm" / "decisions.csv")


def routed(rows: list[dict], lo: float, hi: float) -> list[tuple[str, str]]:
    return sorted((r["record_a"], r["record_b"]) for r in rows if D.route_of(float(r["confidence"]), lo, hi) == "llm")


def write_pairs(path: Path, pairs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_a", "record_b", "weight"])
        for a, b in pairs:
            w.writerow([a, b, "1.0"])


def install_frozen(run: Path) -> None:
    dest = run / "options" / "b_llm"
    dest.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_ASSETS:
        src, dst = DEV / "options" / "b_llm" / name, dest / name
        if dst.exists():
            if dst.read_bytes() != src.read_bytes():
                raise SystemExit(f"{dst} differs from the frozen {src}")
        else:
            shutil.copyfile(src, dst)


def need_new_call(run: Path, pairs) -> int:
    """Distinct prompts of the pair list with no answer in the pair cache (keys built as b_llm_baseline --final-sample builds them)."""
    import b_llm_baseline as B
    from llm_cache import PairCache, sha256_text
    provider = B.make_provider("openai", None, live=False)
    rd = B.RunData(run)
    fewshot = json.loads((run / "options" / "b_llm" / "fewshot.json").read_text(encoding="utf-8"))
    system = B.build_system_prompt(fewshot, rd.cols)
    items = B.make_items(rd, [(a, b) for a, b in pairs], provider, system, provider.settings_sha(), sha256_text(system))
    cache = PairCache(B.CACHE_DIR / f"{provider.name}.jsonl")
    distinct = {it.key for it in items}
    return sum(cache.get(k) is None for k in distinct)


def cmd_list(args) -> int:
    runs = run_dirs(args.runs)
    print(f"{'run':<16}{'test band':>10}{'frozen (0.5, 0.9)':>19}{'lo 0.1, hi 0.9':>16}{'new B calls':>13}{'cost est.':>11}")
    total = 0
    rows_out = []
    for run in runs:
        rows = c_rows(run)
        fz, lw = routed(rows, *SETTINGS["frozen"]), routed(rows, *SETTINGS["lo0.1"])
        assert set(fz) <= set(lw)
        install_frozen(run)
        write_pairs(run / "options" / "d_lo" / "routed_frozen_0.5_0.9.csv", fz)
        write_pairs(run / "options" / "d_lo" / "routed_lo0.1_hi0.9.csv", lw)
        n_new = need_new_call(run, lw)
        total += n_new
        rows_out.append((run.name, len(rows), len(fz), len(lw), n_new))
        print(f"{run.name:<16}{len(rows):>10,}{len(fz):>19}{len(lw):>16}{n_new:>13}{'$' + format(n_new * PILOT_COST, '.4f'):>11}")
    print(f"{'total':<16}{'':>10}{'':>19}{'':>16}{total:>13}{'$' + format(total * PILOT_COST, '.4f'):>11}"
          f"   (pilot ${PILOT_COST} per call; the run's cost gate uses x{GATE_MARGIN} = ${total * PILOT_COST * GATE_MARGIN:.4f})")
    if total > args.max_calls:
        print(f"STOP: {total} new calls exceed {args.max_calls}")
        return 2
    return 0


def cmd_run_b(args) -> int:
    runs = run_dirs(args.runs)
    caps = []
    for run in runs:
        lw = run / "options" / "d_lo" / "routed_lo0.1_hi0.9.csv"
        if not lw.exists():
            raise SystemExit(f"{lw} missing: run 'list' first")
        n_new = need_new_call(run, [(r["record_a"], r["record_b"]) for r in D.read_csv(lw)])
        caps.append((run, lw, n_new))
    total_new = sum(n for *_, n in caps)
    est = total_new * PILOT_COST
    if est * GATE_MARGIN > args.max_usd_total:
        raise SystemExit(f"estimated ${est * GATE_MARGIN:.2f} (with the x{GATE_MARGIN} gate margin) exceeds the total cap ${args.max_usd_total}")
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
    spent_before = None
    for run, lw, n_new in caps:
        cap = max(0.05, round(n_new * PILOT_COST * GATE_MARGIN + 0.02, 3))
        cmd = [sys.executable, str(REPO / "options" / "b_llm_baseline.py"), str(run), "--provider", "openai", "--final-sample", str(lw),
               "--label", LABEL, "--max-usd", str(cap)]
        if not args.approve:
            cmd.append("--estimate-only")
        else:
            cmd.append("--approve-final-run")
        print(f"\n== {run.name}: {n_new} new calls, run cap ${cap}")
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        keep = [ln for ln in r.stdout.splitlines() if re.match(r"(pairs:|projection|cost gate|calls |decisions:|end:|power|final hit|refusing)", ln)]
        print("\n".join("   " + ln[:230] for ln in keep))
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise SystemExit(f"b_llm_baseline exited {r.returncode} for {run.name}")
    return 0


# ---------------------------------------------------------------- report

def truth_entities(run: Path) -> dict:
    csv.field_size_limit(10**8)
    with (run / "orgs_raw_truth.csv").open(newline="", encoding="utf-8-sig") as f:
        return {r["record_id"]: r["true_entity_id"] for r in csv.DictReader(f)}


def counts(decisions: dict, ent: dict) -> dict:
    tp = fp = fn = 0
    for (a, b), d in decisions.items():
        same, said = ent[a] == ent[b], d == "same"
        tp += said and same
        fp += said and not same
        fn += (not said) and same
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def parse_score(text: str) -> dict:
    m = re.search(r"F1 ([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", text)
    q = re.search(r"accuracy [\d.]+\s+precision ([\d.]+)\s+recall ([\d.]+)", text)
    return {"f1": float(m.group(1)), "precision": float(q.group(1)), "recall": float(q.group(2))}


def d_decisions(run: Path, lo: float, hi: float) -> tuple[list[dict], Path]:
    P = D.paths(run)
    c = D.load_c(P["c"])
    b = D.merge_b(D.b_paths(run) + [run / "options" / "b_llm" / "openai" / LABEL / "decisions.csv"])
    rows = D.build(c, b, [(k, 1.0) for k in sorted(c)], lo, hi)
    out = run / "options" / "d_lo" / f"decisions_fullband_lo{lo:g}_hi{hi:g}.csv"
    D.write_decisions(out, rows)
    return rows, out


def score_full(run: Path, decisions: Path, name: str, tag: str) -> dict:
    jp = run / "options" / "d_lo" / f"score_{tag}.json"
    r = subprocess.run([sys.executable, str(REPO / "eval" / "score_option.py"), str(run), "--decisions", str(decisions), "--name", name, "--json-out", str(jp)],
                       cwd=REPO, env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"score_option.py failed for {name}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    (run / "options" / "d_lo" / f"score_report_{tag}.txt").write_text(r.stdout, encoding="utf-8")
    return parse_score(r.stdout)


def cmd_report(args) -> int:
    runs = run_dirs(args.runs)
    per: dict = {}
    L = ["D with a lower lo, on development seeds: frozen rule (lo 0.5, hi 0.9) against lo 0.1, hi 0.9. C is the reference.",
         "Exact full test band for each setting, scored with eval/score_option.py (band F1, precision, recall from its report; FP and FN from the truth",
         "sidecar). Where the LLM is used, B's answers come from the cache (sample, decoy-check and lo01 runs). 'fixed' = C wrong and D right on an LLM-routed pair;",
         "'introduced' = C right and D wrong. No frozen setting was changed.", ""]
    hdr = f"  {'run':<15}{'option':<22}{'band F1':>8}{'prec':>8}{'recall':>8}{'FP':>5}{'FN':>5}{'LLM pairs':>11}{'fixed':>7}{'introduced':>12}"
    L.append(hdr)
    csv_rows = [["run", "option", "band_f1", "precision", "recall", "fp", "fn", "tp", "llm_pairs", "fixed", "introduced", "score_option_f1_check"]]
    tot: dict = {}
    for run in runs:
        ent = truth_entities(run)
        c = D.load_c(D.paths(run)["c"])
        c_dec = {k: v[0] for k, v in c.items()}
        entries = [("C (reference)", c_dec, 0, 0, 0, score_full(run, run / "options" / "c_model" / "c_v1_lgbm" / "decisions.csv", "C v1 LightGBM", "c"))]
        for tag, (lo, hi) in SETTINGS.items():
            rows, out = d_decisions(run, lo, hi)
            dd = {D.key(r["record_a"], r["record_b"]): r["decision"] for r in rows}
            llm = [D.key(r["record_a"], r["record_b"]) for r in rows if r["route"] == "llm"]
            fixed = sum(c_dec[k] != dd[k] and (dd[k] == "same") == (ent[k[0]] == ent[k[1]]) for k in llm)
            intro = sum(c_dec[k] != dd[k] and (c_dec[k] == "same") == (ent[k[0]] == ent[k[1]]) for k in llm)
            entries.append((f"D {tag} (lo {lo:g}, hi {hi:g})", dd, len(llm), fixed, intro, score_full(run, out, f"D lo {lo:g} hi {hi:g}", tag.replace('.', ''))))
        for name, dec, n_llm, fixed, intro, sc in entries:
            m = counts(dec, ent)
            assert abs(m["f1"] - sc["f1"]) < 6e-5, (run.name, name, m["f1"], sc["f1"])         # my counts agree with score_option.py
            per[(run.name, name)] = {**m, "llm": n_llm, "fixed": fixed, "introduced": intro}
            L.append(f"  {run.name:<15}{name:<22}{m['f1']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['fp']:>5}{m['fn']:>5}{n_llm:>11}{fixed:>7}{intro:>12}")
            csv_rows.append([run.name, name, f"{m['f1']:.6f}", f"{m['precision']:.6f}", f"{m['recall']:.6f}", m["fp"], m["fn"], m["tp"], n_llm, fixed, intro, sc["f1"]])
            t = tot.setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "llm": 0, "fixed": 0, "introduced": 0, "f1s": []})
            for k in ("tp", "fp", "fn"):
                t[k] += m[k]
            t["llm"] += n_llm
            t["fixed"] += fixed
            t["introduced"] += intro
            t["f1s"].append(m["f1"])
        L.append("")
    L.append(f"  Totals over the {len(runs)} seeds (pooled counts; the F1 is from the pooled TP, FP, FN; the mean is of the per-seed F1)")
    L.append(f"  {'option':<22}{'pooled F1':>10}{'prec':>8}{'recall':>8}{'FP':>5}{'FN':>5}{'mean F1':>9}{'LLM pairs':>11}{'fixed':>7}{'introduced':>12}")
    for name, t in tot.items():
        p = t["tp"] / (t["tp"] + t["fp"])
        r = t["tp"] / (t["tp"] + t["fn"])
        L.append(f"  {name:<22}{2 * p * r / (p + r):>10.4f}{p:>8.4f}{r:>8.4f}{t['fp']:>5}{t['fn']:>5}{sum(t['f1s']) / len(t['f1s']):>9.4f}{t['llm']:>11}{t['fixed']:>7}{t['introduced']:>12}")
    L += ["", "  Does lo 0.1 beat lo 0.5 on each seed? (band F1, lo 0.1 minus frozen; FP / FN change)"]
    f_name, l_name = "D frozen (lo 0.5, hi 0.9)", "D lo0.1 (lo 0.1, hi 0.9)"
    for run in runs:
        a, b = per[(run.name, f_name)], per[(run.name, l_name)]
        diff = b["f1"] - a["f1"]
        verdict = "better" if diff > 1e-9 else ("worse" if diff < -1e-9 else "the same")
        L.append(f"    {run.name:<15} lo 0.1 is {verdict}: F1 {b['f1']:.4f} vs {a['f1']:.4f} ({diff:+.4f}); FP {b['fp']} vs {a['fp']} ({b['fp'] - a['fp']:+d}), FN {b['fn']} vs {a['fn']} ({b['fn'] - a['fn']:+d}); "
                 f"LLM pairs {b['llm']} vs {a['llm']}")
    better = sum(per[(r.name, l_name)]["f1"] > per[(r.name, f_name)]["f1"] + 1e-9 for r in runs)
    worse = sum(per[(r.name, l_name)]["f1"] < per[(r.name, f_name)]["f1"] - 1e-9 for r in runs)
    L.append(f"    lo 0.1 is better on {better} of {len(runs)} seeds, worse on {worse}, the same on {len(runs) - better - worse}.")
    text = "\n".join(L) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "d_lower_lo.txt").write_text(text, encoding="utf-8")
    with (args.out / "d_lower_lo.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "run-b", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    sub.choices["list"].add_argument("--max-calls", type=int, default=400)
    sub.choices["run-b"].add_argument("--max-usd-total", type=float, required=True)
    sub.choices["run-b"].add_argument("--approve", action="store_true")
    sub.choices["report"].add_argument("--out", type=Path, default=REPO / "results" / "dstudy_summary")
    args = ap.parse_args()
    return {"list": cmd_list, "run-b": cmd_run_b, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

"""
D robustness across development seeds (no LLM calls): how often D's frozen rule (route to the LLM when C's calibrated
p is in [lo, hi)) routes anything, and where C's band errors lie in p.

  python eval/d_study.py prepare --seed 11          # results/dstudy_11: generate, Verity, checks, split, sample, A, C
  python eval/d_study.py report --runs results/dstudy_11 results/dstudy_12 ... [--out results/dstudy_summary]
  python eval/d_study.py fine [--runs ...]    # finer p bins with C's band FP / FN; lo x hi settings: routed pairs and errors in range

prepare runs the same commands as eval/freeze_run.py stages 1 to 6 (rehearsal settings: sample = every same-entity
test band pair + 60 non-matches), with no B call. Seed 7919 is refused. The run folder must not hold data.
report reads each run's C decisions (options/c_model/c_v1_lgbm/decisions.csv: the calibrated p in confidence), C's
cutoff (features.json), the split and the truth sidecar, and D's frozen lo / hi from the dev run's frozen_settings.json.
Per run: test band pairs and true pairs; C's cutoff; pairs with p in [0.5, 0.9) and in [0.1, 0.9); a 10-bin histogram
of p; C's band false positives and false negatives by p (p = 1.0; 0.9 to below 1.0; 0.5 to below 0.9; below 0.5);
C's band precision, recall and F1 (full test band, unweighted); pairs D's frozen rule would route.
Writes <out>/d_study.txt and <out>/d_study.csv. Describes; proposes no change to the rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(REPO / "options"))
from d_hybrid import route_of  # noqa: E402

EVAL_SEED = 7919
FROZEN_SETTINGS = REPO / "results" / "dev_2026_run11" / "options" / "d_hybrid" / "frozen_settings.json"
P_BINS = ("p = 1.0", "0.9 <= p < 1.0", "0.5 <= p < 0.9", "p < 0.5")


def run_step(args: list, env: dict | None = None, save: Path | None = None) -> str:
    full = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1", **{k: str(v) for k, v in (env or {}).items()}}
    r = subprocess.run([str(a) for a in args], cwd=REPO, env=full, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(r.stdout, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{Path(str(args[1])).name} exited {r.returncode}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout


def prepare(seed: int) -> Path:
    if seed == EVAL_SEED:
        raise SystemExit("seed 7919 is the evaluation seed: refused")
    run = REPO / "results" / f"dstudy_{seed}"
    if (run / "orgs_raw.csv").exists() or (run / "verity_data").exists():
        raise SystemExit(f"{run} already holds data")
    run.mkdir(parents=True, exist_ok=True)
    truth = run / "orgs_raw_truth.csv"
    print(f"seed {seed}: generate", flush=True)
    run_step([PY, "data/generate_data_org.py", "--seed", seed, "--output", run / "orgs_raw.csv"])
    print(f"seed {seed}: verity", flush=True)
    run_step([PY, "verity/src/main.py", "--nn-mode", "exact", "--input", (run / "orgs_raw.csv").resolve()], env={"VERITY_DATA_DIR": run / "verity_data"})
    print(f"seed {seed}: checks, split, sample", flush=True)
    run_step([PY, "eval/check_scored_pairs.py", run], save=run / "check_scored_pairs.txt")
    run_step([PY, "eval/band.py", run, "--truth", truth], save=run / "band_report.txt")
    run_step([PY, "eval/split.py", run])
    run_step([PY, "eval/make_sample.py", run, "--out", run / "options" / "b_llm" / "test_sample5000.csv", "--n-nonmatches", 60])
    print(f"seed {seed}: A, C", flush=True)
    run_step([PY, "options/a_threshold.py", run])
    run_step([PY, "eval/score_option.py", run, "--decisions", run / "options" / "a_threshold" / "decisions.csv", "--name", "A. threshold only"],
             save=run / "options" / "a_threshold" / "score_report.txt")
    run_step([PY, "options/c_decision_model.py", run, "--model", "lgbm", "--version", "c_v1_lgbm"])
    cdir = run / "options" / "c_model" / "c_v1_lgbm"
    run_step([PY, "eval/score_option.py", run, "--decisions", cdir / "decisions.csv", "--name", "C v1 LightGBM"], save=cdir / "score_report.txt")
    ri = json.loads((run / "verity_data" / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    if ri.get("nn_mode_resolved") != "exact":
        raise SystemExit(f"neighbor search resolved to {ri.get('nn_mode_resolved')}")
    print(f"seed {seed}: done -> {run}", flush=True)
    return run


def read(path: Path) -> list[dict]:
    csv.field_size_limit(10**8)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def p_bin(p: float) -> str:
    if p >= 1.0:
        return P_BINS[0]
    if p >= 0.9:
        return P_BINS[1]
    if p >= 0.5:
        return P_BINS[2]
    return P_BINS[3]


def analyse(run: Path, lo: float, hi: float | None) -> dict:
    cdir = run / "options" / "c_model" / "c_v1_lgbm"
    ent = {r["record_id"]: r["true_entity_id"] for r in read(run / "orgs_raw_truth.csv")}
    dec = read(cdir / "decisions.csv")
    cutoff = float(json.loads((cdir / "features.json").read_text(encoding="utf-8"))["cutoff"])
    hist = [0] * 10
    fp = dict.fromkeys(P_BINS, 0)
    fn = dict.fromkeys(P_BINS, 0)
    tp = n_true = in_59 = in_19 = routed = 0
    for r in dec:
        p = float(r["confidence"])
        same = ent[r["record_a"]] == ent[r["record_b"]]
        n_true += same
        hist[min(int(p * 10), 9)] += 1
        in_59 += 0.5 <= p < 0.9
        in_19 += 0.1 <= p < 0.9
        routed += route_of(p, lo, hi) == "llm"
        said = r["decision"] == "same"
        if said and same:
            tp += 1
        elif said and not same:
            fp[p_bin(p)] += 1
        elif same and not said:
            fn[p_bin(p)] += 1
    nfp, nfn = sum(fp.values()), sum(fn.values())
    prec = tp / (tp + nfp) if tp + nfp else 0.0
    rec = tp / (tp + nfn) if tp + nfn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    errors_routable = sum(route_of(float(r["confidence"]), lo, hi) == "llm" and ((r["decision"] == "same") != (ent[r["record_a"]] == ent[r["record_b"]]))
                          for r in dec)
    return {"run": run.name, "test_band_pairs": len(dec), "true_pairs": n_true, "c_cutoff": cutoff, "p_in_0.5_0.9": in_59, "p_in_0.1_0.9": in_19,
            "hist": hist, "fp": fp, "fn": fn, "fp_total": nfp, "fn_total": nfn, "tp": tp, "precision": prec, "recall": rec, "f1": f1,
            "routed": routed, "errors_in_routable_range": errors_routable}


def report(runs: list[Path], out: Path) -> int:
    fz = json.loads(FROZEN_SETTINGS.read_text(encoding="utf-8"))
    lo, hi = fz["lo"], fz["hi"]
    res = [analyse(r, lo, hi) for r in runs]
    out.mkdir(parents=True, exist_ok=True)
    L = [f"D robustness across development seeds (no LLM calls). D's frozen rule: route when lo {lo} <= p < hi {hi} (from {FROZEN_SETTINGS.relative_to(REPO).as_posix()}).",
         "p is C's calibrated confidence on the full test band; errors are C's own band decisions against the truth sidecar.", ""]
    hdr = f"  {'run':<16}{'band':>8}{'true':>6}{'cutoff':>9}{'[.5,.9)':>9}{'[.1,.9)':>9}{'routed':>8}{'FP':>5}{'FN':>5}{'prec':>8}{'recall':>8}{'F1':>8}"
    L.append(hdr)
    for r in res:
        L.append(f"  {r['run']:<16}{r['test_band_pairs']:>8,}{r['true_pairs']:>6}{r['c_cutoff']:>9.4f}{r['p_in_0.5_0.9']:>9}{r['p_in_0.1_0.9']:>9}{r['routed']:>8}"
                 f"{r['fp_total']:>5}{r['fn_total']:>5}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}")
    L += ["", "  Histogram of calibrated p (10 bins; the last bin includes p = 1.0)",
          f"  {'run':<16}" + "".join(f"{f'[{i/10:.1f},{(i+1)/10:.1f}' + (']' if i == 9 else ')'):>12}" for i in range(10))]
    for r in res:
        L.append(f"  {r['run']:<16}" + "".join(f"{h:>12,}" for h in r["hist"]))
    L += ["", "  C's band errors by p (FP = said same, different entities; FN = said different, same entity)",
          f"  {'run':<16}" + "".join(f"{'FP ' + b:>20}" for b in P_BINS) + "".join(f"{'FN ' + b:>20}" for b in P_BINS)]
    for r in res:
        L.append(f"  {r['run']:<16}" + "".join(f"{r['fp'][b]:>20}" for b in P_BINS) + "".join(f"{r['fn'][b]:>20}" for b in P_BINS))
    n_runs = len(res)
    empty = sum(r["routed"] == 0 for r in res)
    tiny = sum(0 < r["routed"] <= 10 for r in res)
    errs = sum(r["fp_total"] + r["fn_total"] for r in res)
    routable = sum(r["errors_in_routable_range"] for r in res)
    L += ["", f"  Routing: empty in {empty} of {n_runs} runs; 1 to 10 pairs in {tiny} of {n_runs}; routed counts "
              + ", ".join(f"{r['run']} {r['routed']}" for r in res) + ".",
          f"  C's band errors with p in [{lo}, {hi}) (the routable range): {routable} of {errs} ({routable / errs:.1%})" if errs else "  C made no band errors.",
          "  Per run: " + "; ".join(f"{r['run']} {r['errors_in_routable_range']} of {r['fp_total'] + r['fn_total']}" for r in res) + "."]
    text = "\n".join(L) + "\n"
    (out / "d_study.txt").write_text(text, encoding="utf-8")
    with (out / "d_study.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "test_band_pairs", "true_pairs", "c_cutoff", "p_in_0.5_0.9", "p_in_0.1_0.9", "routed", "tp", "fp_total", "fn_total",
                    "precision", "recall", "f1", "errors_in_routable_range"] + [f"hist_{i}" for i in range(10)]
                   + [f"fp {b}" for b in P_BINS] + [f"fn {b}" for b in P_BINS])
        for r in res:
            w.writerow([r["run"], r["test_band_pairs"], r["true_pairs"], r["c_cutoff"], r["p_in_0.5_0.9"], r["p_in_0.1_0.9"], r["routed"], r["tp"], r["fp_total"],
                        r["fn_total"], f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}", r["errors_in_routable_range"]] + r["hist"]
                       + [r["fp"][b] for b in P_BINS] + [r["fn"][b] for b in P_BINS])
    print(text)
    return 0


FINE_BINS = [(0.0, 0.001), (0.001, 0.005), (0.005, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.0), (1.0, None)]
FINE_LO = (0.005, 0.01, 0.05, 0.1, 0.5)
FINE_HI = (0.9, 0.99, None)
DEFAULT_FINE_RUNS = ("dstudy_11", "dstudy_12", "dstudy_13", "rehearsal_42", "rehearsal_2026")


def fine_bin(p: float) -> int:
    if p >= 1.0:
        return len(FINE_BINS) - 1
    for i, (lo, hi) in enumerate(FINE_BINS[:-1]):
        if lo <= p < hi:
            return i
    return 0


def bin_label(i: int) -> str:
    lo, hi = FINE_BINS[i]
    return "p = 1.0" if hi is None else f"[{lo:g}, {hi:g})"


def fine_rows(run: Path) -> list[tuple[float, bool]]:
    """(calibrated p, C's decision is wrong) for every test band pair of the run."""
    ent = {r["record_id"]: r["true_entity_id"] for r in read(run / "orgs_raw_truth.csv")}
    out = []
    for r in read(run / "options" / "c_model" / "c_v1_lgbm" / "decisions.csv"):
        same = ent[r["record_a"]] == ent[r["record_b"]]
        out.append((float(r["confidence"]), (r["decision"] == "same") != same, (r["decision"] == "same") and not same))
    return out


def fine(runs: list[Path], out: Path) -> int:
    data = {r.name: fine_rows(r) for r in runs}
    names = [r.name for r in runs]
    L = ["Finer error table: C's calibrated p on the test band (no LLM calls, no rule change). Error = C's band decision against the truth sidecar;",
         "FP = decided same, different entities; FN = decided different, same entity. Reads each run's C decisions and truth sidecar only.", ""]
    csv_rows = [["table", "run", "key", "pairs", "fp", "fn"]]
    L.append("1. Pairs, C band FP and FN by calibrated p")
    for name in names:
        rows = data[name]
        L.append(f"  {name}  ({len(rows):,} test band pairs, {sum(e for _, e, _ in rows)} errors)")
        L.append(f"    {'p bin':<18}{'pairs':>9}{'FP':>5}{'FN':>5}")
        for i in range(len(FINE_BINS)):
            g = [(e, f) for p, e, f in rows if fine_bin(p) == i]
            fp, fn = sum(f for _, f in g), sum(e and not f for e, f in g)
            L.append(f"    {bin_label(i):<18}{len(g):>9,}{fp:>5}{fn:>5}")
            csv_rows.append(["bins", name, bin_label(i), len(g), fp, fn])
    L += ["", "2. Per seed and setting: pairs routed (lo <= p < hi; hi none = no upper limit), share of the test band, C band errors inside the routed range"]
    L.append(f"  {'lo':>6} {'hi':>5}  " + "".join(f"{n:>22}" for n in names))
    L.append(f"  {'':>6} {'':>5}  " + "".join(f"{'routed (share) err':>22}" for _ in names))
    tot: dict = {}
    for lo in FINE_LO:
        for hi in FINE_HI:
            cells = []
            for n in names:
                rows = data[n]
                routed = [(e, f) for p, e, f in rows if route_of(p, lo, hi) == "llm"]
                errs = sum(e for e, _ in routed)
                cells.append(f"{len(routed):>6} ({len(routed) / len(rows):6.2%}) {errs:>3}")
                t = tot.setdefault((lo, hi), {"routed": 0, "errors": 0, "per_seed_errors": {}})
                t["routed"] += len(routed)
                t["errors"] += errs
                t["per_seed_errors"][n] = errs
                csv_rows.append(["settings", n, f"lo={lo:g} hi={'none' if hi is None else hi}", len(routed), errs, ""])
            L.append(f"  {lo:>6g} {'none' if hi is None else hi:>5}  " + "".join(f"{c:>22}" for c in cells))
    all_err = sum(e for rows in data.values() for _, e, _ in rows)
    all_pairs = sum(len(rows) for rows in data.values())
    L += ["", f"3. Totals over the {len(names)} seeds ({all_pairs:,} test band pairs, {all_err} C band errors)",
          f"  {'lo':>6} {'hi':>5}{'routed':>9}{'share of band':>15}{'errors in range':>17}{'share of all errors':>21}{'errors per routed pair':>25}"]
    for lo in FINE_LO:
        for hi in FINE_HI:
            t = tot[(lo, hi)]
            L.append(f"  {lo:>6g} {'none' if hi is None else hi:>5}{t['routed']:>9,}{t['routed'] / all_pairs:>15.3%}{t['errors']:>17}{t['errors'] / all_err:>21.1%}"
                     f"{(t['errors'] / t['routed'] if t['routed'] else 0):>25.4f}")
            csv_rows.append(["totals", "all", f"lo={lo:g} hi={'none' if hi is None else hi}", t["routed"], t["errors"], ""])
    best_ratio = max(tot, key=lambda k: tot[k]["errors"] / tot[k]["routed"] if tot[k]["routed"] else 0)
    best_catch = max(tot, key=lambda k: tot[k]["errors"])
    fmt = lambda k: f"lo {k[0]:g}, hi {'none' if k[1] is None else k[1]}"
    br, bc = tot[best_ratio], tot[best_catch]
    L += ["", "4. Reading (descriptive; no change to the rule is recommended)",
          f"  Most errors caught per routed pair: {fmt(best_ratio)} with {br['errors']} errors in {br['routed']} routed pairs ({br['errors'] / br['routed']:.4f} per pair).",
          f"  Most errors caught in total: {fmt(best_catch)} with {bc['errors']} of {all_err} errors ({bc['errors'] / all_err:.1%}) in {bc['routed']:,} routed pairs ({bc['errors'] / bc['routed']:.4f} per pair)."]
    for hi in FINE_HI:
        base = tot[(0.5, hi)]["per_seed_errors"]
        for lo in FINE_LO[:-1]:
            more = [n for n in names if tot[(lo, hi)]["per_seed_errors"][n] > base[n]]
            L.append(f"  Lowering lo from 0.5 to {lo:g} (hi {'none' if hi is None else hi}) adds errors in range on {len(more)} of {len(names)} seeds"
                     + (f": {', '.join(f'{n} +{tot[(lo, hi)]['per_seed_errors'][n] - base[n]}' for n in more)}" if more else "")
                     + f"; total {tot[(lo, hi)]['errors']} vs {tot[(0.5, hi)]['errors']} at lo 0.5, routed {tot[(lo, hi)]['routed']:,} vs {tot[(0.5, hi)]['routed']:,}.")
    text = "\n".join(L) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "d_study_fine.txt").write_text(text, encoding="utf-8")
    with (out / "d_study_fine.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp_ = sub.add_parser("fine")
    fp_.add_argument("--runs", type=Path, nargs="+", default=[REPO / "results" / n for n in DEFAULT_FINE_RUNS])
    fp_.add_argument("--out", type=Path, default=REPO / "results" / "dstudy_summary")
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--runs", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, default=REPO / "results" / "dstudy_summary")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.seed)
        return 0
    if args.cmd == "fine":
        return fine(args.runs, args.out)
    return report(args.runs, args.out)


if __name__ == "__main__":
    sys.exit(main())

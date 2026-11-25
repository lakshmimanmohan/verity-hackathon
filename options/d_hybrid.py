"""
Option D: hybrid. C (calibrated gradient-boosted trees) decides the clear band pairs; the frontier LLM (option B)
decides the ones in between. Routing, with C's calibrated probability p of "same":

    p >= hi          no LLM                              route c_same     (hi may be none)
    lo <= p < hi     the LLM's recorded answer is used   route llm
    p <  lo          no LLM                              route c_different

Two readings of what a pair outside [lo, hi) takes. The primary one is the literal text of selection_rule_v2.txt ("pairs routed
to the LLM take B's answer, all others take C's"): C's own decision (reading "literal"). The forced reading (p >= hi same, p < lo
different, whatever C decided) is reported as a diagnostic, "D (forced reading)"; its files carry the suffix _forced. The two
differ only for a pair C decides against its own probability's side of [lo, hi) (C says same below lo, or different at or above hi).

Two frozen settings exist. "v2" (the setting that counts: selection_rule_v2.txt, frozen_settings_v2.json) is the default;
"original" (selection_rule.txt, frozen_settings.json, lo 0.5, hi 0.9) is the secondary row. The v2 outputs keep the plain
names below; the outputs of the original setting get the suffix _original (decisions_fullband_original.csv, ...).

  python options/d_hybrid.py route results/<run> --lo 0.05 --hi 0.9 [--out decisions.csv] [--reading literal|forced]
  python options/d_hybrid.py sweep results/<run>
  python options/d_hybrid.py list-routed results/<run> [--setting v2|original]   # the full test band's pairs that go to the LLM (no truth)
  python options/d_hybrid.py full results/<run> [--setting v2|original] [--reading literal|forced]   # exact D decisions for every test band pair
  python options/d_hybrid.py readings results/<run> [--setting v2|original]   # pairs where the literal and the forced reading differ
  python options/d_hybrid.py full results/<run> --fallback off   # every LLM-routed pair takes C's own decision; compared with C

Inputs (all under results/<run>/options/):
  c_model/c_v1_lgbm/decisions.csv        C's decision and calibrated confidence for every test band pair
  b_llm/openai/sample5000/decisions.csv  B's recorded answers (decision, confidence = p_same) for the sample
  b_llm/test_sample5000.csv              the fixed weighted sample (record_a, record_b, weight)
D is scored on the sample: a routed sampled pair takes B's recorded answer (no new call); the script fails if a
routed sampled pair has no B answer. Output decisions.csv columns: record_a, record_b, decision, confidence,
route; the confidence is B's p_same on routed pairs and C's p otherwise.

This file never reads the truth file; `sweep` scores each setting by running eval/score_option.py (--sample,
--json-out) and applies the selection rule exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

LO_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
HI_GRID = (0.9, 0.95, 0.99, None)
F1_MARGIN = 0.01
SHARE_MAX = 0.20
LLM_COST_PER_CALL = 0.001527       # option B final run: $7.6220 / 4,993 calls


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10**8)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a.strip(), b.strip())))


def route_of(p: float, lo: float, hi: float | None) -> str:
    """c_same when p >= hi, llm when lo <= p < hi, c_different when p < lo (hi None: no c_same route)."""
    if hi is not None and p >= hi:
        return "c_same"
    if p >= lo:
        return "llm"
    return "c_different"


def valid_settings() -> list[tuple[float, float | None]]:
    return [(lo, hi) for lo in LO_GRID for hi in HI_GRID if hi is None or hi > lo]


def load_c(path: Path) -> dict:
    return {key(r["record_a"], r["record_b"]): (r["decision"], float(r["confidence"])) for r in read_csv(path)}


def load_b(path: Path) -> dict:
    return {key(r["record_a"], r["record_b"]): (r["decision"], float(r["confidence"])) for r in read_csv(path)}


def load_sample(path: Path) -> list[tuple[tuple[str, str], float]]:
    rows = read_csv(path)
    if list(rows[0]) != ["record_a", "record_b", "weight"]:
        raise SystemExit(f"{path}: expected columns record_a, record_b, weight")
    return [(key(r["record_a"], r["record_b"]), float(r["weight"])) for r in rows]


READINGS = ("literal", "forced")


def outside_decision(reading: str, route: str, c_decision: str) -> str:
    """A pair outside [lo, hi): C's own decision (literal, the primary reading) or the route's side (forced)."""
    if reading == "literal":
        return c_decision
    return "same" if route == "c_same" else "different"


def build(c: dict, b: dict, sample: list, lo: float, hi: float | None, reading: str = "literal") -> list[dict]:
    """One row per sampled pair: decision, confidence, route. Fails when a sampled pair has no C answer or a pair
    routed to the LLM has no B answer. Pairs routed to the LLM take B's answer; the others take C's own decision
    (reading "literal") or the route's side (reading "forced")."""
    out = []
    for (k, w) in sample:
        if k not in c:
            raise SystemExit(f"sampled pair {k} has no C decision")
        cd, p = c[k]
        r = route_of(p, lo, hi)
        if r == "llm":
            if k not in b:
                raise SystemExit(f"sampled pair {k} is routed to the LLM but has no recorded B answer")
            d, conf = b[k]
        else:
            d, conf = outside_decision(reading, r, cd), p
        out.append({"record_a": k[0], "record_b": k[1], "decision": d, "confidence": conf, "route": r, "weight": w})
    return out


def write_decisions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_a", "record_b", "decision", "confidence", "route"])
        for r in rows:
            w.writerow([r["record_a"], r["record_b"], r["decision"], f"{r['confidence']:.6f}", r["route"]])


def llm_stats(rows: list[dict], band_pairs: float) -> dict:
    n_llm = sum(r["route"] == "llm" for r in rows)
    wl = sum(r["weight"] for r in rows if r["route"] == "llm")
    share = wl / sum(r["weight"] for r in rows)
    return {"sampled_llm_calls": n_llm, "weighted_llm_pairs": wl, "weighted_llm_share": share,
            "projected_calls": share * band_pairs}


def select(results: list[dict], b_f1: float) -> tuple[dict | None, list[dict]]:
    """Apply the selection rule: among settings with band F1 >= B's band F1 - 0.01 and weighted LLM share <= 0.20,
    the smallest share; ties go to the higher F1, then the higher lo (then, to make the choice unique, the lower
    hi). Returns (chosen or None, qualifying settings)."""
    ok = [r for r in results if r["band_f1"] >= b_f1 - F1_MARGIN and r["weighted_llm_share"] <= SHARE_MAX]
    if not ok:
        return None, []
    ok.sort(key=lambda r: (round(r["weighted_llm_share"], 9), -round(r["band_f1"], 9), -r["lo"],
                           r["hi"] if r["hi"] is not None else 2.0))
    return ok[0], ok


def rule_hash(path: Path) -> str:
    """SHA-256 of the rule file with line endings normalised to LF (so a CRLF checkout hashes the same)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


SETTINGS = {"v2": {"settings": "frozen_settings_v2.json", "rule": "selection_rule_v2.txt", "suffix": ""},
            "original": {"settings": "frozen_settings.json", "rule": "selection_rule.txt", "suffix": "_original"}}


def setting_file(name: str, setting: str, reading: str = "literal") -> str:
    """Name of a D output for a setting and a reading: v2 keeps the plain name, the original setting gets _original and the
    forced reading _forced before the extension."""
    suffix = SETTINGS[setting]["suffix"] + ("_forced" if reading == "forced" else "")
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{suffix}{dot}{ext}" if suffix else name


def paths(run: Path) -> dict:
    o = run / "options"
    return {"c": o / "c_model" / "c_v1_lgbm" / "decisions.csv",
            "b": o / "b_llm" / "openai" / "sample5000" / "decisions.csv",
            "sample": o / "b_llm" / "test_sample5000.csv",
            "dir": o / "d_hybrid", "rule": o / "d_hybrid" / "selection_rule.txt"}


def score(run: Path, decisions: Path, sample: Path, name: str, json_out: Path) -> dict:
    cmd = [sys.executable, str(REPO / "eval" / "score_option.py"), str(run), "--decisions", str(decisions),
           "--sample", str(sample), "--name", name, "--json-out", str(json_out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"score_option.py failed for {name}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return json.loads(json_out.read_text(encoding="utf-8"))


def fmt_hi(hi):
    return "none" if hi is None else f"{hi:g}"


def merge_b(paths: list[Path]) -> dict:
    """B's recorded answers from several runs (sample, decoy check, routed pairs) as one dict. A pair answered in two
    runs must have the same decision (the answers come from one pair cache); a conflict is an error."""
    out: dict = {}
    for path in paths:
        if not path.exists():
            continue
        for k, v in load_b(path).items():
            if k in out and out[k][0] != v[0]:
                raise SystemExit(f"B answered {k} differently in two runs ({out[k][0]} and {v[0]})")
            out.setdefault(k, v)
    return out


def routed_pairs(c_rows: list[dict], lo: float, hi: float | None) -> list[tuple[str, str, float]]:
    """(record_a, record_b, p) of the pairs the rule sends to the LLM, in the orientation the C file has them."""
    out = [(r["record_a"], r["record_b"], float(r["confidence"])) for r in c_rows
           if route_of(float(r["confidence"]), lo, hi) == "llm"]
    return sorted(out)


def b_paths(run: Path) -> list[Path]:
    base = run / "options" / "b_llm" / "openai"
    return [base / "sample5000" / "decisions.csv", base / "decoycheck" / "decisions.csv", base / "dfull" / "decisions.csv",
            base / "dfull_v2" / "decisions.csv"]    # dfull_v2: the dev run's answers for the v2 routed pairs (all from the cache)


def frozen_lo_hi(P: dict, setting: str = "v2") -> tuple[float, float | None]:
    fz = json.loads((P["dir"] / SETTINGS[setting]["settings"]).read_text(encoding="utf-8"))
    if not fz["selected"]:
        raise SystemExit("no frozen setting")
    return fz["lo"], fz["hi"]


def cmd_list_routed(args) -> int:
    P = paths(args.run)
    lo, hi = frozen_lo_hi(P, args.setting)
    c_rows = read_csv(P["c"])
    pairs = routed_pairs(c_rows, lo, hi)
    out = P["dir"] / setting_file("routed_fullband_pairs.csv", args.setting)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_a", "record_b", "weight"])
        for a, b, _ in pairs:
            w.writerow([a, b, "1.0"])
    answered = merge_b(b_paths(args.run))
    have = sum(key(a, b) in answered for a, b, _ in pairs)
    p_range = f"C's p from {min(p for *_, p in pairs):.3f} to {max(p for *_, p in pairs):.3f}" if pairs else "none routed"
    print(f"frozen setting {args.setting}: lo {lo:g}, hi {fmt_hi(hi)}: {len(pairs)} of {len(c_rows):,} test band pairs are routed to the LLM "
          f"({p_range}); {have} already have a recorded B answer in "
          f"the sample / decoy-check / earlier D runs; {len(pairs) - have} do not (the local cache is checked by the B run itself)")
    print(f"written: {out}")
    return 0


def cmd_full(args) -> int:
    P = paths(args.run)
    lo, hi = frozen_lo_hi(P, args.setting)
    c = load_c(P["c"])
    b = merge_b(b_paths(args.run))
    rows = build(c, b, [(k, 1.0) for k in sorted(c)], lo, hi, args.reading)
    out = P["dir"] / setting_file("decisions_fullband.csv", args.setting, args.reading)
    write_decisions(out, rows)
    n_llm = sum(r["route"] == "llm" for r in rows)
    print(f"exact D decisions ({args.setting}, {args.reading} reading) for all {len(rows):,} test band pairs (lo {lo:g}, hi {fmt_hi(hi)}): "
          f"{ {k: sum(r['route'] == k for r in rows) for k in ('c_different', 'llm', 'c_same')} }; LLM share {n_llm / len(rows):.5f} -> {out}")
    return 0


def build_fallback_off(c: dict, lo: float, hi: float | None, reading: str = "literal") -> list[dict]:
    """D with the LLM fallback off: routed exactly as D does, but a pair routed to the LLM takes C's own decision (C's
    calibrated cutoff) instead of B's answer. Every other pair is decided as D decides it (reading literal: C's own
    decision, so the whole result is C's; reading forced: the route's side)."""
    rows = []
    for k in sorted(c):
        cd, p = c[k]
        r = route_of(p, lo, hi)
        d = cd if r == "llm" else outside_decision(reading, r, cd)
        rows.append({"record_a": k[0], "record_b": k[1], "decision": d, "confidence": p, "route": r, "weight": 1.0})
    return rows


def readings_differ(c: dict, lo: float, hi: float | None) -> list[tuple[str, str]]:
    """Pairs outside [lo, hi) where C's own decision and the route's side disagree: the pairs on which the literal and the forced
    reading of the rule give different decisions."""
    out = []
    for k in sorted(c):
        cd, p = c[k]
        r = route_of(p, lo, hi)
        if r != "llm" and outside_decision("forced", r, cd) != cd:
            out.append(k)
    return out


def cmd_readings(args) -> int:
    P = paths(args.run)
    lo, hi = frozen_lo_hi(P, args.setting)
    c = load_c(P["c"])
    diff = readings_differ(c, lo, hi)
    doc = {"setting": args.setting, "lo": lo, "hi": hi, "pairs": len(c), "pairs_where_readings_differ": len(diff),
           "c_same_below_lo": sum(1 for k in diff if c[k][0] == "same"), "c_different_at_or_above_hi": sum(1 for k in diff if c[k][0] == "different")}
    (P["dir"] / setting_file("readings_diff.json", args.setting)).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{args.setting} (lo {lo:g}, hi {fmt_hi(hi)}): the literal and the forced reading differ on {len(diff)} of {len(c):,} test band pairs "
          f"({doc['c_same_below_lo']} C same below lo, {doc['c_different_at_or_above_hi']} C different at or above hi)")
    return 0


def cmd_full_off(args) -> int:
    P = paths(args.run)
    lo, hi = frozen_lo_hi(P, args.setting)
    c = load_c(P["c"])
    rows = build_fallback_off(c, lo, hi, args.reading)
    out = P["dir"] / setting_file("decisions_fullband_fallback_off.csv", args.setting, args.reading)
    write_decisions(out, rows)
    differing = [(r["record_a"], r["record_b"]) for r in rows if r["decision"] != c[(r["record_a"], r["record_b"])][0]]
    routed = sum(r["route"] == "llm" for r in rows)
    doc = {"pairs": len(rows), "llm_routed_pairs": routed, "differing_from_c": len(differing), "identical_to_c": not differing,
           "lo": lo, "hi": hi, "compared_with": "options/c_model/c_v1_lgbm/decisions.csv (C's calibrated cutoff)",
           "differing_examples": [list(x) for x in differing[:5]]}
    if differing:
        score_out = P["dir"] / setting_file("score_report_fallback_off.txt", args.setting, args.reading)
        r = subprocess.run([sys.executable, str(REPO / "eval" / "score_option.py"), str(args.run), "--decisions", str(out), "--name",
                            "D, LLM fallback off (full band)", "--json-out", str(P["dir"] / setting_file("score_fallback_off.json", args.setting, args.reading))], capture_output=True, text=True, encoding="utf-8")
        score_out.write_text(r.stdout, encoding="utf-8")
        doc["scored"] = score_out.name
    (P["dir"] / setting_file("fallback_off_check.json", args.setting, args.reading)).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"D with the LLM fallback off, {len(rows):,} test band pairs: {routed} LLM-routed pairs take C's own decision; "
          f"{len(differing)} pairs differ from C v1 LightGBM's decisions ({'identical' if not differing else 'NOT identical: scored with score_option.py'}) -> {out}")
    return 0


def cmd_route(args) -> int:
    P = paths(args.run)
    hi = None if args.hi.lower() == "none" else float(args.hi)
    rows = build(load_c(P["c"]), load_b(P["b"]), load_sample(P["sample"]), args.lo, hi, args.reading)
    out = args.out or P["dir"] / f"decisions_lo{args.lo:g}_hi{fmt_hi(hi)}{'_forced' if args.reading == 'forced' else ''}.csv"
    write_decisions(out, rows)
    st = llm_stats(rows, sum(r["weight"] for r in rows))
    print(f"lo {args.lo:g}, hi {fmt_hi(hi)}: {len(rows)} sampled pairs; routes "
          f"{ {k: sum(r['route'] == k for r in rows) for k in ('c_different', 'llm', 'c_same')} }; "
          f"LLM calls {st['sampled_llm_calls']}, weighted LLM share {st['weighted_llm_share']:.4f} -> {out}")
    return 0


def cmd_sweep(args) -> int:
    P = paths(args.run)
    rule = P["rule"]
    if not rule.exists():
        raise SystemExit(f"{rule} must exist (and be committed) before any sweep.")
    c, b, sample = load_c(P["c"]), load_b(P["b"]), load_sample(P["sample"])
    band_pairs = sum(w for _, w in sample)
    work = Path(tempfile.mkdtemp(prefix="d_sweep_"))
    settings = valid_settings()
    jobs = []
    for lo, hi in settings:
        rows = build(c, b, sample, lo, hi, "forced")      # the sweep that chose the original rule used the forced reading; kept as it was
        dec = work / f"d_lo{lo:g}_hi{fmt_hi(hi)}.csv"
        write_decisions(dec, rows)
        jobs.append((lo, hi, dec, llm_stats(rows, band_pairs)))
    jobs.append(("B", None, P["b"], None))

    def run_one(job):
        lo, hi, dec, st = job
        name = "B" if lo == "B" else f"D lo={lo:g} hi={fmt_hi(hi)}"
        return job, score(args.run, dec, P["sample"], name, work / f"{dec.stem}.json")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        scored = list(ex.map(run_one, jobs))
    b_res = next(res for (lo, _, _, _), res in scored if lo == "B")
    results = []
    for (lo, hi, dec, st), res in scored:
        if lo == "B":
            continue
        results.append({"lo": lo, "hi": hi, **st, **res})
    chosen, ok = select(results, b_res["band_f1"])
    L = []
    L.append(f"Option D routing sweep on the 5,000-pair test band sample ({band_pairs:,.0f} band pairs), run {args.run.name}")
    L.append(f"B: band F1 {b_res['band_f1']:.4f} [{b_res['band_f1_ci'][0]:.4f}, {b_res['band_f1_ci'][1]:.4f}]; selection needs band F1 >= "
             f"{b_res['band_f1'] - F1_MARGIN:.4f} and weighted LLM share <= {SHARE_MAX:.2f}")
    L.append(f"rule file sha256 {rule_hash(rule)[:16]}... (LF-normalised); {len(settings)} settings; scored with eval/score_option.py "
             f"--sample (1,000 bootstrap resamples, seed 0); {time.time() - t0:.0f} s")
    L.append(f"C's calibrated p has few distinct values: pairs with p >= 0.005 are the only ones that can reach the LLM.")
    L.append("")
    L.append(f"{'lo':>6} {'hi':>5} | {'LLM calls':>9} {'LLM share':>10} {'proj. calls':>12} | {'band F1 [95% CI]':>26} {'band prec':>9} {'band rec':>9} | "
             f"{'overall P':>9} {'overall R':>9} | {'ECE':>7} | ok")
    okset = {(r['lo'], r['hi']) for r in ok}
    for r in results:
        L.append(f"{r['lo']:>6g} {fmt_hi(r['hi']):>5} | {r['sampled_llm_calls']:>9} {r['weighted_llm_share']:>10.4f} {r['projected_calls']:>12,.0f} | "
                 f"{r['band_f1']:.4f} [{r['band_f1_ci'][0]:.4f}, {r['band_f1_ci'][1]:.4f}] {r['band_precision']:>9.4f} {r['band_recall']:>9.4f} | "
                 f"{r['overall_precision']:>9.4f} {r['overall_recall']:>9.4f} | {r['ece'] if r['ece'] is None else format(r['ece'], '.4f'):>7} | "
                 f"{'*' if (r['lo'], r['hi']) in okset else ''}{' <== CHOSEN' if chosen and (r['lo'], r['hi']) == (chosen['lo'], chosen['hi']) else ''}")
    L.append("")
    L.append(f"* = meets the rule (band F1 >= B - {F1_MARGIN}, weighted LLM share <= {SHARE_MAX}): {len(ok)} of {len(results)} settings")
    if chosen:
        L.append(f"CHOSEN (smallest weighted LLM share; ties: higher F1, then higher lo): lo {chosen['lo']:g}, hi {fmt_hi(chosen['hi'])}: "
                 f"LLM share {chosen['weighted_llm_share']:.4f}, band F1 {chosen['band_f1']:.4f}")
    else:
        L.append("NO SETTING MEETS THE RULE: nothing is chosen and nothing is frozen.")
    text = "\n".join(L) + "\n"
    (P["dir"] / "sweep.txt").write_text(text, encoding="utf-8")
    print(text)
    frozen = {"date": time.strftime("%Y-%m-%d"), "rule_file": "selection_rule.txt", "rule_file_sha256_lf": rule_hash(rule),
              "selected": chosen is not None, "lo": chosen["lo"] if chosen else None,
              "hi": chosen["hi"] if chosen else None, "b_band_f1": b_res["band_f1"],
              "f1_floor": b_res["band_f1"] - F1_MARGIN, "share_max": SHARE_MAX, "qualifying_settings": len(ok),
              "chosen_metrics": {k: chosen[k] for k in ("sampled_llm_calls", "weighted_llm_share", "projected_calls", "band_f1",
                                                          "band_f1_ci", "band_precision", "band_recall", "overall_precision",
                                                          "overall_recall", "ece")} if chosen else None,
              "applies_to": "seed 7919 unchanged, with C retrained on that seed's train split; B's frozen prompt for the LLM route",
              "dev_run": args.run.name}
    (P["dir"] / "frozen_settings.json").write_text(json.dumps(frozen, indent=1) + "\n", encoding="utf-8")
    return 0 if chosen else 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Option D: hybrid routing of C and the LLM.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("route")
    r.add_argument("run", type=Path)
    r.add_argument("--lo", type=float, required=True)
    r.add_argument("--hi", default="none")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--reading", choices=READINGS, default="literal")
    s = sub.add_parser("sweep")
    s.add_argument("run", type=Path)
    s.add_argument("--workers", type=int, default=4)
    for name in ("list-routed", "full", "readings"):
        sp = sub.add_parser(name)
        sp.add_argument("run", type=Path)
        sp.add_argument("--setting", choices=tuple(SETTINGS), default="v2")
        if name == "full":
            sp.add_argument("--fallback", choices=("on", "off"), default="on")
            sp.add_argument("--reading", choices=READINGS, default="literal")
    args = ap.parse_args()
    if args.cmd == "full" and args.fallback == "off":
        return cmd_full_off(args)
    return {"route": cmd_route, "sweep": cmd_sweep, "list-routed": cmd_list_routed, "full": cmd_full, "readings": cmd_readings}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

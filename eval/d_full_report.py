"""
Report on option D's exact decisions for the whole test band (no sampling, no weights). Scoring only: the truth
file is read for test records after the decisions exist; no API call.

  python eval/d_full_report.py results/<run>

Reads <run>/options/d_hybrid/decisions_fullband.csv (python options/d_hybrid.py full) and score_fullband.json
(eval/score_option.py ... --json-out) and writes <run>/options/d_hybrid/d_fullband_report.txt:
  - every error of the exact D, both records side by side (decision, probability, route)
  - the sample-based D estimate (compare_d_vs_b.txt) next to the exact numbers, and whether each exact number falls
    inside the sample's interval (or inside the zero-error bound where the sample's interval is degenerate)
  - where D and B differ (only where B has an answer: the sample, the decoy check and the routed pairs) and where D
    and C differ (all band pairs), as pair counts and lists, with the truth of each
  - D's decision on each of the 68 decoy pairs: route, decision, and B's p_same (routed) or C's p
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "options"))
import d_hybrid as D  # noqa: E402
import error_bounds as EB  # noqa: E402

HARD = ("acronym", "truncation", "po_box")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    P = D.paths(run)
    dh = P["dir"]
    lo, hi = D.frozen_lo_hi(P)
    drows = {D.key(r["record_a"], r["record_b"]): r for r in D.read_csv(dh / "decisions_fullband.csv")}
    c = D.load_c(P["c"])
    b = D.merge_b(D.b_paths(run))
    exact = json.loads((dh / "score_fullband.json").read_text(encoding="utf-8"))
    side = {r["record_id"]: r["side"] for r in D.read_csv(run / "split_records.csv")}
    truth = {r["record_id"]: r for r in D.read_csv(run / "orgs_raw_truth.csv") if side.get(r["record_id"]) == "test"}
    raw = D.read_csv(run / "orgs_raw.csv")
    cls = json.loads((run / "verity_data" / "column_classification.json").read_text(encoding="utf-8"))
    keymap = {k.lower(): k for k in raw[0]}
    rid = keymap[str(cls["record_id_column"]).lower()]
    cols = [keymap[x.lower()] for x in cls["content_columns"] if x.lower() in keymap]
    records = {r[rid]: r for r in raw}

    def info(k):
        ta, tb = truth[k[0]], truth[k[1]]
        same = ta["true_entity_id"] == tb["true_entity_id"]
        tags = set(ta["variants"].split("+")) | set(tb["variants"].split("+"))
        decoy = bool(ta["decoy_group"]) and ta["decoy_group"] == tb["decoy_group"] and not same
        return same, tags, decoy

    L: list[str] = []
    say = L.append
    n = len(drows)
    say(f"Option D, exact decisions for all {n:,} test band pairs (lo {lo:g}, hi {D.fmt_hi(hi)}), run {run.name}")
    routes = Counter(r["route"] for r in drows.values())
    say(f"routes: {dict(routes)}; LLM share {routes['llm'] / n:.5f} ({routes['llm']} pairs: {sum(k in b for k in drows if drows[k]['route'] == 'llm')} with a B answer)")

    def side_by_side(k, r, same, tags, decoy):
        out = [f"  {k[0]} | {k[1]}   decision {r['decision']}, p {float(r['confidence']):.3f}, route {r['route']}, "
               f"tags {'+'.join(sorted(tags))}{', decoy' if decoy else ''}"]
        for col in cols:
            va, vb = (records[k[0]].get(col) or "(missing)"), (records[k[1]].get(col) or "(missing)")
            out.append(f"      {col:<16} {va[:44]:<44} | {vb[:44]}")
        return out

    fps, fns = [], []
    for k, r in sorted(drows.items()):
        same, tags, decoy = info(k)
        if r["decision"] == "same" and not same:
            fps.append((k, r, same, tags, decoy))
        if r["decision"] != "same" and same:
            fns.append((k, r, same, tags, decoy))
    say(f"\nExact errors: {len(fps)} false matches, {len(fns)} false non-matches")
    for title, errs in (("False matches (different organizations decided same)", fps),
                        ("False non-matches (same organization decided different)", fns)):
        say(f"\n{title}: {len(errs)}, all shown")
        for e in errs:
            L.extend(side_by_side(*e))

    # sample estimate against the exact numbers
    txt = (dh / "compare_d_vs_b.txt").read_text(encoding="utf-8")
    est = {}
    block = txt.split("D (hybrid)", 1)[1].split("B (gpt-5.4)", 1)[0]
    for name in ("band F1", "band precision", "band recall"):
        m = re.search(rf"{name}\s+([\d.]+)\s+95% CI \[([\d.]+), ([\d.]+)\]", block)
        est[name] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    zb = EB.worst_case(0, 4660, 7, 340, 333 / 340, band_true=340, band_false=44542)
    say("\nSample-based estimate (5,000-pair sample, compare_d_vs_b.txt) against the exact full-band numbers")
    say(f"  {'':<16}{'sample estimate':>16}  {'sample 95% interval':<22}{'exact':>8}   inside?")
    exact_vals = {"band F1": exact["band_f1"], "band precision": exact["band_precision"], "band recall": exact["band_recall"]}
    for name, (pt, l, h) in est.items():
        ev = exact_vals[name]
        inside = l - 1e-9 <= ev <= h + 1e-9
        note = "yes" if inside else "NO"
        if name == "band precision" and l == h:
            note = (f"NO: the interval is degenerate [{l:.4f}, {h:.4f}] (0 false matches sampled); the zero-error bound "
                    f"(worst-case precision >= {zb['precision']:.4f}) "
                    f"{'contains' if ev >= zb['precision'] else 'does not contain'} the exact value")
        say(f"  {name:<16}{pt:>16.4f}  [{l:.4f}, {h:.4f}]{'':<6}{ev:>8.4f}   {note}")
    say(f"  overall precision: sample 1.0000, exact {exact['overall_precision']:.4f}; overall recall: sample 0.5791, exact "
        f"{exact['overall_recall']:.4f}; ECE: sample 0.0002, exact {exact['ece']:.4f}")
    say(f"  why the sample missed the false matches: the {len(fps)} exact false matches are "
        f"{sum(r['route'] == 'c_same' for _, r, *_ in fps)} sure-same pairs from C (p = 1.0) and "
        f"{sum(r['route'] == 'llm' for _, r, *_ in fps)} the LLM answered same; each non-match has a 1-in-9.56 chance of being sampled, "
        f"so a handful of false matches can all be missed (the sample's 95% bound allowed up to 29)")

    # D against C, D against B
    diff_c = [(k, r) for k, r in sorted(drows.items()) if (r["decision"], ) != (c[k][0], )]
    say(f"\nWhere D and C differ: {len(diff_c)} of {n:,} band pairs (D differs from C only on the LLM-routed pairs)")
    ct = Counter()
    for k, r in diff_c:
        same, _, _ = info(k)
        ct[("D right" if (r["decision"] == "same") == same else "D wrong")] += 1
        say(f"  {k[0]} {k[1]}   C {c[k][0]} (p {c[k][1]:.3f}) -> D {r['decision']}   truth {'same' if same else 'different'}")
    say(f"  of those {len(diff_c)}: {dict(ct)}")
    both = [(k, r) for k, r in sorted(drows.items()) if k in b]
    diff_b = [(k, r) for k, r in both if r["decision"] != b[k][0]]
    ctb = Counter()
    for k, r in diff_b:
        same, _, _ = info(k)
        ctb["D right, B wrong" if (r["decision"] == "same") == same else "D wrong, B right"] += 1
    say(f"\nWhere D and B differ: B has an answer for {len(both):,} band pairs (the 5,000 sampled, 64 decoy-check and routed pairs); "
        f"they differ on {len(diff_b)} of them; {dict(ctb)}. On the {n - len(both):,} band pairs B never saw, the comparison is not possible")
    for k, r in diff_b:
        same, tags, decoy = info(k)
        say(f"  {k[0]} {k[1]}   D {r['decision']} (route {r['route']}, p {float(r['confidence']):.3f})  B {b[k][0]} (p_same {b[k][1]:.2f})   "
            f"truth {'same' if same else 'different'}   {'+'.join(sorted(tags))}")

    # the 68 decoy pairs
    dpairs = [D.key(r["record_a"], r["record_b"]) for r in D.read_csv(run / "options" / "b_llm" / "decoy_check_pairs.csv")]
    say(f"\nD's decision on each of the {len(dpairs)} decoy pairs (different organizations that share a decoy group)")
    say(f"  {'pair':<26}{'route':<13}{'D decision':<12}{'C p':>7}{'B p_same':>10}")
    cnt = Counter()
    for k in dpairs:
        r = drows[k]
        cnt[(r["route"], r["decision"])] += 1
        say(f"  {k[0] + ' ' + k[1]:<26}{r['route']:<13}{r['decision']:<12}{c[k][1]:>7.3f}{(f'{b[k][1]:.2f}' if r['route'] == 'llm' else '-'):>10}"
            f"{'   <== FALSE MATCH' if r['decision'] == 'same' else ''}")
    say(f"  summary: {dict(cnt)}; wrongly matched {sum(r['decision'] == 'same' for r in (drows[k] for k in dpairs))} of {len(dpairs)} "
        f"(B alone: 1 of 68)")
    text = "\n".join(L) + "\n"
    (dh / "d_fullband_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

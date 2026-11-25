"""
Error and routing report for option D on the fixed 5,000-pair sample (scoring only: reads the truth file for test
records after the run; no API call).

  python eval/d_report.py results/<run>

Reads <run>/options/d_hybrid/frozen_settings.json (the chosen lo and hi), builds D's decisions the same way as
options/d_hybrid.py and writes <run>/options/d_hybrid/d_report.txt:
  - routes: sampled and weighted pairs per route, the weighted LLM share, projected calls and cost per 1,000 band
    pairs at $0.001527 per call (option B's measured cost per call)
  - every false negative and false positive among the sampled pairs, both records side by side, with decision,
    probability, route and the weight; errors by tag; zero-error bounds wherever a count is 0
  - C's full-band errors (false matches and missed true pairs of c_v1_lgbm on all test band pairs): which of them the
    chosen rule sends to the LLM, which are in the sample (so B's recorded answer exists), and what B and D decided
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "options"))
import error_bounds as EB  # noqa: E402
import d_hybrid as D  # noqa: E402

HARD = ("acronym", "truncation", "po_box")
PLAIN = {"clean", "standard", ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    P = D.paths(run)
    fz = json.loads((P["dir"] / "frozen_settings.json").read_text(encoding="utf-8"))
    if not fz["selected"]:
        raise SystemExit("no setting was selected")
    lo, hi = fz["lo"], fz["hi"]
    c, b, sample = D.load_c(P["c"]), D.load_b(P["b"]), D.load_sample(P["sample"])
    rows = D.build(c, b, sample, lo, hi)
    side = {r["record_id"]: r["side"] for r in D.read_csv(run / "split_records.csv")}
    truth = {r["record_id"]: r for r in D.read_csv(run / "orgs_raw_truth.csv") if side.get(r["record_id"]) == "test"}
    raw = D.read_csv(run / "orgs_raw.csv")
    cls = json.loads((run / "verity_data" / "column_classification.json").read_text(encoding="utf-8"))
    keymap = {k.lower(): k for k in raw[0]}
    rid = keymap[str(cls["record_id_column"]).lower()]
    cols = [keymap[x.lower()] for x in cls["content_columns"] if x.lower() in keymap]
    records = {r[rid]: r for r in raw}

    def info(a, b_):
        ta, tb = truth[a], truth[b_]
        same = ta["true_entity_id"] == tb["true_entity_id"]
        tags = set(ta["variants"].split("+")) | set(tb["variants"].split("+"))
        decoy = bool(ta["decoy_group"]) and ta["decoy_group"] == tb["decoy_group"] and not same
        return same, tags, decoy

    for r in rows:
        r["true"], r["tags"], r["decoy"] = info(r["record_a"], r["record_b"])
        r["pred"] = r["decision"] == "same"
    band_pairs = sum(r["weight"] for r in rows)
    st = D.llm_stats(rows, band_pairs)
    L: list[str] = []
    say = L.append
    say(f"Option D (hybrid) on the 5,000-pair test band sample, run {run.name}: lo {lo:g}, hi {D.fmt_hi(hi)} "
        f"(chosen by selection_rule.txt, frozen_settings.json)")
    say(f"\nRoutes (C's calibrated p: p >= hi decided same, lo <= p < hi the LLM, p < lo decided different)")
    say(f"  {'route':<12}{'sampled':>9}{'weighted (test band)':>22}{'share':>9}")
    for rt in ("c_different", "llm", "c_same"):
        g = [r for r in rows if r["route"] == rt]
        w = sum(r["weight"] for r in g)
        say(f"  {rt:<12}{len(g):>9}{w:>22,.1f}{w / band_pairs:>9.4f}")
    say(f"  LLM calls: {st['sampled_llm_calls']} sampled; weighted LLM share of the band {st['weighted_llm_share']:.4f} "
        f"= about {st['projected_calls']:.0f} calls on the {band_pairs:,.0f} band pairs")
    say(f"  projected cost: ${st['weighted_llm_share'] * 1000 * D.LLM_COST_PER_CALL:.4f} per 1,000 band pairs "
        f"(${D.LLM_COST_PER_CALL} per call, B's measured cost), about ${st['projected_calls'] * D.LLM_COST_PER_CALL:.3f} for the band; "
        f"B alone costs ${1000 * D.LLM_COST_PER_CALL:.3f} per 1,000 band pairs")

    n_true = sum(r["true"] for r in rows)
    n_false = len(rows) - n_true
    fp = sum(r["pred"] and not r["true"] for r in rows)
    fn = sum((not r["pred"]) and r["true"] for r in rows)
    bt = sum(r["weight"] for r in rows if r["true"])
    bf = sum(r["weight"] for r in rows if not r["true"])
    say(f"\nErrors among the sampled pairs: {fn} false non-matches, {fp} false matches ({n_true} same, {n_false} different sampled)")
    for line in EB.bound_lines(fp, n_false, fn, n_true, (n_true - fn) / n_true, band_true=bt, band_false=bf):
        say(line)

    def side_by_side(x):
        out = [f"  {x['record_a']} | {x['record_b']}   decision {x['decision']}, p {x['confidence']:.3f}, route {x['route']}, weight "
               f"{x['weight']:.4f}, tags {'+'.join(sorted(x['tags']))}{', decoy' if x['decoy'] else ''}"]
        for col in cols:
            va, vb = (records[x["record_a"]].get(col) or "(missing)"), (records[x["record_b"]].get(col) or "(missing)")
            out.append(f"      {col:<16} {va[:44]:<44} | {vb[:44]}")
        return out

    for title, cond in (("False negatives (same organization, decided different)", lambda x: x["true"] and not x["pred"]),
                        ("False positives (different organizations, decided same)", lambda x: (not x["true"]) and x["pred"])):
        errs = sorted([x for x in rows if cond(x)], key=lambda x: (x["record_a"], x["record_b"]))
        say(f"\n{title}: {len(errs)}, all shown")
        for x in errs:
            L.extend(side_by_side(x))

    say("\nErrors by tag (sample counts; recall and false-match rate weighted to the test band)")
    say(f"  {'tag':<10}{'pairs':>7}{'same':>6}{'diff':>6}{'FN':>5}{'FP':>5}   recall (w)   false-match rate (w)   FP-rate bound")
    groups = [(t, [x for x in rows if t in x["tags"]]) for t in HARD]
    groups.append(("decoy", [x for x in rows if x["decoy"]]))
    groups.append(("easy", [x for x in rows if not x["decoy"] and not (x["tags"] & set(HARD)) and not x["tags"] <= PLAIN]))
    groups.append(("no tag", [x for x in rows if not x["decoy"] and x["tags"] <= PLAIN]))
    for name, g in groups:
        gs = [x for x in g if x["true"]]
        gd = [x for x in g if not x["true"]]
        wr = (sum(x["weight"] for x in gs if x["pred"]) / sum(x["weight"] for x in gs)) if gs else None
        wf = (sum(x["weight"] for x in gd if x["pred"]) / sum(x["weight"] for x in gd)) if gd else None
        gfp = sum(x["pred"] for x in gd)
        bound = f"<= {EB.cp_upper(0, len(gd)):.4f}" if gd and gfp == 0 else ""
        say(f"  {name:<10}{len(g):>7}{len(gs):>6}{len(gd):>6}{sum(not x['pred'] for x in gs):>5}{gfp:>5}   "
            f"{('-' if wr is None else f'{wr:.4f}'):>10}   {('-' if wf is None else f'{wf:.5f}'):>20}   {bound}")

    # C's errors on the full band
    full = [(k, v) for k, v in c.items()]
    c_fp, c_fn = [], []
    for (a, b_), (dec, p) in full:
        same, tags, decoy = info(a, b_)
        if dec == "same" and not same:
            c_fp.append((a, b_, p, tags, decoy))
        if dec != "same" and same:
            c_fn.append((a, b_, p, tags, decoy))
    in_sample = {(r["record_a"], r["record_b"]): r for r in rows}
    say(f"\nC v1 LightGBM's errors on the full test band: {len(c_fp)} false matches and {len(c_fn)} missed true pairs; "
        f"where the chosen rule (lo {lo:g}, hi {D.fmt_hi(hi)}) sends each pair")
    for title, errs in (("C false matches", c_fp), ("C missed true pairs", c_fn)):
        say(f"\n  {title}: {len(errs)}; routed to the LLM {sum(D.route_of(p, lo, hi) == 'llm' for _, _, p, _, _ in errs)}, "
            f"c_same {sum(D.route_of(p, lo, hi) == 'c_same' for _, _, p, _, _ in errs)}, "
            f"c_different {sum(D.route_of(p, lo, hi) == 'c_different' for _, _, p, _, _ in errs)}")
        say(f"    {'pair':<26}{'C p':>6}  {'route':<12}{'in sample':<10}B answer (recorded)       D decision   tags")
        for a, b_, p, tags, decoy in sorted(errs):
            rt = D.route_of(p, lo, hi)
            k = D.key(a, b_)
            bans = f"{b[k][0]} {b[k][1]:.2f}" if k in b else "not in sample (no B answer)"
            drow = in_sample.get(k)
            dd = drow["decision"] if drow else "(not sampled)"
            say(f"    {a + ' ' + b_:<26}{p:>6.3f}  {rt:<12}{'yes' if drow else 'no':<10}{bans:<27}{dd:<13}{'+'.join(sorted(tags))}{',decoy' if decoy else ''}")
    llm_c_err = [(a, b_) for a, b_, p, _, _ in c_fp + c_fn if D.route_of(p, lo, hi) == "llm"]
    say(f"\n  {len(llm_c_err)} of C's {len(c_fp) + len(c_fn)} full-band errors are routed to the LLM; "
        f"the other {len(c_fp) + len(c_fn) - len(llm_c_err)} keep C's decision.")
    text = "\n".join(L) + "\n"
    (P["dir"] / "d_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

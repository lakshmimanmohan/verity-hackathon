"""
Report on an option B tuning round over TRAIN band pairs (labels read for train records only).

  python options/b_tune_report.py results/<run> --label tune1 [--provider openai]
      [--true-weight-pop 410] [--false-weight-pop 62863]

Reads <run>/options/b_llm/<provider>/<label>/decisions.csv and calls.jsonl and writes tune_report.txt
next to them:
  - sample composition (same / decoy / other different)
  - confusion matrix, precision, recall, F1 (same-entity = positive), unweighted
  - the same weighted to the real train band mix: every true pair weighs true_pop / n_true and every
    false pair false_pop / n_false (the populations given on the command line), with a bootstrap 95%
    interval (1,000 resamples of the rows, seed 0, weights kept). Also shown: weights by sampling
    stratum (true / decoy / other different) from the sampler pool, because the sampler draws decoys
    at a much higher rate than other false pairs.
  - errors by tag: acronym, truncation, po_box (either record carries it), decoy (same decoy group,
    different organization), easy (no hard tag, not a decoy, some other variation such as case, typo,
    typo_address or missing_phone), no tag (both records only clean / standard)
  - up to 15 false negatives and 15 false positives with both records side by side, decision and
    confidence (the most confident errors first)
  - confidence (the p_same of the answer; "confidence" in round 1) in 5 bins against truth, read as P(same);
    answers whose value contradicts their decision (decision different with a value of 0.5 or more,
    decision same with a value below 0.5) are counted
  - when an error count is 0 the bootstrap interval is degenerate and is replaced by exact one-sided 95%
    Clopper-Pearson upper bounds on the false-match and false-non-match rates and the worst-case
    precision and recall they imply on a test band of 340 true and 44,542 false pairs (eval/error_bounds.py)
  - with --reused-from: calls made and cache hits, the prompt hashes of both rounds, results in three groups
    (fresh same pairs, reused decoy pairs, fresh other pairs: errors, p_same bins, inconsistent answers) and
    the reused decoys compared with the earlier round (decisions that changed, p_same before and after)
  - when <out dir>/acceptance_rule.txt exists (written before the run), its text and one verdict line per
    rule
  - tokens, cached tokens, hit ratio (overall and excluding each key's first call), cost per call
    and total, latency p50 / p95, refusals, parse failures and API errors, from calls.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))
import b_llm_baseline as B  # noqa: E402
import error_bounds as EB  # noqa: E402

HARD = ("acronym", "truncation", "po_box")
PLAIN = {"clean", "standard", ""}
RESAMPLES, SEED = 1000, 0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def boot(pred, tru, w):
    rng = np.random.default_rng(SEED)
    n = len(pred)
    tp_v, fp_v, fn_v = w * (pred & tru), w * (pred & ~tru), w * (~pred & tru)
    out = {"p": [], "r": [], "f1": []}
    for start in range(0, RESAMPLES, 100):
        idx = rng.integers(0, n, size=(min(100, RESAMPLES - start), n))
        tp, fp, fn = tp_v[idx].sum(1), fp_v[idx].sum(1), fn_v[idx].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["p"].append(np.where(tp + fp > 0, tp / (tp + fp), 0.0))
            out["r"].append(np.where(tp + fn > 0, tp / (tp + fn), 0.0))
            out["f1"].append(np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0))
    return {k: tuple(float(x) for x in np.percentile(np.concatenate(v), [2.5, 97.5])) for k, v in out.items()}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--true-weight-pop", type=float, default=410)
    ap.add_argument("--false-weight-pop", type=float, default=62863)
    ap.add_argument("--reused-from", default=None, metavar="LABEL",
                    help="an earlier round whose pairs this round reuses (its decoy pairs): reports the three groups "
                         "(fresh same, reused decoys, fresh other) and compares the reused decoys with that round")
    ap.add_argument("--seed", type=int, default=1, help="sampler seed of the round (for the stratum pool sizes)")
    args = ap.parse_args()
    run = args.run
    rd = B.RunData(run)
    truth = B.TrainTruth(run, rd.side)
    bdir = run / "options" / "b_llm"
    out_dir = bdir / args.provider / args.label
    dec = B.read_csv(out_dir / "decisions.csv")
    calls = [json.loads(x) for x in (out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    L: list[str] = []
    say = L.append

    rows = []
    for r in dec:
        a, b = r["record_a"], r["record_b"]
        same = truth.same(a, b)
        tags = truth.tags(a, b)
        conf = float(r["confidence"]) if r["confidence"] else None
        rows.append({"a": a, "b": b, "pred": r["decision"] == "same", "true": same, "conf": conf,
                     "decision": r["decision"], "decoy": truth.is_decoy_pair(a, b), "tags": tags, "model": r["model"]})
    n = len(rows)
    n_true = sum(x["true"] for x in rows)
    n_decoy = sum(x["decoy"] for x in rows)
    n_other = n - n_true - n_decoy
    say(f"Option B tuning round {args.label}: {args.provider}, {n} TRAIN band pairs (labels from the truth file, train records only)")
    say(f"models reported: {dict(Counter(x['model'] for x in rows))}; settings: {dec[0]['settings'] if dec else ''}")
    say(f"sample: {n_true} same, {n_decoy} decoy different, {n_other} other different")
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8")) if (out_dir / "run_meta.json").exists() else {}
    prior_pairs, prior_rows = set(), {}
    if args.reused_from:
        pdir_ = bdir / args.provider / args.reused_from
        pmeta = json.loads((pdir_ / "run_meta.json").read_text(encoding="utf-8"))
        prior_rows = {(r["record_a"], r["record_b"]): r for r in B.read_csv(pdir_ / "decisions.csv")}
        prior_pairs = set(prior_rows)
        say(f"calls made {meta.get('calls')}, cache hits {meta.get('cache_hits')} (answers served from the local pair cache). "
            f"The pair-cache key includes the SHA-256 of the system prompt, so round {args.reused_from} answers "
            f"(prompt hash {str(pmeta.get('system_prompt_sha256'))[:12]}) cannot be reused by this round "
            f"(prompt hash {str(meta.get('system_prompt_sha256'))[:12]}).")
        say(f"The {len(prior_pairs & {(x['a'], x['b']) for x in rows if x['decoy']})} decoy pairs are REUSED from {args.reused_from}; "
            f"all other pairs are fresh. Why: {args.reused_from} had 0 decision errors, so no pair-specific edit was made, "
            f"and the only change between the rounds is the output field's name and definition (confidence -> p_same). "
            f"The fresh pool had no decoy pairs left (84 in the pool, {args.reused_from} used 68, the rest were in the earlier "
            f"smoke and diagnostic pools), so reusing them keeps the decoy class in the gate and gives a before/after.")

    pred = np.array([x["pred"] for x in rows])
    tru = np.array([x["true"] for x in rows])
    tp, fp = int((pred & tru).sum()), int((pred & ~tru).sum())
    fn, tn = int((~pred & tru).sum()), int((~pred & ~tru).sum())
    p, r, f1 = prf(tp, fp, fn)
    say("\nConfusion matrix (unweighted; same-entity = positive)")
    say(f"                    truth same   truth different")
    say(f"  decided same      {tp:>10}   {fp:>15}")
    say(f"  decided different {fn:>10}   {tn:>15}")
    say(f"  precision {p:.4f}  recall {r:.4f}  F1 {f1:.4f}  accuracy {(tp + tn) / n:.4f}")

    n_false = n - n_true

    def weighted(label, w):
        wtp, wfp, wfn = float((w * (pred & tru)).sum()), float((w * (pred & ~tru)).sum()), float((w * (~pred & tru)).sum())
        wp, wr, wf = prf(wtp, wfp, wfn)
        ci = boot(pred, tru, w)
        bw = EB.worst_case(fp, n_false, fn, n_true, wr)
        say(f"\nWeighted to the train band mix: {label}")

        def shown(name, val, key, bound_text):
            lo, hi = ci[key]
            if lo == hi:                                  # degenerate: no errors of the relevant kind
                return f"{name} {val:.4f} (bootstrap interval degenerate; {bound_text})"
            return f"{name} {val:.4f} [{lo:.4f}, {hi:.4f}]"

        say("  " + shown("precision", wp, "p", f"false-match rate <= {bw['fmr_ub']:.5f}, worst-case precision >= {bw['precision']:.4f}")
            + "   " + shown("recall", wr, "r", f"false-non-match rate <= {bw['fnmr_ub']:.5f}, recall >= {bw['recall']:.4f}"))
        say("  " + shown("F1", wf, "f1", f"worst-case F1 >= {bw['f1']:.4f}")
            + f"   (bootstrap 95%, {RESAMPLES} resamples, seed {SEED})")
        say(f"  weighted TP {wtp:.1f}  FP {wfp:.1f}  FN {wfn:.1f}")

    for line in EB.bound_lines(fp, n_false, fn, n_true, r):
        say(line.replace("  zero-error bounds", "\nZero-error bounds", 1) if line.startswith("  zero-error") else line)

    wt, wf_ = args.true_weight_pop / n_true, args.false_weight_pop / (n - n_true)
    weighted(f"true weight {args.true_weight_pop:g} / {n_true} = {wt:.4f}, false weight {args.false_weight_pop:g} / "
             f"{n - n_true} = {wf_:.4f} (as specified)", np.where(tru, wt, wf_))

    fs = json.loads((bdir / "fewshot.json").read_text(encoding="utf-8"))
    fs_ids = {(e["ids"][0], e["ids"][1]) for e in fs}
    fs_rec = {x for pr in fs_ids for x in pr}
    tb = rd.band("train")
    pool = B.sampler_pool(tb, fs_ids, fs_rec)        # the population the sample represents (earlier-run pairs stay in)
    pt = sum(truth.same(q[0], q[1]) for q in pool)
    pd_ = sum(truth.is_decoy_pair(q[0], q[1]) for q in pool)
    po = len(pool) - pt - pd_
    w_strat = np.array([pt / n_true if x["true"] else (pd_ / n_decoy if x["decoy"] else po / n_other) for x in rows])
    weighted(f"by sampling stratum from the train band minus few-shot pairs ({len(pool):,} pairs: {pt} same, {pd_} decoy, {po:,} other): "
             f"same {pt / n_true:.3f}, decoy {pd_ / n_decoy:.3f}, other {po / n_other:.3f}", w_strat)

    say("\nErrors by tag (a pair counts under every hard tag either record carries)")
    say(f"  {'tag':<12}{'pairs':>7}{'same':>7}{'diff':>7}{'FN':>6}{'FP':>6}   recall   FP rate")
    groups = [(t, [x for x in rows if t in x["tags"]]) for t in HARD]
    groups.append(("decoy", [x for x in rows if x["decoy"]]))
    groups.append(("easy", [x for x in rows if not x["decoy"] and not (x["tags"] & set(HARD)) and not x["tags"] <= PLAIN]))
    groups.append(("no tag", [x for x in rows if not x["decoy"] and x["tags"] <= PLAIN]))
    for name, g in groups:
        gs = sum(x["true"] for x in g)
        gfn = sum(x["true"] and not x["pred"] for x in g)
        gfp = sum((not x["true"]) and x["pred"] for x in g)
        rec = f"{(gs - gfn) / gs:.3f}" if gs else "  -  "
        fpr = f"{gfp / (len(g) - gs):.3f}" if len(g) - gs else "  -  "
        say(f"  {name:<12}{len(g):>7}{gs:>7}{len(g) - gs:>7}{gfn:>6}{gfp:>6}   {rec:>6}   {fpr:>7}")

    def side_by_side(x):
        out = [f"  {x['a']} | {x['b']}   decision {x['decision']}, confidence {x['conf']}, tags "
               f"{'+'.join(sorted(x['tags']))}{', decoy' if x['decoy'] else ''}"]
        for c in rd.cols:
            va, vb = rd.records[x["a"]][c] or "(missing)", rd.records[x["b"]][c] or "(missing)"
            out.append(f"      {c:<16} {va[:44]:<44} | {vb[:44]}")
        return out

    for title, cond in (("False negatives (same organization, decided different)", lambda x: x["true"] and not x["pred"]),
                        ("False positives (different organizations, decided same)", lambda x: not x["true"] and x["pred"])):
        errs = sorted([x for x in rows if cond(x)], key=lambda x: (-(x["conf"] or 0), x["a"], x["b"]))
        say(f"\n{title}: {len(errs)} in all, up to 15 shown (most confident first)")
        for x in errs[:15]:
            L.extend(side_by_side(x))

    say("\nConfidence / p_same in 5 bins, read as P(same)")
    say(f"  {'bin':<12}{'pairs':>7}{'same':>7}{'frac same':>11}{'mean P(same)':>14}{'decided same':>14}")
    ps = [(x["conf"], x["true"], x["pred"]) for x in rows if x["conf"] is not None]
    for i in range(5):
        lo, hi = i / 5, (i + 1) / 5
        g = [q for q in ps if (lo <= q[0] < hi) or (i == 4 and q[0] == 1.0)]
        if g:
            say(f"  [{lo:.1f}, {hi:.1f}{']' if i == 4 else ')'}  {len(g):>7}{sum(t for _, t, _ in g):>7}"
                f"{sum(t for _, t, _ in g) / len(g):>11.3f}{sum(v for v, _, _ in g) / len(g):>14.3f}"
                f"{sum(d for _, _, d in g):>14}")
        else:
            say(f"  [{lo:.1f}, {hi:.1f}{']' if i == 4 else ')'}  {0:>7}")
    c_same = sum(1 for v, _, d in ps if d and v < 0.5)
    n_diff = sum(1 for *_, d in ps if not d)
    c_diff = sum(1 for v, _, d in ps if not d and v >= 0.5)
    vals = Counter(f"{v:.2f}" for v, _, d in ps if not d and v >= 0.5)
    inconsistent = c_same + c_diff
    say(f"  inconsistent answers: {inconsistent} of {len(ps)} ({100 * inconsistent / max(len(ps), 1):.2f}%): {c_same} 'same' "
        f"with a value below 0.5, {c_diff} of {n_diff} 'different' with a value of 0.5 or more (most common values "
        f"{dict(vals.most_common(5))})")

    if args.reused_from:
        def grp(x):
            if x["decoy"] and (x["a"], x["b"]) in prior_pairs:
                return "reused decoys"
            return "fresh same" if x["true"] else "fresh other"
        names = ("fresh same", "reused decoys", "fresh other")
        say("\nResults in three groups")
        say(f"  {'group':<15}{'pairs':>7}{'FN':>5}{'FP':>5}{'decided same':>14}{'inconsistent':>14}{'mean p_same':>13}")
        gl = {nm: [x for x in rows if grp(x) == nm] for nm in names}
        for nm in names:
            g = gl[nm]
            gfn = sum(x["true"] and not x["pred"] for x in g)
            gfp = sum((not x["true"]) and x["pred"] for x in g)
            ginc = sum((x["pred"] and x["conf"] < 0.5) or ((not x["pred"]) and x["conf"] >= 0.5) for x in g)
            say(f"  {nm:<15}{len(g):>7}{gfn:>5}{gfp:>5}{sum(x['pred'] for x in g):>14}{ginc:>14}"
                f"{sum(x['conf'] for x in g) / max(len(g), 1):>13.3f}")
        for nm in names:
            g = gl[nm]
            say(f"\n  p_same in 5 bins, {nm} ({len(g)} pairs)")
            say(f"    {'bin':<12}{'pairs':>7}{'truly same':>12}{'decided same':>14}")
            for i in range(5):
                lo, hi = i / 5, (i + 1) / 5
                gg = [x for x in g if (lo <= x["conf"] < hi) or (i == 4 and x["conf"] == 1.0)]
                say(f"    [{lo:.1f}, {hi:.1f}{']' if i == 4 else ')'}  {len(gg):>7}{sum(x['true'] for x in gg):>12}{sum(x['pred'] for x in gg):>14}")
        g = gl["reused decoys"]
        old = [(x, prior_rows[(x["a"], x["b"])]) for x in g]
        changed = [(x, o) for x, o in old if o["decision"] != x["decision"]]
        old_inc = sum((o["decision"] == "same" and float(o["confidence"]) < 0.5) or (o["decision"] != "same" and float(o["confidence"]) >= 0.5) for _, o in old)
        new_inc = sum((x["pred"] and x["conf"] < 0.5) or ((not x["pred"]) and x["conf"] >= 0.5) for x in g)
        say(f"\nReused decoys against {args.reused_from} ({len(g)} pairs; the same pairs, the same records, only the prompt's "
            f"field name and definition differ)")
        say(f"  decisions that changed: {len(changed)} of {len(g)}")
        say(f"  mean value of the field: {args.reused_from} (confidence) {sum(float(o['confidence']) for _, o in old) / max(len(old), 1):.3f}, "
            f"this round (p_same) {sum(x['conf'] for x in g) / max(len(g), 1):.3f}")
        say(f"  inconsistent answers: {args.reused_from} {old_inc} of {len(old)}, this round {new_inc} of {len(g)}")
        say(f"  value of the field, {args.reused_from} against this round (differences of 0.1 or more: "
            f"{sum(abs(float(o['confidence']) - x['conf']) >= 0.1 for x, o in old)} pairs)")
        for x, o in sorted(old, key=lambda t: -abs(float(t[1]["confidence"]) - t[0]["conf"]))[:8]:
            say(f"    {x['a']} | {x['b']}   {args.reused_from}: {o['decision']} {float(o['confidence']):.2f}   this round: {x['decision']} {x['conf']:.2f}")

    live = [c for c in calls if c.get("mode") == "live"]
    inp = sum((c.get("usage") or {}).get("input_tokens", 0) or 0 for c in live)
    cac = sum((c.get("usage") or {}).get("cached_tokens", 0) or 0 for c in live)
    outt = sum((c.get("usage") or {}).get("output_tokens", 0) or 0 for c in live)
    seen, inp_ex, cac_ex = set(), 0, 0
    for c in live:
        u = c.get("usage") or {}
        if c.get("slot") in seen:
            inp_ex += u.get("input_tokens", 0) or 0
            cac_ex += u.get("cached_tokens", 0) or 0
        seen.add(c.get("slot"))
    usd = [c["usd"] for c in live]
    lat = [c["latency_s"] for c in live if c.get("latency_s") is not None]
    refusals = sum(bool(c.get("refused")) for c in live)
    api_err = sum(bool(c.get("error")) and (not c.get("usage") or str(c.get("error")).startswith("response failed"))
                  for c in live)
    parse_fail = sum(bool(c.get("error")) and bool(c.get("usage")) and not str(c.get("error")).startswith("response failed")
                     for c in live)
    say("\nCalls (calls.jsonl)")
    say(f"  live calls {len(live)}; tokens: input {inp:,}, cached {cac:,}, output {outt:,} "
        f"(per call: input {inp / max(len(live), 1):.0f}, cached {cac / max(len(live), 1):.0f}, output {outt / max(len(live), 1):.1f})")
    say(f"  hit ratio (cached / input tokens): overall {cac / inp if inp else 0:.3f}; excluding each key's first call "
        f"{cac_ex / inp_ex if inp_ex else 0:.3f} ({len(seen)} keys)")
    say(f"  cost: total ${sum(usd):.4f}, per call ${sum(usd) / max(len(usd), 1):.6f} (min ${min(usd, default=0):.6f}, "
        f"max ${max(usd, default=0):.6f})")
    say(f"  latency per call: p50 {pct(lat, 0.5):.2f} s, p95 {pct(lat, 0.95):.2f} s, max {max(lat, default=0):.2f} s")
    say(f"  refusals {refusals}, parse failures {parse_fail}, API errors {api_err}")
    rule = out_dir / "acceptance_rule.txt"
    if rule.exists():
        hr_ex = cac_ex / inp_ex if inp_ex else 0.0
        errors = fp + fn
        checks = [("(a) decision errors at most 3 of 1,000", errors <= 3, f"{errors} errors ({fn} false non-matches, {fp} false matches) of {n}"),
                  ("(b) at most 1% inconsistent answers", inconsistent <= 0.01 * len(ps), f"{inconsistent} of {len(ps)} ({100 * inconsistent / max(len(ps), 1):.2f}%)"),
                  ("(c) hit ratio excluding first calls at least 0.80", hr_ex >= 0.80, f"{hr_ex:.3f}"),
                  ("(d) zero parse failures", parse_fail == 0, f"{parse_fail} parse failures, {api_err} API errors")]
        say("\nAcceptance rule (written before the run, from " + rule.name + ")")
        say("  " + rule.read_text(encoding="utf-8").strip().replace("\n", "\n  "))
        say("\nVerdicts")
        for name, ok, detail in checks:
            say(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        say(f"  OVERALL: {'all four rules pass: freeze' if all(c[1] for c in checks) else 'at least one rule failed: report and stop (one more edit round at most)'}")
    text = "\n".join(L) + "\n"
    (out_dir / "tune_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

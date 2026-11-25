"""
Error and run report for option B's final run on the fixed sample of test band pairs. Scoring only:
it reads the truth file for test records, after the run, and makes no API call.

  python eval/b_sample_report.py results/<run> --label sample5000 [--provider openai]
      [--sample results/<run>/options/b_llm/test_sample5000.csv]

Writes <run>/options/b_llm/<provider>/<label>/sample_report.txt:
  - every false negative and false positive among the sampled pairs, both records side by side, with the
    decision, p_same (the decisions file's confidence column) and the pair's sample weight
  - errors by tag: acronym, truncation, po_box (either record carries it), decoy (same decoy group,
    different organization), easy (no hard tag, not a decoy, some other variation), no tag (both records
    only clean / standard); sample counts and weighted estimates
  - p_same in 5 bins against truth (sample counts and weighted)
  - zero-error bounds (eval/error_bounds.py) wherever an error count is 0, overall and per tag
  - from calls.jsonl and run_meta.json: hit ratio overall and excluding each key's first call, cost per
    call and total, latency p50 and p95, refusals, parse failures, API errors, total run time
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import error_bounds as EB  # noqa: E402

HARD = ("acronym", "truncation", "po_box")
PLAIN = {"clean", "standard", ""}


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10**8)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--sample", type=Path, default=None)
    args = ap.parse_args()
    run = args.run
    out_dir = run / "options" / "b_llm" / args.provider / args.label
    sample = args.sample or run / "options" / "b_llm" / "test_sample5000.csv"
    side = {r["record_id"]: r["side"] for r in read_csv(run / "split_records.csv")}
    truth = {r["record_id"]: r for r in read_csv(run / "orgs_raw_truth.csv") if side.get(r["record_id"]) == "test"}
    raw = read_csv(run / "orgs_raw.csv")
    cls = json.loads((run / "verity_data" / "column_classification.json").read_text(encoding="utf-8"))
    keymap = {k.lower(): k for k in raw[0]}
    rid = keymap[str(cls["record_id_column"]).lower()]
    cols = [keymap[c.lower()] for c in cls["content_columns"] if c.lower() in keymap]
    records = {r[rid]: r for r in raw}
    weights = {(r["record_a"], r["record_b"]): float(r["weight"]) for r in read_csv(sample)}
    dec = read_csv(out_dir / "decisions.csv")
    assert {(r["record_a"], r["record_b"]) for r in dec} == set(weights), "decisions do not cover the sample exactly"

    rows = []
    for r in dec:
        a, b = r["record_a"], r["record_b"]
        ta, tb = truth[a], truth[b]
        same = ta["true_entity_id"] == tb["true_entity_id"]
        tags = set(ta["variants"].split("+")) | set(tb["variants"].split("+"))
        decoy = bool(ta["decoy_group"]) and ta["decoy_group"] == tb["decoy_group"] and not same
        rows.append({"a": a, "b": b, "pred": r["decision"] == "same", "true": same, "p": float(r["confidence"]),
                     "w": weights[(a, b)], "tags": tags, "decoy": decoy})
    L: list[str] = []
    say = L.append
    n = len(rows)
    n_true = sum(x["true"] for x in rows)
    n_false = n - n_true
    fp = sum(x["pred"] and not x["true"] for x in rows)
    fn = sum((not x["pred"]) and x["true"] for x in rows)
    band_true = sum(x["w"] for x in rows if x["true"])
    band_false = sum(x["w"] for x in rows if not x["true"])
    say(f"Option B final run {args.label} on {sample.name}: {n} sampled test band pairs ({n_true} same, {n_false} different; "
        f"weights sum to {band_true + band_false:,.1f} = {band_true:,.0f} same + {band_false:,.0f} different in the test band)")
    say(f"errors among the sampled pairs: {fn} false non-matches, {fp} false matches")
    for line in EB.bound_lines(fp, n_false, fn, n_true, (n_true - fn) / n_true if n_true else 0.0,
                               band_true=band_true, band_false=band_false):
        say(line)

    def side_by_side(x):
        out = [f"  {x['a']} | {x['b']}   decision {'same' if x['pred'] else 'different'}, p_same {x['p']:.2f}, weight "
               f"{x['w']:.4f}, tags {'+'.join(sorted(x['tags']))}{', decoy' if x['decoy'] else ''}"]
        for c in cols:
            va, vb = (records[x["a"]].get(c) or "(missing)"), (records[x["b"]].get(c) or "(missing)")
            out.append(f"      {c:<16} {va[:44]:<44} | {vb[:44]}")
        return out

    for title, cond in (("False negatives (same organization, decided different)", lambda x: x["true"] and not x["pred"]),
                        ("False positives (different organizations, decided same)", lambda x: (not x["true"]) and x["pred"])):
        errs = sorted([x for x in rows if cond(x)], key=lambda x: (x["a"], x["b"]))
        say(f"\n{title}: {len(errs)}, all shown")
        for x in errs:
            L.extend(side_by_side(x))

    say("\nErrors by tag (a pair counts under every hard tag either record carries); weighted = estimate for the test band")
    say(f"  {'tag':<12}{'pairs':>7}{'same':>6}{'diff':>6}{'FN':>5}{'FP':>5}   recall (w)   false-match rate (w)   FP-rate bound")
    groups = [(t, [x for x in rows if t in x["tags"]]) for t in HARD]
    groups.append(("decoy", [x for x in rows if x["decoy"]]))
    groups.append(("easy", [x for x in rows if not x["decoy"] and not (x["tags"] & set(HARD)) and not x["tags"] <= PLAIN]))
    groups.append(("no tag", [x for x in rows if not x["decoy"] and x["tags"] <= PLAIN]))
    for name, g in groups:
        gs = [x for x in g if x["true"]]
        gd = [x for x in g if not x["true"]]
        gfn = sum(not x["pred"] for x in gs)
        gfp = sum(x["pred"] for x in gd)
        wr = (sum(x["w"] for x in gs if x["pred"]) / sum(x["w"] for x in gs)) if gs else None
        wf = (sum(x["w"] for x in gd if x["pred"]) / sum(x["w"] for x in gd)) if gd else None
        bound = f"<= {EB.cp_upper(0, len(gd)):.4f}" if gd and gfp == 0 else ""
        say(f"  {name:<12}{len(g):>7}{len(gs):>6}{len(gd):>6}{gfn:>5}{gfp:>5}   "
            f"{('-' if wr is None else f'{wr:.4f}'):>10}   {('-' if wf is None else f'{wf:.5f}'):>20}   {bound}")

    say("\np_same in 5 bins against truth (sample counts; weighted counts estimate the test band)")
    say(f"  {'bin':<12}{'pairs':>7}{'same':>6}{'decided same':>14}{'weighted pairs':>16}{'weighted frac same':>20}")
    for i in range(5):
        lo, hi = i / 5, (i + 1) / 5
        g = [x for x in rows if (lo <= x["p"] < hi) or (i == 4 and x["p"] == 1.0)]
        W = sum(x["w"] for x in g)
        say(f"  [{lo:.1f}, {hi:.1f}{']' if i == 4 else ')'}  {len(g):>7}{sum(x['true'] for x in g):>6}{sum(x['pred'] for x in g):>14}"
            f"{W:>16,.1f}{(sum(x['w'] for x in g if x['true']) / W if W else 0):>20.4f}")
    inc = sum((x["pred"] and x["p"] < 0.5) or ((not x["pred"]) and x["p"] >= 0.5) for x in rows)
    say(f"  inconsistent answers (decision against p_same at 0.5): {inc} of {n}")

    calls = [json.loads(x) for x in (out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    live = [c for c in calls if c.get("mode") == "live"]
    inp = sum((c.get("usage") or {}).get("input_tokens", 0) or 0 for c in live)
    cac = sum((c.get("usage") or {}).get("cached_tokens", 0) or 0 for c in live)
    outt = sum((c.get("usage") or {}).get("output_tokens", 0) or 0 for c in live)
    seen, ie, ce = set(), 0, 0
    for c in live:
        u = c.get("usage") or {}
        if c.get("slot") in seen:
            ie += u.get("input_tokens", 0) or 0
            ce += u.get("cached_tokens", 0) or 0
        seen.add(c.get("slot"))
    usd = [c["usd"] for c in live]
    lat = [c["latency_s"] for c in live if c.get("latency_s") is not None]
    errors = [c for c in live if c.get("error")]
    api_err = sum(1 for c in errors if not c.get("usage") or str(c["error"]).startswith("response failed"))
    say("\nRun (calls.jsonl, run_meta.json)")
    say(f"  calls {len(live)} (log lines), cache hits {meta.get('cache_hits')}, pairs {meta.get('pairs')}, distinct prompts "
        f"{meta.get('distinct_prompts')}; models reported {meta.get('models_reported')}")
    say(f"  tokens: input {inp:,}, cached {cac:,}, output {outt:,}; hit ratio overall {cac / inp if inp else 0:.3f}, "
        f"excluding each key's first call {ce / ie if ie else 0:.3f} ({len(seen)} keys)")
    say(f"  cost: total ${sum(usd):.4f}, per call ${sum(usd) / max(len(usd), 1):.6f}")
    say(f"  latency per call: p50 {pct(lat, 0.5):.2f} s, p95 {pct(lat, 0.95):.2f} s, max {max(lat, default=0):.2f} s")
    say(f"  refusals {meta.get('refusals')}, parse failures {meta.get('unusable_answers')}, API errors {api_err} "
        f"(error lines {len(errors)}), stopped {meta.get('stopped')}")
    say(f"  run time {meta.get('run_time_minutes')} minutes ({meta.get('started_utc')} to {meta.get('ended_utc')})")
    text = "\n".join(L) + "\n"
    (out_dir / "sample_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

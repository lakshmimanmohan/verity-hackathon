"""
Report on option B's supplementary decoy check (every test band pair whose records share a decoy group but have
different entities). No bar. Scoring only: reads the truth file for test records after the run; no API call.

  python eval/decoy_check_report.py results/<run> --label decoycheck [--provider openai]

Every pair in this set is a different-organization pair, so every "same" answer is a false match. Writes
<run>/options/b_llm/<provider>/<label>/decoy_report.txt with the count, the errors (both records side by side, the
decision and p_same), the zero-error bound (eval/error_bounds.py) when there are none, the p_same distribution,
cost and cache figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import error_bounds as EB  # noqa: E402
import make_sample as ms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--provider", default="openai")
    args = ap.parse_args()
    run = args.run
    out_dir = run / "options" / "b_llm" / args.provider / args.label
    side = {r["record_id"]: r["side"] for r in ms.read_csv(run / "split_records.csv")}
    truth = {r["record_id"]: r for r in ms.read_csv(run / "orgs_raw_truth.csv") if side.get(r["record_id"]) == "test"}
    raw = ms.read_csv(run / "orgs_raw.csv")
    cls = json.loads((run / "verity_data" / "column_classification.json").read_text(encoding="utf-8"))
    keymap = {k.lower(): k for k in raw[0]}
    rid = keymap[str(cls["record_id_column"]).lower()]
    cols = [keymap[c.lower()] for c in cls["content_columns"] if c.lower() in keymap]
    records = {r[rid]: r for r in raw}
    dec = ms.read_csv(out_dir / "decisions.csv")
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    rows = []
    for r in dec:
        a, b = r["record_a"], r["record_b"]
        ta, tb = truth[a], truth[b]
        assert ta["decoy_group"] and ta["decoy_group"] == tb["decoy_group"] and ta["true_entity_id"] != tb["true_entity_id"], (a, b)
        rows.append({"a": a, "b": b, "pred": r["decision"] == "same", "p": float(r["confidence"])})
    n = len(rows)
    fp = sum(x["pred"] for x in rows)
    L: list[str] = []
    say = L.append
    say(f"Option B (frozen prompt, gpt-5.4) on every test band pair whose records share a decoy group but have different "
        f"entities, run {run.name}: {n} pairs (supplementary check, no bar)")
    say(f"every pair is a different-organization pair: false matches {fp} of {n}")
    for line in EB.bound_lines(fp, n, 1, 1, 1.0, band_true=1, band_false=n)[:2]:
        say(line)
    if fp:
        say(f"  false-match rate {fp / n:.3f}; exact one-sided 95% upper bound {EB.cp_upper(fp, n):.4f}")
    say(f"  (with 0 false matches in {n} pairs the exact one-sided 95% bound on the false-match rate for pairs like these is "
        f"{EB.cp_upper(0, n):.4f}, about 3 / {n})" if not fp else "")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    say("\np_same distribution (all pairs are truly different)")
    for lo, hi in bins:
        g = [x for x in rows if lo <= x["p"] < hi]
        say(f"  [{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}  {len(g):>4}   decided same {sum(x['pred'] for x in g)}")
    say(f"  highest p_same {max(x['p'] for x in rows):.2f}, mean {sum(x['p'] for x in rows) / n:.3f}")
    errs = [x for x in rows if x["pred"]]
    say(f"\nFalse matches (decided same): {len(errs)}, all shown")
    for x in errs:
        say(f"  {x['a']} | {x['b']}   decision same, p_same {x['p']:.2f}")
        for c in cols:
            say(f"      {c:<16} {(records[x['a']].get(c) or '(missing)')[:44]:<44} | {(records[x['b']].get(c) or '(missing)')[:44]}")
    near = sorted([x for x in rows if not x["pred"]], key=lambda x: -x["p"])[:3]
    say(f"\nThe {len(near)} highest-p_same pairs that were decided different (closest calls), both records side by side")
    for x in near:
        say(f"  {x['a']} | {x['b']}   decision different, p_same {x['p']:.2f}")
        for c in cols:
            say(f"      {c:<16} {(records[x['a']].get(c) or '(missing)')[:44]:<44} | {(records[x['b']].get(c) or '(missing)')[:44]}")
    say(f"\nRun: calls {meta['calls']}, cache hits {meta['cache_hits']} (pairs already answered in the sample run), spent "
        f"${meta['spent_usd']:.4f}; hit ratio excluding each key's first call "
        f"{'n/a (no live calls)' if meta['prompt_cache'].get('hit_ratio_excl_first') is None else format(meta['prompt_cache']['hit_ratio_excl_first'], '.3f')}; refusals {meta['refusals']}, parse failures "
        f"{meta['unusable_answers']}, API errors {meta['api_errors']}; run time {meta['run_time_minutes']} minutes")
    text = "\n".join(x for x in L if x != "") + "\n"
    (out_dir / "decoy_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Live concurrency test for option B (spends real money; hard cap --max-usd for the whole test, enforced by the ledger).

  python options/b_concurrency_test.py results/dev_2026_run11 --max-usd 0.60

Frozen prompt (checked against frozen_prompt.json), same model string, settings, ledger and pair cache as the final
run; TRAIN band pairs that no earlier run used (not few-shot, not in any run folder's calls.jsonl, not in the caching
diagnostic's pool); 16 prompt-cache keys assigned round-robin; the per-key limiter is OFF. 100 calls per level at 2, 5
and 10 workers. Per level: calls per minute (whole level, and after the solo first call), latency p50, p95 and max,
hit ratio excluding each key's first call in the level, errors. Stops at the first 429 or other API error (after the
SDK's own retries) and reports the last clean level. Prints the power plan, AC status and the start and end time.
Never prints or logs a key. Writes <run>/options/b_llm/openai/concurrency/{concurrency.json, report.txt, level<N>/}.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b_llm_baseline as B  # noqa: E402
from llm_ledger import Ledger  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--max-usd", type=float, default=0.60)
    ap.add_argument("--calls", type=int, default=100, help="calls per level")
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--keys", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    run = args.run
    bdir = run / "options" / "b_llm"
    out_dir = bdir / "openai" / "concurrency"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    power = B.power_status()
    plan = "unknown"
    if sys.platform == "win32":
        r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        plan = (r.stdout or "").strip()
    t_start = time.time()
    say(f"concurrency test, start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))} local ({time.strftime('%H:%M:%SZ', time.gmtime(t_start))})")
    say(f"power plan: {plan}")
    say(f"AC status: {power}")
    if power == "battery" and not args.allow_battery:
        raise SystemExit("refusing: on battery; plug in the laptop.")

    provider = B.make_provider("openai", None, live=True)
    rd = B.RunData(run)
    fewshot = json.loads((bdir / "fewshot.json").read_text(encoding="utf-8"))
    fs_ids = {(e["ids"][0], e["ids"][1]) for e in fewshot}
    fs_records = {r for pr in fs_ids for r in pr}
    system = B.build_system_prompt(fewshot, rd.cols)
    B.check_frozen(bdir / "frozen_prompt.json", system, provider)           # the frozen prompt, 8-key scheme
    provider.transport = {**provider.transport, "prompt_cache_keys": args.keys}
    settings_sha, system_sha = provider.settings_sha(), B.sha256_text(system)

    truth = B.TrainTruth(run, rd.side)                                       # train labels, only for the earlier-pair exclusion
    tb = rd.band("train")
    excl = set(fs_ids) | B.earlier_live_pairs(rd, truth, tb, fs_ids, fs_records, bdir)
    pool = B.sampler_pool(tb, excl, fs_records)
    cache = B.PairCache(B.CACHE_DIR / "openai.jsonl")
    need = args.calls * len(args.levels)
    rng = random.Random(args.seed)
    picked, keys_seen = [], set()
    for q in rng.sample(pool, len(pool)):
        k = B.make_items(rd, [q], provider, system, settings_sha, system_sha)[0].key
        if k in keys_seen or cache.get(k) is not None:
            continue
        keys_seen.add(k)
        picked.append(q)
        if len(picked) == need:
            break
    if len(picked) < need:
        raise SystemExit(f"only {len(picked)} fresh train pairs for {need} calls")
    say(f"fresh train band pairs: {len(pool):,} in the pool after excluding {len(excl):,} earlier or few-shot pairs; using {need} "
        f"(seed {args.seed}); {args.keys} cache keys round-robin, per-key limiter off; prompt hash {system_sha[:12]}")

    ledger = Ledger(B.DEFAULT_LEDGER)
    run_id = f"{run.name}/b_llm/openai/concurrency"
    levels, last_clean, stopped = [], None, None
    B.keep_awake(True)
    try:
        for i, w in enumerate(args.levels):
            pairs = picked[i * args.calls:(i + 1) * args.calls]
            items = B.make_items(rd, pairs, provider, system, settings_sha, system_sha)
            say(f"\nlevel {w} workers: {len(items)} calls, start {time.strftime('%H:%M:%S')}")
            t0 = time.perf_counter()
            oc = B.run_items(provider, items, system, ledger=ledger, cache=cache, run_id=run_id, run_cap_usd=args.max_usd,
                             mode="live", concurrency=w, out_dir=out_dir / f"level{w}", stop_on_error=True,
                             rate_limiter=None, hit_report_every=0, check_prefix=False, log=lambda *a: None)
            wall = time.perf_counter() - t0
            lat = oc.latencies
            n_ok = len(oc.latencies)
            after_first = wall - (lat[0] if lat else 0.0)
            h = oc.hits or {}
            errs = oc.errors
            text_err = oc.stopped or ""
            lv = {"workers": w, "calls_requested": len(items), "calls_completed": n_ok, "wall_s": round(wall, 2),
                  "calls_per_minute": round(n_ok / wall * 60, 1),
                  "calls_per_minute_after_first": round((n_ok - 1) / max(after_first, 1e-9) * 60, 1) if n_ok > 1 else None,
                  "latency_p50_s": round(pct(lat, 0.5), 3), "latency_p95_s": round(pct(lat, 0.95), 3),
                  "latency_max_s": round(max(lat, default=0), 3),
                  "hit_ratio_excl_first": round(h["hit_ratio_excl_first"], 4) if h.get("hit_ratio_excl_first") is not None else None,
                  "hit_ratio_overall": round(h["hit_ratio"], 4) if h.get("hit_ratio") is not None else None,
                  "errors": errs, "stopped": oc.stopped, "spent_usd": round(oc.spent_usd, 4),
                  "rate_limit_429": ("429" in text_err) or ("rate" in text_err.lower())}
            levels.append(lv)
            say(f"  completed {n_ok}/{len(items)}; wall {wall:.1f} s; {lv['calls_per_minute']} calls per minute ({lv['calls_per_minute_after_first']} after the solo first call)")
            say(f"  latency p50 {lv['latency_p50_s']} s, p95 {lv['latency_p95_s']} s, max {lv['latency_max_s']} s; hit ratio excluding first calls "
                f"{lv['hit_ratio_excl_first']} (overall {lv['hit_ratio_overall']}); errors {errs}; spent ${lv['spent_usd']}")
            if oc.stopped or errs or n_ok < len(items):
                stopped = f"level {w} workers: {oc.stopped or 'incomplete'}"
                say(f"  STOPPED: {stopped}")
                break
            last_clean = w
    finally:
        B.keep_awake(False)
    t_end = time.time()
    say(f"\nend {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_end))} local; total {(t_end - t_start) / 60:.1f} minutes; last clean level: "
        f"{last_clean if last_clean is not None else 'none'}{'; ' + stopped if stopped else ''}")
    st = ledger.state()
    say(f"ledger total ${st.total_micro / 1e6:.4f}; this test ${st.by_run_micro.get(run_id, 0) / 1e6:.4f}")
    doc = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
           "power_plan": plan, "ac_status": power, "keys": args.keys, "limiter": "off", "calls_per_level": args.calls,
           "prompt_sha256": system_sha, "model": provider.model, "levels": levels, "last_clean_workers": last_clean, "stopped": stopped,
           "note": "the SDK retries 429 and 5xx twice before an error reaches this test; latency max includes such retries"}
    (out_dir / "concurrency.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if stopped is None else 2


if __name__ == "__main__":
    sys.exit(main())

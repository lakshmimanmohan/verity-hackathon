"""
Per-pair latency of option D including the LLM fallback, as a SIMULATION from measured distributions.

  OMP_NUM_THREADS=1 python options/d_latency_mixture.py results/dev_2026_run11 [--draws 1000000] [--seed 0]

Measured inputs
  C: the per-pair latency of C on the test band pairs, single core, features included, 3 repeats (this script measures
     them with the same code as options/latency_bench.py and keeps every per-pair value). The routing step adds about
     0.0003 ms and is left out.
  B: every measured live call latency of the final-run code path at one thread level (2 workers): the sample5000 run,
     tune2, the decoy check, the D routed-pair run and the 2-worker level of the concurrency test. The 5- and 10-worker
     levels are not used. The first call of a run (alone, cold cache) is included; calls with an error are not.
Simulation (draws, seed): a pair takes a latency drawn at random from C's pooled per-pair values; with probability
38 / 44,882 (the exact share of the band that D's frozen rule routes to the LLM) it also takes one B call latency drawn at
random from the pooled B values, added to C's. Writes <run>/options/latency/d_latency_mixture.json and .txt, the pooled B
latencies (b_call_latencies.csv) and C's per-pair values (c_per_pair_ms.npz) so the simulation can be repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d_hybrid as D  # noqa: E402
import latency_bench as LB  # noqa: E402

B_SOURCES = [("sample5000", "sample5000/calls.jsonl"), ("tune2", "tune2/calls.jsonl"), ("decoycheck", "decoycheck/calls.jsonl"),
             ("dfull", "dfull/calls.jsonl"), ("concurrency, 2 workers", "concurrency/level2/calls.jsonl")]
PCTS = (50, 95, 99, 99.9, 99.95, 99.99)


def b_latencies(run: Path) -> tuple[list[tuple[str, float]], dict]:
    base = run / "options" / "b_llm" / "openai"
    rows, per_source = [], {}
    for name, rel in B_SOURCES:
        f = base / rel
        n = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("error") or c.get("latency_s") is None or c.get("mode") != "live":
                continue
            rows.append((name, float(c["latency_s"])))
            n += 1
        per_source[name] = n
    return rows, per_source


def stats(x_ms: np.ndarray) -> dict:
    out = {f"p{q:g}_ms": round(float(np.percentile(x_ms, q)), 4) for q in PCTS}
    out["max_ms"] = round(float(x_ms.max()), 4)
    out["mean_ms"] = round(float(x_ms.mean()), 4)
    return out


def simulate(c_ms: np.ndarray, b_ms: np.ndarray, share: float, draws: int, seed: int) -> dict:
    """draws pairs: C latency, plus one B call with probability share."""
    rng = np.random.default_rng(seed)
    c = c_ms[rng.integers(0, len(c_ms), draws)]
    fallback = rng.random(draws) < share
    b = b_ms[rng.integers(0, len(b_ms), draws)]
    d = c + np.where(fallback, b, 0.0)
    return {"d": d, "c": c, "fallback_draws": int(fallback.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--draws", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-arrays", action="store_true", help="redo the simulation from the saved c_per_pair_ms.npz (no new C measurement)")
    args = ap.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("OMP_NUM_THREADS must be 1")
    run = args.run
    P = D.paths(run)
    lo, hi = D.frozen_lo_hi(P)
    b_answers = {k: v[0] for k, v in D.merge_b(D.b_paths(run)).items()}
    npz = run / "options" / "latency" / "c_per_pair_ms.npz"
    if args.reuse_arrays:
        reps = [r.astype(float) for r in np.load(npz)["repeats"]]
        n_pairs_band = len(reps[0])
        rows = None
    else:
        rows = LB.load_rows(run)
        model = LB.load_model(run, "c_v1_lgbm")
        LB.time_pairs(rows[:200], model, lo, hi, b_answers)
        reps = []
        for k in range(args.repeats):
            c_s, _, _, _ = LB.time_pairs(rows, model, lo, hi, b_answers)
            reps.append(np.array(c_s) * 1000.0)
            print(f"C repeat {k + 1}: p50 {np.percentile(reps[-1], 50):.4f} ms, p95 {np.percentile(reps[-1], 95):.4f} ms, max {reps[-1].max():.3f} ms", flush=True)
        n_pairs_band = len(rows)
    for k, r in enumerate(reps, 1):
        print(f"C repeat {k}: p50 {np.percentile(r, 50):.4f} ms, p95 {np.percentile(r, 95):.4f} ms, max {r.max():.3f} ms (used)", flush=True) if args.reuse_arrays else None
    c_ms = np.concatenate(reps)
    brows, per_source = b_latencies(run)
    b_ms = np.array([x for _, x in brows]) * 1000.0
    routed = sum(1 for _ in (P["dir"] / "routed_fullband_pairs.csv").open(encoding="utf-8")) - 1
    n_band = n_pairs_band
    share = routed / n_band
    sim = simulate(c_ms, b_ms, share, args.draws, args.seed)
    out = run / "options" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    if not args.reuse_arrays:
        np.savez_compressed(out / "c_per_pair_ms.npz", repeats=np.array(reps, dtype=np.float32))
    with (out / "b_call_latencies.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "latency_s"])
        w.writerows([(n, f"{x:.4f}") for n, x in brows])
    ppr = 20.0854
    res = {"kind": "simulation from measured distributions (not a direct measurement)", "draws": args.draws, "seed": args.seed,
           "fallback_share": share, "fallback_share_text": f"{routed} / {n_band:,}", "fallback_draws": sim["fallback_draws"],
           "c_pairs_pooled": int(len(c_ms)), "c_repeats": len(reps), "c_pooled": stats(c_ms), "b_calls_pooled": int(len(b_ms)),
           "b_calls_by_source": per_source, "b_pooled": stats(b_ms), "d": stats(sim["d"]), "c_only_draws": stats(sim["c"]),
           "per_record": {"band_pairs_per_record": ppr, "b_calls_per_record": ppr, "d_expected_llm_calls_per_record": round(ppr * share, 4),
                          "d_records_with_any_llm_call_pct": round(100 * (1 - (1 - share) ** ppr), 2),
                          "b_seconds_per_record_at_10_workers_mean": round(ppr / 10 * 1.041, 2), "b_seconds_per_record_at_10_workers_whole_rounds": round(np.ceil(ppr / 10) * 1.041, 2),
                          "b_p50_s_at_10_workers": 1.041},
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "d_latency_mixture.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = ["Per-pair latency of D including the LLM fallback: a SIMULATION from measured distributions (not a direct measurement)",
         f"  C: {len(c_ms):,} measured per-pair latencies ({len(reps)} repeats x {n_band:,} test band pairs), one core, features included",
         f"  B: {len(b_ms):,} measured live call latencies at one thread level (2 workers): " + ", ".join(f"{k} {v:,}" for k, v in per_source.items()),
         f"  fallback: probability {routed} / {n_band:,} = {share:.6f} per pair (the exact routed share); {args.draws:,} draws, seed {args.seed}; "
         f"{sim['fallback_draws']:,} draws took a B call",
         "", f"  {'':<22}" + "".join(f"{('p' + format(q, 'g')):>10}" for q in PCTS) + f"{'max':>12}{'mean':>10}   (ms)"]
    for name, st in (("C alone (measured)", res["c_pooled"]), ("B call (measured)", res["b_pooled"]), ("D, simulated", res["d"])):
        L.append(f"  {name:<22}" + "".join(f"{st[f'p{q:g}_ms']:>10,.3f}" for q in PCTS) + f"{st['max_ms']:>12,.1f}{st['mean_ms']:>10,.3f}")
    L.append("")
    L.append(f"  The fallback is {100 * share:.3f}% of pairs, so p50, p95 and p99 are C's latency; p99.9 falls inside the fallback share (0.085% of draws "
             f"take a call, 0.1% is the top 1,000 draws), so it is C's extreme tail (the slowest of the non-fallback draws), not an LLM latency; "
             f"p99.95 and above show the LLM call (about 1 second).")
    pr = res["per_record"]
    L.append(f"  Per record: a record has about {ppr:.1f} band pairs. B makes about {ppr:.0f} LLM calls per record: at 10 workers (measured p50 1.041 s per call) "
             f"that is about {pr['b_seconds_per_record_at_10_workers_mean']} s per record ({ppr:.1f} calls / 10 workers x 1.041 s), or {pr['b_seconds_per_record_at_10_workers_whole_rounds']} s "
             f"counting whole rounds of 10 calls; queueing is not included.")
    L.append(f"  D makes {pr['d_expected_llm_calls_per_record']} LLM calls per record on average; {pr['d_records_with_any_llm_call_pct']}% of records need any LLM call, "
             f"and those take about one call (about 1 s) on top of C's {ppr:.0f} x {res['c_pooled']['mean_ms']:.3f} ms.")
    text = "\n".join(L) + "\n"
    (out / "d_latency_mixture.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

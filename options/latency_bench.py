"""
Per-pair latency of option C and option D on the test band pairs, single core, features included, repeated; then a
10-process parallel test.

  OMP_NUM_THREADS=1 python options/latency_bench.py results/dev_2026_run11 [--version c_v1_lgbm] [--repeats 3] [--processes 10]

C per pair: build the feature vector from the scored_pairs.csv row (parse the JSON fields), predict one row with the
saved model, apply the isotonic calibration and the cutoff (the same code path as options/c_latency.py; reading the
file is not timed). D per pair = the C time plus the routing step: the frozen rule (options/d_hybrid.py route_of with
the lo and hi of frozen_settings.json) on C's calibrated p, and for a pair routed to the LLM the lookup of B's recorded
answer instead of a call (no LLM latency is included: D's LLM calls are asynchronous and measured separately).
Single-core runs: `repeats` full passes over all test band pairs in one process, each reports p50, p95, max and mean per
pair; the summary is the median across repeats of each statistic. Every decision is checked against C's decisions.csv.

Parallel test: `processes` worker processes, one thread each (the caller sets OMP_NUM_THREADS=1), each loads its own model
and takes a disjoint contiguous slice of the band; they start together (barrier) and each times its pairs one at a time.
Reported: total pairs per second (all pairs / wall time of the slowest worker) and per-pair p95 under load (pooled over all
workers), for C and for D.

Writes <run>/options/latency/latency_summary.json and latency_summary.txt (and nothing large).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verity" / "src"))
from c_features import FeatureBuilder  # noqa: E402
from config import host_info  # noqa: E402
import d_hybrid as D  # noqa: E402


def load_rows(run: Path):
    csv.field_size_limit(10**8)
    side = {r["record_id"]: r["side"] for r in csv.DictReader((run / "split_records.csv").open(newline="", encoding="utf-8"))}
    rows = []
    with (run / "verity_data" / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["decision_status"] == "review" and side[r["record_a"]] == side[r["record_b"]] == "test":
                rows.append(r)
    return rows


def load_model(run: Path, version: str):
    data, vdir = run / "verity_data", run / "options" / "c_model" / version
    meta = json.loads((vdir / "features.json").read_text(encoding="utf-8"))
    fb = FeatureBuilder(json.loads((data / "scored_pairs_columns.json").read_text(encoding="utf-8")))
    cal = json.loads((vdir / "calibrator.json").read_text(encoding="utf-8"))
    cx, cy = np.array(cal["x"]), np.array(cal["y"])
    cutoff = float(meta["cutoff"])
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(vdir / "model.txt"))
    return fb, booster, cx, cy, cutoff


def time_pairs(rows, model, lo, hi, b_answers):
    """Per pair: seconds for C (features + predict + calibrate + decide) and for the routing step; plus C's decisions
    and D's decisions."""
    fb, booster, cx, cy, cutoff = model
    c_s, r_s, c_dec, d_dec = [], [], [], []
    pc = time.perf_counter
    for r in rows:
        a = pc()
        x = np.array(fb.features(r), dtype=float)
        p = float(np.interp(float(booster.predict(x.reshape(1, -1), num_threads=1)[0]), cx, cy))
        decision = "same" if p >= cutoff else "different"
        b = pc()
        route = D.route_of(p, lo, hi)
        if route == "llm":
            dd = b_answers[(min(r["record_a"], r["record_b"]), max(r["record_a"], r["record_b"]))]
        else:
            dd = "same" if route == "c_same" else "different"
        c = pc()
        c_s.append(b - a)
        r_s.append(c - b)
        c_dec.append(decision)
        d_dec.append(dd)
    return c_s, r_s, c_dec, d_dec


def stats_ms(seconds) -> dict:
    ms = np.array(seconds) * 1000.0
    return {"p50_ms": round(float(np.percentile(ms, 50)), 4), "p95_ms": round(float(np.percentile(ms, 95)), 4),
            "max_ms": round(float(ms.max()), 4), "mean_ms": round(float(ms.mean()), 4), "pairs": int(len(ms))}


def worker(args):
    idx, run, version, lo, hi, b_answers, rows, barrier, q = args
    model = load_model(Path(run), version)
    time_pairs(rows[:20], model, lo, hi, b_answers)                  # warm the code paths (not reported)
    barrier.wait()
    t0 = time.perf_counter()
    c_s, r_s, c_dec, d_dec = time_pairs(rows, model, lo, hi, b_answers)
    wall = time.perf_counter() - t0
    q.put((idx, wall, c_s, r_s, c_dec, d_dec))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--version", default="c_v1_lgbm")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--processes", type=int, default=10)
    ap.add_argument("--parallel-repeats", type=int, default=3)
    args = ap.parse_args()
    run = args.run
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp != "1":
        raise SystemExit("OMP_NUM_THREADS must be 1 (single core per process)")
    P = D.paths(run)
    lo, hi = D.frozen_lo_hi(P)
    b_answers = {k: v[0] for k, v in D.merge_b(D.b_paths(run)).items()}
    rows = load_rows(run)
    expected = {(r["record_a"], r["record_b"]): r["decision"] for r in csv.DictReader((run / "options" / "c_model" / args.version / "decisions.csv").open(newline="", encoding="utf-8"))}
    full_d = {}
    fp = P["dir"] / "decisions_fullband.csv"
    if fp.exists():
        full_d = {(r["record_a"], r["record_b"]): r["decision"] for r in D.read_csv(fp)}
    n = len(rows)
    print(f"{n:,} test band pairs; OMP_NUM_THREADS={omp}; lo {lo:g}, hi {D.fmt_hi(hi)}; {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    model = load_model(run, args.version)
    time_pairs(rows[:200], model, lo, hi, b_answers)                  # warm-up pass, not reported
    reps_c, reps_d, reps_r, mism_c, mism_d = [], [], [], 0, 0
    for k in range(args.repeats):
        t0 = time.perf_counter()
        c_s, r_s, c_dec, d_dec = time_pairs(rows, model, lo, hi, b_answers)
        wall = time.perf_counter() - t0
        d_s = [a + b for a, b in zip(c_s, r_s)]
        for r, cd, dd in zip(rows, c_dec, d_dec):
            key = (r["record_a"], r["record_b"])
            mism_c += cd != expected[key]
            if full_d:
                mism_d += dd != full_d[key]
        sc, sd, sr = stats_ms(c_s), stats_ms(d_s), stats_ms(r_s)
        sc["total_s"] = sd["total_s"] = round(wall, 3)
        reps_c.append(sc)
        reps_d.append(sd)
        reps_r.append(sr)
        print(f"repeat {k + 1}: C p50 {sc['p50_ms']} ms, p95 {sc['p95_ms']} ms, max {sc['max_ms']} ms, mean {sc['mean_ms']} ms | "
              f"D p50 {sd['p50_ms']}, p95 {sd['p95_ms']}, max {sd['max_ms']}, mean {sd['mean_ms']} | routing step mean {sr['mean_ms']} ms | pass {wall:.1f} s", flush=True)

    def med(reps):
        return {k: round(statistics.median(r[k] for r in reps), 4) for k in ("p50_ms", "p95_ms", "max_ms", "mean_ms")}

    summary = {"pairs": n, "omp_num_threads": omp, "host": host_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "frozen_lo_hi": [lo, hi], "decision_mismatches_C_vs_decisions_csv": int(mism_c),
               "decision_mismatches_D_vs_decisions_fullband_csv": int(mism_d) if full_d else None,
               "C": {"repeats": reps_c, "median_of_repeats": med(reps_c)}, "D": {"repeats": reps_d, "median_of_repeats": med(reps_d)},
               "routing_step": {"repeats": reps_r, "median_of_repeats": med(reps_r)},
               "single_core_pairs_per_second_C": round(1000.0 / med(reps_c)["mean_ms"], 1),
               "single_core_pairs_per_second_D": round(1000.0 / med(reps_d)["mean_ms"], 1)}

    # parallel test: `repeats` passes (a pass takes under a second per worker), each with fresh worker processes
    P_n = args.processes
    ctx = mp.get_context("spawn")
    size = n // P_n
    slices = [rows[i * size:(i + 1) * size if i < P_n - 1 else n] for i in range(P_n)]
    assert sum(len(s_) for s_ in slices) == n

    def one_parallel_pass() -> dict:
        barrier, q = ctx.Barrier(P_n), ctx.Queue()
        procs = [ctx.Process(target=worker, args=((i, str(run), args.version, lo, hi, b_answers, slices[i], barrier, q),)) for i in range(P_n)]
        for pr in procs:
            pr.start()
        results = [q.get() for _ in procs]
        for pr in procs:
            pr.join()
        total_wall = max(r[1] for r in results)
        pooled_c = [x for r in results for x in r[2]]
        pooled_r = [x for r in results for x in r[3]]
        pooled_d = [a_ + b_ for a_, b_ in zip(pooled_c, pooled_r)]
        mism = 0
        by_idx = {r[0]: r for r in results}
        for i in range(P_n):
            for row, cd in zip(slices[i], by_idx[i][4]):
                mism += cd != expected[(row["record_a"], row["record_b"])]
        sc, sd = stats_ms(pooled_c), stats_ms(pooled_d)
        return {"processes": P_n, "threads_per_process": 1, "pairs": n, "wall_s": round(total_wall, 3),
                "pairs_per_second_total": round(n / total_wall, 1), "p50_ms_under_load": sc["p50_ms"], "p95_ms_under_load": sc["p95_ms"],
                "max_ms_under_load": sc["max_ms"], "mean_ms_under_load": sc["mean_ms"],
                "per_process_p95_ms": sorted(round(float(np.percentile(np.array(r[2]) * 1000, 95)), 4) for r in results),
                "worker_wall_s": sorted(round(r[1], 2) for r in results), "decision_mismatches_C": int(mism),
                "D": {"pairs_per_second_total": round(n / total_wall, 1), "p50_ms_under_load": sd["p50_ms"], "p95_ms_under_load": sd["p95_ms"],
                      "max_ms_under_load": sd["max_ms"], "mean_ms_under_load": sd["mean_ms"]}}

    passes = [one_parallel_pass() for _ in range(args.parallel_repeats)]
    for i, ps in enumerate(passes, 1):
        print(f"parallel pass {i}: {ps['pairs_per_second_total']} pairs per second, wall {ps['wall_s']} s, C p95 {ps['p95_ms_under_load']} ms", flush=True)
    med_pass = sorted(passes, key=lambda x: x["pairs_per_second_total"])[len(passes) // 2]
    summary["parallel"] = {**med_pass, "reported": "the median pass by pairs per second", "passes": [{k: v[k] for k in ("pairs_per_second_total", "wall_s", "p95_ms_under_load", "p50_ms_under_load")} for v in passes],
                           "scaling_vs_one_core": round(med_pass["pairs_per_second_total"] / summary["single_core_pairs_per_second_C"], 2)}
    out = run / "options" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    L = [f"Per-pair latency on {n:,} test band pairs, features included, one pair at a time (OMP_NUM_THREADS=1), {args.repeats} repeats",
         f"  {'':<10}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}{'mean ms':>9}"]
    for name, reps, key in (("C", reps_c, "C"), ("D", reps_d, "D"), ("routing", reps_r, "routing_step")):
        for i, r in enumerate(reps, 1):
            L.append(f"  {name + ' r' + str(i):<10}{r['p50_ms']:>9}{r['p95_ms']:>9}{r['max_ms']:>9}{r['mean_ms']:>9}")
        m = summary[key]["median_of_repeats"]
        L.append(f"  {name + ' median':<10}{m['p50_ms']:>9}{m['p95_ms']:>9}{m['max_ms']:>9}{m['mean_ms']:>9}")
    L.append(f"  single core: {summary['single_core_pairs_per_second_C']} pairs per second (C), {summary['single_core_pairs_per_second_D']} (D)")
    L.append(f"  decision mismatches: C vs decisions.csv {mism_c}; D vs decisions_fullband.csv {summary['decision_mismatches_D_vs_decisions_fullband_csv']}")
    pl = summary["parallel"]
    L.append(f"\nParallel: {P_n} processes, one thread each, disjoint slices of {size:,} pairs; wall {pl['wall_s']} s; {pl['pairs_per_second_total']} pairs per second in total "
             f"({pl['scaling_vs_one_core']}x one core)")
    L.append(f"  per-pair under load, pooled: C p50 {pl['p50_ms_under_load']} ms, p95 {pl['p95_ms_under_load']} ms, max {pl['max_ms_under_load']} ms; "
             f"D p50 {pl['D']['p50_ms_under_load']}, p95 {pl['D']['p95_ms_under_load']}, max {pl['D']['max_ms_under_load']}")
    L.append(f"  per-process p95 (C), sorted: {pl['per_process_p95_ms']}; decision mismatches {pl['decision_mismatches_C']}")
    text = "\n".join(L) + "\n"
    (out / "latency_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())

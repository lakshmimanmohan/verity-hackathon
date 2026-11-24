"""
Option C latency, single core, per test band pair (protocol bar: p95 <= 20 ms, feature
extraction included).

  OMP_NUM_THREADS=1 python options/c_latency.py results/<run> --version c_v1_lgbm

Loads the saved model and calibrator from results/<run>/options/c_model/<version>/, reads the
test band rows of scored_pairs.csv (reading the file is not timed), then for each pair times:
features (parse the row's JSON fields and build the feature vector) and predict (model
probability for one row, isotonic calibration, decision). LightGBM runs with num_threads=1.
Checks that every decision equals decisions.csv. Writes timings.json in the shared timing
format, with the host block.
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
sys.path.insert(0, str(HERE.parent / "verity" / "src"))
from c_features import FeatureBuilder  # noqa: E402
from config import host_info, record_timing, recorded_timings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    csv.field_size_limit(10**8)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        print("WARNING: OMP_NUM_THREADS is not 1; the bar is measured on a single core")
    t0 = time.perf_counter()
    run, data = args.run, args.run / "verity_data"
    vdir = run / "options" / "c_model" / args.version
    meta = json.loads((vdir / "features.json").read_text(encoding="utf-8"))
    side = {}
    with (run / "split_records.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            side[r["record_id"]] = r["side"]
    rows = []
    with (data / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["decision_status"] == "review" and side[r["record_a"]] == side[r["record_b"]] == "test":
                rows.append(r)
    expected = {}
    with (vdir / "decisions.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            expected[(r["record_a"], r["record_b"])] = r["decision"]
    t_read = time.perf_counter() - t0

    t_load = time.perf_counter()
    fb = FeatureBuilder(json.loads((data / "scored_pairs_columns.json").read_text(encoding="utf-8")))
    cal = json.loads((vdir / "calibrator.json").read_text(encoding="utf-8"))
    cx, cy = np.array(cal["x"]), np.array(cal["y"])
    cutoff = float(meta["cutoff"])
    if meta["kind"] == "lgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(vdir / "model.txt"))

        def prob(x: np.ndarray) -> float:
            return float(booster.predict(x.reshape(1, -1), num_threads=1)[0])
    else:
        m = json.loads((vdir / "model.json").read_text(encoding="utf-8"))
        mean, scale = np.array(m["mean"]), np.array(m["scale"])
        coef, icpt = np.array(m["coef"]), float(m["intercept"])

        def prob(x: np.ndarray) -> float:
            nan = np.isnan(x)
            z = np.hstack([np.nan_to_num(x, nan=0.0), nan.astype(float)])
            v = float(((z - mean) / scale) @ coef + icpt)
            return 1.0 / (1.0 + np.exp(-v))
    load_s = time.perf_counter() - t_load

    feat_s, pred_s, pair_s = [], [], []
    mismatches = 0
    t_loop = time.perf_counter()
    for r in rows:
        a = time.perf_counter()
        x = np.array(fb.features(r), dtype=float)
        b = time.perf_counter()
        p = float(np.interp(prob(x), cx, cy))
        decision = "same" if p >= cutoff else "different"
        c = time.perf_counter()
        feat_s.append(b - a)
        pred_s.append(c - b)
        pair_s.append(c - a)
        if decision != expected[(r["record_a"], r["record_b"])]:
            mismatches += 1
    loop_s = time.perf_counter() - t_loop
    total_s = time.perf_counter() - t0

    record_timing("optionC.read_inputs", t_read, note="scored_pairs.csv, split, decisions.csv (not part of the bar)")
    record_timing("optionC.model_load", load_s, note=f"{meta['model']}, calibrator, feature builder")
    record_timing("optionC.per_pair", loop_s, items=len(rows), item="pair", per_item_s=pair_s,
                  note="features + predict + calibrate + decide, one pair at a time")
    record_timing("optionC.features", float(sum(feat_s)), parent="optionC.per_pair", items=len(rows),
                  item="pair", per_item_s=feat_s, note="parse JSON fields, build 72 features")
    record_timing("optionC.predict", float(sum(pred_s)), parent="optionC.per_pair", items=len(rows),
                  item="pair", per_item_s=pred_s, note="single-row model probability, isotonic, cutoff")
    record_timing("optionC.per_pair.other", max(0.0, loop_s - sum(pair_s)), parent="optionC.per_pair",
                  note="loop overhead")
    record_timing("optionC.total", total_s)
    tim = recorded_timings()
    pp = next(t for t in tim if t["stage"] == "optionC.per_pair")["per_item_ms"]
    doc = {"option": "C", "model": meta["model"], "host": host_info(), "timings": tim,
           "timing_summary": {"per_pair_p50_ms": pp["p50"], "per_pair_p95_ms": pp["p95"],
                              "per_pair_max_ms": pp["max"], "bar_p95_ms": 20.0,
                              "meets_bar": pp["p95"] <= 20.0, "pairs": len(rows),
                              "decision_mismatches_vs_decisions_csv": mismatches}}
    (vdir / "timings.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"{meta['model']}: {len(rows)} pairs, per pair p50 {pp['p50']:.4f} ms, p95 {pp['p95']:.4f} ms, "
          f"max {pp['max']:.4f} ms (bar p95 <= 20 ms: {'PASS' if pp['p95'] <= 20 else 'FAIL'}); "
          f"decision mismatches vs decisions.csv: {mismatches}; OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

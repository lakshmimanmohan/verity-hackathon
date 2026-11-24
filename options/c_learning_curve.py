"""
Option C learning curve: LightGBM trained on nested subsets of the train band pairs
(n = 100, 250, 500, 1,000, 2,000) with a 25% true share (at most all true pairs), plus all
train band pairs (the c_v1_lgbm model). Each point is scored once on the full test band with
eval/score_option.py. These are reported results only; nothing is chosen from them.

  v1 (one sampling seed):  python options/c_learning_curve.py results/<run>
  v2 (several seeds):      python options/c_learning_curve.py results/<run> --seeds 0 1 2 3 4 --subdir lc_v2

Subsets are nested within each sampling seed; folds, model seed and grid stay at seed 0.
v1 writes c_model/lc_n<n>/ and c_model/learning_curve.{csv,png}. With --subdir, everything goes
under c_model/<subdir>/: s<seed>_n<n>/ per point, learning_curve_points.csv (every point),
learning_curve.csv (per size: mean, min, max of band F1 and band average precision) and
learning_curve.png (mean with min-max range). Plots label each point with total and true
training labels, e.g. "500 (125 true)", and draw A's band F1 (and B's when its score report
exists) as reference lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
F1_RE = re.compile(r"F1 ([0-9.]+)\s+95% CI \[([0-9.]+), ([0-9.]+)\]")
OVERALL_RE = re.compile(r"Overall .*\n\s+precision ([0-9.]+)\s+recall ([0-9.]+)")
AP_RE = re.compile(r"Band average precision \(cutoff-free[^:]*: ([0-9.]+)")


def parse_report(text: str) -> dict:
    f1 = F1_RE.search(text)
    ov = OVERALL_RE.search(text)
    ap = AP_RE.search(text)
    return {"band_f1": float(f1.group(1)), "ci_low": float(f1.group(2)), "ci_high": float(f1.group(3)),
            "band_ap": float(ap.group(1)) if ap else None,
            "overall_precision": float(ov.group(1)), "overall_recall": float(ov.group(2))}


def run(cmd: list[str]) -> str:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0")
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", env=env)
    if res.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    return res.stdout


def train_counts(cv_text: str) -> tuple[int, int]:
    m = re.search(r"Training set: (\d+) pairs \((\d+) true", cv_text)
    return int(m.group(1)), int(m.group(2))


def reference_lines(ax, run_dir: Path) -> None:
    a_rep = run_dir / "options" / "a_threshold" / "score_report.txt"
    if a_rep.is_file():
        a = parse_report(a_rep.read_text(encoding="utf-8"))
        ax.axhline(a["band_f1"], color="#888888", linestyle="--", linewidth=1,
                   label=f"A threshold only ({a['band_f1']:.3f})")
    b_rep = run_dir / "options" / "b_llm" / "score_report.txt"
    if b_rep.is_file():
        b = parse_report(b_rep.read_text(encoding="utf-8"))
        ax.axhline(b["band_f1"], color="#c0392b", linestyle="-.", linewidth=1.2,
                   label=f"B LLM baseline ({b['band_f1']:.3f})")
        ax.axhspan(b["band_f1"] - 0.02, b["band_f1"] + 0.02, color="#c0392b", alpha=0.08)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 2000])
    ap.add_argument("--true-share", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--subdir", default="", help="Write under c_model/<subdir>/ (required with more than one seed)")
    args = ap.parse_args()
    if len(args.seeds) > 1 and not args.subdir:
        raise SystemExit("--subdir is required with more than one seed")
    py = sys.executable
    cdir = args.run / "options" / "c_model"
    odir = cdir / args.subdir if args.subdir else cdir
    odir.mkdir(parents=True, exist_ok=True)

    points = []
    for seed in args.seeds:
        for n in args.sizes:
            ver = f"{args.subdir}/s{seed}_n{n}" if args.subdir else f"lc_n{n}"
            run([py, "options/c_decision_model.py", str(args.run), "--model", "lgbm", "--version", ver,
                 "--n", str(n), "--true-share", str(args.true_share), "--sample-seed", str(seed)])
            vdir = cdir / ver
            rep = run([py, "eval/score_option.py", str(args.run), "--decisions", str(vdir / "decisions.csv"),
                       "--name", f"C learning curve seed {seed} n={n}"])
            (vdir / "score_report.txt").write_text(rep, encoding="utf-8")
            meta = json.loads((vdir / "features.json").read_text(encoding="utf-8"))
            tot, tr = train_counts((vdir / "cv_report.txt").read_text(encoding="utf-8"))
            points.append({"version": ver, "seed": seed, "n": tot, "n_true": tr, "k_folds": meta["k_folds"],
                           "cutoff": meta["cutoff"], **parse_report(rep)})
            print(f"{ver}: F1 {points[-1]['band_f1']:.4f}  AP {points[-1]['band_ap']}")

    # "All train band pairs" is the c_v1_lgbm model (one point, independent of the sampling seed).
    allv = cdir / "c_v1_lgbm"
    rep = run([py, "eval/score_option.py", str(args.run), "--decisions", str(allv / "decisions.csv"),
               "--name", "C v1 LightGBM (all train band pairs)"])
    (odir / "all_score_report.txt").write_text(rep, encoding="utf-8")
    meta = json.loads((allv / "features.json").read_text(encoding="utf-8"))
    tot, tr = train_counts((allv / "cv_report.txt").read_text(encoding="utf-8"))
    points.append({"version": "c_v1_lgbm", "seed": None, "n": tot, "n_true": tr, "k_folds": meta["k_folds"],
                   "cutoff": meta["cutoff"], **parse_report(rep)})

    pfile = odir / ("learning_curve_points.csv" if args.subdir else "learning_curve.csv")
    with pfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        w.writeheader()
        w.writerows(points)

    # Per-size summary across seeds.
    sizes = sorted({(p["n"], p["n_true"]) for p in points})
    summary = []
    for n, nt in sizes:
        ps = [p for p in points if p["n"] == n]
        f1s = [p["band_f1"] for p in ps]
        aps = [p["band_ap"] for p in ps if p["band_ap"] is not None]
        summary.append({"n": n, "n_true": nt, "seeds": len(ps),
                        "f1_mean": statistics.mean(f1s), "f1_min": min(f1s), "f1_max": max(f1s),
                        "ap_mean": statistics.mean(aps) if aps else None,
                        "ap_min": min(aps) if aps else None, "ap_max": max(aps) if aps else None,
                        "overall_recall_mean": statistics.mean(p["overall_recall"] for p in ps),
                        "overall_precision_min": min(p["overall_precision"] for p in ps)})
    if args.subdir:
        with (odir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [s["n"] for s in summary]
    ax.plot(xs, [s["f1_mean"] for s in summary], marker="o", label="C v1 LightGBM band F1 (mean over seeds)")
    ax.fill_between(xs, [s["f1_min"] for s in summary], [s["f1_max"] for s in summary], alpha=0.2,
                    label="band F1 min-max over seeds")
    if all(s["ap_mean"] is not None for s in summary):
        ax.plot(xs, [s["ap_mean"] for s in summary], marker="s", linestyle=":",
                label="band average precision (mean, cutoff-free)")
    for s in summary:
        ax.annotate(f"{s['n']:,} ({s['n_true']} true)", (s["n"], s["f1_mean"]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7)
    reference_lines(ax, args.run)
    ax.set_xscale("log")
    ax.set_xlabel("labelled train band pairs (log scale)")
    ax.set_ylabel("on the full test band")
    ax.set_ylim(0, 1.02)
    ax.set_title("Option C learning curve" + (f" ({len(args.seeds)} sampling seeds)" if len(args.seeds) > 1 else ""))
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, f"Subsets nested within each sampling seed ({', '.join(map(str, args.seeds))}); "
             f"{args.true_share:.0%} true labels (at most all 418 true train band pairs); the last point is all "
             "train band pairs (one model). Reported only; nothing is chosen from these points.", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(odir / "learning_curve.png", dpi=120)
    print(f"wrote {pfile}, {odir / 'learning_curve.png'}" + (f", {odir / 'learning_curve.csv'}" if args.subdir else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

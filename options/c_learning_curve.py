"""
Option C learning curve: LightGBM trained on nested, seed-0 subsets of the train band pairs
(n = 100, 250, 500, 1,000, 2,000) with a 25% true share (at most all true pairs), plus all
train band pairs (the c_v1_lgbm model). Each point is scored once on the full test band with
eval/score_option.py.

  python options/c_learning_curve.py results/<run> [--sizes 100 250 500 1000 2000] [--true-share 0.25]

Writes results/<run>/options/c_model/learning_curve.csv and learning_curve.png; the plot
labels each point with total and true training labels, e.g. "500 (125 true)", draws A's band
F1 as a reference line and B's when results/<run>/options/b_llm/score_report.txt exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
F1_RE = re.compile(r"F1 ([0-9.]+)\s+95% CI \[([0-9.]+), ([0-9.]+)\]")
OVERALL_RE = re.compile(r"Overall .*\n\s+precision ([0-9.]+)\s+recall ([0-9.]+)")


def parse_report(text: str) -> dict:
    f1 = F1_RE.search(text)
    ov = OVERALL_RE.search(text)
    return {"band_f1": float(f1.group(1)), "ci_low": float(f1.group(2)), "ci_high": float(f1.group(3)),
            "overall_precision": float(ov.group(1)), "overall_recall": float(ov.group(2))}


def run(cmd: list[str]) -> str:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0")
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", env=env)
    if res.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    return res.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 2000])
    ap.add_argument("--true-share", type=float, default=0.25)
    args = ap.parse_args()
    py = sys.executable
    cdir = args.run / "options" / "c_model"
    points = []
    for n in args.sizes:
        ver = f"lc_n{n}"
        run([py, "options/c_decision_model.py", str(args.run), "--model", "lgbm", "--version", ver,
             "--n", str(n), "--true-share", str(args.true_share)])
        meta = json.loads((cdir / ver / "features.json").read_text(encoding="utf-8"))
        rep = run([py, "eval/score_option.py", str(args.run), "--decisions", str(cdir / ver / "decisions.csv"),
                   "--name", f"C learning curve n={n}"])
        (cdir / ver / "score_report.txt").write_text(rep, encoding="utf-8")
        cv = (cdir / ver / "cv_report.txt").read_text(encoding="utf-8")
        m = re.search(r"Training set: (\d+) pairs \((\d+) true", cv)
        points.append({"version": ver, "n": int(m.group(1)), "n_true": int(m.group(2)), "k_folds": meta["k_folds"],
                       "cutoff": meta["cutoff"], **parse_report(rep)})
        print(f"{ver}: {points[-1]}")
    # The "all" point is the official c_v1_lgbm model.
    allrep = (cdir / "c_v1_lgbm" / "score_report.txt").read_text(encoding="utf-8")
    cv = (cdir / "c_v1_lgbm" / "cv_report.txt").read_text(encoding="utf-8")
    m = re.search(r"Training set: (\d+) pairs \((\d+) true", cv)
    meta = json.loads((cdir / "c_v1_lgbm" / "features.json").read_text(encoding="utf-8"))
    points.append({"version": "c_v1_lgbm", "n": int(m.group(1)), "n_true": int(m.group(2)),
                   "k_folds": meta["k_folds"], "cutoff": meta["cutoff"], **parse_report(allrep)})

    with (cdir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        w.writeheader()
        w.writerows(points)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [p["n"] for p in points]
    ys = [p["band_f1"] for p in points]
    ax.errorbar(xs, ys, yerr=[[p["band_f1"] - p["ci_low"] for p in points], [p["ci_high"] - p["band_f1"] for p in points]],
                marker="o", capsize=3, label="C v1 LightGBM (band F1, 95% CI)")
    for p in points:
        ax.annotate(f"{p['n']:,} ({p['n_true']} true)", (p["n"], p["band_f1"]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7)
    a_rep = args.run / "options" / "a_threshold" / "score_report.txt"
    if a_rep.is_file():
        a = parse_report(a_rep.read_text(encoding="utf-8"))
        ax.axhline(a["band_f1"], color="#888888", linestyle="--", linewidth=1,
                   label=f"A threshold only ({a['band_f1']:.3f})")
    b_rep = args.run / "options" / "b_llm" / "score_report.txt"
    if b_rep.is_file():
        b = parse_report(b_rep.read_text(encoding="utf-8"))
        ax.axhline(b["band_f1"], color="#c0392b", linestyle="-.", linewidth=1.2,
                   label=f"B LLM baseline ({b['band_f1']:.3f})")
        ax.axhspan(b["band_f1"] - 0.02, b["band_f1"] + 0.02, color="#c0392b", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel("labelled train band pairs (log scale)")
    ax.set_ylabel("band F1 on the full test band")
    ax.set_ylim(0, 1.02)
    ax.set_title("Option C learning curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, f"Subsets nested, seed 0, {args.true_share:.0%} true labels (at most all 418 true train band "
             "pairs); the last point is all train band pairs. Each point scored once on the test band.", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(cdir / "learning_curve.png", dpi=120)
    print(f"wrote {cdir / 'learning_curve.csv'} and {cdir / 'learning_curve.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

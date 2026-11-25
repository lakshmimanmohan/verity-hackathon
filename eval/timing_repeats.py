"""
Repeated Verity timing runs: save each run's numbers, delete its large files, and summarize the repeats.

  python eval/timing_repeats.py save-and-clean results/dev_2026_timing_<setting>_r<n>
  python eval/timing_repeats.py summarize [--root results] [--out results/dev_2026_timing_summary]

save-and-clean reads <folder>/verity_data/run_info.json (the last run entry), writes <folder>/timing_run.json (every
timing record, the host block, the commit, the model-cache flag) and then deletes everything in the folder except that
file, run_info.json, verity_run.log, match_candidates_check.txt and evaluator output (the large files: store,
embeddings, scored_pairs.csv, match_candidates.csv, the input csv and so on).

summarize reads the runs results/dev_2026_timing_<setting>_r1..r3 for the settings "allthreads" and "1thread" and
writes <out>/verity_timing.json and <out>/verity_timing.txt: per stage the median, min and max across repeats; cold =
the first repeat; warm = the median of the others; the model load (verity.embed.model_load) separately; the total's
spread (max - min) / median and whether the repeats agree within 10%; and whether each run's match_candidates.csv was
byte-identical to run 11's.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

KEEP = {"timing_run.json", "run_info.json", "verity_run.log", "match_candidates_check.txt", "evaluator_output.txt"}
SETTINGS = ("allthreads", "1thread")
REPEATS = 3
WITHIN = 0.10


def save_and_clean(folder: Path) -> None:
    ri = folder / "verity_data" / "run_info.json"
    run = json.loads(ri.read_text(encoding="utf-8"))["runs"][-1]
    doc = {"folder": folder.name, "git_commit": run.get("git_commit"), "git_src_uncommitted_changes": run.get("git_src_uncommitted_changes"),
           "model_cache_hit": run.get("model_cache_hit"), "nn_mode_resolved": run.get("nn_mode_resolved"), "pythonhashseed": run.get("pythonhashseed"),
           "n_records_in_index": run.get("n_records_in_index"), "host": run.get("host"), "timings": run.get("timings"),
           "timing_summary": run.get("timing_summary"), "started_utc": run.get("started_utc")}
    (folder / "timing_run.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for child in list(folder.iterdir()):
        if child.name == "verity_data":
            for g in list(child.iterdir()):
                if g.name != "run_info.json":
                    shutil.rmtree(g) if g.is_dir() else g.unlink()
        elif child.name not in KEEP:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    print(f"saved {folder / 'timing_run.json'}; large files deleted")


def stage_seconds(doc: dict) -> dict:
    return {t["stage"]: float(t["seconds"]) for t in doc["timings"]}


def summarize(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"records": None, "settings": {}}
    lines: list[str] = []
    for setting in SETTINGS:
        runs = []
        for n in range(1, REPEATS + 1):
            f = root / f"dev_2026_timing_{setting}_r{n}"
            if not (f / "timing_run.json").exists():
                raise SystemExit(f"{f}/timing_run.json is missing")
            doc = json.loads((f / "timing_run.json").read_text(encoding="utf-8"))
            chk = (f / "match_candidates_check.txt").read_text(encoding="utf-8").strip() if (f / "match_candidates_check.txt").exists() else "missing"
            runs.append((f.name, doc, chk))
        summary["records"] = runs[0][1]["n_records_in_index"]
        names = []
        for _, d, _ in runs:
            for t in d["timings"]:
                if t["stage"] not in names:
                    names.append(t["stage"])
        secs = [stage_seconds(d) for _, d, _ in runs]
        stages = {}
        for st in names:
            vals = [s.get(st) for s in secs]
            if any(v is None for v in vals):
                continue
            warm = statistics.median(vals[1:])
            stages[st] = {"values": [round(v, 3) for v in vals], "median": round(statistics.median(vals), 3), "min": round(min(vals), 3),
                          "max": round(max(vals), 3), "cold": round(vals[0], 3), "warm": round(warm, 3)}
        total = stages["verity.total"]
        spread = (total["max"] - total["min"]) / total["median"]
        host = runs[0][1]["host"]
        summary["settings"][setting] = {
            "repeats": [r[0] for r in runs], "stage": stages, "total_spread_pct": round(100 * spread, 2), "total_within_10pct": spread <= WITHIN,
            "model_load_s": stages.get("verity.embed.model_load"), "model_cache_hit": [r[1]["model_cache_hit"] for r in runs],
            "match_candidates_vs_run11": [r[2] for r in runs], "commits": sorted({r[1]["git_commit"] for r in runs}),
            "uncommitted_changes_flags": [r[1]["git_src_uncommitted_changes"] for r in runs], "host": host,
            "started_utc": [r[1]["started_utc"] for r in runs]}
        lines.append(f"== {setting}: {', '.join(r[0] for r in runs)}")
        lines.append(f"   host: {host.get('cpu_model')}, {host.get('logical_cores')} logical cores, torch_threads {host.get('torch_threads')}, "
                     f"omp_num_threads {host.get('omp_num_threads')}; commit {summary['settings'][setting]['commits']}; model_cache_hit {summary['settings'][setting]['model_cache_hit']}")
        lines.append(f"   byte comparison with run 11: {summary['settings'][setting]['match_candidates_vs_run11']}")
        lines.append(f"   {'stage':<32}{'r1 (cold)':>11}{'r2':>9}{'r3':>9}{'median':>9}{'min':>9}{'max':>9}{'warm':>9}")
        for st, v in stages.items():
            indent = "  " if st.count(".") > 1 else ""
            lines.append(f"   {indent + st:<32}{v['values'][0]:>11.2f}{v['values'][1]:>9.2f}{v['values'][2]:>9.2f}{v['median']:>9.2f}{v['min']:>9.2f}{v['max']:>9.2f}{v['warm']:>9.2f}")
        ml = stages.get("verity.embed.model_load")
        lines.append(f"   model load (verity.embed.model_load), reported separately: cold {ml['cold']:.2f} s, warm {ml['warm']:.2f} s, median {ml['median']:.2f} s" if ml else "   model load: not recorded")
        lines.append(f"   total: median {total['median']:.1f} s, min {total['min']:.1f}, max {total['max']:.1f}, cold {total['cold']:.1f}, warm {total['warm']:.1f}; "
                     f"spread (max - min) / median {100 * spread:.1f}% -> {'agree within 10%' if spread <= WITHIN else 'DO NOT agree within 10%'}")
        lines.append("")
    (out / "verity_timing.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    text = "\n".join(lines)
    (out / "verity_timing.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save-and-clean")
    s.add_argument("folder", type=Path)
    m = sub.add_parser("summarize")
    m.add_argument("--root", type=Path, default=Path("results"))
    m.add_argument("--out", type=Path, default=Path("results/dev_2026_timing_summary"))
    args = ap.parse_args()
    if args.cmd == "save-and-clean":
        save_and_clean(args.folder)
    else:
        summarize(args.root, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

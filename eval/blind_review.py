"""
Blind review set for the pairs one or more runs scored on the code path, and agreement scoring once labels are filled in. Works on any run folders: no file or
column names are assumed (everything comes from each run's column_classification.json, run_info.json, scored_pairs.csv and input CSV).

  python eval/blind_review.py <run-folder> [<run-folder> ...] --out-dir <dir> [--seed 0] [--total 40] [--max-match 20]
  python eval/blind_review.py --score --out-dir <dir>

Each <run-folder> holds verity_data/ (for example results/product_v4_raw). With several runs the pairs are pooled.

Writes into --out-dir:
  review_pairs.csv  one row per pair, shuffled: pair_id, run, record_a, record_b, then every content column of both records side by side (the union over the runs,
                    blank where a run's file has no such column), and an empty your_label column. It carries no Verity decision, score, group or match type.
  review_key.csv    pair_id, the sampling group, Verity's decision (match_type, decision_status, score, verity_match). Do not open it before labeling.

Sample (seed 0 unless changed), filled in this order until --total pairs are chosen: every code_signal match, up to --max-match; then code-path pairs whose
decision_status is review; then other code-path pairs, at random. With --max-conflict N the second step takes up to N code_conflict pairs in review instead, and
every remaining place is filled with random code-path pairs. A code-path pair is a neighbor-scored pair for which the multi-word test fails (the two records
do not each have a space in some NAME, DESCRIPTIVE or STREET_ADDRESS value), the same test match.py applies. The folder is gitignored; never commit it.

--score reads review_pairs.csv (your_label: 1/yes/same or 0/no/different; blank or anything else is skipped) and review_key.csv, and prints agreement with
Verity's decision, overall and per group. It prints counts and rates only, never a pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "verity" / "src"))
from column_classification import col_key  # noqa: E402

WRITE_TYPES = {"full_signal", "weak_signal", "embedding_only", "code_signal"}
GROUP_NAMES = ("code_signal_match", "in_review", "random_code_path", "code_conflict_review")
YES, NO = {"1", "y", "yes", "same", "true", "match"}, {"0", "n", "no", "different", "false", "non_match", "nonmatch"}
csv.field_size_limit(10**9)


def read_rows(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_inputs(run: Path, input_csv: Path | None):
    V = run / "verity_data"
    cls = json.loads((V / "column_classification.json").read_text(encoding="utf-8"))
    ri = json.loads((V / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    src = input_csv or Path(ri["input_csv"])
    with src.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    rid_col = next(h for h in header if col_key(h) == col_key(cls["record_id_column"]))
    by_id = {str(r[rid_col]).strip(): r for r in rows}
    sem = {col_key(k): str(v).strip().upper() for k, v in cls["semantic_types"].items()}
    guard_cols = [h for h in header if sem.get(col_key(h)) in ("DESCRIPTIVE", "NAME", "STREET_ADDRESS")]
    show_cols = [h for h in cls["content_columns"] if h in header]
    return V, by_id, guard_cols, show_cols


def is_code_path(a: dict, b: dict, guard_cols: list[str]) -> bool:
    """The multi-word test of match.py fails: not both records have a space in some guard column value (and a guard column exists)."""
    if not guard_cols:
        return False
    has_a = any(" " in str(a.get(c, "") or "").strip() for c in guard_cols)
    has_b = any(" " in str(b.get(c, "") or "").strip() for c in guard_cols)
    return not (has_a and has_b)


def code_path_pairs(run: Path, input_csv: Path | None) -> tuple[list[dict], list[str], list[str]]:
    """The neighbor-scored code-path pairs of one run (rows tagged with the run name and the two records' content), and its content columns."""
    V, by_id, guard_cols, show_cols = load_inputs(run, input_csv)
    out = []
    for r in read_rows(V / "scored_pairs.csv"):
        if r["pair_source"] != "neighbor":
            continue
        a, b = by_id.get(r["record_a"]), by_id.get(r["record_b"])
        if a is None or b is None or not is_code_path(a, b, guard_cols):
            continue
        out.append({**r, "run": run.name, "_a": a, "_b": b})
    return out, show_cols, guard_cols


def sample(pairs: list[dict], seed: int, total: int, max_match: int, max_conflict: int | None = None):
    pairs = sorted(pairs, key=lambda r: (r["run"], r["record_a"], r["record_b"]))
    rng = random.Random(seed)
    chosen: list[tuple[str, dict]] = []
    taken: set[tuple[str, str, str]] = set()

    def take(group: str, pool: list[dict], n: int):
        pool = [r for r in pool if (r["run"], r["record_a"], r["record_b"]) not in taken]
        for r in rng.sample(pool, max(0, min(n, len(pool)))):
            taken.add((r["run"], r["record_a"], r["record_b"]))
            chosen.append((group, r))

    take(GROUP_NAMES[0], [r for r in pairs if r["match_type"] == "code_signal" and r["decision_status"] == "match"], min(max_match, total))
    if max_conflict is None:
        take(GROUP_NAMES[1], [r for r in pairs if r["decision_status"] == "review"], total - len(chosen))
    else:                                                                       # up to max_conflict code_conflict review pairs, then random code-path pairs
        take(GROUP_NAMES[3], [r for r in pairs if r["match_type"] == "code_conflict" and r["decision_status"] == "review"], min(max_conflict, total - len(chosen)))
    take(GROUP_NAMES[2], pairs, total - len(chosen))
    rng.shuffle(chosen)
    return chosen


def write_review(runs: list[Path], input_csvs: list[Path | None], out_dir: Path, seed: int, total: int, max_match: int, max_conflict: int | None = None) -> dict:
    pairs: list[dict] = []
    show_cols: list[str] = []
    for run, inp in zip(runs, input_csvs):
        p, cols, _ = code_path_pairs(run, inp)
        pairs += p
        show_cols += [c for c in cols if c not in show_cols]
    chosen = sample(pairs, seed, total, max_match, max_conflict)
    out_dir.mkdir(parents=True, exist_ok=True)
    rid_cols = [f"{c}_{s}" for c in show_cols for s in ("a", "b")]
    with (out_dir / "review_pairs.csv").open("w", newline="", encoding="utf-8") as f, (out_dir / "review_key.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(f)
        k = csv.writer(g)
        w.writerow(["pair_id", "run", "record_a", "record_b", *rid_cols, "your_label"])
        k.writerow(["pair_id", "group", "match_type", "decision_status", "score", "verity_match"])
        for i, (group, r) in enumerate(chosen, 1):
            pid = f"P{i:02d}"
            w.writerow([pid, r["run"], r["record_a"], r["record_b"], *[x for c in show_cols for x in (r["_a"].get(c, ""), r["_b"].get(c, ""))], ""])
            k.writerow([pid, group, r["match_type"], r["decision_status"], r["score"], int(r["match_type"] in WRITE_TYPES)])
    return {"code_path_pairs_available": len(pairs), "written": len(chosen), "by_group": dict(Counter(g for g, _ in chosen)), "by_run": dict(Counter(r["run"] for _, r in chosen)),
            "out_dir": str(out_dir)}


def parse_label(x: str) -> int | None:
    x = (x or "").strip().lower()
    return 1 if x in YES else 0 if x in NO else None


def score(out_dir: Path) -> dict:
    labels = {r["pair_id"]: parse_label(r.get("your_label", "")) for r in read_rows(out_dir / "review_pairs.csv")}
    key = {r["pair_id"]: r for r in read_rows(out_dir / "review_key.csv")}
    res = {"pairs": len(key), "labeled": 0, "skipped": 0, "agree": 0, "by_group": {}, "confusion": Counter()}
    per: dict[str, Counter] = {}
    for pid, kr in key.items():
        lab = labels.get(pid)
        if lab is None:
            res["skipped"] += 1
            continue
        v = int(kr["verity_match"])
        res["labeled"] += 1
        res["agree"] += int(lab == v)
        res["confusion"][f"verity_match={v},label={lab}"] += 1
        c = per.setdefault(kr["group"], Counter())
        c["labeled"] += 1
        c["agree"] += int(lab == v)
    res["by_group"] = {g: {"labeled": c["labeled"], "agree": c["agree"], "agreement": round(c["agree"] / c["labeled"], 4)} for g, c in per.items()}
    res["agreement"] = round(res["agree"] / res["labeled"], 4) if res["labeled"] else None
    tp, fp, fn = (res["confusion"].get(f"verity_match=1,label=1", 0), res["confusion"].get("verity_match=1,label=0", 0), res["confusion"].get("verity_match=0,label=1", 0))
    res["precision_of_verity_matches"] = round(tp / (tp + fp), 4) if tp + fp else None
    res["recall_among_labeled_same"] = round(tp / (tp + fn), 4) if tp + fn else None
    res["confusion"] = dict(res["confusion"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Blind review set for code-path pairs, and agreement scoring.")
    ap.add_argument("runs", type=Path, nargs="*", help="run folders that hold verity_data/ (pooled)")
    ap.add_argument("--input-csv", type=Path, action="append", default=None, help="a run's input CSV, once per run in the same order (default: input_csv in run_info.json)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total", type=int, default=40)
    ap.add_argument("--max-match", type=int, default=20)
    ap.add_argument("--max-conflict", type=int, default=None, help="take up to this many code_conflict review pairs instead of any review pairs; the rest are random code-path pairs")
    ap.add_argument("--score", action="store_true", help="report agreement from the filled review_pairs.csv and review_key.csv")
    args = ap.parse_args()
    if args.score:
        print(json.dumps(score(args.out_dir), indent=2))
        return 0
    if not args.runs:
        ap.error("give at least one run folder")
    inputs = args.input_csv or [None] * len(args.runs)
    if len(inputs) != len(args.runs):
        ap.error("--input-csv must be given once per run folder, or not at all")
    print(json.dumps(write_review(args.runs, inputs, args.out_dir, args.seed, args.total, args.max_match, args.max_conflict), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

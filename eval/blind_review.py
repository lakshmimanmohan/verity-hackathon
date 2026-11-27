"""
Blind review set for the pairs a run scored on the code path, and agreement scoring once labels are filled in. Works on any run folder: no file or column
names are assumed (everything comes from the run's column_classification.json, run_info.json, scored_pairs.csv and the input CSV).

  python eval/blind_review.py <run-folder> [--input-csv <csv>] [--out-dir <run-folder>/review] [--seed 0] [--n-match 20] [--n-band 10] [--n-random 10]
  python eval/blind_review.py <run-folder> --score [--out-dir ...]

<run-folder> is the folder that holds verity_data/ (for example results/product_v3_raw).

Writes into --out-dir:
  review_pairs.csv  one row per pair, shuffled: pair_id, record_a, record_b, then every content column of both records side by side, and an empty your_label
                    column. It carries no Verity decision, score, group or match type.
  review_key.csv    pair_id, the sampling group, Verity's decision (match_type, decision_status, score, verity_match). Do not open it before labeling.

Sample (seed 0 unless changed): n-match pairs whose match_type is code_signal; n-band code-path pairs inside the review band (decision_status review), not already
chosen; n-random other code-path pairs, not already chosen. A code-path pair is a neighbor-scored pair for which the multi-word test fails (the two records do not
each have a space in some NAME, DESCRIPTIVE or STREET_ADDRESS value), the same test match.py applies. The folder is gitignored; never commit it.

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
GROUP_NAMES = ("code_signal_match", "near_threshold", "random_code_path")
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


def sample(V: Path, by_id: dict, guard_cols: list[str], seed: int, n_match: int, n_band: int, n_random: int):
    pairs = []
    for r in read_rows(V / "scored_pairs.csv"):
        if r["pair_source"] != "neighbor":
            continue
        a, b = by_id.get(r["record_a"]), by_id.get(r["record_b"])
        if a is None or b is None or not is_code_path(a, b, guard_cols):
            continue
        pairs.append(r)
    pairs.sort(key=lambda r: (r["record_a"], r["record_b"]))
    rng = random.Random(seed)
    chosen: list[tuple[str, dict]] = []
    taken: set[tuple[str, str]] = set()

    def take(group: str, pool: list[dict], n: int):
        pool = [r for r in pool if (r["record_a"], r["record_b"]) not in taken]
        for r in rng.sample(pool, min(n, len(pool))):
            taken.add((r["record_a"], r["record_b"]))
            chosen.append((group, r))

    take(GROUP_NAMES[0], [r for r in pairs if r["match_type"] == "code_signal"], n_match)
    take(GROUP_NAMES[1], [r for r in pairs if r["decision_status"] == "review"], n_band)
    take(GROUP_NAMES[2], pairs, n_random)
    rng.shuffle(chosen)
    return chosen, len(pairs)


def write_review(run: Path, input_csv: Path | None, out_dir: Path, seed: int, n_match: int, n_band: int, n_random: int) -> dict:
    V, by_id, guard_cols, show_cols = load_inputs(run, input_csv)
    chosen, n_code_pairs = sample(V, by_id, guard_cols, seed, n_match, n_band, n_random)
    out_dir.mkdir(parents=True, exist_ok=True)
    rid_cols = [f"{c}_{s}" for c in show_cols for s in ("a", "b")]
    with (out_dir / "review_pairs.csv").open("w", newline="", encoding="utf-8") as f, (out_dir / "review_key.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(f)
        k = csv.writer(g)
        w.writerow(["pair_id", "record_a", "record_b", *rid_cols, "your_label"])
        k.writerow(["pair_id", "group", "match_type", "decision_status", "score", "verity_match"])
        for i, (group, r) in enumerate(chosen, 1):
            pid = f"P{i:02d}"
            a, b = by_id[r["record_a"]], by_id[r["record_b"]]
            w.writerow([pid, r["record_a"], r["record_b"], *[x for c in show_cols for x in (a.get(c, ""), b.get(c, ""))], ""])
            k.writerow([pid, group, r["match_type"], r["decision_status"], r["score"], int(r["match_type"] in WRITE_TYPES)])
    return {"code_path_pairs_available": n_code_pairs, "written": len(chosen), "by_group": dict(Counter(g for g, _ in chosen)), "out_dir": str(out_dir)}


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
    ap.add_argument("run", type=Path, help="run folder that holds verity_data/")
    ap.add_argument("--input-csv", type=Path, default=None, help="the run's input CSV (default: input_csv in run_info.json)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <run>/review")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-match", type=int, default=20)
    ap.add_argument("--n-band", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--score", action="store_true", help="report agreement from the filled review_pairs.csv and review_key.csv")
    args = ap.parse_args()
    out = args.out_dir or args.run / "review"
    if args.score:
        print(json.dumps(score(out), indent=2))
        return 0
    print(json.dumps(write_review(args.run, args.input_csv, out, args.seed, args.n_match, args.n_band, args.n_random), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

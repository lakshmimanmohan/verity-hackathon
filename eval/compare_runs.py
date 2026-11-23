"""
Compare two run folders (e.g. results/dev_2026_run2 and results/dev_2026_run3) file by file.

Ignored, because they differ between runs by design:
  - JSON keys plot_path, source_csv, column_classification_path (absolute paths that name
    the run folder), at any depth
  - golden_id values: the golden_id column in CSV files, and in verity_store.sqlite any
    cell whose whole value is a golden_id from either run's golden_records.csv (a golden_id
    embedded inside a longer text value is not masked and would show as a difference)
  - the "Source file:" line in learned_rules.md (absolute input path) and the "Generated:"
    line in match_explorer.html (generation time)

Everything else is compared exactly. README.txt and *.log are run notes and are listed,
not compared. A CSV whose rows match only after sorting is reported as SAME_ROWS_REORDERED.

Usage:
  python eval/compare_runs.py results/dev_2026_run2 results/dev_2026_run3

Exit code 0 if every compared file is the same (SAME or SAME_IGNORING), 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

IGNORED_JSON_KEYS = frozenset({"plot_path", "source_csv", "column_classification_path"})
GOLDEN_ID_COLUMN = "golden_id"
GOLDEN_CSV_NAME = "golden_records.csv"
# Per-file line patterns to drop before comparing (whole line must match).
IGNORED_LINES = {
    "learned_rules.md": re.compile(r"^Source file: `[^`]*`\s*$"),
    "match_explorer.html": re.compile(r"^\s*<span>Generated: <strong>[^<]*</strong></span>\s*$"),
}
NOT_COMPARED_NAMES = frozenset({"README.txt"})
NOT_COMPARED_SUFFIXES = frozenset({".log"})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in IGNORED_JSON_KEYS}
    if isinstance(obj, list):
        return [strip_keys(v) for v in obj]
    return obj


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    return (rows[0] if rows else []), rows[1:]


def golden_ids(run: Path) -> set[str]:
    out: set[str] = set()
    for p in run.rglob(GOLDEN_CSV_NAME):
        header, rows = read_csv_rows(p)
        if GOLDEN_ID_COLUMN in header:
            i = header.index(GOLDEN_ID_COLUMN)
            out.update(r[i] for r in rows if len(r) > i)
    return out


def compare_json(a: Path, b: Path) -> tuple[str, str]:
    ja = json.loads(a.read_text(encoding="utf-8"))
    jb = json.loads(b.read_text(encoding="utf-8"))
    if ja == jb:
        return "SAME", ""
    sa, sb = strip_keys(ja), strip_keys(jb)
    if sa == sb:
        return "SAME_IGNORING", "differs only in " + ", ".join(sorted(IGNORED_JSON_KEYS))
    keys = sorted(set(sa) | set(sb)) if isinstance(sa, dict) and isinstance(sb, dict) else []
    differing = [k for k in keys if sa.get(k) != sb.get(k)]
    return "DIFF", f"differing keys: {', '.join(differing[:8]) or '(structure)'}"


def compare_csv(a: Path, b: Path) -> tuple[str, str]:
    ha, ra = read_csv_rows(a)
    hb, rb = read_csv_rows(b)
    if ha != hb:
        return "DIFF", "headers differ"
    if GOLDEN_ID_COLUMN in ha:
        i = ha.index(GOLDEN_ID_COLUMN)
        ra = [r[:i] + r[i + 1:] for r in ra]
        rb = [r[:i] + r[i + 1:] for r in rb]
        note = "golden_id ignored"
    else:
        note = ""
    if len(ra) != len(rb):
        return "DIFF", f"{len(ra)} vs {len(rb)} rows"
    if ra == rb:
        return ("SAME_IGNORING", note) if note else ("SAME", "")
    if sorted(ra) == sorted(rb):
        return "SAME_ROWS_REORDERED", note
    n = sum(1 for x, y in zip(ra, rb) if x != y)
    return "DIFF", f"{n} of {len(ra)} rows differ" + (f" ({note})" if note else "")


def sqlite_dump(path: Path, mask: set[str]) -> dict[str, list[tuple]]:
    # immutable=1: read without creating -shm/-wal files next to the run's store.
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out: dict[str, list[tuple]] = {}
        for t in tables:
            rows = []
            for row in conn.execute(f'SELECT * FROM "{t}"'):
                rows.append(tuple(
                    "<golden_id>" if isinstance(v, str) and v in mask else v for v in row
                ))
            out[t] = sorted(rows, key=repr)
        return out
    finally:
        conn.close()


def compare_sqlite(a: Path, b: Path, mask: set[str]) -> tuple[str, str]:
    da, db = sqlite_dump(a, mask), sqlite_dump(b, mask)
    if set(da) != set(db):
        return "DIFF", "tables differ"
    differing = [t for t in da if da[t] != db[t]]
    if differing:
        return "DIFF", "tables with differing rows: " + ", ".join(differing)
    return "SAME_IGNORING", "rows compared table by table, golden_id masked, order ignored"


def compare_text_ignoring(a: Path, b: Path, pattern: re.Pattern[str]) -> tuple[str, str]:
    la = a.read_text(encoding="utf-8").splitlines()
    lb = b.read_text(encoding="utf-8").splitlines()
    ka = [l for l in la if not pattern.match(l)]
    kb = [l for l in lb if not pattern.match(l)]
    if ka == kb:
        n = len(la) - len(ka)
        return "SAME_IGNORING", f"{n} line(s) ignored"
    return "DIFF", first_text_diff(a, b)


def first_text_diff(a: Path, b: Path) -> str:
    try:
        la = a.read_text(encoding="utf-8").splitlines()
        lb = b.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return "binary"
    for i, (x, y) in enumerate(zip(la, lb), start=1):
        if x != y:
            return f"line {i}: {x.strip()[:70]!r} vs {y.strip()[:70]!r}"
    return f"{len(la)} vs {len(lb)} lines"


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two run folders file by file.")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    args = p.parse_args()
    a_root, b_root = args.run_a.resolve(), args.run_b.resolve()
    for r in (a_root, b_root):
        if not r.is_dir():
            raise SystemExit(f"Not a folder: {r}")

    files_a = {f.relative_to(a_root).as_posix() for f in a_root.rglob("*") if f.is_file()}
    files_b = {f.relative_to(b_root).as_posix() for f in b_root.rglob("*") if f.is_file()}
    mask = golden_ids(a_root) | golden_ids(b_root)

    results: list[tuple[str, str, str]] = []
    for rel in sorted(files_a | files_b):
        fa, fb = a_root / rel, b_root / rel
        name = Path(rel).name
        if rel not in files_a or rel not in files_b:
            results.append(("ONLY_IN_" + ("A" if rel in files_a else "B"), rel, ""))
            continue
        if name in NOT_COMPARED_NAMES or Path(rel).suffix in NOT_COMPARED_SUFFIXES:
            results.append(("NOT_COMPARED", rel, "run notes / log"))
            continue
        if sha256(fa) == sha256(fb):
            results.append(("SAME", rel, ""))
            continue
        suffix = Path(rel).suffix.lower()
        if suffix == ".json":
            status, note = compare_json(fa, fb)
        elif suffix == ".csv":
            status, note = compare_csv(fa, fb)
        elif suffix == ".sqlite":
            status, note = compare_sqlite(fa, fb, mask)
        elif name in IGNORED_LINES:
            status, note = compare_text_ignoring(fa, fb, IGNORED_LINES[name])
        else:
            status, note = "DIFF", first_text_diff(fa, fb)
        results.append((status, rel, note))

    print(f"A: {a_root}\nB: {b_root}\n")
    width = max(len(s) for s, _, _ in results)
    for status, rel, note in results:
        print(f"{status:<{width}}  {rel}" + (f"  ({note})" if note else ""))

    ok = {"SAME", "SAME_IGNORING", "NOT_COMPARED"}
    bad = [r for r in results if r[0] not in ok]
    counts: dict[str, int] = {}
    for s, _, _ in results:
        counts[s] = counts.get(s, 0) + 1
    print("\n" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("RESULT: " + ("all compared files match" if not bad else f"{len(bad)} file(s) differ"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

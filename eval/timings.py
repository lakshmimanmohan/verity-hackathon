"""
Print one timing table for a run folder.

  python eval/timings.py results/<run>

Reads every timing record in the run folder:
  - verity_data/run_info.json: "timings", "host" and "timing_summary" of each run entry
    (Verity, written by verity/src/main.py)
  - any timings.json below the run folder (option scripts), same record format:
      {"host": {...}, "timings": [ {"stage": ..., "parent": ..., "seconds": ...,
       "items": ..., "item": ..., "per_item_ms": {"p50": .., "p95": .., "max": ..},
       "note": ...}, ... ]}

For each source it prints the host, then the stages (children indented under their
parent), seconds, share of the source's total, item counts and per-item latency, and a
check that the top-level stages add up to the total.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_sources(run: Path) -> list[tuple[str, dict, list[dict], dict]]:
    out = []
    ri = run / "verity_data" / "run_info.json"
    if ri.is_file():
        runs = json.loads(ri.read_text(encoding="utf-8")).get("runs") or []
        for i, entry in enumerate(runs):
            if entry.get("timings"):
                out.append((f"{ri.relative_to(run)} run {i} ({entry.get('run_mode', '?')})",
                            entry.get("host") or {}, entry["timings"],
                            entry.get("timing_summary") or {}))
    for tj in sorted(run.rglob("timings.json")):
        doc = json.loads(tj.read_text(encoding="utf-8"))
        out.append((str(tj.relative_to(run)), doc.get("host") or {}, doc.get("timings") or [],
                    doc.get("timing_summary") or {}))
    return out


def fmt_items(t: dict) -> str:
    if t.get("items") is None:
        return ""
    return f"{t['items']:,} {t.get('item') or ''}".strip()


def print_source(name: str, host: dict, timings: list[dict], summary: dict) -> bool:
    print(f"== {name}")
    if host:
        print("   host: " + ", ".join(f"{k}={v}" for k, v in host.items()))
    total = next((t["seconds"] for t in timings if t["stage"].endswith(".total")), None)
    children: dict[str | None, list[dict]] = {}
    for t in timings:
        if t["stage"].endswith(".total"):
            continue
        children.setdefault(t.get("parent"), []).append(t)

    print(f"   {'stage':<32} {'seconds':>9} {'% total':>8} {'items':>16} "
          f"{'p50 ms':>8} {'p95 ms':>8} {'max ms':>9}  note")

    def row(t: dict, depth: int) -> None:
        pct = f"{100 * t['seconds'] / total:.1f}%" if total else ""
        pim = t.get("per_item_ms") or {}
        label = ("  " * depth) + t["stage"]
        print(f"   {label:<32} {t['seconds']:>9.3f} {pct:>8} {fmt_items(t):>16} "
              f"{pim.get('p50', ''):>8} {pim.get('p95', ''):>8} {pim.get('max', ''):>9}  "
              f"{t.get('note', '')}")
        for c in children.get(t["stage"], []):
            row(c, depth + 1)

    for t in children.get(None, []):
        row(t, 0)
    ok = True
    if total is not None:
        top = sum(t["seconds"] for t in children.get(None, []))
        gap = (total - top) / total if total else 0.0
        ok = abs(gap) <= 0.05
        print(f"   {'TOTAL':<32} {total:>9.3f}")
        print(f"   top-level stages sum to {top:.3f}s, {100 * (1 - gap):.1f}% of total "
              f"({'within' if ok else 'NOT within'} 5%)")
        for parent, kids in children.items():
            if parent is None:
                continue
            p = next((t["seconds"] for t in timings if t["stage"] == parent), None)
            if p:
                s = sum(k["seconds"] for k in kids)
                print(f"   children of {parent} sum to {s:.3f}s of {p:.3f}s")
    if summary:
        print("   summary: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    print()
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Print the timing table for a run folder.")
    p.add_argument("run", type=Path)
    args = p.parse_args()
    sources = load_sources(args.run)
    if not sources:
        print(f"No timing records found in {args.run}")
        return 1
    ok = all([print_source(*s) for s in sources])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

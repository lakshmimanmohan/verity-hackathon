"""
Compare a rehearsal run folder with the dev run it should reproduce (same seed, same code paths).

  python eval/compare_rehearsal.py results/rehearsal_2026 results/dev_2026_run11 [--part verity-c|d-b|all]

Checks, each reported as a pair count of differences:
  verity-c  match_candidates.csv byte-identical (sha256); C (c_v1_lgbm) decisions and confidences pair by pair; C's cutoff
  d-b       D's routed set (routed_fullband_pairs.csv); the full-band D decisions (decisions_fullband.csv: decision,
            confidence, route); B: calls made for routed pairs not in the rehearsal sample (the dfull run: count, cost,
            cache hits); every B answer in the rehearsal (sample5000, dfull, decoycheck) against the dev run's answer for
            the same pair where the dev run answered it (sample5000, dfull, decoycheck, tune1, tune2, smoke5)
Reads no truth file. Exit 1 if anything differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

B_LABELS_DEV = ("sample5000", "dfull", "decoycheck", "tune1", "tune2", "smoke5")
B_LABELS_REH = ("sample5000", "dfull", "decoycheck")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path) -> list[dict]:
    csv.field_size_limit(10**8)
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def by_pair(p: Path) -> dict:
    return {key(r["record_a"], r["record_b"]): r for r in rows(p)}


def diff_fields(x: dict, y: dict, fields: tuple[str, ...]) -> tuple[int, int, int]:
    """(only in x, only in y, in both with any field different)"""
    both = x.keys() & y.keys()
    return len(x.keys() - y.keys()), len(y.keys() - x.keys()), sum(any(x[k][f] != y[k][f] for f in fields) for k in both)


def part_verity_c(reh: Path, dev: Path, out: list[str]) -> bool:
    ok = True
    a, b = reh / "verity_data" / "match_candidates.csv", dev / "verity_data" / "match_candidates.csv"
    same = sha(a) == sha(b)
    ok &= same
    out.append(f"a. match_candidates.csv: {'byte-identical' if same else 'DIFFERENT'} (sha256 {sha(a)[:16]}... vs {sha(b)[:16]}...; "
               f"{a.stat().st_size:,} vs {b.stat().st_size:,} bytes)")
    if not same:
        x, y = by_pair(a), by_pair(b)
        cols = tuple(c for c in next(iter(x.values())) if c not in ("record_a", "record_b"))
        o1, o2, d = diff_fields(x, y, cols)
        out.append(f"   pairs only in the rehearsal {o1}, only in the dev run {o2}, in both with a different field {d}")
    cr, cd = reh / "options" / "c_model" / "c_v1_lgbm", dev / "options" / "c_model" / "c_v1_lgbm"
    x, y = by_pair(cr / "decisions.csv"), by_pair(cd / "decisions.csv")
    o1, o2, dd = diff_fields(x, y, ("decision",))
    _, _, dc = diff_fields(x, y, ("confidence",))
    ok &= not (o1 or o2 or dd or dc)
    out.append(f"b. C decisions: {len(x):,} vs {len(y):,} pairs; only in the rehearsal {o1}, only in the dev run {o2}; "
               f"decision differs {dd}; confidence differs {dc}")
    fr = json.loads((cr / "features.json").read_text(encoding="utf-8"))["cutoff"]
    fd = json.loads((cd / "features.json").read_text(encoding="utf-8"))["cutoff"]
    same_cut = fr == fd
    ok &= same_cut and round(float(fr), 4) == 0.8014
    out.append(f"   C cutoff: rehearsal {float(fr):.6f}, dev run {float(fd):.6f} ({'identical' if same_cut else 'DIFFERENT'}; rounds to {float(fr):.4f})")
    return ok


def b_answers(run: Path, labels) -> dict:
    out = {}
    for lab in labels:
        p = run / "options" / "b_llm" / "openai" / lab / "decisions.csv"
        if p.exists():
            for k, r in by_pair(p).items():
                out.setdefault(lab, {})[k] = r
    return out


def part_d_b(reh: Path, dev: Path, out: list[str]) -> bool:
    ok = True
    dr, dd = reh / "options" / "d_hybrid", dev / "options" / "d_hybrid"
    x, y = set(by_pair(dr / "routed_fullband_pairs.csv")), set(by_pair(dd / "routed_fullband_pairs.csv"))
    ok &= x == y
    out.append(f"c. D routed set: rehearsal {len(x)}, dev run {len(y)}; only in the rehearsal {len(x - y)}, only in the dev run {len(y - x)}")
    fx, fy = by_pair(dr / "decisions_fullband.csv"), by_pair(dd / "decisions_fullband.csv")
    o1, o2, d1 = diff_fields(fx, fy, ("decision", "route"))
    _, _, d2 = diff_fields(fx, fy, ("confidence",))
    ok &= not (o1 or o2 or d1 or d2)
    out.append(f"   D full-band decisions: {len(fx):,} vs {len(fy):,} pairs; only in the rehearsal {o1}, only in the dev run {o2}; "
               f"decision or route differs {d1}; confidence differs {d2}")
    sample = set(by_pair(reh / "options" / "b_llm" / "test_sample5000.csv"))
    meta_p = reh / "options" / "b_llm" / "openai" / "dfull" / "run_meta.json"
    not_in_sample = len(x - sample)
    if meta_p.exists():
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        out.append(f"   B for routed pairs not in the rehearsal sample: {not_in_sample} pairs; dfull run: {m['calls']} calls, cache hits "
                   f"{m['cache_hits']}, spent ${m['spent_usd']:.4f}")
    else:
        out.append(f"   B for routed pairs not in the rehearsal sample: {not_in_sample} pairs; no dfull run in the rehearsal")
    rb, db = b_answers(reh, B_LABELS_REH), b_answers(dev, B_LABELS_DEV)
    dev_all: dict = {}
    for lab in B_LABELS_DEV:
        for k, r in db.get(lab, {}).items():
            dev_all.setdefault(k, []).append((lab, r["decision"]))
    for lab in B_LABELS_REH:
        ans = rb.get(lab, {})
        compared = [k for k in ans if k in dev_all]
        diff = [k for k in compared if any(d != ans[k]["decision"] for _, d in dev_all[k])]
        ok &= not diff
        out.append(f"   B {lab}: {len(ans)} answers; {len(compared)} answered before in the dev run; differ {len(diff)}; "
                   f"new (never answered in the dev run) {len(ans) - len(compared)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rehearsal", type=Path)
    ap.add_argument("dev", type=Path)
    ap.add_argument("--part", choices=("verity-c", "d-b", "all"), default="all")
    args = ap.parse_args()
    out: list[str] = [f"Rehearsal {args.rehearsal.name} vs {args.dev.name}"]
    ok = True
    if args.part in ("verity-c", "all"):
        ok &= part_verity_c(args.rehearsal, args.dev, out)
    if args.part in ("d-b", "all"):
        ok &= part_d_b(args.rehearsal, args.dev, out)
    out.append("result: " + ("no differences" if ok else "DIFFERENCES FOUND"))
    text = "\n".join(out) + "\n"
    print(text)
    (args.rehearsal / f"compare_with_{args.dev.name}{'' if args.part == 'all' else '_' + args.part}.txt").write_text(text, encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

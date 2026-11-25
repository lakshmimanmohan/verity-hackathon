"""
Data profile of a generated run (the data realism table). Reads the run's orgs_raw.csv and its truth sidecar.

  python eval/data_profile.py results/<run>

Writes <run>/data_profile.txt and <run>/data_profile.csv (metric, value) with:
  records; entities; entity-size distribution; duplicate records (every record beyond the first of its entity);
  share of duplicates with at least one defect beyond a legal-suffix change, and with two or more; counts per variant
  tag (truncation, acronym, po_box, case, missing_phone, typo, typo_address); decoy records and groups; blank share per
  column; phone format counts (digits masked as 9); ZIP+4 share.

A "defect" is one of the seven tagged variants. The first record of an entity is tagged "clean"; a duplicate with no
tagged variant is "standard": it carries only the generator's untagged formatting noise (legal-suffix form, street
abbreviation, phone format, ZIP+4, website prefix, city case), so it has no defect beyond a legal-suffix change.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

TAGS = ("truncation", "acronym", "po_box", "case", "missing_phone", "typo", "typo_address")
ZIP4 = re.compile(r"^\d{5}-\d{4}$")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mask_phone(s: str) -> str:
    return re.sub(r"\d", "9", s)


def profile(records: list[dict], truth: list[dict]) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    n = len(records)
    ents = Counter(t["true_entity_id"] for t in truth)
    out.append(("records", n))
    out.append(("entities", len(ents)))
    sizes = Counter(min(v, 8) for v in ents.values())
    for k in sorted(sizes):
        out.append((f"entities with {k}{'+' if k == 8 else ''} record{'s' if k > 1 else ''}", sizes[k]))
    out.append(("mean records per entity", round(n / len(ents), 4)))
    dup = [t for t in truth if t["variants"] != "clean"]
    out.append(("duplicate records (beyond the first of each entity)", len(dup)))
    ndef = [sum(tag in TAGS for tag in t["variants"].split("+")) for t in dup]
    out.append(("share of duplicates with at least one defect beyond a legal-suffix change", round(sum(x >= 1 for x in ndef) / max(len(dup), 1), 4)))
    out.append(("share of duplicates with two or more defects", round(sum(x >= 2 for x in ndef) / max(len(dup), 1), 4)))
    tag_counts = Counter(tag for t in truth for tag in t["variants"].split("+"))
    for tag in TAGS:
        out.append((f"records tagged {tag}", tag_counts.get(tag, 0)))
    out.append(("records tagged standard (duplicates with formatting noise only)", tag_counts.get("standard", 0)))
    decoy = [t for t in truth if t["decoy_group"]]
    out.append(("decoy records", len(decoy)))
    out.append(("decoy groups", len({t["decoy_group"] for t in decoy})))
    cols = [c for c in records[0] if c != "record_id"]
    for c in cols:
        out.append((f"blank share: {c}", round(sum(not (r.get(c) or "").strip() for r in records) / n, 4)))
    phones = Counter(mask_phone(r["phone"].strip()) if (r.get("phone") or "").strip() else "(blank)" for r in records) if "phone" in records[0] else Counter()
    for fmt, k in phones.most_common():
        out.append((f"phone format {fmt}", k))
    if "postal_code" in records[0]:
        z = [r["postal_code"].strip() for r in records if (r.get("postal_code") or "").strip()]
        out.append(("ZIP+4 share (of non-blank postal codes)", round(sum(bool(ZIP4.match(x)) for x in z) / max(len(z), 1), 4)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    records = read_csv(args.run / "orgs_raw.csv")
    truth = read_csv(args.run / "orgs_raw_truth.csv")
    if len(records) != len(truth):
        raise SystemExit(f"{len(records)} records but {len(truth)} truth rows")
    rows = profile(records, truth)
    with (args.run / "data_profile.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    text = f"Data profile of {args.run.name} (orgs_raw.csv and its truth sidecar)\n" + "\n".join(f"  {k:<74}{v}" for k, v in rows) + "\n"
    (args.run / "data_profile.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

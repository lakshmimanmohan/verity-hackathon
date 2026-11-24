"""
Hand-check of generator labels (protocol: 100 pairs are hand-checked and agreement is reported).

  python eval/hand_check.py --sample results/<run>   [--truth <truth.csv>] [--force]
  python eval/hand_check.py --score  results/<run>

--sample picks 100 pairs with a fixed seed (0) from the run's scored_pairs.csv and the
truth file, in these strata:
  25 true pairs in the band, 25 false pairs in the band, 15 decoy pairs (same decoy group,
  different entity), 15 true pairs above the band with a hard-case tag (5 acronym,
  5 truncation, 5 po_box), 10 false pairs above the band, 10 random true pairs.
A pair is used once. The band is each row's effective_threshold +/- 0.05, inclusive, by score.
The 100 pairs are shuffled and written to two files in the run folder:
  hand_check_pairs.csv  pair_no, every content column of record A and record B side by side,
                        and an empty your_label column (same / different / unsure).
                        No ids, truth, score, match_type or hard-case tag.
  hand_check_key.csv    the answers: ids, stratum, truth, entity ids, tags, score, match_type,
                        region. Do not open it before labelling.
--sample refuses to overwrite an existing hand_check_pairs.csv unless --force is given.

--score reads the filled-in hand_check_pairs.csv and the key and reports agreement with the
generator's labels, overall and per stratum (unsure and blank labels excluded from the
denominator and counted separately), and lists every disagreement with both records.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SEED = 0
HALF_WIDTH = 0.05
HARD_TAGS = ("acronym", "truncation", "po_box")
PAIRS_NAME = "hand_check_pairs.csv"
KEY_NAME = "hand_check_key.csv"
LABELS = {"same": "same", "s": "same", "different": "different", "diff": "different",
          "d": "different", "unsure": "unsure", "u": "unsure", "?": "unsure"}
# (stratum name, count)
PLAN = [
    ("band_true", 25),
    ("band_false", 25),
    ("decoy", 15),
    ("above_true_acronym", 5),
    ("above_true_truncation", 5),
    ("above_true_po_box", 5),
    ("above_false", 10),
    ("random_true", 10),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sample(run: Path, truth_path: Path, force: bool) -> int:
    out_pairs, out_key = run / PAIRS_NAME, run / KEY_NAME
    if out_pairs.exists() and not force:
        raise SystemExit(f"{out_pairs} exists (it may hold your labels). Use --force to overwrite.")
    csv.field_size_limit(10**8)
    data = run / "verity_data"
    cls = json.loads((data / "column_classification.json").read_text(encoding="utf-8"))
    rid_col = cls["record_id_column"]
    content = list(cls.get("content_columns") or [])

    raw_rows = read_csv(run / "orgs_raw.csv")
    header = list(raw_rows[0].keys())
    lower_to_actual = {h.lower(): h for h in header}
    content_actual = [lower_to_actual[c.lower()] for c in content if c.lower() in lower_to_actual]
    rid_actual = lower_to_actual[rid_col.lower()]
    records = {r[rid_actual]: r for r in raw_rows}

    truth = read_csv(truth_path)
    entity = {r["record_id"]: r["true_entity_id"] for r in truth}
    tags = {r["record_id"]: set(r["variants"].split("+")) for r in truth}
    decoy_group = {r["record_id"]: r["decoy_group"] for r in truth}
    by_entity: dict[str, list[str]] = defaultdict(list)
    by_decoy: dict[str, list[str]] = defaultdict(list)
    for rid, e in entity.items():
        by_entity[e].append(rid)
        if decoy_group[rid]:
            by_decoy[decoy_group[rid]].append(rid)
    true_pairs = sorted({tuple(sorted(p)) for ids in by_entity.values() for p in combinations(ids, 2)})
    decoy_pairs = sorted({tuple(sorted(p)) for ids in by_decoy.values() for p in combinations(ids, 2)
                          if entity[p[0]] != entity[p[1]]})

    scored: dict[tuple[str, str], dict[str, str]] = {}
    region: dict[tuple[str, str], str] = {}
    for r in read_csv(data / "scored_pairs.csv"):
        k = (r["record_a"], r["record_b"])
        # Whole millionths of the written 6-decimal values, so band edges are exact.
        s = round(float(r["score"]) * 1_000_000)
        t = round(float(r["effective_threshold"]) * 1_000_000)
        scored[k] = r
        hw = round(HALF_WIDTH * 1_000_000)
        region[k] = "below" if s < t - hw else ("in" if s <= t + hw else "above")
    true_set = set(true_pairs)
    keys_sorted = sorted(scored)

    def tagged(k: tuple[str, str], tag: str) -> bool:
        return tag in tags[k[0]] or tag in tags[k[1]]

    pools = {
        "band_true": [k for k in keys_sorted if region[k] == "in" and k in true_set],
        "band_false": [k for k in keys_sorted if region[k] == "in" and k not in true_set],
        "decoy": decoy_pairs,
        "above_false": [k for k in keys_sorted if region[k] == "above" and k not in true_set],
        "random_true": true_pairs,
    }
    for tag in HARD_TAGS:
        pools[f"above_true_{tag}"] = [k for k in keys_sorted
                                      if region[k] == "above" and k in true_set and tagged(k, tag)]

    rng = random.Random(SEED)
    used: set[tuple[str, str]] = set()
    picked: list[tuple[str, tuple[str, str]]] = []
    for name, n in PLAN:
        pool = [k for k in pools[name] if k not in used]
        if len(pool) < n:
            print(f"WARNING: stratum {name} has only {len(pool)} pairs; wanted {n}")
        for k in rng.sample(pool, min(n, len(pool))):
            used.add(k)
            picked.append((name, k))
    rng.shuffle(picked)

    pair_fields = ["pair_no"]
    for c in content_actual:
        pair_fields += [f"{c}_A", f"{c}_B"]
    pair_fields.append("your_label")
    key_fields = ["pair_no", "record_a", "record_b", "stratum", "truth", "entity_a", "entity_b",
                  "tags_a", "tags_b", "in_scored_pairs", "score", "match_type", "region"]
    with out_pairs.open("w", newline="", encoding="utf-8") as fp, \
            out_key.open("w", newline="", encoding="utf-8") as fk:
        wp = csv.DictWriter(fp, fieldnames=pair_fields)
        wk = csv.DictWriter(fk, fieldnames=key_fields)
        wp.writeheader()
        wk.writeheader()
        for i, (name, (a, b)) in enumerate(picked, start=1):
            # Randomize which record is shown as A, so id order gives nothing away.
            if rng.random() < 0.5:
                a, b = b, a
            row = {"pair_no": i, "your_label": ""}
            for c in content_actual:
                row[f"{c}_A"] = records[a][c]
                row[f"{c}_B"] = records[b][c]
            wp.writerow(row)
            k = tuple(sorted((a, b)))
            sr = scored.get(k)
            wk.writerow({
                "pair_no": i, "record_a": a, "record_b": b, "stratum": name,
                "truth": "same" if entity[a] == entity[b] else "different",
                "entity_a": entity[a], "entity_b": entity[b],
                "tags_a": "+".join(sorted(tags[a])), "tags_b": "+".join(sorted(tags[b])),
                "in_scored_pairs": "yes" if sr else "no",
                "score": sr["score"] if sr else "", "match_type": sr["match_type"] if sr else "",
                "region": region.get(k, ""),
            })
    counts = Counter(name for name, _ in picked)
    print(f"Wrote {len(picked)} pairs to {out_pairs}")
    print(f"Wrote the answers to {out_key} (do not open it before labelling)")
    print("Pairs per stratum: " + ", ".join(f"{n} {counts[n]}" for n, _ in PLAN))
    print("Fill in your_label with same / different / unsure, then run --score.")
    return 0


def score(run: Path) -> int:
    pairs = {r["pair_no"]: r for r in read_csv(run / PAIRS_NAME)}
    key = {r["pair_no"]: r for r in read_csv(run / KEY_NAME)}
    if set(pairs) != set(key):
        raise SystemExit(f"{PAIRS_NAME} and {KEY_NAME} have different pair numbers.")
    value_cols = [c for c in next(iter(pairs.values())) if c.endswith("_A")]

    per = defaultdict(lambda: Counter())
    disagreements = []
    bad_labels = []
    for no in sorted(pairs, key=int):
        raw = (pairs[no].get("your_label") or "").strip().lower()
        label = LABELS.get(raw) if raw else "blank"
        if label is None:
            bad_labels.append((no, raw))
            label = "blank"
        k = key[no]
        for group in (k["stratum"], "ALL"):
            per[group][label] += 1
            if label in ("same", "different"):
                per[group]["labelled"] += 1
                per[group]["agree"] += label == k["truth"]
        if label in ("same", "different") and label != k["truth"]:
            disagreements.append((no, label, k))

    print(f"Run: {run.resolve()}\n")
    if bad_labels:
        print("Unrecognised labels (counted as blank): "
              + ", ".join(f"pair {n}: {v!r}" for n, v in bad_labels) + "\n")
    order = ["ALL"] + [n for n, _ in PLAN if n in per]
    print(f"  {'stratum':<24} {'pairs':>5} {'labelled':>8} {'agree':>6} {'agreement':>9} {'unsure':>6} {'blank':>5}")
    for g in order:
        c = per[g]
        n = sum(c[x] for x in ("same", "different", "unsure", "blank"))
        pct = f"{100 * c['agree'] / c['labelled']:.1f}%" if c["labelled"] else "n/a"
        print(f"  {g:<24} {n:>5} {c['labelled']:>8} {c['agree']:>6} {pct:>9} {c['unsure']:>6} {c['blank']:>5}")

    print(f"\nDisagreements: {len(disagreements)}")
    for no, label, k in disagreements:
        print(f"\npair {no}  you: {label}  generator: {k['truth']}  stratum {k['stratum']}  "
              f"entities {k['entity_a']} / {k['entity_b']}  tags {k['tags_a']} / {k['tags_b']}  "
              f"score {k['score'] or '-'}  match_type {k['match_type'] or '-'}")
        print(f"    {'field':<16} {'record A (' + k['record_a'] + ')':<44} record B ({k['record_b']})")
        for c in value_cols:
            base = c[:-2]
            print(f"    {base:<16} {pairs[no][c][:42]:<44} {pairs[no][base + '_B'][:42]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Hand-check of generator labels.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=Path, metavar="RUN")
    g.add_argument("--score", type=Path, metavar="RUN")
    p.add_argument("--truth", type=Path, default=None,
                   help="Truth file (default: <run>/orgs_raw_truth.csv)")
    p.add_argument("--force", action="store_true", help="Overwrite an existing hand_check_pairs.csv")
    args = p.parse_args()
    if args.sample:
        truth = args.truth or args.sample / "orgs_raw_truth.csv"
        return sample(args.sample, truth, args.force)
    return score(args.score)


if __name__ == "__main__":
    sys.exit(main())

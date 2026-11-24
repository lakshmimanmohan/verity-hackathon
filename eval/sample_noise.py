"""
Sampling-noise check (reporting only): score one option's full-band decisions on fixed samples drawn
with different seeds (eval/make_sample.py) and compare the weighted band F1 and band precision with
the full-band values. Nothing is chosen from these numbers.

  python eval/sample_noise.py results/<run> --decisions <decisions.csv> [--size 5000] [--seeds 20]

Every sample holds all same-entity test band pairs, so band recall is the same in every sample; only
the non-matches drawn differ. Band F1 is the same weighted quantity eval/score_option.py reports.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sample as ms  # noqa: E402


def band_prf(pred: dict, entity: dict, weights: dict):
    tp = sum(w for k, w in weights.items() if pred[k] and entity[k[0]] == entity[k[1]])
    fp = sum(w for k, w in weights.items() if pred[k] and entity[k[0]] != entity[k[1]])
    fn = sum(w for k, w in weights.items() if not pred[k] and entity[k[0]] == entity[k[1]])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--size", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    band = ms.test_band_pairs(args.run)
    entity = ms.test_entities(args.run)
    pred = {tuple(sorted((r["record_a"], r["record_b"]))): r["decision"].strip().lower() == "same"
            for r in ms.read_csv(args.decisions)}
    missing = [k for k in band if k not in pred]
    if missing:
        raise SystemExit(f"{len(missing)} test band pairs have no decision, e.g. {missing[:3]}")
    full = band_prf(pred, entity, {k: 1.0 for k in band})
    print(f"full test band ({len(band):,} pairs): band precision {full[0]:.4f}  recall {full[1]:.4f}  F1 {full[2]:.4f}")
    rows = []
    for seed in range(args.seeds):
        pairs, weights, *_ = ms.draw_sample(band, entity, args.size, seed)
        rows.append((seed, *band_prf(pred, entity, weights)))
        print(f"  seed {seed:>2}: precision {rows[-1][1]:.4f}  recall {rows[-1][2]:.4f}  F1 {rows[-1][3]:.4f}")
    for name, idx, ref in (("band F1", 3, full[2]), ("band precision", 1, full[0])):
        v = [r[idx] for r in rows]
        print(f"{name:<15} mean {statistics.mean(v):.4f}  std {statistics.stdev(v):.4f}  min {min(v):.4f}  max {max(v):.4f}  "
              f"(full band {ref:.4f}; max |sample - full| {max(abs(x - ref) for x in v):.4f}; "
              f"range {max(v) - min(v):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Post-freeze exploratory exhibit: what Verity learned from the freeze run's data and what was fixed, and how long the setup took. The output starts with the
line "EXPLORATORY: not a protocol result; no setting was chosen from these numbers." Read-only on results; writes only post_freeze_config.md into --out-dir
(default results/eval_7919/post_freeze/). No run, no call.

  python eval/post_freeze_config.py [--run results/eval_7919] [--freeze results/freeze_2026-11-25] [--out-dir results/eval_7919/post_freeze]

Reads results/eval_7919/verity_data/{attribute_weights.json, column_classification.json, learned_threshold.json, learned_rules.md, scored_pairs_columns.json,
run_info.json} (and the first row of scored_pairs.csv, to see which weights the score used), options/d_hybrid/frozen_settings_v2.json, and the stage 2
log of the freeze run (the command line).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEADER_LINE = "EXPLORATORY: not a protocol result; no setting was chosen from these numbers."
TOP_STAGES = (("verity.startup", "start-up (imports, data folder, store check)"), ("verity.rule_learning", "rule learning (column types, weights, threshold)"),
              ("verity.build_graph", "graph build"), ("verity.embed", "embedding"), ("verity.match", "matching (neighbors, scoring, decisions, outputs)"),
              ("verity.merge", "merge"), ("verity.explore", "explorer files"))


class Reader:
    """Reads only under the allowed roots; data and truth files are refused."""

    def __init__(self, roots: list[Path]):
        self.roots = [Path(r).resolve() for r in roots]
        self.read: list[Path] = []

    def check(self, p: Path) -> Path:
        q = Path(p).resolve()
        if q.name.startswith("orgs_raw") or "truth" in q.name or "hand_check" in q.parts:
            raise SystemExit(f"refusing to read {q.name}")
        if not any(q == r or r in q.parents for r in self.roots):
            raise SystemExit(f"refusing to read {p}: outside the allowed folders")
        return q

    def text(self, p: Path) -> str:
        q = self.check(p)
        self.read.append(q)
        return q.read_text(encoding="utf-8")

    def json(self, p: Path):
        return json.loads(self.text(p))

    def first_csv_row(self, p: Path) -> dict:
        q = self.check(p)
        self.read.append(q)
        csv.field_size_limit(10**9)
        with q.open(newline="", encoding="utf-8-sig") as f:
            return next(csv.DictReader(f))


def pick_weights_used(first_row: dict, columns: dict) -> str:
    """Which learned weight the score used: compare the first scored pair's field_weights with the two learned weights."""
    fw = json.loads(first_row["field_weights"])
    close = lambda a, b: abs(a - b) <= 5e-6
    if fw and all(close(fw[c], columns[c]["normalized_discriminative_weight"]) for c in fw if c in columns):
        return "normalized_discriminative_weight"
    if fw and all(close(fw[c], columns[c]["composite_weight"]) for c in fw if c in columns):
        return "composite_weight"
    return "neither (see the file)"


def build(run: Path, freeze: Path, R: Reader) -> dict:
    rel = lambda p: Path(p).resolve().relative_to(REPO).as_posix() if REPO in Path(p).resolve().parents else Path(p).resolve().as_posix()
    V = run / "verity_data"
    res = {"sources": {}}
    res["weights"] = R.json(V / "attribute_weights.json")
    res["classification"] = R.json(V / "column_classification.json")
    res["threshold"] = R.json(V / "learned_threshold.json")
    res["rules_md"] = R.text(V / "learned_rules.md")
    res["scored_cols"] = R.json(V / "scored_pairs_columns.json")
    res["run_info"] = R.json(V / "run_info.json")["runs"][-1]
    res["settings_v2"] = R.json(run / "options" / "d_hybrid" / "frozen_settings_v2.json")
    res["weights_used"] = pick_weights_used(R.first_csv_row(V / "scored_pairs.csv"), res["weights"]["columns"])
    log = R.text(freeze / "logs" / "stage_02.log")
    m = re.search(r"^\$ (.+)$", log, re.M)
    res["command"] = re.sub(r"\s*\[.*\]\s*$", "", m.group(1)) if m else "missing"
    res["files"] = {k: rel(V / k) for k in ("attribute_weights.json", "column_classification.json", "learned_threshold.json", "learned_rules.md", "scored_pairs_columns.json", "run_info.json")}
    res["files"]["frozen_settings_v2.json"] = rel(run / "options" / "d_hybrid" / "frozen_settings_v2.json")
    res["files"]["stage_02.log"] = rel(freeze / "logs" / "stage_02.log")
    return res


def md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def setup_times(run_info: dict) -> dict:
    """Seconds per top-level stage and the total, from the run's timings."""
    by = {t["stage"]: t["seconds"] for t in run_info["timings"] if t.get("parent") is None}
    rows = [(label, by[stage]) for stage, label in TOP_STAGES if stage in by]
    total = by["verity.total"]
    return {"rows": rows, "sum": sum(s for _, s in rows), "total": total}


def render(res: dict) -> str:
    w, cl, th, sc, ri, sv = res["weights"], res["classification"], res["threshold"], res["scored_cols"], res["run_info"], res["settings_v2"]
    cols = w["columns"]
    f = res["files"]
    order = sorted(cols, key=lambda c: -cols[c]["normalized_discriminative_weight"])
    L = [HEADER_LINE, "", "# Learned configuration and setup time (freeze run, seed 7919)", "",
         "Read from the freeze run's Verity outputs. Nothing was rerun. Every value is marked **learned** (computed from this run's data) or **fixed** (a setting that is the same for every run, or chosen before the freeze).", "",
         "## 1. Learned configuration", "", "### Per column", "",
         md_table(["column", "semantic type", "uniqueness ratio", "null rate", "discriminative score", "reliability", "normalized discriminative weight", "composite weight", "origin"],
                  [[c, cols[c]["semantic_type"], f"{cols[c]['uniqueness_ratio']:.4f}", f"{cols[c]['null_rate']:.4f}", f"{cols[c]['discriminative_score']:.4f}",
                    f"{cols[c]['reliability_score']:.4f}", f"{cols[c]['normalized_discriminative_weight']:.6f}", f"{cols[c]['composite_weight']:.6f}", "learned from the data"] for c in order]), "",
         f"Source: `{f['attribute_weights.json']}` ({w['n_rows']:,} rows; reliability and the threshold use {w['n_pairs_sampled']:,} random row pairs). "
         f"discriminative score = uniqueness ratio x (1 - null rate); the normalized weights sum to 1; the composite weight merges the normalized weight with the cross-field reliability. "
         f"The score used **{res['weights_used']}** as the field weights (checked on the first scored pair's `field_weights` in `scored_pairs.csv`).", "",
         "Columns left out of learning (classified from the data, not named by hand):", "",
         md_table(["column", "decision and reason (from the classifier)"], [[c, cl["reasoning"][c]] for c in cl["metadata_columns"]]), "",
         f"Source: `{f['column_classification.json']}` (record id column: `{cl['record_id_column']}`).", "",
         "### Thresholds, margin and blend weights", "",
         md_table(["item", "value", "learned or fixed", "how"],
                  [["learned threshold", f"{th['threshold']:.6f}", "learned from the data",
                    f"KMeans (k=2) on {th['n_pairs']:,} random row pairs: lower cluster ({th['low_cluster_n']} pairs) mean {th['low_cluster_mean']:.4f} + {th['sd_multiplier']:g} x sd {th['low_cluster_std']:.4f}"],
                   ["standard-deviation multiplier", f"{th['sd_multiplier']:g}", "fixed setting", "part of the threshold rule"],
                   ["pairs sampled for the threshold and the weights", f"{th['n_pairs']:,}", "fixed setting", "same for every run"],
                   ["effective threshold", f"{sc['effective_threshold']}", "fixed floor applied to the learned value",
                    f"the larger of the learned threshold ({sc['learned_threshold']}) and the fixed floor 0.82; here the floor applies"],
                   ["review margin", f"{sc['review_margin']}", "fixed setting", f"review when |score - effective threshold| <= {sc['review_margin']} (the band)"],
                   ["embedding blend weight (record cosine)", f"{sc['score_formula']['w_embed_blend']}", "fixed setting", "score = embedding blend x record cosine + field blend x weighted field similarities"],
                   ["field blend weight", f"{sc['score_formula']['w_field_blend']}", "fixed setting", "same formula"],
                   ["within-field blend (NAME, STREET_ADDRESS, DESCRIPTIVE)", f"{sc['score_formula']['hybrid_embed_weight']} cosine + {sc['score_formula']['hybrid_char_weight']} character Jaccard", "fixed setting",
                    "per field, for the types listed in the formula"],
                   ["D routing, lo", f"{sv['lo']}", "fixed setting, chosen before the freeze", f"rule `{sv['rule_file']}` applied to development seeds {', '.join(str(x) for x in sv['development_seeds'])}; not learned on seed 7919"],
                   ["D routing, hi", f"{sv['hi']}", "fixed setting, chosen before the freeze", "unchanged by the v2 rule"]]), "",
         f"Sources: `{f['learned_threshold.json']}`, `{f['scored_pairs_columns.json']}` (effective threshold, review margin, score formula), `{f['learned_rules.md']}` (the same threshold and blend, in words), "
         f"`{f['frozen_settings_v2.json']}` (D's lo and hi).", "",
         "## 2. Setup time", ""]
    t = setup_times(ri)
    steps = dict(t["rows"])
    start, rule = steps.get("start-up (imports, data folder, store check)", 0.0), steps.get("rule learning (column types, weights, threshold)", 0.0)
    L += [f"From the input file to the learned configuration and the matches, one run, all threads (`{f['run_info.json']}`, timings; interpreter start-up excluded):", "",
          md_table(["step", "seconds"], [[label, f"{s:.1f}"] for label, s in t["rows"]] + [["**total (verity.total)**", f"**{t['total']:.1f}**"]]), "",
          f"The steps listed add up to {t['sum']:.2f} s of the {t['total']:.2f} s total ({t['total'] - t['sum']:.2f} s untimed). The learned configuration (column types, weights, threshold) is ready after start-up and rule learning "
          f"({start + rule:.1f} s); the {ri['scored_pairs_rows']:,} scored pairs are written at the end of matching. {ri['n_records_in_index']:,} records, neighbor search `{ri['nn_mode_resolved']}`.", "",
          f"**No column mapping or parameter was set by hand.** The command line of the run (`{f['stage_02.log']}`) is `{res['command']}`: an input file and the neighbor-search mode, nothing else. "
          "The column types, the weights and the learned threshold above were derived from the data; the fixed settings in the table are the same for every run.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Learned configuration and setup time of the freeze run.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "eval_7919")
    ap.add_argument("--freeze", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()
    freeze = (args.freeze or next(iter(sorted((REPO / "results").glob("freeze_????-??-??"))), None))
    if freeze is None:
        raise SystemExit("no results/freeze_<date> folder")
    R = Reader([run, Path(freeze)])
    res = build(run, Path(freeze).resolve(), R)
    out = args.out_dir or run / "post_freeze"
    out.mkdir(parents=True, exist_ok=True)
    (out / "post_freeze_config.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'post_freeze_config.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

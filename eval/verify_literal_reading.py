"""
Check that d_hybrid.py's primary D decisions (reading "literal": pairs routed to the LLM take B's answer, all others take C's own
decision) equal the decisions the v2 selection measured (eval/d_select_lo.py literal_decisions) on every development seed, that
their band F1 equals the committed per-seed figures (results/dstudy_summary/d_select_lo.csv), and count the pairs on which the
literal and the forced reading differ, per seed and setting. Reads the truth sidecars to score; calls no LLM.

  python eval/verify_literal_reading.py [--out results/dstudy_summary/literal_reading_check.txt]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "options"))
import d_select_lo as S  # noqa: E402

D, L = S.D, S.L
SEEDS = {"dstudy_11": 11, "dstudy_12": 12, "dstudy_13": 13, "rehearsal_42": 42, "rehearsal_2026": 2026}
SETTINGS = (("v2", 0.005), ("original", 0.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=S.REPO / "results" / "dstudy_summary" / "literal_reading_check.txt")
    args = ap.parse_args()
    rows_ = list(csv.DictReader((S.REPO / "results" / "dstudy_summary" / "d_select_lo.csv").open(encoding="utf-8")))
    committed = {float(r["lo"]): r for r in rows_ if not r["lo"].startswith("share_")}
    lines = ["Primary D decisions (literal reading) against the decisions the v2 selection measured (eval/d_select_lo.py literal_decisions)",
             f"  {'run':<16}{'setting':<10}{'lo':>6}{'pairs':>8}{'routed':>8}{'decisions differ':>18}{'band F1':>9}{'committed F1':>14}{'F1 equal':>10}{'readings differ':>17}"]
    bad = 0
    for name, seed in SEEDS.items():
        run = S.REPO / "results" / name
        c = D.load_c(D.paths(run)["c"])
        b = D.merge_b(D.b_paths(run) + [run / "options" / "b_llm" / "openai" / n / "decisions.csv" for n in ("lo01", S.LABEL)])
        ent = L.truth_entities(run)
        for setting, lo in SETTINGS:
            rows = D.build(c, b, [(k, 1.0) for k in sorted(c)], lo, S.HI, "literal")
            mine = {(r["record_a"], r["record_b"]): r["decision"] for r in rows}
            ref, n_llm = S.literal_decisions(run, lo, b, c)
            n_diff = sum(1 for k in ref if mine[k] != ref[k]) + len(set(ref) ^ set(mine))
            f1 = L.counts(mine, ent)["f1"]
            want = float(committed[lo][f"f1_{name}"])
            n_read = len(D.readings_differ(c, lo, S.HI))
            ok = n_diff == 0 and abs(f1 - want) < 5e-7
            bad += not ok
            lines.append(f"  {name:<16}{setting:<10}{lo:>6g}{len(c):>8,}{n_llm:>8}{n_diff:>18}{f1:>9.4f}{want:>14.4f}{'yes' if abs(f1 - want) < 5e-7 else 'NO':>10}{n_read:>17}")
    lines += ["", f"result: {'primary D decisions equal the committed d_select_lo results on all 5 seeds, for both settings' if not bad else f'{bad} MISMATCHES'}"]
    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

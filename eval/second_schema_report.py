"""
Exploratory report for a Verity run on a second input file (no labels, no accuracy claimed). Prints column names, counts, hashes, similarities and
weights only: no record value is written. Reads the run's verity_data folder (local, gitignored) and writes, into --summary-dir, the report and small
copies of five Verity outputs with local paths removed.

  python eval/second_schema_report.py --verity-data results/second_schema/verity_data --input-csv "<path>" --summary-dir results/second_schema/summary

Checks done by the same script: no local path remains in the copies, and no cell of the input file appears in any file of the summary folder
(alphabetic cells of 4 or more characters as whole words after the column names are masked, not counting words that are Verity vocabulary; digit-only cells
of 5 or more characters as whole tokens, separating matches that are counts the run reports).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HEADER_LINE = "EXPLORATORY: not a protocol result; no labels exist for this file, so no accuracy is claimed."
PATH_REMOVED = "(local path removed)"
COPIES = ("column_classification.json", "attribute_weights.json", "learned_threshold.json", "learned_rules.md", "run_info.json")
EARLIER_WEIGHTS = {"Alias Account": 0.362878, "Street Name": 0.138752, "House Number1": 0.114763, "Box1 Number": 0.001647, "Postal Code": 0.258904, "City Name": 0.123056}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(p: Path):
    csv.field_size_limit(10**9)
    with Path(p).open(newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


def md_table(headers: list[str], rows: list[list]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


# ---------------------------------------------------------------------------------------------- redaction and checks

def redact_text(text: str, paths: list[str]) -> str:
    """Replace every occurrence of the given local paths (as written, JSON-escaped, and with forward slashes) with the removal marker."""
    for p in sorted({x for x in paths if x}, key=len, reverse=True):
        for form in (p, p.replace("\\", "\\\\"), p.replace("\\", "/")):
            text = text.replace(form, PATH_REMOVED)
    return text


def redact_attribute_weights(doc: dict) -> dict:
    out = dict(doc)
    for k in ("source_csv", "column_classification_path"):
        if k in out:
            out[k] = PATH_REMOVED
    return out


def local_path_left(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]:\\|[A-Za-z]:/|/Users/|\\Users\\|OneDrive", text))


def input_cells(csv_path: Path) -> tuple[set[str], set[str]]:
    """(alphabetic cells of 4+ characters, digit-only cells of 5+ characters) of the input file; used only to check that none appears in a committed file."""
    words, nums = set(), set()
    for row in read_rows(csv_path):
        for v in row.values():
            v = (v or "").strip()
            if len(v) >= 4 and re.search(r"[A-Za-z]", v):
                words.add(v)
            elif len(v) >= 5 and v.isdigit():
                nums.add(v)
    return words, nums


def scan_for_cells(folder: Path, words: set[str], nums: set[str], allowed_numbers: set[str], column_names: list[str] = (), vocabulary_text: str = "") -> dict:
    """Input cells found in the files of the folder. Alphabetic cells: whole-word, case-sensitive matches after the column names are masked out; a match that is also a
    whole word of the vocabulary text (Verity's own outputs for an unrelated dataset) is counted as vocabulary, not as a leak. Digit cells: whole-token matches; tokens
    that are also counts the run reports are separated from the rest."""
    word_re = lambda w: re.compile(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])")
    found_words, vocab_words = set(), set()
    found_nums, found_allowed = set(), set()
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        masked = text
        for c in sorted(column_names, key=len, reverse=True):
            masked = masked.replace(c, " ")
        for w in words:
            if w in masked and word_re(w).search(masked):
                (vocab_words if word_re(w).search(vocabulary_text) else found_words).add(w)
        tokens = set(re.findall(r"(?<![\d.])\d{5,}(?![\d])", text))
        for t in tokens & nums:
            (found_allowed if t in allowed_numbers else found_nums).add(t)
    return {"word_cells_found_as_values": len(found_words), "word_cells_found_only_as_vocabulary": len(vocab_words), "number_cells_found_not_counts": len(found_nums),
            "number_cells_found_that_are_stated_counts": len(found_allowed)}


# ---------------------------------------------------------------------------------------------- the facts

def facts(V: Path) -> dict:
    cl = json.loads((V / "column_classification.json").read_text(encoding="utf-8"))
    aw = json.loads((V / "attribute_weights.json").read_text(encoding="utf-8"))
    th = json.loads((V / "learned_threshold.json").read_text(encoding="utf-8"))
    sc = json.loads((V / "scored_pairs_columns.json").read_text(encoding="utf-8"))
    ri = json.loads((V / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    status, mtype, cross = Counter(), Counter(), Counter()
    first = None
    first_code = None
    n = 0
    for r in read_rows(V / "scored_pairs.csv"):
        n += 1
        status[r["decision_status"]] += 1
        mtype[r["match_type"]] += 1
        cross[(r["decision_status"], r["match_type"])] += 1
        if r["decision_status"] == "match" and r.get("pair_source") != "exact":                 # the first matched pair that is not an exact duplicate
            k = tuple(sorted((r["record_a"], r["record_b"])))
            if first is None or k < first[0]:
                first = (k, r)
        if r["match_type"] == "code_signal" and r["decision_status"] == "match":                  # the first code_signal match by sorted record ids
            k = tuple(sorted((r["record_a"], r["record_b"])))
            if first_code is None or k < first_code[0]:
                first_code = (k, r)
    golden = Counter(int(r["member_count"]) for r in read_rows(V / "golden_records.csv"))
    groups: dict[str, int] = {}
    for r in read_rows(V / "match_summary.csv"):
        groups[r["match_id"]] = int(r["group_size"])
    matched_groups = Counter(s for m, s in groups.items() if not m.startswith("U"))
    singles = sum(1 for m in groups if m.startswith("U"))
    return {"cl": cl, "aw": aw, "th": th, "sc": sc, "ri": ri, "rows": n, "status": status, "mtype": mtype, "cross": cross, "first": first, "first_code": first_code, "golden": golden,
            "matched_groups": matched_groups, "singletons": singles, "n_records_in_summary": len(groups)}


def weight_comparison(columns: dict, earlier: dict) -> list[list]:
    rows = []
    for c, old in earlier.items():
        if c in columns:
            new = columns[c]["normalized_discriminative_weight"]
            d = new - old
            rows.append([c, f"{new:.6f}", f"{old:.6f}", f"{d:+.6f}", "no (within 6-decimal rounding)" if abs(d) < 5.5e-7 else "yes"])
        else:
            rows.append([c, "not in this run", f"{old:.6f}", "n/a", "n/a"])
    return rows


def explanation_card(first) -> tuple[list[list], dict]:
    (a, b), r = first
    if not r["field_similarities"].strip():                               # an exact-duplicate pair: score 1.0 by rule, no per-field data is stored
        return [], {"score": r["score"], "match_type": r["match_type"], "decision_status": r["decision_status"], "pair_source": r["pair_source"],
                    "per-field similarities, weights and evidence": "not stored: exact-duplicate pairs are scored 1.0 by rule"}
    sims, wts, esims = json.loads(r["field_similarities"]), json.loads(r["field_weights"]), json.loads(r["evidence_field_similarities"])
    rows = [[f, f"{sims[f]:.4f}", f"{wts.get(f, float('nan')):.4f}", f"{esims[f]:.4f}" if f in esims else "n/a"] for f in sims]
    return rows, {"score": r["score"], "match_type": r["match_type"], "decision_status": r["decision_status"], "evidence_mass": r["evidence_mass"], "conflict_mass": r["conflict_mass"],
                  "evidence_passes": r["evidence_passes"]}


def key_lists(V: Path, cl: dict) -> dict:
    """Names and counts only: the semantic-type keys match.py builds (col_key form), the stored attribute types, and how many records have a space in a NAME/DESCRIPTIVE/STREET_ADDRESS value."""
    import sqlite3
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verity" / "src"))
    from column_classification import col_key
    sem = {col_key(k): str(v).upper() for k, v in cl["semantic_types"].items()}
    out = {"sem_keys": sorted(sem), "sem_guard_keys": sorted(k for k, v in sem.items() if v in ("NAME", "DESCRIPTIVE", "STREET_ADDRESS")), "store_keys": [], "records": None, "with_space": None, "with_value": None}
    for db in sorted(V.glob("*.sqlite")):
        c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        out["store_keys"] = sorted(r[0].lower() for r in c.execute("select distinct attr_type from attrs"))
        g = tuple(k.upper() for k in out["sem_guard_keys"])
        q = ",".join("?" * len(g))
        out["records"] = c.execute("select count(distinct id) from records").fetchone()[0]
        out["with_value"] = c.execute(f"select count(distinct id) from attrs where upper(attr_type) in ({q})", g).fetchone()[0]
        out["with_space"] = c.execute(f"select count(distinct id) from attrs where upper(attr_type) in ({q}) and instr(value, ' ') > 0", g).fetchone()[0]
        c.close()
    return out


def finding_single_token(F: dict) -> list[str]:
    ri, K = F["ri"], F.get("keys") or {}
    return ["### Finding: no neighbor pair was scored", "",
            f"Of {ri['neighbor_pairs']:,} neighbor pairs, {ri['skipped_single_token_semantic_fields']:,} were skipped before scoring by the single-token guard (`verity/src/match.py` lines 2036-2040: "
            "`has_a = any(\" \" in str(aa.get(k, \"\")) for k in sem_fields) ...`: both records need a space in at least one NAME, DESCRIPTIVE or STREET_ADDRESS value), "
            f"{ri.get('skipped_no_semantic_field', 0):,} by the no-semantic-field guard, and 0 were scored. The only pairs in the output are the exact duplicates ({ri['exact_duplicate_pairs']:,}), "
            "which Verity scores 1.0 by rule. So this run did no similarity scoring: there is no review band, no non-match row and no per-field evidence.", "",
            "Key lists (names only; the two forms agree, so the column-key mismatch of the second-schema run is not the cause here):", "",
            md_table(["list", "keys"], [["semantic-type keys as match.py builds them (`col_key` form)", ", ".join(f"`{k}`" for k in K.get("sem_keys", []))],
                                         ["of those, the keys the guard looks up (NAME, DESCRIPTIVE, STREET_ADDRESS)", ", ".join(f"`{k}`" for k in K.get("sem_guard_keys", []))],
                                         ["attribute types in the record store (lower-cased)", ", ".join(f"`{k}`" for k in K.get("store_keys", []))]]), "",
            f"Counts: {K.get('records')} records in the store; {K.get('with_value')} have a value in at least one guard field; {K.get('with_space')} have a space in at least one such value. "
            "Likely cause (a reading of the counters, not tested by a fix): the guard fields hold no multi-word values, so every pair fails the space test. The identifier and quasi-identifier columns are not guard fields. No change was made to Verity.", ""]


def finding(F: dict, earlier_run_note: bool = True) -> list[str]:
    """A plain statement when the neighbor search scored nothing: every neighbor pair was skipped before scoring. When pairs were scored, a short note on what was scored and skipped."""
    ri = F["ri"]
    if ri.get("scored_neighbor_pairs", 0) != 0 and "code_path_pairs" in ri:
        sk = {k: ri.get(k, 0) for k in ("skipped_missing_store_context", "skipped_missing_embeddings", "skipped_no_semantic_field", "skipped_no_code_field")}
        return ["### Neighbor pairs scored, by path", "",
                f"Of {ri['neighbor_pairs']:,} neighbor pairs, **{ri['scored_neighbor_pairs']:,} were scored** and {sum(sk.values()):,} were skipped before scoring "
                f"({', '.join(f'{k.replace(chr(95), chr(32))}: {v:,}' for k, v in sk.items())}). "
                f"**Text path:** {ri['text_path_pairs']:,} pairs (embedding cosine blended with field similarity, the original scoring). "
                f"**Code path:** {ri['code_path_pairs']:,} pairs that fail the multi-word test and were scored on code-like columns only (no embedding cosine); "
                f"code-like columns chosen from the data: {', '.join(f'`{c}`' for c in ri.get('code_like_columns', [])) or '(none)'}. "
                f"The exact-duplicate pairs ({ri['exact_duplicate_pairs']:,}) are scored 1.0 by rule. "
                + ("The first run of this file (`results/second_schema/summary`) scored none of the neighbor pairs; this run follows the key fix and the code path. " if earlier_run_note else "")
                + "No labels exist, so nothing here says whether a pair is right.", ""]
    if ri.get("scored_neighbor_pairs", 0) != 0:
        sk = {k: ri.get(k, 0) for k in ("skipped_missing_store_context", "skipped_missing_embeddings", "skipped_no_semantic_field", "skipped_single_token_semantic_fields")}
        return ["### Neighbor pairs scored", "",
                f"Of {ri['neighbor_pairs']:,} neighbor pairs, **{ri['scored_neighbor_pairs']:,} were scored** and {sum(sk.values()):,} were skipped before scoring "
                f"({', '.join(f'{k.replace(chr(95), chr(32))}: {v:,}' for k, v in sk.items())}). The exact-duplicate pairs ({ri['exact_duplicate_pairs']:,}) are scored 1.0 by rule. "
                + ("The first run of this file (`results/second_schema/summary`) scored none of the neighbor pairs, because column names with a space did not match the record store's snake_case keys; "
                   "this run follows the key fix (`col_key`). " if earlier_run_note else "")
                + "No labels exist, so nothing here says whether a pair is right.", ""]
    if not ri.get("neighbor_pairs"):
        return []
    if ri.get("skipped_single_token_semantic_fields", 0) == ri["neighbor_pairs"]:
        return finding_single_token(F)
    return ["### Finding: no neighbor pair was scored", "",
            f"Of {ri['neighbor_pairs']:,} neighbor pairs, {ri['skipped_no_semantic_field']:,} were skipped before scoring by the guard \"a record has no NAME/DESCRIPTIVE/STREET_ADDRESS value\" "
            "(`verity/src/match.py` lines 2028-2035: `if not any(k in aa for k in sem_fields) ...`), and 0 were scored. The only pairs in the output are the exact duplicates "
            f"({ri['exact_duplicate_pairs']:,}), which Verity scores 1.0 by rule. So this run did no similarity scoring: there is no review band, no non-match row and no per-field evidence.", "",
            "Likely cause (not tested by a fix, and a hypothesis from names only): the guard looks up the semantic-type keys, which are the lower-cased input column names with their spaces "
            "(for example `alias account`), in each record's attributes, and the record store keeps the attributes under snake_case keys (for example `alias_account`; the store's attribute "
            "types are upper-case snake_case). With names that contain a space the two never match, so every pair looks as if it had no such value. The first dataset's column names have no spaces and "
            "are unaffected. No change was made to Verity.", ""]


def render(F: dict, input_facts: dict, command: str, title: str = "Second-schema run (exploratory)", what: str = "a second, unlabeled input file with a different schema",
           compare_earlier: bool = True) -> str:
    cl, aw, th, sc, ri = F["cl"], F["aw"], F["th"], F["sc"], F["ri"]
    cols = aw["columns"]
    L = [HEADER_LINE, "", f"# {title}", "",
         f"A Verity run on {what}. The report holds column names, counts, hashes, similarities and weights only; no record value is printed. "
         "Nothing about accuracy can be said: no labels exist for this file.", "",
         "## a. Input facts", "",
         md_table(["item", "value"], [["rows (data rows, header excluded)", f"{input_facts['rows']:,}"], ["columns", f"{len(input_facts['columns'])}: " + ", ".join(f"`{c}`" for c in input_facts["columns"])],
                                      ["file sha256", f"`{input_facts['sha256']}`"], ["file size", f"{input_facts['bytes']:,} bytes"]]), "",
         "Command (environment `PYTHONHASHSEED=0 PYTHONUTF8=1`, plugged in, all threads, one run): ", "", f"`{command}`", "",
         "The only arguments are the input path and `--nn-mode exact`: no column mapping and no parameter was given. The directory of the input path is replaced by "
         f"\"{PATH_REMOVED}\" in this report.", "",
         "## b. Classification", "",
         md_table(["column", "role", "semantic type", "classifier's reason"],
                  [[c, "content", cl["semantic_types"].get(c, ""), cl["reasoning"].get(c, "")] for c in cl["content_columns"]] +
                  [[c, "excluded", "", cl["reasoning"].get(c, "")] for c in cl["metadata_columns"]]), "",
         f"Record id column: `{cl['record_id_column']}`.", "",
         "## c. Weights", "",
         md_table(["column", "semantic type", "uniqueness ratio", "null rate", "discriminative score", "normalized weight"],
                  [[c, cols[c]["semantic_type"], f"{cols[c]['uniqueness_ratio']:.4f}", f"{cols[c]['null_rate']:.4f}", f"{cols[c]['discriminative_score']:.4f}",
                    f"{cols[c]['normalized_discriminative_weight']:.6f}"] for c in sorted(cols, key=lambda k: -cols[k]["normalized_discriminative_weight"])]), "",
         *(["Against the normalized weights of an earlier run of the same file (values given in the request):", "",
            md_table(["column", "this run", "earlier run", "difference", "differs"], weight_comparison(cols, EARLIER_WEIGHTS)), ""] if compare_earlier else []),
         "## d. Threshold", "",
         md_table(["item", "value"], [["learned threshold", f"{th['threshold']:.6f}"], ["effective threshold", f"{sc['effective_threshold']}"], ["review margin", f"{sc['review_margin']}"],
                                      ["rule", "effective threshold = the larger of the learned threshold and the fixed floor 0.82"]]), "",
         "## e. Counts", "",
         f"Scored pairs: **{F['rows']:,}** (`scored_pairs_rows` in run_info: {ri['scored_pairs_rows']:,}); records in the index: {ri['n_records_in_index']:,}; neighbor search `{ri['nn_mode_resolved']}`.", "",
         md_table(["neighbor-search counter (run_info.json)", "pairs"],
                  [[k, f"{ri.get(k):,}"] for k in ("neighbor_pairs", "skipped_missing_store_context", "skipped_missing_embeddings", "skipped_no_semantic_field",
                                                   "skipped_single_token_semantic_fields", "scored_neighbor_pairs", "exact_duplicate_pairs", "exact_only_pairs")] +
                  [[k, f"{ri.get(k):,}"] for k in ("text_path_pairs", "code_path_pairs", "skipped_no_code_field") if k in ri]), "",
         md_table(["decision status", "pairs"], [[k, f"{v:,}"] for k, v in sorted(F["status"].items())]), "",
         md_table(["match type", "pairs"], [[k, f"{v:,}"] for k, v in sorted(F["mtype"].items())]), "",
         md_table(["decision status", "match type", "pairs"], [[a, b, f"{v:,}"] for (a, b), v in sorted(F["cross"].items())]), "",
         f"Merge groups: **{sum(F['golden'].values()):,} golden records** (`golden_records.csv`). `match_summary.csv` has {F['n_records_in_summary']:,} distinct group ids: "
         f"{sum(F['matched_groups'].values()):,} multi-record groups and {F['singletons']:,} unmatched single records. "
         f"The two files give the same number of groups at the sizes {', '.join(str(s) for s in sorted(set(F['matched_groups']) | set(F['golden'])) if s > 1 and F['matched_groups'].get(s, 0) == F['golden'].get(s, 0)) or '(none)'} "
         f"and differ at the sizes {', '.join(str(s) for s in sorted(set(F['matched_groups']) | set(F['golden'])) if F['matched_groups'].get(s, 0) != F['golden'].get(s, 0))} (not investigated).",
         "",
         md_table(["group size", "multi-record groups (match_summary)", "golden records (member_count)"],
                  [[s, f"{F['matched_groups'].get(s, 0):,}", f"{F['golden'].get(s, 0):,}"] for s in sorted(set(F["matched_groups"]) | set(F["golden"]))]), "",
         *finding(F, compare_earlier),
         "## f. Timing (run_info.json)", "",
         md_table(["stage", "parent", "seconds", "items"], [[t["stage"], t.get("parent") or "", f"{t['seconds']:.3f}", t.get("items") if t.get("items") is not None else ""] for t in ri["timings"]]), "",
         f"Total (`verity.total`): {[t['seconds'] for t in ri['timings'] if t['stage'] == 'verity.total'][0]:.1f} s. Host: {ri['host']['cpu_model']}, {ri['host']['logical_cores']} logical cores.", ""]
    code_card = "code_path_pairs" in ri
    card_pair = F.get("first_code") if code_card else F["first"]
    rows, extra = explanation_card(card_pair) if card_pair else ([], {"result": "no code_signal match" if code_card else "no matched pair that is not an exact duplicate"})
    L += ["## g. Explanation card: the first code_signal match, by sorted record ids" if code_card else
          "## g. Explanation card: the first matched pair that is not an exact duplicate, by sorted record ids", "",
          "Field names, similarities, weights and evidence similarities only; no values.", "",
          md_table(["field", "similarity", "weight", "evidence similarity"], rows) if rows else
          ("No code_signal match exists in this run, so there is no card to show. " if code_card else "No matched pair that is not an exact duplicate exists in this run, so there is no card to show. ") + " Exact duplicates are scored 1.0 by rule and carry no per-field data (section e).", "",
          md_table(["item", "value"], [[k, v] for k, v in extra.items()]), "",
          "## Copies in this folder", "",
          "Small copies of `column_classification.json`, `attribute_weights.json`, `learned_threshold.json`, `learned_rules.md` and `run_info.json`. In `attribute_weights.json` the values of "
          f"`source_csv` and `column_classification_path` were replaced by \"{PATH_REMOVED}\". The same marker replaces the input path in `learned_rules.md` and `input_csv` in `run_info.json`. "
          "Nothing else was changed.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Report and small copies for a second-schema Verity run.")
    ap.add_argument("--verity-data", type=Path, required=True)
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--summary-dir", type=Path, required=True)
    ap.add_argument("--vocabulary-dir", type=Path, default=None, help="Verity outputs of an unrelated dataset (json and md files): words found there are vocabulary, not values")
    ap.add_argument("--report-name", default="second_schema_report.md", help="file name of the report in --summary-dir")
    ap.add_argument("--title", default="Second-schema run (exploratory)")
    ap.add_argument("--what", default="a second, unlabeled input file with a different schema")
    ap.add_argument("--no-earlier-comparison", action="store_true", help="leave out the comparison with the earlier second-schema weights and run")
    ap.add_argument("--command", default=None, help="the command line to print (default: reconstructed from the input name)")
    args = ap.parse_args()
    V, src, out = args.verity_data.resolve(), args.input_csv.resolve(), args.summary_dir
    out.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        columns = next(r)
        n_rows = sum(1 for _ in r)
    inp = {"rows": n_rows, "columns": columns, "sha256": sha256_file(src), "bytes": src.stat().st_size}
    paths = [str(src), str(src.parent), str(V), str(V.parent), str(Path(V).resolve().parents[2]) if len(V.parents) > 2 else ""]
    # copies with local paths removed
    for name in COPIES:
        text = (V / name).read_text(encoding="utf-8")
        if name == "attribute_weights.json":
            text = json.dumps(redact_attribute_weights(json.loads(text)), indent=2) + "\n"
        text = redact_text(text, paths)
        (out / name).write_text(text, encoding="utf-8")
    F = facts(V)
    F["keys"] = key_lists(V, F["cl"])
    try:
        rel_data = V.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_data = "(data folder)"
    command = args.command or f'python verity/src/main.py --nn-mode exact --input "{PATH_REMOVED}\\{src.name}"   [VERITY_DATA_DIR={rel_data}]'
    (out / args.report_name).write_text(render(F, inp, command, args.title, args.what, not args.no_earlier_comparison), encoding="utf-8")
    left = [n for n in list(COPIES) + [args.report_name] if local_path_left((out / n).read_text(encoding="utf-8"))]
    words, nums = input_cells(src)
    stated = {str(inp["rows"]), str(F["rows"]), str(sum(F["golden"].values())), str(F["n_records_in_summary"])}
    stated |= {str(v) for c in (F["status"], F["mtype"], F["golden"], F["matched_groups"]) for v in c.values()} | {str(v) for v in F["cross"].values()}
    for name in ("run_info.json", "attribute_weights.json", "learned_threshold.json"):                 # counts the run itself reports (rows, pairs, items), not record values
        stated |= set(re.findall(r"(?<![\d.])\d{5,}(?![\d.])", (V / name).read_text(encoding="utf-8")))
    vocab = "".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(Path(args.vocabulary_dir).glob("*")) if f.suffix in (".json", ".md")) if args.vocabulary_dir else ""
    scan = scan_for_cells(out, words, nums, stated, columns, vocab)
    print(f"wrote the report and {len(COPIES)} copies into {out}; files with a local path left: {left or 'none'}; cell scan: {scan}")
    return 1 if left or scan["word_cells_found_as_values"] or scan["number_cells_found_not_counts"] else 0


if __name__ == "__main__":
    sys.exit(main())

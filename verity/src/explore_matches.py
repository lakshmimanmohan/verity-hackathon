"""
Generate a standalone HTML match explorer from match_candidates and a customer CSV.

Usage (from repo root):
    python src/explore_matches.py --input PATH/to/customers.csv

Record details for side-by-side comparison come from the same ``input_path`` CSV the
pipeline uses (joined on the record id column from ``column_classification.json`` to
``record_a`` / ``record_b``).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import html as html_module
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from config import data_dir

ROOT = Path(__file__).resolve().parent.parent
DATA = data_dir()
MATCH_CSV = DATA / "match_candidates.csv"
WEIGHTS_JSON = DATA / "attribute_weights.json"
COLUMN_CLASSIFICATION_NAME = "column_classification.json"
DEFAULT_CLASS_JSON = DATA / COLUMN_CLASSIFICATION_NAME
OUT_HTML = DATA / "match_explorer.html"

SAMPLE_PER_STRATUM = 50
SAMPLE_SEED = 42


def norm(s: str) -> str:
    return (s or "").strip().lower()


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(norm(a).split()) if norm(a) else set()
    tb = set(norm(b).split()) if norm(b) else set()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def row_compare_class(va: str, vb: str) -> str:
    na, nb = norm(va), norm(vb)
    if not na and not nb:
        return "cell-missing"
    if not na or not nb:
        return "cell-missing"
    if na == nb:
        return "cell-exact"
    if jaccard_tokens(va, vb) > 0.5:
        return "cell-similar"
    return "cell-diff"


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


_PO_BOX_RE_LOCAL = re.compile(r"^\s*p\.?\s*o\.?\s*box\b", re.IGNORECASE)

# Matches a leading street-number block: digits, optionally hyphenated or with
# a letter suffix (e.g. "123", "4B", "10-12"). Used only to separate the
# numeric anchor of an address from the word tokens that follow — no
# assumption about what those words mean.
_LEADING_NUMBER_RE = re.compile(r"^(\d{1,6}[A-Za-z]?(?:-\d{1,6})?)\b")


def normalize_street_address(s: str) -> str:
    """
    Canonicalize an address string for cross-format comparison: lowercase,
    strip punctuation, collapse whitespace, and split off the leading
    street-number block as a separate anchor (so "222 Bergen Turnpike"
    becomes ("222", "bergen turnpike")). The number anchor and the word
    portion are compared separately downstream: numbers must match
    exactly (a different street number is a different building), while
    the word portion is compared with sequence-similarity rather than an
    exact match, so spelling/abbreviation differences like "Turnpike" vs
    "TPKE" still register as a strong match without needing a fixed
    abbreviation vocabulary. Pure value-pattern transform — no word list,
    no column names, no domain assumptions. Applies identically to a
    street address, a vendor's mailing address, or any other
    STREET_ADDRESS-classified field.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def address_number_and_words(normalized: str) -> tuple[str, str]:
    """Splits a normalize_street_address() output into (number_anchor, word_part)."""
    if not normalized:
        return "", ""
    m = _LEADING_NUMBER_RE.match(normalized)
    if not m:
        return "", normalized
    return m.group(1), normalized[m.end():].strip()


def is_po_box(s: str) -> bool:
    """True if the address string is a PO Box rather than a street address."""
    return bool(_PO_BOX_RE_LOCAL.match((s or "").strip()))


def address_field_keys(semantic_by_col_lower: dict[str, str]) -> list[str]:
    """
    Column keys classified STREET_ADDRESS in attribute_weights.json.
    Falls back to an empty list (no address-aware gating) on schemas where
    rule_learner hasn't run or the classification predates this field type —
    the cohesion check then behaves exactly as it did before this fix.
    """
    return sorted(
        k for k, v in semantic_by_col_lower.items()
        if str(v).strip().upper() == "STREET_ADDRESS"
    )


def name_field_keys(semantic_by_col_lower: dict[str, str]) -> list[str]:
    """Column keys classified NAME in attribute_weights.json."""
    return sorted(
        k for k, v in semantic_by_col_lower.items()
        if str(v).strip().upper() == "NAME"
    )


def parse_shared_attributes(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not s or not str(s).strip():
        return out
    for part in str(s).split("|"):
        part = part.strip()
        if not part:
            continue
        idx = part.find(":")
        if idx == -1:
            continue
        field, val = part[:idx], part[idx + 1 :]
        out.append((field.strip(), val.strip()))
    return out


def load_json(path: Path, default):
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_column_classification_path(csv_path: Path) -> Path:
    """``column_classification.json`` in the data folder."""
    if DEFAULT_CLASS_JSON.is_file():
        return DEFAULT_CLASS_JSON
    raise FileNotFoundError(
        f"Missing column classification file. Expected:\n"
        f"  {DEFAULT_CLASS_JSON}"
    )


def stratified_sample_pairs(
    unique_pairs: list[dict], per: int, rng: random.Random
) -> list[dict]:
    """
    Take ``per`` pairs from top scores, middle third, lower third, then ``per``
    random from the remainder (deduped). Sort result by score descending.
    """
    if not unique_pairs:
        return []
    sorted_desc = sorted(
        unique_pairs,
        key=lambda x: (-float(x["score"]), x["record_a"], x["record_b"]),
    )
    n = len(sorted_desc)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    cap = min(4 * per, n)

    def add_from_pool(pool: list[dict], max_add: int) -> None:
        nonlocal out, seen
        added = 0
        for p in pool:
            if added >= max_add or len(out) >= cap:
                break
            key = canonical_pair(p["record_a"], p["record_b"])
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            added += 1

    add_from_pool(sorted_desc[: min(per, n)], per)

    lo, hi = n // 3, (2 * n) // 3
    mid_pool = list(sorted_desc[lo:hi])
    rng.shuffle(mid_pool)
    add_from_pool(mid_pool, per)

    low_pool = list(sorted_desc[(2 * n) // 3 :])
    rng.shuffle(low_pool)
    add_from_pool(low_pool, per)

    remainder = [p for p in sorted_desc if canonical_pair(p["record_a"], p["record_b"]) not in seen]
    rng.shuffle(remainder)
    add_from_pool(remainder, per)

    out.sort(key=lambda x: (-float(x["score"]), x["record_a"], x["record_b"]))
    return out


class UnionFind:
    def __init__(self) -> None:
        self._p: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._p:
            self._p[x] = x
        if self._p[x] != x:
            self._p[x] = self.find(self._p[x])
        return self._p[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def main(input_path: Path | None = None) -> None:
    if input_path is None:
        raise ValueError(
            "input_path is required (the customer CSV used by the pipeline, "
            "e.g. run_explore(input_path=...) from main.py)."
        )
    customers_csv = input_path.expanduser().resolve()
    if not customers_csv.is_file():
        raise FileNotFoundError(f"Customer CSV not found: {customers_csv}")

    cls_path = resolve_column_classification_path(customers_csv)
    with cls_path.open(encoding="utf-8") as f:
        class_data = json.load(f)
    rid_col = str(class_data.get("record_id_column") or "").strip()
    if not rid_col:
        raise ValueError(f"{cls_path} is missing record_id_column.")

    weights_data = load_json(WEIGHTS_JSON, {})
    col_weights: dict[str, float] = {}
    cols_meta = weights_data.get("columns") or {}
    for cname, meta in cols_meta.items():
        if isinstance(meta, dict):
            w = meta.get("composite_weight")
            if w is not None:
                col_weights[norm(cname)] = float(w)
    semantic_by_col: dict[str, str] = {
        norm(k): str(v) for k, v in (class_data.get("semantic_types") or {}).items()
    }

    content_columns: list[str] = list(class_data.get("content_columns") or [])
    # Metadata columns (other than the record id) shown as badges on each card.
    badge_columns: list[str] = [
        c for c in (class_data.get("metadata_columns") or []) if norm(c) != norm(rid_col)
    ]

    with customers_csv.open(encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError(f"CSV has no header row: {customers_csv}")
        fields = list(rdr.fieldnames)
        norm_record_id = norm(rid_col)
        csv_id_field: str | None = None
        for name in fields:
            if name is None:
                continue
            if norm(name) == norm_record_id:
                csv_id_field = name
                break
        if csv_id_field is None:
            raise ValueError(
                f"CSV {customers_csv.name} has no column matching "
                f"record_id_column {rid_col!r} from {cls_path}. "
                f"Headers: {fields!r}"
            )
        rid_col = csv_id_field

        customers: dict[str, dict[str, str]] = {}
        for row in rdr:
            cid = (row.get(rid_col) or "").strip()
            if cid:
                customers[cid] = {k: (v or "") for k, v in row.items()}

    # Full candidate list (both directions in file)
    raw_rows: list[dict] = []
    with MATCH_CSV.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            raw_rows.append(dict(row))

    # Unique undirected pairs — keep row with max score if duplicates
    best_by_pair: dict[tuple[str, str], dict] = {}
    for row in raw_rows:
        a = (row.get("record_a") or "").strip()
        b = (row.get("record_b") or "").strip()
        if not a or not b:
            continue
        key = canonical_pair(a, b)
        try:
            sc = float(row.get("score") or 0.0)
        except ValueError:
            sc = 0.0
        prev = best_by_pair.get(key)
        if prev is None or sc > float(prev.get("score") or 0):
            best_by_pair[key] = {
                "record_a": a,
                "record_b": b,
                "score": sc,
                "shared_attributes": row.get("shared_attributes") or "",
                "match_type": (row.get("match_type") or "").strip() or "unknown",
                "field_contributions": row.get("field_contributions") or {},
                "field_similarities": row.get("field_similarities") or {},
            }

    unique_pairs = list(best_by_pair.values())
    n_unique = len(unique_pairs)
    full_signal_n = sum(1 for p in unique_pairs if p["match_type"] == "full_signal")
    full_signal_pct = (100.0 * full_signal_n / n_unique) if n_unique else 0.0
    avg_score = sum(p["score"] for p in unique_pairs) / n_unique if n_unique else 0.0

    # Estimated golden: UF on all unique pairs + singletons for customers not in graph
    uf = UnionFind()
    in_graph: set[str] = set()
    for p in unique_pairs:
        a, b = p["record_a"], p["record_b"]
        in_graph.add(a)
        in_graph.add(b)
        uf.union(a, b)
    roots_in_graph = {uf.find(x) for x in in_graph}
    k_matched = len(roots_in_graph)
    n_customers = len(customers)
    n_isolated = n_customers - len(in_graph)
    estimated_golden = k_matched + max(0, n_isolated)

    rng = random.Random(SAMPLE_SEED)
    display_pairs = stratified_sample_pairs(unique_pairs, SAMPLE_PER_STRATUM, rng)
    n_shown = len(display_pairs)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def esc(s: str) -> str:
        return html_module.escape(s, quote=True)

    cards_html: list[str] = []
    for p in display_pairs:
        a, b = p["record_a"], p["record_b"]
        ra = customers.get(a, {})
        rb = customers.get(b, {})
        score = float(p["score"])
        mt = p["match_type"]
        if score >= 0.95:
            bar_cls = "bar-green"
        elif score >= 0.85:
            bar_cls = "bar-yellow"
        elif score >= 0.75:
            bar_cls = "bar-orange"
        else:
            bar_cls = "bar-red"
        mt_cls = {
            "full_signal": "mt-full",
            "weak_signal": "mt-weak",
            "embedding_only": "mt-emb",
        }.get(mt, "mt-other")

        shared = parse_shared_attributes(p["shared_attributes"])
        try:
            fc: dict[str, float] = json.loads(p.get("field_contributions") or "{}")
        except (ValueError, TypeError):
            fc = {}
        try:
            fs: dict[str, float] = json.loads(p.get("field_similarities") or "{}")
        except (ValueError, TypeError):
                fs = {}

        # Display-only contribution:
        # % Contribution in the explorer should reconcile with the visible
        # Weight and Similarity columns, so compute it as:
        # normalized(composite_weight * displayed_similarity).
        display_contrib: dict[str, float] = {}
        for field, sim in fs.items():
            w = col_weights.get(field)
            if w is None or sim is None:
                continue
            display_contrib[field] = max(0.0, float(w)) * max(0.0, float(sim))

        display_contrib_total = sum(display_contrib.values()) or 1.0

        # Sort by displayed contribution so the order matches the visible table math.
        shared = sorted(
            parse_shared_attributes(p["shared_attributes"]),
            key=lambda x: display_contrib.get(x[0], 0.0),
            reverse=True,
        )

        # fields already shown via shared_attributes
        shown_fields: set[str] = {field for field, _ in shared}

        # additional fields present in field_similarities but not exact-matched
        extra_rows: list[str] = []
        
        for field in sorted(fs.keys(), key=lambda x: display_contrib.get(x, 0.0), reverse=True):
            if field in shown_fields:
                continue

            raw_sim = fs.get(field)
            display_raw_contrib = display_contrib.get(field)

            if (raw_sim is None or raw_sim == 0.0) and (
                display_raw_contrib is None or display_raw_contrib == 0.0
            ):
                continue

            st = semantic_by_col.get(field, "UNKNOWN")

            w = col_weights.get(field)
            wtxt = f"{w:.4f}" if w is not None else "—"
            sim_pct = f"{raw_sim * 100.0:.1f}%" if raw_sim is not None else "—"
            pct_txt = (
                f"{display_raw_contrib / display_contrib_total * 100.0:.1f}%"
                if display_raw_contrib is not None
                else "—"
            )
            va = esc(str(ra.get(field, "") or ""))
            vb = esc(str(rb.get(field, "") or ""))
            flag = ""
            if raw_sim is not None and w is not None and w > 0.25 and raw_sim < 0.70:
                flag = " <span style='color:#c45c00' title='High-weight field with low similarity'>⚠</span>"
            extra_rows.append(
                f"<tr style='opacity:0.75'><td>{esc(field)}</td>"
                f"<td><span class='stype'>{esc(st)}</span></td>"
                f"<td class='wval'>{esc(wtxt)}</td>"
                f"<td class='wval'>{esc(sim_pct)}</td>"
                f"<td class='wval'>{esc(pct_txt)}{flag}</td></tr>"
            )

        why_rows: list[str] = []
        for field, sval in shared:
            st = semantic_by_col.get(field, "UNKNOWN")
            w = col_weights.get(field)
            wtxt = f"{w:.4f}" if w is not None else "—"

            raw_sim = fs.get(field)
            sim_pct = f"{raw_sim * 100.0:.1f}%" if raw_sim is not None else "—"

            display_raw_contrib = display_contrib.get(field)
            pct_txt = (
                f"{display_raw_contrib / display_contrib_total * 100.0:.1f}%"
                if display_raw_contrib is not None
                else "—"
            )

            flag = ""
            if raw_sim is not None and w is not None and w > 0.25 and raw_sim < 0.70:
                flag = " <span style='color:#c45c00' title='High-weight field with low similarity'>⚠</span>"
            why_rows.append(
                f"<tr><td>{esc(field)}</td>"
                f"<td><span class='stype'>{esc(st)}</span></td>"
                f"<td class='wval'>{esc(wtxt)}</td>"
                f"<td class='wval'>{esc(sim_pct)}</td>"
                f"<td class='wval'>{esc(pct_txt)}{flag}</td></tr>"
            )

        why_block = (
            "<table class='why-table'><thead><tr>"
            "<th>Field</th><th>Type</th><th>Weight</th>"
            "<th>Similarity</th><th>% Contribution</th>"
            "</tr></thead><tbody>"
            + "".join(why_rows)
            + ("".join(extra_rows) if extra_rows else "")
            + "</tbody></table>"
        )

        cmp_rows: list[str] = []
        for col in content_columns:
            va = ra.get(col, "")
            vb = rb.get(col, "")
            cls = row_compare_class(va, vb)
            cmp_rows.append(
                f"<tr class='{cls}'><td class='fname'>{esc(col)}</td>"
                f"<td>{esc(va)}</td><td>{esc(vb)}</td></tr>"
            )

        sys_a = esc(" / ".join(str(ra.get(c, "") or "") for c in badge_columns).strip(" /") or "—")
        sys_b = esc(" / ".join(str(rb.get(c, "") or "") for c in badge_columns).strip(" /") or "—")
        bar_width_pct = f"{score * 100.0:.6f}".rstrip("0").rstrip(".")
        records_attr = esc(f"{a} {b}")
        score_attr = esc(f"{score}")
        cards_html.append(
            f'<article class="card" data-type="{esc(mt)}" data-score="{score_attr}" '
            f'data-records="{records_attr}">'
            f'<div class="card-head">'
            f'<div class="score-wrap"><div class="score-label">Score <strong>{score:.4f}</strong></div>'
            f'<div class="bar-track"><div class="bar-fill {bar_cls}" style="width:{bar_width_pct}%"></div></div></div>'
            f'<span class="badge {mt_cls}">{esc(mt)}</span>'
            f'<span class="pair-ids">{esc(a)} <span class="vs">↔</span> {esc(b)}</span>'
            f'<div class="sys-badges"><span class="sys-badge">{sys_a}</span>'
            f'<span class="sys-badge sys-b">{sys_b}</span></div></div>'
            f'<div class="compare-wrap"><table class="compare"><thead><tr>'
            f'<th>Field</th><th>{esc(a)}</th><th>{esc(b)}</th></tr></thead><tbody>'
            f'{"".join(cmp_rows)}</tbody></table></div>'
            f'<details class="why" open><summary>Why it matched</summary>{why_block}</details>'
            f"</article>"
        )

    cards_joined = "\n".join(cards_html)

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Match Explorer — MDM Knowledge Graph</title>
<style>
:root {{
  --navy: #0d1b2a;
  --navy2: #1b263b;
  --muted: #778da9;
  --bg: #f8f9fb;
  --card: #fff;
  --text: #1a1a2e;
  --border: #e2e6ef;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.45;
}}
.site-header {{
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy2) 100%);
  color: #e0e1dd; padding: 1.25rem 1.5rem 1.5rem;
  box-shadow: 0 2px 12px rgba(13,27,42,0.35);
}}
.site-header h1 {{
  margin: 0 0 0.75rem; font-size: 1.45rem; font-weight: 600; letter-spacing: -0.02em;
}}
.stats-bar {{
  display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem;
  font-size: 0.88rem; color: #c9d1e0;
}}
.stats-bar span {{ white-space: nowrap; }}
.stats-bar strong {{ color: #fff; font-weight: 600; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 1rem 1.25rem 2.5rem; }}
.filters {{
  display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-end;
  background: var(--card); padding: 1rem 1.1rem; border-radius: 10px;
  margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  border: 1px solid var(--border);
}}
.filters label {{ font-size: 0.78rem; color: #555; display: block; margin-bottom: 0.25rem; }}
.filters select, .filters input[type="text"] {{
  padding: 0.45rem 0.6rem; border: 1px solid var(--border); border-radius: 6px;
  font-size: 0.9rem; min-width: 10rem;
}}
.score-slider-wrap {{ flex: 1; min-width: 200px; max-width: 320px; }}
.score-slider-wrap .range-row {{ display: flex; align-items: center; gap: 0.5rem; }}
.score-slider-wrap input[type="range"] {{ flex: 1; }}
.counts {{ font-size: 0.9rem; color: #555; margin-bottom: 0.75rem; }}
.counts strong {{ color: var(--navy2); }}
.card {{
  background: var(--card); border-radius: 12px; padding: 1rem 1.1rem;
  margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.07);
  border: 1px solid var(--border);
}}
.card.hidden {{ display: none; }}
.card-head {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 1rem;
  margin-bottom: 0.75rem; padding-bottom: 0.65rem; border-bottom: 1px solid var(--border);
}}
.score-wrap {{ flex: 1; min-width: 180px; max-width: 280px; }}
.score-label {{ font-size: 0.82rem; color: #555; margin-bottom: 0.2rem; }}
.bar-track {{
  height: 10px; background: #e8ecf4; border-radius: 6px; overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.2s; }}
.bar-green {{ background: linear-gradient(90deg, #2d6a4f, #40916c); }}
.bar-yellow {{ background: linear-gradient(90deg, #b69100, #d4a017); }}
.bar-orange {{ background: linear-gradient(90deg, #c45c00, #e85d04); }}
.bar-red {{ background: linear-gradient(90deg, #9d0208, #d00000); }}
.badge {{
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; padding: 0.25rem 0.55rem; border-radius: 6px; color: #fff;
}}
.mt-full {{ background: #1b7f5a; }}
.mt-weak {{ background: #b58900; }}
.mt-emb {{ background: #5c4d7d; }}
.mt-other {{ background: #6c757d; }}
.pair-ids {{ font-weight: 600; font-size: 0.95rem; }}
.vs {{ color: var(--muted); font-weight: 400; }}
.sys-badges {{ margin-left: auto; display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.sys-badge {{
  font-size: 0.75rem; background: var(--navy2); color: #e0e1dd;
  padding: 0.2rem 0.5rem; border-radius: 5px;
}}
.compare-wrap {{ overflow-x: auto; }}
table.compare {{
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
}}
table.compare th, table.compare td {{
  border: 1px solid var(--border); padding: 0.35rem 0.5rem; text-align: left; vertical-align: top;
}}
table.compare th {{ background: #eef1f7; font-weight: 600; }}
table.compare .fname {{
  font-weight: 600; color: #415a77; width: 8rem; background: #fafbfc;
}}
tr.cell-exact td:not(.fname) {{ background: #d8f3dc; }}
tr.cell-similar td:not(.fname) {{ background: #fff3bf; }}
tr.cell-diff td:not(.fname) {{ background: #ffccd5; }}
tr.cell-missing td:not(.fname) {{ background: #e9ecef; color: #6c757d; }}
.why {{ margin-top: 0.5rem; }}
.why summary {{
  cursor: pointer; font-weight: 600; color: var(--navy2); font-size: 0.9rem;
}}
.why-table {{
  width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.5rem;
}}
.why-table th, .why-table td {{
  border: 1px solid var(--border); padding: 0.3rem 0.45rem; text-align: left;
}}
.why-table th {{ background: #eef1f7; }}
.stype {{ font-size: 0.78rem; color: #415a77; }}
.wval {{ font-family: ui-monospace, monospace; }}
.sval {{ word-break: break-word; max-width: 28rem; }}
</style>
</head>
<body>
<header class="site-header">
  <h1>Match Explorer — MDM Knowledge Graph</h1>
  <div class="stats-bar">
    <span>Total candidate pairs (unique): <strong>{n_unique}</strong></span>
    <span><code>full_signal</code>: <strong>{full_signal_n}</strong> ({full_signal_pct:.1f}%)</span>
    <span>Average score: <strong>{avg_score:.4f}</strong></span>
    <span>Est. golden records: <strong>{estimated_golden}</strong></span>
    <span>Showing <strong>{n_shown}</strong> of <strong>{n_unique}</strong> pairs</span>
    <span>Generated: <strong>{esc(generated)}</strong></span>
  </div>
</header>
<main>
  <div class="filters">
    <div>
      <label for="flt-type">Match type</label>
      <select id="flt-type">
        <option value="all">All</option>
        <option value="full_signal">full_signal</option>
        <option value="weak_signal">weak_signal</option>
        <option value="embedding_only">embedding_only</option>
      </select>
    </div>
    <div class="score-slider-wrap">
      <label>Score range: <span id="score-min-lbl">0.000</span> – <span id="score-max-lbl">1.000</span></label>
      <div class="range-row">
        <span style="font-size:0.75rem">min</span>
        <input type="range" id="flt-score-min" min="0" max="1" step="0.001" value="0"/>
      </div>
      <div class="range-row">
        <span style="font-size:0.75rem">max</span>
        <input type="range" id="flt-score-max" min="0" max="1" step="0.001" value="1"/>
      </div>
    </div>
    <div>
      <label for="flt-search">Search record ID</label>
      <input type="text" id="flt-search" placeholder="e.g. C1" autocomplete="off"/>
    </div>
  </div>
  <p class="counts" id="counts-line">Total cards: <strong>{n_shown}</strong> · Visible: <strong>{n_shown}</strong></p>
  <div id="cards">
{cards_joined}
  </div>
</main>
<script>
(function() {{
  var cards = Array.prototype.slice.call(document.querySelectorAll("#cards .card"));
  var elType = document.getElementById("flt-type");
  var elMin = document.getElementById("flt-score-min");
  var elMax = document.getElementById("flt-score-max");
  var elSearch = document.getElementById("flt-search");
  var elCounts = document.getElementById("counts-line");
  var lblMin = document.getElementById("score-min-lbl");
  var lblMax = document.getElementById("score-max-lbl");

  function applyFilters() {{
    var t = elType.value;
    var smin = parseFloat(elMin.value) || 0;
    var smax = parseFloat(elMax.value) || 1;
    if (smin > smax) {{
      var tmp = smin; smin = smax; smax = tmp;
    }}
    lblMin.textContent = smin.toFixed(3);
    lblMax.textContent = smax.toFixed(3);
    var q = (elSearch.value || "").trim().toLowerCase();
    var vis = 0;
    for (var i = 0; i < cards.length; i++) {{
      var c = cards[i];
      var mt = c.getAttribute("data-type") || "";
      var sc = parseFloat(c.getAttribute("data-score") || "0");
      var ids = (c.getAttribute("data-records") || "").toLowerCase();
      var okType = t === "all" || mt === t;
      var okScore = sc >= smin - 1e-9 && sc <= smax + 1e-9;
      var okSearch = !q || ids.indexOf(q) !== -1;
      var show = okType && okScore && okSearch;
      if (show) {{ vis++; c.classList.remove("hidden"); }}
      else {{ c.classList.add("hidden"); }}
    }}
    elCounts.innerHTML = "Total cards: <strong>" + cards.length + "</strong> · Visible: <strong>" + vis + "</strong>";
  }}

  elType.addEventListener("change", applyFilters);
  elMin.addEventListener("input", function() {{
    var a = parseFloat(elMin.value), b = parseFloat(elMax.value);
    if (a > b) elMax.value = String(a);
    applyFilters();
  }});
  elMax.addEventListener("input", function() {{
    var a = parseFloat(elMin.value), b = parseFloat(elMax.value);
    if (b < a) elMin.value = String(b);
    applyFilters();
  }});
  elSearch.addEventListener("input", applyFilters);
  applyFilters();
}})();
</script>
</body>
</html>
"""

    OUT_HTML.write_text(html_out, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({n_shown} cards sampled, {n_unique} unique pairs in source)")

MATCH_SUMMARY_CSV = DATA / "match_summary.csv"


def run_match_summary(input_path: Path | None = None) -> None:
    """
    Write match_summary.csv: one row per record in the input CSV, with a Match ID
    shared by all records in the same match group. Isolated records get a unique Match ID.
    Match ID format: M0001, M0002, ... for matched groups; U0001, U0002, ... for unmatched.

    After building connected components via union-find, any component that looks
    like a transitive-closure chain rather than a genuine cluster is split apart.
    Two independent structural checks decide this, run as separate gates rather
    than one blended score, because they catch different failure modes:

      1. Address cohesion — do the STREET_ADDRESS-classified fields across the
         component actually agree once abbreviation/formatting noise is removed?
         "222 Bergen Turnpike" and "222 Bergen TPKE" normalize to the same
         string and count as cohesive. A component spanning many distinct
         normalized addresses (different PO boxes, different streets, different
         cities) is NOT cohesive on location grounds, regardless of how similar
         the names look.

      2. Name divergence — among records that share the dominant normalized
         address, do the NAME-classified fields actually look like the same
         entity, or do they diverge enough to suggest separate organizational
         units that merely share a building or location? Measured purely as
         token overlap between name values — no vocabulary list, no domain
         assumptions — so it applies the same way to location, customer,
         vendor, and product records alike.

    Neither gate inspects column names or assumes a domain; both operate only
    on fields whose semantic_type (assigned by rule_learner) is STREET_ADDRESS
    or NAME. If a schema has neither type populated, both gates are skipped
    and the component falls back to the original single-score cohesion check.
    """
    if input_path is None:
        raise ValueError("input_path is required")
    customers_csv = input_path.expanduser().resolve()
    if not customers_csv.is_file():
        raise FileNotFoundError(f"Customer CSV not found: {customers_csv}")

    cls_path = resolve_column_classification_path(customers_csv)
    with cls_path.open(encoding="utf-8") as f:
        class_data = json.load(f)
    rid_col = str(class_data.get("record_id_column") or "").strip()
    if not rid_col:
        raise ValueError(f"{cls_path} is missing record_id_column.")

    # Semantic types — same source/fallback order as match.py's
    # load_column_semantic_types, so STREET_ADDRESS/NAME classification here
    # always matches what scored the pairs in match_candidates.csv.
    semantic_by_col_lower: dict[str, str] = {
        norm(k): str(v).strip().upper()
        for k, v in (class_data.get("semantic_types") or {}).items()
    }
    if not semantic_by_col_lower:
        weights_data = load_json(WEIGHTS_JSON, {})
        for cname, meta in (weights_data.get("columns") or {}).items():
            if isinstance(meta, dict):
                semantic_by_col_lower[norm(cname)] = str(
                    meta.get("semantic_type", "DESCRIPTIVE")
                ).strip().upper()

    addr_fields = address_field_keys(semantic_by_col_lower)
    name_fields = name_field_keys(semantic_by_col_lower)

    all_records: list[str] = []
    rows_by_id: dict[str, dict[str, str]] = {}
    with customers_csv.open(encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames or [])
        csv_id_field: str | None = None
        field_lower_map: dict[str, str] = {}
        for name in fields:
            if not name:
                continue
            field_lower_map[norm(name)] = name
            if norm(name) == norm(rid_col):
                csv_id_field = name
        if csv_id_field is None:
            raise ValueError(f"CSV has no column matching record_id_column {rid_col!r}")
        for row in rdr:
            rid = (row.get(csv_id_field) or "").strip()
            if not rid:
                continue
            all_records.append(rid)
            # Only keep the columns the cohesion gates actually need — avoids
            # holding the full wide row for every record in memory.
            if addr_fields or name_fields:
                kept: dict[str, str] = {}
                for lk in (*addr_fields, *name_fields):
                    src_col = field_lower_map.get(lk)
                    if src_col is not None:
                        kept[lk] = (row.get(src_col) or "").strip()
                rows_by_id[rid] = kept

    # build connected components from match_candidates.csv
    uf = UnionFind()
    for rid in all_records:
        uf.find(rid)

    score_lookup: dict[tuple[str, str], float] = {}
    if MATCH_CSV.is_file():
        with MATCH_CSV.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                a = (row.get("record_a") or "").strip()
                b = (row.get("record_b") or "").strip()
                if not a or not b:
                    continue
                try:
                    score = float(row.get("score") or 0.0)
                except ValueError:
                    score = 0.0
                key = canonical_pair(a, b)
                if key not in score_lookup or score > score_lookup[key]:
                    score_lookup[key] = score
                uf.union(a, b)

    # group by component root
    root_to_members: dict[str, list[str]] = {}
    for rid in all_records:
        root = uf.find(rid)
        root_to_members.setdefault(root, []).append(rid)

    COHESION_FLOOR = 0.55             # fallback floor when no address/name fields exist
    MAX_MEMBERS_FOR_FULL_CHECK = 200  # cap on exhaustive normalization pass per component
    ADDRESS_COHESION_FLOOR = 0.50     # below this, too many distinct normalized addresses
    NAME_DIVERGENCE_CEILING = 0.65    # above this, same address but names look unrelated

    def normalized_addresses_for(members: list[str]) -> list[tuple[str, str]]:
        """
        (raw_joined_address, skeleton_normalized_address) per member, or
        ("", "") if no address value. is_po_box() must run on the raw form
        since "PO BOX" reduces to "p bx" after skeleton normalization and
        would no longer match the PO Box pattern.
        """
        out = []
        for rid in members:
            row = rows_by_id.get(rid) or {}
            parts = [str(row.get(f, "") or "") for f in addr_fields]
            joined = " ".join(p for p in parts if p)
            if joined:
                out.append((joined, normalize_street_address(joined)))
            else:
                out.append(("", ""))
        return out

    def normalized_names_for(members: list[str]) -> list[str]:
        out = []
        for rid in members:
            row = rows_by_id.get(rid) or {}
            parts = [str(row.get(f, "") or "") for f in name_fields]
            joined = " ".join(p for p in parts if p)
            out.append(joined.strip().lower())
        return out

    ADDRESS_WORD_SIM_FLOOR = 0.75  # sequence-similarity floor for the word portion

    def _addresses_equivalent(a: str, b: str) -> bool:
        """
        True if two normalize_street_address() outputs represent the same
        physical address. The leading number anchor must match exactly (a
        different street number is a different building); the remaining
        word portion is compared with difflib sequence-similarity rather
        than requiring an exact match, so spelling and abbreviation
        differences ("Turnpike" vs "TPKE", "Bergen" vs "Bergan") still
        register as the same address without any fixed word list. The
        similarity floor is the same general-purpose threshold used for
        string comparison elsewhere in this codebase, not an address-
        specific rule.
        """
        if a == b:
            return True
        if not a or not b:
            return False
        num_a, words_a = address_number_and_words(a)
        num_b, words_b = address_number_and_words(b)
        if num_a != num_b:
            return False
        if not words_a or not words_b:
            return words_a == words_b
        ratio = difflib.SequenceMatcher(None, words_a, words_b).ratio()
        return ratio >= ADDRESS_WORD_SIM_FLOOR

    def _equivalence_classes(addr_pairs: list[tuple[str, str]]) -> list[list[int]]:
        """
        Partitions address indices into equivalence classes. ``addr_pairs``
        is a list of (raw_address, skeleton_normalized_address) tuples.
        PO Box detection runs on the raw form (skeleton normalization
        collapses "PO BOX" to "p bx", which would no longer match the PO
        Box pattern). PO Box addresses are bucketed together regardless of
        box number, since different regional box numbers for one company
        are a weak signal compared to two different street addresses,
        which point at two different physical buildings. Non-PO-Box
        addresses are compared via _addresses_equivalent on the normalized
        form. Returns a list of classes (each a list of indices into
        ``addr_pairs``), sorted largest-first.
        """
        bucket_keys = [
            ("PO_BOX" if raw and is_po_box(raw) else norm_addr)
            for raw, norm_addr in addr_pairs
        ]
        assigned = [-1] * len(addr_pairs)
        classes: list[list[int]] = []
        for i, bi in enumerate(bucket_keys):
            if not bi or assigned[i] != -1:
                continue
            cls = [i]
            assigned[i] = len(classes)
            for j in range(i + 1, len(addr_pairs)):
                if not bucket_keys[j] or assigned[j] != -1:
                    continue
                if bi == "PO_BOX" and bucket_keys[j] == "PO_BOX":
                    cls.append(j)
                    assigned[j] = len(classes)
                elif bi != "PO_BOX" and _addresses_equivalent(bi, bucket_keys[j]):
                    cls.append(j)
                    assigned[j] = len(classes)
            classes.append(cls)
        classes.sort(key=len, reverse=True)
        return classes

    def address_cohesion(members: list[str]) -> float | None:
        """
        Fraction of address-bearing members that fall in the largest
        structural-equivalence class (see _equivalence_classes). None when
        no address field exists or no member has an address value (gate
        skipped).
        """
        if not addr_fields:
            return None
        addr_pairs = normalized_addresses_for(members)
        non_empty_idx = [i for i, (raw, _) in enumerate(addr_pairs) if raw]
        if not non_empty_idx:
            return None
        non_empty_pairs = [addr_pairs[i] for i in non_empty_idx]
        classes = _equivalence_classes(non_empty_pairs)
        if not classes:
            return None
        return len(classes[0]) / len(non_empty_pairs)

    def _idf_weights_for(values: list[str]) -> dict[str, float]:
        """
        Token weights derived from this specific set of values: a token
        appearing in every value gets weight ~0, a token appearing in only
        one value gets weight close to 1. Computed fresh per component, no
        external vocabulary.
        """
        tokenized = [frozenset(norm(v).split()) for v in values if norm(v)]
        tokenized = [t for t in tokenized if t]
        if not tokenized:
            return {}
        n = len(tokenized)
        doc_freq: dict[str, int] = {}
        for toks in tokenized:
            for t in toks:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        return {t: 1.0 - (df / n) for t, df in doc_freq.items()}

    def _weighted_jaccard(a: str, b: str, weight: dict[str, float]) -> float:
        """IDF-weighted jaccard similarity between two name strings, using
        weights already computed for the surrounding component."""
        ta, tb = frozenset(norm(a).split()), frozenset(norm(b).split())
        union = ta | tb
        if not union:
            return 1.0
        inter = ta & tb
        w_union = sum(weight.get(t, 0.0) for t in union)
        w_inter = sum(weight.get(t, 0.0) for t in inter)
        if w_union <= 0.0:
            return len(inter) / len(union) if union else 1.0
        return w_inter / w_union

    def _idf_weighted_divergence(values: list[str]) -> float | None:
        """
        Average pairwise name divergence, weighted by how distinctive each
        token is within this specific component — not a fixed vocabulary,
        just inverse document frequency computed from ``values`` itself.

        Plain token-jaccard treats every word equally, which misses a real
        failure mode: many records share a long common name prefix (e.g.
        "Nutrien Ag Solutions") and differ only in a trailing branch/city
        token ("- Tucson AZ" vs "- Yuma AZ"). The shared prefix dominates a
        flat jaccard score and keeps divergence low even though the
        trailing token is exactly what distinguishes one branch from
        another. Weighting each token's contribution by how many of the
        compared values contain it (rare tokens count more, near-universal
        tokens count less) makes the branch-distinguishing tokens drive the
        score instead of being diluted by the shared prefix. The weights
        are derived entirely from this component's own values — no
        external word list, no domain assumption.
        """
        values = [v for v in values if norm(v)]
        if len(values) < 2:
            return None
        weight = _idf_weights_for(values)
        pairs = list(combinations(values, 2))
        if len(pairs) > 435:  # cap at C(30,2) worth of comparisons
            pairs = pairs[:435]
        distances = [1.0 - _weighted_jaccard(a, b, weight) for a, b in pairs]
        return sum(distances) / len(distances) if distances else None

    def name_divergence(members: list[str]) -> float | None:
        """
        Among members whose address falls in the largest structural-
        equivalence class, the average pairwise divergence between NAME
        values, weighted by token distinctiveness within this component
        (see _idf_weighted_divergence). None when no name field exists or
        fewer than two members share that dominant address class.
        """
        if not addr_fields or not name_fields:
            return None
        addr_pairs = normalized_addresses_for(members)
        names = normalized_names_for(members)
        non_empty_idx = [i for i, (raw, _) in enumerate(addr_pairs) if raw]
        if not non_empty_idx:
            return None
        non_empty_pairs = [addr_pairs[i] for i in non_empty_idx]
        classes = _equivalence_classes(non_empty_pairs)
        if not classes:
            return None
        dominant_local_idx = classes[0]
        dominant_global_idx = [non_empty_idx[i] for i in dominant_local_idx]
        same_addr_names = [names[i] for i in dominant_global_idx if names[i]]
        if len(same_addr_names) < 2:
            return None
        return _idf_weighted_divergence(same_addr_names[:30])

    def component_avg_pairwise_score(members: list[str]) -> float:
        n = len(members)
        if n <= 1:
            return 1.0
        total_pairs = n * (n - 1) // 2
        if total_pairs > 4000:
            sample_size = min(4000, total_pairs)
            pairs_seen = set()
            scores: list[float] = []
            attempts = 0
            while len(scores) < sample_size and attempts < sample_size * 4:
                attempts += 1
                a, b = random.sample(members, 2)
                key = canonical_pair(a, b)
                if key in pairs_seen:
                    continue
                pairs_seen.add(key)
                scores.append(score_lookup.get(key, 0.0))
            return sum(scores) / len(scores) if scores else 0.0
        scores = []
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_pair(members[i], members[j])
                scores.append(score_lookup.get(key, 0.0))
        return sum(scores) / len(scores) if scores else 0.0

    # Pairwise similarity floor for name sub-clustering — independent from
    # NAME_DIVERGENCE_CEILING, which is an AVERAGE divergence across all
    # pairs in a component and is not the right threshold for individual
    # pairwise decisions. Empirically, exact-duplicate name pairs score
    # ~1.0 under IDF-weighted jaccard, while genuinely different branches
    # sharing a long corporate prefix plus a common short suffix token
    # (e.g. two different cities both abbreviated to the same state, "CA"
    # or "ID") score no higher than ~0.4 even after IDF weighting, since
    # state/region abbreviations recur often enough to retain real weight.
    # 0.70 sits in the gap between those two clusters of evidence.
    NAME_SIM_FLOOR = 0.70

    def _greedy_name_clusters(member_idx: list[int], names: list[str]) -> list[list[int]]:
        """
        Within one address-equivalence group, partitions members into
        sub-clusters by pairwise IDF-weighted name similarity. Greedy
        single-link clustering: a member joins the first existing cluster
        where its name similarity to that cluster's seed member clears
        NAME_SIM_FLOOR; otherwise it starts a new cluster. Members with no
        name value form their own singleton clusters, since there's no
        signal to group them on safely.
        Weights are computed once from all names in this address group, so
        a token frequent within the group (e.g. a shared corporate prefix)
        is down-weighted the same way for every comparison.
        """
        named = [(i, names[pos]) for pos, i in enumerate(member_idx) if names[pos]]
        unnamed = [i for pos, i in enumerate(member_idx) if not names[pos]]
        clusters: list[list[int]] = [[i] for i in unnamed]
        if not named:
            return clusters
        weight = _idf_weights_for([n for _, n in named])
        cluster_seed_name: list[str] = []
        cluster_members: list[list[int]] = []
        for idx, nm in named:
            placed = False
            for ci, seed in enumerate(cluster_seed_name):
                if _weighted_jaccard(nm, seed, weight) >= NAME_SIM_FLOOR:
                    cluster_members[ci].append(idx)
                    placed = True
                    break
            if not placed:
                cluster_seed_name.append(nm)
                cluster_members.append([idx])
        clusters.extend(cluster_members)
        return clusters

    def recluster_component(members: list[str]) -> list[list[str]]:
        """
        Splits an over-merged component into structurally-justified
        sub-groups instead of blanking it to singletons. Two-stage
        partition, mirroring the should_split gates:

          1. Partition by address-equivalence class (_equivalence_classes).
             Each distinct physical address/PO-Box-bucket becomes its own
             candidate group — this alone fixes chains spanning unrelated
             locations (e.g. NUTRIEN records scattered across many street
             addresses).
          2. Within each address group, sub-partition by name similarity
             (_greedy_name_clusters). This catches same-address chains
             where names diverge (e.g. many "Nutrien Ag Solutions- <city>"
             branch records at one corporate address) while keeping
             genuine exact-duplicate pairs together (e.g. two identical
             "NUTRIEN AG SOLUTIONS" records at that same address).

        Members with no address value are name-clustered directly against
        the whole component (no location signal to partition on first).
        Returns a list of member-id groups; singletons are valid output
        when nothing else in the component is structurally similar.
        """
        if len(members) <= 2:
            return [members]
        addr_pairs = normalized_addresses_for(members)
        names = normalized_names_for(members)
        has_addr_idx = [i for i, (raw, _) in enumerate(addr_pairs) if raw]
        no_addr_idx = [i for i in range(len(members)) if i not in set(has_addr_idx)]

        result: list[list[int]] = []
        if has_addr_idx and addr_fields:
            addr_subset = [addr_pairs[i] for i in has_addr_idx]
            classes = _equivalence_classes(addr_subset)
            for cls in classes:
                global_idx = [has_addr_idx[i] for i in cls]
                if name_fields and len(global_idx) > 2:
                    sub_names = [names[i] for i in global_idx]
                    result.extend(_greedy_name_clusters(global_idx, sub_names))
                else:
                    result.append(global_idx)
        else:
            no_addr_idx = list(range(len(members)))

        if no_addr_idx:
            if name_fields and len(no_addr_idx) > 2:
                sub_names = [names[i] for i in no_addr_idx]
                result.extend(_greedy_name_clusters(no_addr_idx, sub_names))
            else:
                result.append(no_addr_idx)

        return [[members[i] for i in group] for group in result]

    def should_split(members: list[str]) -> tuple[bool, str]:
        """Decide whether a component should be re-clustered. Returns (split, reason)."""
        if len(members) <= 2:
            # a direct pair always has a real edge between its two members — no chain risk
            return False, ""
        if len(members) <= MAX_MEMBERS_FOR_FULL_CHECK and (addr_fields or name_fields):
            addr_coh = address_cohesion(members)
            if addr_coh is not None and addr_coh < ADDRESS_COHESION_FLOOR:
                return True, (
                    f"address cohesion {addr_coh:.2f} < {ADDRESS_COHESION_FLOOR} "
                    f"(component spans too many distinct normalized addresses)"
                )
            name_div = name_divergence(members)
            if name_div is not None and name_div > NAME_DIVERGENCE_CEILING:
                return True, (
                    f"name divergence {name_div:.2f} > {NAME_DIVERGENCE_CEILING} "
                    f"(shared address, but names indicate separate entities)"
                )
            if addr_coh is not None or name_div is not None:
                # at least one structural gate ran and passed — trust it over
                # the generic score floor, since it's a more targeted signal
                return False, ""
        # fallback: no address/name fields available for this schema, or
        # component too large to check structurally — use the original
        # blended-score floor as a safety net.
        avg_score = component_avg_pairwise_score(members)
        if avg_score < COHESION_FLOOR:
            return True, f"avg pairwise score {avg_score:.3f} < {COHESION_FLOOR}"
        return False, ""

    def resolve_component(root: str, members: list[str], depth: int = 0) -> dict[str, list[str]]:
        """
        Recursively applies should_split + recluster_component. Each
        accepted sub-group is re-checked (depth-limited to avoid pathological
        recursion on adversarial input) so a re-clustered group that still
        fails the gates gets split further rather than accepted on one pass.
        """
        split, reason = should_split(members)
        if not split:
            return {root: members}
        if depth >= 4 or len(members) <= 2:
            # depth guard / nothing left to structurally divide — fall back
            # to singletons only as a last resort, not the default outcome
            out: dict[str, list[str]] = {}
            for i, rid in enumerate(members):
                out[f"{root}__s{i}"] = [rid]
            return out
        print(
            f"INFO: component of {len(members)} members re-clustered — {reason}."
        )
        sub_groups = recluster_component(members)
        out: dict[str, list[str]] = {}
        for gi, grp in enumerate(sub_groups):
            sub_root = f"{root}__c{gi}"
            out.update(resolve_component(sub_root, grp, depth + 1))
        return out

    final_root_to_members: dict[str, list[str]] = {}
    for root, members in root_to_members.items():
        final_root_to_members.update(resolve_component(root, members))

    root_to_members = final_root_to_members

    root_to_match_id: dict[str, str] = {}
    m_counter = 1
    u_counter = 1
    for root, members in root_to_members.items():
        if len(members) > 1:
            root_to_match_id[root] = f"M{m_counter:04d}"
            m_counter += 1
    for root, members in root_to_members.items():
        if len(members) == 1:
            root_to_match_id[root] = f"U{u_counter:04d}"
            u_counter += 1

    rid_to_root: dict[str, str] = {}
    for root, members in root_to_members.items():
        for rid in members:
            rid_to_root[rid] = root

    MATCH_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MATCH_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record_id", "match_id", "group_size"])
        w.writeheader()
        for rid in all_records:
            root = rid_to_root[rid]
            match_id = root_to_match_id[root]
            group_size = len(root_to_members[root])
            w.writerow({
                "record_id": rid,
                "match_id": match_id,
                "group_size": group_size,
            })

    matched = sum(1 for rid in all_records if len(root_to_members[rid_to_root[rid]]) > 1)
    print(
        f"Wrote {MATCH_SUMMARY_CSV} — "
        f"{len(all_records)} records, {matched} in matched groups, "
        f"{m_counter - 1} match groups, {u_counter - 1} unmatched singletons"
    )


def run_explore(input_path: Path | None = None) -> None:
    """Pipeline entry point (same as ``main``)."""
    main(input_path=input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build match_explorer.html from match_candidates and a customer CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the same CSV the MDM pipeline used (classification supplies record id column).",
    )
    cli_args = parser.parse_args()
    main(input_path=cli_args.input)
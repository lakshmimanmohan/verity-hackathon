"""
Build golden entities from scored match pairs: graph communities, golden records, metrics.
Record nodes use the same record label as ``build_graph`` (derived from the input CSV stem).
Survivorship uses semantic types from ``attribute_weights.json`` (data-driven), read from
the data folder (``config.data_dir()``). Ground truth uses that CSV and the record store as fallback.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import re
import sys
import uuid
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv

from column_classification import apply_global_content_overlay
from config import data_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = None
GOLDEN_CSV_PATH = data_dir() / "golden_records.csv"
COLUMN_CLASSIFICATION_NAME = "column_classification.json"

SKIP_NEO_PROPS = frozenset({"embedding"})

# Community detection edges: merge-quality signal types (incl. embedding-only neighbors).
COMMUNITY_DETECTION_MATCH_TYPES = frozenset(
    {"full_signal", "weak_signal", "embedding_only"}
)

# After community detection: cap golden-record size; larger groups split by confidence.
MAX_GOLDEN_COMMUNITY_SIZE = 15

# Above this node count, use label propagation (O(n)) instead of greedy modularity (O(n log n)).
COMMUNITY_DETECTION_MODULARITY_MAX_NODES = 50_000


def _process_rss_bytes() -> int:
    """Best-effort resident set size for the current process (bytes). Returns -1 if unknown."""
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return int(ru.ru_maxrss)
        if sys.platform.startswith("linux"):
            return int(ru.ru_maxrss) * 1024
    except (ImportError, OSError, ValueError):
        pass
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            GetCurrentProcess = kernel32.GetCurrentProcess
            GetCurrentProcess.restype = ctypes.wintypes.HANDLE
            GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [
                ctypes.wintypes.HANDLE,
                ctypes.c_void_p,
                ctypes.wintypes.DWORD,
            ]
            GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
        except (OSError, AttributeError, ValueError):
            pass
    return -1


def _format_memory_mb() -> str:
    b = _process_rss_bytes()
    if b < 0:
        return "RSS unavailable on this platform"
    return f"{b / (1024 * 1024):.1f} MiB RSS"


def resolve_input_csv(input_path: str | Path) -> Path:
    """Resolve CSV path; relative paths are taken from ``PROJECT_ROOT``."""
    p = Path(input_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def record_label_from_csv(csv_path: Path) -> str:
    """
    Derive the record label from the CSV filename stem: strip trailing
    ``_raw``, ``_clean``, ``_input``, ``_data`` (case-insensitive), PascalCase
    words, append ``Record``. E.g. ``customers_raw.csv`` → ``CustomerRecord``.
    """
    stem = csv_path.stem
    base = re.sub(r"[_\-](raw|clean|input|data)([_\-]\w+)?$", "", stem, flags=re.IGNORECASE)
    parts = [p for p in re.split(r"[_\s\-]+", base) if p]
    words: list[str] = []
    for p in parts:
        w = p.strip().lower()
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w.capitalize())
    return "".join(words) + "Record"


def canon_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def load_attribute_semantics(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return (logical_field -> semantic_type, columns_excluded_from_learning)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    cols = data.get("columns") or {}
    sem: dict[str, str] = {}
    for name, meta in cols.items():
        key = str(name).strip().lower()
        if not key:
            continue
        sem[key] = str(meta.get("semantic_type", "DESCRIPTIVE")).upper()
    excluded = [str(x).strip().lower() for x in (data.get("columns_excluded_from_learning") or [])]
    return sem, excluded


def resolve_column_classification_path(csv_path: Path) -> Path:
    path = data_dir() / COLUMN_CLASSIFICATION_NAME
    if path.is_file():
        return path
    raise SystemExit(
        f"Missing column classification file. Expected:\n"
        f"  {path}\n"
        "Run src/rule_learner.py on your CSV first."
    )


def load_column_classification(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def csv_row_field_ci(row: dict[str, Any], name_lower: str) -> str:
    for k, v in row.items():
        if str(k).strip().lower() == name_lower:
            return "" if v is None else str(v).strip()
    return ""


def load_ingestion_row_index(path: Path, record_id_column: str) -> dict[str, int]:
    """Record id (``record_id_column`` from classification) -> CSV row order for recency."""
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    rid_key = str(record_id_column).strip().lower()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            cid = csv_row_field_ci(row, rid_key)
            if cid:
                out[cid] = idx
    return out


def resolve_neo_property(
    logical_field: str,
    prop_keys: set[str],
    record_id_column_lower: str,
) -> str | None:
    """Map logical / CSV column name to a record property key in the store."""
    lf = logical_field.strip().lower()
    low = {k.lower(): k for k in prop_keys}
    if lf in low:
        return low[lf]
    rid = record_id_column_lower.strip().lower()
    if rid and lf == rid and "id" in prop_keys:
        return "id"
    return None


def discover_eval_label_column(classification: dict[str, Any]) -> str | None:
    """
    Ground-truth label column: ``semantic_types`` value ``ANSWER_KEY``, or reasoning
    indicates the answer_key exclusion rule (``[answer_key]``, etc.).
    """
    hits: list[str] = []
    sem = classification.get("semantic_types") or {}
    for col, st in sem.items():
        if str(st).strip().upper() == "ANSWER_KEY":
            hits.append(str(col).strip())
    reasoning = classification.get("reasoning") or {}
    for col, why in reasoning.items():
        w = str(why).lower()
        if "[answer_key]" in w or "answer-key" in w or "answer key" in w:
            hits.append(str(col).strip())
    uniq = sorted({h for h in hits if h})
    return uniq[0] if uniq else None


def survivorship_rule(semantic: str) -> str:
    s = semantic.upper()
    if s in {"IDENTIFIER", "QUASI_IDENTIFIER"}:
        return "mode"
    if s == "NAME":
        return "longest"
    if s in {"DESCRIPTIVE", "DATE", "STREET_ADDRESS"}:
        return "recency"
    return "mode"


def pick_mode(values: list[str]) -> tuple[str, list[str]]:
    nn = [v for v in values if v and str(v).strip()]
    if not nn:
        return "", []
    cnt = Counter(nn)
    best = max(cnt.keys(), key=lambda v: (cnt[v], v))
    alts = sorted({v for v in nn if v != best})
    return best, alts


def pick_longest(values: list[str]) -> tuple[str, list[str]]:
    nn = [v for v in values if v and str(v).strip()]
    if not nn:
        return "", []
    best = max(nn, key=lambda v: (len(v), v))
    alts = sorted({v for v in nn if v != best}, key=len, reverse=True)
    return best, alts


def pick_recency(
    member_ids: list[str],
    neo_field: str,
    rows: dict[str, dict[str, Any]],
    ingestion_index: dict[str, int],
) -> tuple[str, list[str]]:
    has_index = any(ingestion_index.get(mid, -1) >= 0 for mid in member_ids)
    best_key: tuple[int, str] = (-1, "")
    best_val = ""
    seen: list[str] = []
    for mid in member_ids:
        rr = rows.get(mid) or {}
        raw = rr.get(neo_field)
        v = "" if raw is None else str(raw).strip()
        if not v:
            continue
        seen.append(v)
        idx = int(ingestion_index.get(mid, -1)) if has_index else 0
        key = (idx, str(mid)) if has_index else (0, str(mid))
        if key > best_key:
            best_key = key
            best_val = v
    if not best_val:
        return "", []
    alts = sorted({x for x in seen if x != best_val})
    return best_val, alts


def survivorship_for_field(
    logical_field: str,
    semantic: str,
    neo_field: str,
    member_ids: list[str],
    rows: dict[str, dict[str, Any]],
    ingestion_index: dict[str, int],
) -> tuple[str, list[str]]:
    vals = []
    for mid in member_ids:
        rr = rows.get(mid) or {}
        raw = rr.get(neo_field)
        vals.append("" if raw is None else str(raw).strip())

    rule = survivorship_rule(semantic)
    if rule == "mode":
        return pick_mode(vals)
    if rule == "longest":
        return pick_longest(vals)
    return pick_recency(member_ids, neo_field, rows, ingestion_index)


def stream_match_candidates_into_weights_and_graph(
    G: nx.Graph,
    path: Path,
) -> tuple[dict[tuple[str, str], float], int]:
    """
    Stream ``match_candidates.csv`` with ``csv.DictReader`` and merge duplicate pairs to
    max score in ``score_lookup``. Whenever the running max for a pair improves, update
    ``G`` with ``add_edge`` / ``weight`` so edges are applied in one pass (no intermediate
    list of all edges). Only rows with ``match_type`` in ``COMMUNITY_DETECTION_MATCH_TYPES``
    contribute. Returns the count of CSV rows whose ``match_type`` is
    ``insufficient_evidence`` for logging.
    """
    weights: dict[tuple[str, str], float] = {}
    insufficient_evidence_rows = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mt = csv_row_field_ci(row, "match_type").strip().lower()
            if mt == "insufficient_evidence":
                insufficient_evidence_rows += 1
            if mt not in COMMUNITY_DETECTION_MATCH_TYPES:
                continue
            a = csv_row_field_ci(row, "record_a")
            b = csv_row_field_ci(row, "record_b")
            if not a or not b:
                continue
            score_raw = csv_row_field_ci(row, "score")
            try:
                s = float(score_raw) if score_raw != "" else 0.0
            except ValueError:
                s = 0.0
            k = canon_pair(a, b)
            prev = weights.get(k)
            if prev is not None and s <= prev:
                continue
            weights[k] = s
            if G.has_node(a) and G.has_node(b):
                G.add_edge(a, b, weight=s)
    return weights, insufficient_evidence_rows


def fetch_all_customer_ids(store, record_label: str) -> list[str]:
    return store.all_record_ids(record_label)


def fetch_customer_rows(
    store, ids: list[str], record_label: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not ids:
        return out
    rows = store.record_props_rows(list(ids), record_label)
    for r in rows:
        oid = str(r["id"])
        props = dict(r["props"] or {})
        props.pop("embedding", None)
        flat: dict[str, Any] = {"id": oid}
        for k, v in props.items():
            ks = str(k)
            if ks in SKIP_NEO_PROPS or ks.startswith("_"):
                continue
            if ks == "id":
                continue
            if isinstance(v, (list, dict)):
                continue
            if v is None:
                flat[ks] = ""
            elif isinstance(v, bool):
                flat[ks] = v
            elif isinstance(v, (int, float)):
                flat[ks] = v
            else:
                flat[ks] = str(v).strip()
        out[oid] = flat
    return out


def average_intra_community_score_for_member(
    member: str,
    community_members: list[str],
    score_lookup: dict[tuple[str, str], float],
) -> float:
    """Mean match score from ``member`` to every other node in the community (0 if no edge in lookup)."""
    others = [x for x in community_members if x != member]
    if not others:
        return 1.0
    total = sum(score_lookup.get(canon_pair(member, o), 0.0) for o in others)
    return float(total / len(others))


def split_oversized_communities(
    communities: list[Any],
    G: nx.Graph,
    score_lookup: dict[tuple[str, str], float],
    max_members: int = MAX_GOLDEN_COMMUNITY_SIZE,
) -> list[set[str]]:
    """
    If any community has more than ``max_members`` nodes, keep the top ``max_members`` by
    average intra-community pairwise score as one cluster; re-cluster the rest on the
    induced subgraph (or singletons when there are no edges).
    """
    out: list[set[str]] = []
    for comm in communities:
        members = sorted(str(x) for x in comm)
        if len(members) <= max_members:
            out.append(set(members))
            continue
        ranked = sorted(
            members,
            key=lambda m: (
                -average_intra_community_score_for_member(m, members, score_lookup),
                m,
            ),
        )
        core = ranked[:max_members]
        rest = ranked[max_members:]
        print(
            f"WARNING: Community of {len(members)} members split — likely "
            f"over-merged. Review recommended."
        )
        out.append(set(core))
        if not rest:
            continue
        H = G.subgraph(rest).copy()
        sub = list(
            nx.algorithms.community.greedy_modularity_communities(H, weight="weight")
        )
        for sc in sub:
            out.append({str(x) for x in sc})
    return out


def community_pairwise_confidence(
    member_ids: list[str],
    score_lookup: dict[tuple[str, str], float],
) -> float:
    if len(member_ids) <= 1:
        return 1.0
    scores: list[float] = []
    for u, v in combinations(sorted(member_ids), 2):
        k = canon_pair(u, v)
        s = score_lookup.get(k)
        if s is not None:
            scores.append(s)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


GOLDEN_UF_SENTINEL = "__UF_GOLDEN__:"


def load_csv_record_ids(path: Path, record_id_column: str) -> set[str]:
    """Distinct non-empty record ids from the CSV (same keying as merge survivorship)."""
    out: set[str] = set()
    rid_l = str(record_id_column).strip().lower()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = csv_row_field_ci(row, rid_l)
            if cid:
                out.add(cid)
    return out


def load_existing_golden_artifacts(
    path: Path,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, dict[str, str]]]:
    """
    From ``golden_records.csv``: member_record_id -> golden_id, golden_id -> member set,
    and golden_id -> raw CSV row (string values) for rows we do not rewrite.
    """
    member_to_golden: dict[str, str] = {}
    golden_to_members: dict[str, set[str]] = {}
    row_by_golden: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return member_to_golden, golden_to_members, row_by_golden
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = csv_row_field_ci(row, "golden_id").strip()
            if not gid:
                continue
            raw_members = csv_row_field_ci(row, "member_record_ids").strip()
            members = {m.strip() for m in raw_members.split(";") if m.strip()}
            golden_to_members[gid] = members
            row_by_golden[gid] = {k: "" if v is None else str(v) for k, v in row.items()}
            for m in members:
                if m not in member_to_golden:
                    member_to_golden[m] = gid
    return member_to_golden, golden_to_members, row_by_golden


class _StringUF:
    """Disjoint-set for string keys (record ids and golden sentinels)."""

    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _incremental_golden_work_items(
    match_path: Path,
    new_ids: set[str],
    member_to_golden: dict[str, str],
    golden_to_members: dict[str, set[str]],
) -> tuple[list[tuple[str, list[str]]], set[str]]:
    """
    Union-find on new records (and sentinel keys for existing goldens) from merge-quality
    pairs only. Returns (list of (golden_id, sorted_member_ids), absorbed_existing_golden_ids)
    when multiple prior goldens merge into one canonical id.
    """
    uf = _StringUF()
    for nid in new_ids:
        uf.add(nid)

    with match_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mt = csv_row_field_ci(row, "match_type").strip().lower()
            if mt not in COMMUNITY_DETECTION_MATCH_TYPES:
                continue
            a = csv_row_field_ci(row, "record_a")
            b = csv_row_field_ci(row, "record_b")
            if not a or not b:
                continue
            na = a in new_ids
            nb = b in new_ids
            if not na and not nb:
                continue
            if na and nb:
                uf.union(a, b)
            elif na and not nb:
                gid_b = member_to_golden.get(b)
                if gid_b:
                    uf.union(a, f"{GOLDEN_UF_SENTINEL}{gid_b}")
                else:
                    uf.union(a, b)
            else:
                gid_a = member_to_golden.get(a)
                if gid_a:
                    uf.union(b, f"{GOLDEN_UF_SENTINEL}{gid_a}")
                else:
                    uf.union(a, b)

    nodes = list(uf.parent.keys())
    root_to_nodes: dict[str, list[str]] = {}
    for n in nodes:
        r = uf.find(n)
        root_to_nodes.setdefault(r, []).append(n)

    work: list[tuple[str, list[str]]] = []
    absorbed: set[str] = set()
    for _root, comp in root_to_nodes.items():
        sentinel_gids: list[str] = []
        regular: list[str] = []
        for n in comp:
            if n.startswith(GOLDEN_UF_SENTINEL):
                sentinel_gids.append(n[len(GOLDEN_UF_SENTINEL) :])
            else:
                regular.append(n)
        if sentinel_gids:
            sg_set = set(sentinel_gids)
            canonical = min(sg_set)
            for g in sg_set:
                if g != canonical:
                    absorbed.add(g)
            members: set[str] = set(regular)
            for g in sg_set:
                members |= golden_to_members.get(g, set())
            work.append((canonical, sorted(members)))
        else:
            gid_new = str(uuid.uuid4())
            work.append((gid_new, sorted(set(regular))))

    return work, absorbed


def load_true_labels_from_raw(
    path: Path,
    record_id_column: str,
    label_column: str,
) -> dict[str, int | None]:
    """Map record id -> integer label from raw CSV (``label_column`` is usually an answer-key field)."""
    by_id: dict[str, int | None] = {}
    if not path.is_file():
        return by_id
    rid_l = str(record_id_column).strip().lower()
    lbl_l = str(label_column).strip().lower()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = csv_row_field_ci(row, rid_l)
            if not cid:
                continue
            te = csv_row_field_ci(row, lbl_l)
            try:
                by_id[cid] = int(te) if te != "" else None
            except ValueError:
                by_id[cid] = None
    return by_id


def pairwise_cluster_metrics(
    record_ids: list[str],
    pred_community: dict[str, int],
    true_label: dict[str, int | None],
) -> tuple[int, int, int, int, float, float, float]:
    TP = FP = FN = TN = 0
    n = len(record_ids)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = record_ids[i], record_ids[j]
            same_pred = pred_community[a] == pred_community[b]
            ta, tb = true_label.get(a), true_label.get(b)
            same_true = (
                ta is not None
                and tb is not None
                and ta == tb
            )
            if same_pred and same_true:
                TP += 1
            elif same_pred and not same_true:
                FP += 1
            elif (not same_pred) and same_true:
                FN += 1
            else:
                TN += 1

    if TP + FP == 0:
        precision = 1.0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 1.0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return TP, FP, FN, TN, precision, recall, f1


def _execute_incremental_merge(
    *,
    eval_path: Path,
    record_label: str,
    record_id_column: str,
    record_id_lower: str,
    content_logical: frozenset[str],
    metadata_logical: frozenset[str],
    semantics: dict[str, str],
    ingestion_index: dict[str, int],
    match_path: Path,
    new_record_ids: set[str] | None,
) -> dict[str, Any]:
    """
    Delta merge: assign new records to existing goldens via ``match_candidates`` pairs,
    union-find new–new links, singletons for unmatched new ids. Preserves existing
    ``golden_id`` values; does not run graph-wide community detection or clear stored
    golden nodes.
    """
    new_ids: set[str] = (
        {str(x) for x in new_record_ids}
        if new_record_ids is not None
        else load_csv_record_ids(eval_path, record_id_column)
    )
    if not new_ids:
        raise SystemExit("Incremental merge: no new record ids (empty input CSV).")

    member_to_golden, golden_to_members, row_by_golden = load_existing_golden_artifacts(
        GOLDEN_CSV_PATH
    )
    empty_g = nx.Graph()
    score_lookup, n_insufficient_evidence = (
        stream_match_candidates_into_weights_and_graph(empty_g, match_path)
    )
    print(
        f"(Incremental) Using {len(score_lookup)} merged pair keys from CSV "
        f"({n_insufficient_evidence} insufficient_evidence rows); "
        "skipping graph-wide community detection."
    )

    work_raw, absorbed = _incremental_golden_work_items(
        match_path, new_ids, member_to_golden, golden_to_members
    )
    by_gid: dict[str, set[str]] = {}
    for gid, mems in work_raw:
        by_gid.setdefault(gid, set()).update(mems)
    final_items = [(g, sorted(ms)) for g, ms in sorted(by_gid.items())]

    pre_existing = set(row_by_golden.keys())
    n_new = sum(1 for g, _ in final_items if g not in pre_existing)
    written_existing = {g for g, _ in final_items if g in pre_existing}
    n_updated = len(written_existing) + len(absorbed)

    from record_store import open_store

    store = open_store()

    diversity_rows: list[dict[str, Any]] = []
    fieldname_union: set[str] = {
        "golden_id",
        "member_record_ids",
        "member_count",
        "confidence",
        "alternates_json",
    }
    rows_by_id: dict[str, dict[str, Any]] = {}
    golden_jsonl_tmp = (
        GOLDEN_CSV_PATH.parent / f"{GOLDEN_CSV_PATH.stem}_incr_buffer.jsonl"
    )
    GOLDEN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with golden_jsonl_tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as golden_jf:
        store.prepare_golden(clear=False)

        for gid, member_ids in final_items:
            batch = fetch_customer_rows(store, member_ids, record_label)
            rows_by_id.update(batch)

            sample_keys: set[str] = set()
            for mid in member_ids:
                sample_keys |= set((batch.get(mid) or {}).keys())

            survivorship_targets: list[tuple[str, str, str]] = []
            for logical in sorted(content_logical & set(semantics.keys())):
                sem = semantics[logical]
                neo_f = resolve_neo_property(logical, sample_keys, record_id_lower)
                if neo_f is None or neo_f == "id":
                    continue
                survivorship_targets.append((logical, sem, neo_f))

            best: dict[str, str] = {}
            alts: dict[str, list[str]] = {}
            for logical, sem, neo_f in survivorship_targets:
                primary, alt_list = survivorship_for_field(
                    logical, sem, neo_f, member_ids, batch, ingestion_index
                )
                best[neo_f] = primary
                alts[neo_f] = alt_list

            rep_id = member_ids[0]
            rep_row = batch.get(rep_id) or {}
            for logical in sorted(metadata_logical):
                neo_f = resolve_neo_property(logical, sample_keys, record_id_lower)
                if neo_f is None or neo_f == "id":
                    continue
                if neo_f in best:
                    continue
                raw = rep_row.get(neo_f)
                v = "" if raw is None else str(raw).strip()
                if v != "":
                    best[neo_f] = v

            conf = community_pairwise_confidence(member_ids, score_lookup)

            dyn_props: dict[str, Any] = {}
            for neo_f, val in best.items():
                if val != "":
                    dyn_props[neo_f] = val
            for neo_f, alt_list in alts.items():
                if alt_list:
                    dyn_props[f"alts_{neo_f}"] = list(alt_list)

            payload = {
                "golden_id": gid,
                "confidence": float(conf),
                "member_count": len(member_ids),
                "dyn_props": dyn_props,
                "member_ids": member_ids,
            }

            store.write_golden(payload, record_label)

            ss_neo = resolve_neo_property("source_system", sample_keys, record_id_lower)
            ss_key = ss_neo if ss_neo is not None else "source_system"
            distinct_sources = sorted(
                {
                    (batch[mid].get(ss_key) or "").strip()
                    for mid in member_ids
                }
                - {""}
            )
            diversity_rows.append(
                {
                    "golden_id": gid,
                    "n_sources": len(distinct_sources),
                    "sources_display": ";".join(distinct_sources),
                    "member_count": len(member_ids),
                }
            )

            alts_compact = {k: v for k, v in alts.items() if v}
            csv_row: dict[str, Any] = {
                "golden_id": gid,
                "member_record_ids": ";".join(member_ids),
                "member_count": len(member_ids),
                "confidence": f"{conf:.6f}",
                "alternates_json": json.dumps(alts_compact, sort_keys=True),
            }
            for neo_f, val in best.items():
                csv_row[neo_f] = val
            fieldname_union.update(csv_row.keys())
            json.dump(csv_row, golden_jf, ensure_ascii=False)
            golden_jf.write("\n")

    store.close()

    written_ids = {g for g, _ in final_items}
    out_rows_by_gid: dict[str, dict[str, str]] = {}
    for g, old_row in row_by_golden.items():
        if g in absorbed:
            continue
        if g in written_ids:
            continue
        out_rows_by_gid[g] = dict(old_row)

    with golden_jsonl_tmp.open("r", encoding="utf-8") as jf:
        for line in jf:
            line = line.strip()
            if not line:
                continue
            gr = json.loads(line)
            gid = str(gr["golden_id"])
            out_rows_by_gid[gid] = {
                k: "" if v is None else str(v) for k, v in gr.items()
            }

    extra_keys = {k for k in fieldname_union if k not in {"alternates_json"}}
    fieldnames = [
        "golden_id",
        "member_record_ids",
        "member_count",
        "confidence",
        *sorted(
            k
            for k in extra_keys
            if k
            not in {
                "golden_id",
                "member_record_ids",
                "member_count",
                "confidence",
            }
        ),
        "alternates_json",
    ]
    out_rows = [out_rows_by_gid[k] for k in sorted(out_rows_by_gid.keys())]

    with GOLDEN_CSV_PATH.open("w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for gr in out_rows:
            writer.writerow({k: gr.get(k, "") for k in fieldnames})

    try:
        golden_jsonl_tmp.unlink()
    except OSError:
        pass

    print(
        f"Incremental merge: {n_new} new golden records created, "
        f"{n_updated} existing golden records updated"
    )
    print(f"Wrote {len(out_rows)} total golden rows -> {GOLDEN_CSV_PATH}")

    community_sizes: list[int] = []
    for gr in out_rows:
        raw_mc = str(gr.get("member_count", "0")).strip()
        try:
            community_sizes.append(int(raw_mc))
        except ValueError:
            community_sizes.append(0)

    diversity_rows.sort(
        key=lambda r: (-r["n_sources"], -r["member_count"], r["golden_id"])
    )
    top_merged = diversity_rows[:5]

    return {
        "n_customer_records": len(new_ids),
        "n_golden": len(out_rows),
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "pairwise_eval_skipped": True,
        "pairwise_eval_skip_reason": "incremental_merge",
        "community_sizes": community_sizes,
        "top_merged_by_sources": top_merged,
    }


def main(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    record_label_from_csv_path: str | Path | None = None,
    *,
    incremental: bool = False,
    new_record_ids: set[str] | None = None,
) -> dict[str, Any]:
    """
    Read ``attribute_weights.json``, ``column_classification.json``, and
    ``match_candidates.csv`` from the data folder (``config.data_dir()``).
    ``content_columns`` are unioned with ``column_classification_global.json`` there when
    that file exists. Survivorship runs only on those content columns; ``metadata_columns``
    are copied from one representative member record without merge rules.
    When ``incremental`` is True, skip full community detection; update ``golden_records.csv``
    and stored goldens using delta pair logic (see ``_execute_incremental_merge``).
    """
    eval_path = resolve_input_csv(input_path)
    if not eval_path.is_file():
        raise SystemExit(f"Input CSV not found: {eval_path}")
    with eval_path.open(newline="", encoding="utf-8-sig") as f:
        if not f.readline().strip():
            raise SystemExit(f"Input CSV has no header row: {eval_path}")

    label_path = (
        resolve_input_csv(record_label_from_csv_path)
        if record_label_from_csv_path is not None
        else eval_path
    )
    record_label = record_label_from_csv(label_path)
    weights_path = data_dir() / "attribute_weights.json"
    match_path = data_dir() / "match_candidates.csv"

    load_dotenv(PROJECT_ROOT / ".env")

    if not weights_path.is_file():
        raise SystemExit(f"Missing {weights_path}. Run rule_learner.py first.")

    cls_path = resolve_column_classification_path(eval_path)
    classification = apply_global_content_overlay(load_column_classification(cls_path))
    semantics, _excluded_raw = load_attribute_semantics(weights_path)

    record_id_column = str(classification.get("record_id_column") or "").strip()
    if not record_id_column:
        raise SystemExit(f"{cls_path} is missing record_id_column.")
    record_id_lower = record_id_column.lower()
    content_logical = frozenset(
        str(c).strip().lower()
        for c in (classification.get("content_columns") or [])
        if str(c).strip()
    )
    metadata_logical = frozenset(
        str(c).strip().lower()
        for c in (classification.get("metadata_columns") or [])
        if str(c).strip()
    )
    if not content_logical:
        raise SystemExit(f"{cls_path} has no content_columns.")

    print(f"Loaded attribute_weights: {weights_path}")
    print(
        f"Loaded column_classification: {cls_path}\n"
        f"  record_id_column={record_id_column!r}, "
        f"content={len(content_logical)}, metadata={len(metadata_logical)}"
    )

    ingestion_index = load_ingestion_row_index(eval_path, record_id_column)

    label_col = discover_eval_label_column(classification)
    eval_skip = label_col is None

    if not match_path.is_file():
        raise SystemExit(f"Missing {match_path}")

    if incremental:
        return _execute_incremental_merge(
            eval_path=eval_path,
            record_label=record_label,
            record_id_column=record_id_column,
            record_id_lower=record_id_lower,
            content_logical=content_logical,
            metadata_logical=metadata_logical,
            semantics=semantics,
            ingestion_index=ingestion_index,
            match_path=match_path,
            new_record_ids=new_record_ids,
        )

    diversity_rows: list[dict[str, Any]] = []

    from record_store import open_store

    store = open_store()
    all_ids = fetch_all_customer_ids(store, record_label)

    if not all_ids:
        store.close()
        raise SystemExit(f"No {record_label} records in the record store.")

    G = nx.Graph()
    G.add_nodes_from(all_ids)
    score_lookup, n_insufficient_evidence = stream_match_candidates_into_weights_and_graph(
        G, match_path
    )
    print(
        f"Using {len(score_lookup)} merged pair keys from CSV "
        f"({n_insufficient_evidence} insufficient_evidence rows); "
        f"graph has {G.number_of_edges()} edges between records"
    )
    print(
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
        f"(from {match_path.name}, streamed with csv.DictReader + nx.add_edge)"
    )

    print(f"Memory before community detection: {_format_memory_mb()}")
    n_nodes = G.number_of_nodes()
    if n_nodes > COMMUNITY_DETECTION_MODULARITY_MAX_NODES:
        print(
            f"Community detection: label_propagation_communities "
            f"({n_nodes} nodes > {COMMUNITY_DETECTION_MODULARITY_MAX_NODES}: "
            "greedy modularity is O(n log n) and is slower at this scale; "
            "label propagation is O(n) and scales better for large graphs.)"
        )
        communities = list(
            nx.algorithms.community.label_propagation_communities(G)
        )
        cd_algo_name = "label propagation"
    else:
        print(
            f"Community detection: greedy_modularity_communities "
            f"({n_nodes} nodes <= {COMMUNITY_DETECTION_MODULARITY_MAX_NODES}: "
            "modularity optimization fits this graph size.)"
        )
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        )
        cd_algo_name = "greedy modularity"
    n_comm_raw = len(communities)
    communities = split_oversized_communities(communities, G, score_lookup)
    print(f"Memory after community detection: {_format_memory_mb()}")
    print(
        f"Communities ({cd_algo_name}): {n_comm_raw} raw; "
        f"{len(communities)} after max-{MAX_GOLDEN_COMMUNITY_SIZE} member sanity pass"
    )

    pred: dict[str, int] = {}
    for idx, comm in enumerate(communities):
        for rid in comm:
            pred[str(rid)] = idx

    GOLDEN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    golden_jsonl_tmp = GOLDEN_CSV_PATH.parent / f"{GOLDEN_CSV_PATH.stem}_write_buffer.jsonl"
    fieldname_union: set[str] = {
        "golden_id",
        "member_record_ids",
        "member_count",
        "confidence",
        "alternates_json",
    }
    n_golden_written = 0

    with golden_jsonl_tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as golden_jf:
        store.prepare_golden(clear=True)

        rows_by_id: dict[str, dict[str, Any]] = {}

        for comm in communities:
            member_ids = sorted(str(x) for x in comm)
            batch = fetch_customer_rows(store, member_ids, record_label)
            rows_by_id.update(batch)

            sample_keys = set()
            for mid in member_ids:
                sample_keys |= set((batch.get(mid) or {}).keys())

            survivorship_targets: list[tuple[str, str, str]] = []
            for logical in sorted(content_logical & set(semantics.keys())):
                sem = semantics[logical]
                neo_f = resolve_neo_property(logical, sample_keys, record_id_lower)
                if neo_f is None or neo_f == "id":
                    continue
                survivorship_targets.append((logical, sem, neo_f))

            best: dict[str, str] = {}
            alts: dict[str, list[str]] = {}
            for logical, sem, neo_f in survivorship_targets:
                primary, alt_list = survivorship_for_field(
                    logical, sem, neo_f, member_ids, batch, ingestion_index
                )
                best[neo_f] = primary
                alts[neo_f] = alt_list

            rep_id = member_ids[0]
            rep_row = batch.get(rep_id) or {}
            for logical in sorted(metadata_logical):
                neo_f = resolve_neo_property(logical, sample_keys, record_id_lower)
                if neo_f is None or neo_f == "id":
                    continue
                if neo_f in best:
                    continue
                raw = rep_row.get(neo_f)
                v = "" if raw is None else str(raw).strip()
                if v != "":
                    best[neo_f] = v

            conf = community_pairwise_confidence(member_ids, score_lookup)
            gid = str(uuid.uuid4())

            dyn_props: dict[str, Any] = {}
            for neo_f, val in best.items():
                if val != "":
                    dyn_props[neo_f] = val
            for neo_f, alt_list in alts.items():
                if alt_list:
                    dyn_props[f"alts_{neo_f}"] = list(alt_list)

            payload = {
                "golden_id": gid,
                "confidence": float(conf),
                "member_count": len(member_ids),
                "dyn_props": dyn_props,
                "member_ids": member_ids,
            }

            store.write_golden(payload, record_label)

            ss_neo = resolve_neo_property("source_system", sample_keys, record_id_lower)
            ss_key = ss_neo if ss_neo is not None else "source_system"
            distinct_sources = sorted(
                {
                    (batch[mid].get(ss_key) or "").strip()
                    for mid in member_ids
                }
                - {""}
            )
            diversity_rows.append(
                {
                    "golden_id": gid,
                    "n_sources": len(distinct_sources),
                    "sources_display": ";".join(distinct_sources),
                    "member_count": len(member_ids),
                }
            )

            alts_compact = {k: v for k, v in alts.items() if v}
            csv_row: dict[str, Any] = {
                "golden_id": gid,
                "member_record_ids": ";".join(member_ids),
                "member_count": len(member_ids),
                "confidence": f"{conf:.6f}",
                "alternates_json": json.dumps(alts_compact, sort_keys=True),
            }
            for neo_f, val in best.items():
                csv_row[neo_f] = val
            fieldname_union.update(csv_row.keys())
            json.dump(csv_row, golden_jf, ensure_ascii=False)
            golden_jf.write("\n")
            n_golden_written += 1

    store.close()

    extra_keys = {k for k in fieldname_union if k not in {"alternates_json"}}
    fieldnames = [
        "golden_id",
        "member_record_ids",
        "member_count",
        "confidence",
        *sorted(k for k in extra_keys if k not in {"golden_id", "member_record_ids", "member_count", "confidence"}),
        "alternates_json",
    ]
    with (
        golden_jsonl_tmp.open("r", encoding="utf-8") as jf,
        GOLDEN_CSV_PATH.open("w", newline="", encoding="utf-8") as outf,
    ):
        writer = csv.DictWriter(outf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for line in jf:
            line = line.strip()
            if not line:
                continue
            gr = json.loads(line)
            writer.writerow({k: gr.get(k, "") for k in fieldnames})
    try:
        golden_jsonl_tmp.unlink()
    except OSError:
        pass

    print(f"Wrote {n_golden_written} golden rows -> {GOLDEN_CSV_PATH}")

    lbl_neo_global: str | None = None
    if label_col is not None:
        sk_all: set[str] = set()
        for rr in rows_by_id.values():
            sk_all |= set(rr.keys())
        lbl_neo_global = resolve_neo_property(
            str(label_col).strip().lower(), sk_all, record_id_lower
        )

    if eval_skip:
        TP = FP = FN = TN = 0
        precision = recall = f1 = None
        print(
            "\nPairwise merge quality vs ground-truth labels: skipped "
            "(no column with semantic_type ANSWER_KEY or answer_key in classification reasoning)"
        )
    else:
        assert label_col is not None
        true_from_raw = load_true_labels_from_raw(
            eval_path, record_id_column, label_col
        )
        true_label: dict[str, int | None] = {}
        for rid in all_ids:
            if rid in true_from_raw:
                true_label[rid] = true_from_raw[rid]
                continue
            rr = rows_by_id.get(rid)
            tid = None
            if rr is not None and lbl_neo_global is not None:
                tid = rr.get(lbl_neo_global)
            if tid is not None:
                try:
                    true_label[rid] = int(tid)
                except (TypeError, ValueError):
                    true_label[rid] = None
            else:
                true_label[rid] = None

        TP, FP, FN, TN, precision, recall, f1 = pairwise_cluster_metrics(
            all_ids, pred, true_label
        )
        print(f"\nPairwise merge quality vs label column {label_col!r} (raw + record store fallback):")
        print(f"  TP={TP} FP={FP} FN={FN} TN={TN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
    print("Done.")

    community_sizes = [len(c) for c in communities]
    diversity_rows.sort(
        key=lambda r: (-r["n_sources"], -r["member_count"], r["golden_id"])
    )
    top_merged = diversity_rows[:5]

    return {
        "n_customer_records": len(all_ids),
        "n_golden": n_golden_written,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "tn": TN,
        "pairwise_eval_skipped": eval_skip,
        "community_sizes": community_sizes,
        "top_merged_by_sources": top_merged,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build golden records from match candidates and the record store."
    )
    ap.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "Source customer CSV path (absolute, or relative to project root); "
            "weights and match_candidates are read from its directory. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    ap.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Delta merge: keep existing golden_ids, assign from match_candidates only "
            "(no full community detection)."
        ),
    )
    args = ap.parse_args()
    main(
        input_path=args.csv_path if args.csv_path is not None else DEFAULT_INPUT_PATH,
        incremental=bool(args.incremental),
    )

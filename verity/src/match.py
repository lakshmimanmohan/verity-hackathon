"""
Score candidate record-node pairs from embedding neighbors + graph overlap.
Record node labels match ``build_graph`` (derived from the input CSV stem).
Loads discriminative weights and match threshold from rule_learner JSON outputs.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import pickle
import re
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors

try:
    from annoy import AnnoyIndex

    _ANNOY_AVAILABLE = True
except ImportError:
    AnnoyIndex = None  # type: ignore[misc, assignment]
    _ANNOY_AVAILABLE = False

from column_classification import apply_global_content_overlay, col_key
from config import data_dir, nn_mode, record_timing, update_run_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = None
OUTPUT_PATH = data_dir() / "match_candidates.csv"
# Align with ``merge.COMMUNITY_DETECTION_MATCH_TYPES`` — only these rows become merge edges.
MATCH_CANDIDATES_WRITE_MATCH_TYPES = frozenset(
    {"full_signal", "weak_signal", "embedding_only", "code_signal"}
)
MATCH_CANDIDATES_FIELDNAMES = [
    "record_a",
    "record_b",
    "score",
    "shared_attributes",
    "match_type",
    "evidence_mass",
    "conflict_mass",
    "evidence_detail",
    "field_contributions",
    "field_similarities",
]
EXACT_DUPLICATE_SHARED_ATTRIBUTES = "exact_duplicate"
# Every scored pair, whatever its match_type, plus exact-duplicate pairs; one row per pair.
SCORED_PAIRS_PATH = data_dir() / "scored_pairs.csv"
# Written once per run: column -> semantic type, and the constants of the score formula.
SCORED_PAIRS_COLUMNS_PATH = data_dir() / "scored_pairs_columns.json"
SCORED_PAIRS_FIELDNAMES = [
    "record_a",
    "record_b",
    "score",
    "score_uncapped",
    "cosine",
    "match_type",
    "decision_status",
    "pair_source",
    "applied_threshold",
    "effective_threshold",
    "evidence_passes",
    "evidence_mass",
    "conflict_mass",
    "field_similarities",
    "field_weights",
    "evidence_field_similarities",
]
# Review band for scored_pairs.csv decision_status: a pair is "review" when its score is
# within this margin of the effective threshold, inclusive (protocol.md borderline band).
REVIEW_MARGIN = 0.05
# Semantic types whose field similarity blends the record cosine with char Jaccard.
HYBRID_SEMANTIC_TYPES = ("DESCRIPTIVE", "NAME", "STREET_ADDRESS")
COLUMN_CLASSIFICATION_NAME = "column_classification.json"
DATA_COLUMN_CLASSIFICATION_PATH = data_dir() / COLUMN_CLASSIFICATION_NAME
GLOBAL_EMBEDDINGS_PATH = data_dir() / "embeddings_global.pkl"
MERGED_INPUT_CSV = data_dir() / "merged_input.csv"
# Must match ``embed.EMBEDDINGS_STORE_META_KEY`` (not a record id).
EMBEDDINGS_STORE_META_KEY = "__embeddings_store_meta__"
# Optional: ``{"files": {"customers_raw_1.csv": ["C1", "C2"], ...}}`` or flat ``{"C1": "file_a.csv"}``.
EMBEDDINGS_FILE_REGISTRY_PATH = data_dir() / "embeddings_file_registry.json"

TOP_K = 30
APPROXIMATE_NN_THRESHOLD = 10000
_ID_BUCKET_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
# Minimum score floor: effective threshold = max(learned_threshold, this).
LEARNED_THRESHOLD_MIN_FLOOR = 0.82

# Cosine vs field blend: cap embedding weight so scores lean on shared field values.
EMBED_BLEND_MIN = 0.12
EMBED_BLEND_MAX = 0.65  # implies field weight >= 0.35
HYBRID_EMBED_WEIGHT = 0.65
HYBRID_CHAR_WEIGHT = 0.35

# Schema-agnostic evidence gates.
# These are intentionally not tied to column names.
IDENTITY_FIELD_SIM_MIN = 0.85
DESCRIPTIVE_FIELD_SIM_MIN = 0.85
CONFLICT_FIELD_SIM_MAX = 0.50
MIN_EVIDENCE_MASS = 0.35
MAX_CONFLICT_MASS = 0.25
SEMANTIC_ONLY_COSINE_MIN = 0.88
EMBEDDING_ONLY_COSINE_MIN = 0.92
FIELD_BLEND_MIN = 1.0 - EMBED_BLEND_MAX
CSV_FALLBACK_JACCARD_MIN = 0.8

# Code path: pairs whose text fields hold no multi-word value (part numbers, codes, ids) are
# compared on code-like columns only. Chosen per run from the data, never by column name.
CODE_NOSPACE_SHARE_MIN = 0.90          # a content column is code-like when this share of its non-empty values has no space
_CODE_DROP = re.compile(r"[\W_]+")

# Internal delimiter for attribute fingerprint (av.type + sep + av.value)
_ATTR_SEP = "\x1f"


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


def resolve_match_column_classification_path(csv_path: Path) -> Path:
    """
    Prefer ``{{input_stem}}/column_classification.json`` in the data folder (file-scoped
    artifact). Fall back to ``column_classification.json`` there with a warning when missing.
    """
    stem_scoped = data_dir() / csv_path.stem / COLUMN_CLASSIFICATION_NAME
    if stem_scoped.is_file():
        return stem_scoped
    print(
        f"Warning: file-scoped column classification not found at {stem_scoped} — "
        f"falling back to {DATA_COLUMN_CLASSIFICATION_PATH}"
    )
    if DATA_COLUMN_CLASSIFICATION_PATH.is_file():
        return DATA_COLUMN_CLASSIFICATION_PATH
    raise SystemExit(
        f"Missing column classification file. Expected:\n"
        f"  {stem_scoped}\n"
        f"or:\n"
        f"  {DATA_COLUMN_CLASSIFICATION_PATH}\n"
        "Run src/rule_learner.py on your CSV first."
    )

def load_column_classification(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def graph_label_attribute_key(label: str) -> str:
    """Lowercase attribute/column name from a graph shared label ``type:value``."""
    s = str(label).strip()
    if ":" in s:
        return s.split(":", 1)[0].strip().lower()
    return s.lower()


def drop_metadata_shared_labels(
    labels: list[str],
    metadata_columns_lower: frozenset[str],
) -> list[str]:
    """Remove graph overlap labels whose attribute type is listed as metadata."""
    return [
        lb
        for lb in labels
        if graph_label_attribute_key(lb) not in metadata_columns_lower
    ]


def drop_metadata_csv_columns(
    cols: list[str],
    metadata_columns_lower: frozenset[str],
) -> list[str]:
    """Remove column names listed as metadata (defensive; csv fallback uses content_columns)."""
    return [c for c in cols if col_key(c) not in metadata_columns_lower]


def shared_content_attribute_keys_graph(labels: list[str]) -> frozenset[str]:
    """Distinct attribute/column names in graph overlap labels (metadata already dropped)."""
    return frozenset(graph_label_attribute_key(lb) for lb in labels)


def shared_content_attribute_keys_csv(columns: list[str]) -> frozenset[str]:
    """Distinct CSV column names, lowercased (metadata already dropped)."""
    return frozenset(col_key(c) for c in columns)


def load_column_semantic_types(
    classification: dict[str, Any],
    weights_path: Path,
) -> dict[str, str]:
    """
    Per-column ``semantic_type`` from ``column_classification.json`` key ``semantic_types``
    (written by rule_learner). Keys are normalized to lowercase for lookup.

    Legacy files without ``semantic_types`` fall back to ``attribute_weights.json``.
    """
    out: dict[str, str] = {}
    raw = classification.get("semantic_types")
    if isinstance(raw, dict):
        for k, v in raw.items():
            lk = col_key(k)
            if lk:
                out[lk] = str(v).strip().upper()
    if out:
        return out
    if weights_path.is_file():
        with weights_path.open(encoding="utf-8") as f:
            wj: dict[str, Any] = json.load(f)
        for col_name, meta in (wj.get("columns") or {}).items():
            lk = col_key(col_name)
            if lk:
                out[lk] = str(meta.get("semantic_type", "DESCRIPTIVE")).strip().upper()
    return out


@dataclass(frozen=True)
class EvidenceDecision:
    """Schema-agnostic evidence decision for a candidate pair."""

    passes: bool
    match_type: str | None
    evidence_mass: float
    conflict_mass: float
    detail: str


def evaluate_schema_agnostic_evidence(
    *,
    shared_keys: frozenset[str],
    field_similarities: dict[str, float],
    semantic_by_col_lower: dict[str, str],
    discriminative_by_col: dict[str, float],
    profile: "EvidenceProfile",
    cosine_similarity_score: float,
    effective_threshold: float,
    attrs_a: dict[str, str] | None = None,
    attrs_b: dict[str, str] | None = None,
    cross_file: bool = False,
) -> EvidenceDecision:
    """
    Decide whether a candidate has enough evidence to become a merge edge.

    ``cross_file``: the two records come from different input files (both known).
    A pair with no shared field values passes (embedding_only) only across files.

    Uses only:
      - semantic type
      - learned discriminative score
      - field similarity
      - embedding cosine
      - conflict mass

    No column-name logic.
    """
    if not shared_keys:
        if (
            cosine_similarity_score >= max(EMBEDDING_ONLY_COSINE_MIN, effective_threshold)
            and not cross_file
        ):
            return EvidenceDecision(
                passes=False,
                match_type=None,
                evidence_mass=0.0,
                conflict_mass=0.0,
                detail="embedding-only blocked: both records from the same input file",
            )
        if cosine_similarity_score >= max(EMBEDDING_ONLY_COSINE_MIN, effective_threshold):
            return EvidenceDecision(
                passes=True,
                match_type="embedding_only",
                evidence_mass=0.0,
                conflict_mass=0.0,
                detail="embedding-only pass with high cosine",
            )
        return EvidenceDecision(
            passes=False,
            match_type=None,
            evidence_mass=0.0,
            conflict_mass=0.0,
            detail="no shared evidence",
        )

    total_weight = sum(
        max(0.0, discriminative_by_col.get(k, 0.0))
        for k in profile.all_content_fields
    ) or 1.0

    evidence_mass = 0.0
    conflict_mass = 0.0
    descriptive_hits = 0
    street_address_hits = 0
    identity_hits = 0

    _attrs_a = attrs_a or {}
    _attrs_b = attrs_b or {}
    for k in shared_keys:
        # Stored attribute type names are snake_case (built by column_to_attr_type
        # in column_classification.py which replaces non-alphanumerics with "_").
        # semantic_by_col_lower and discriminative_by_col are keyed by the plain
        # lowercased CSV column name which may contain spaces (e.g. "addr line1").
        # Every column-keyed dict and set is built with col_key (lowercase snake), so k
        # is looked up as it is; no space/underscore conversion is needed.
        k_lookup = k
        sem = str(semantic_by_col_lower.get(k_lookup, semantic_by_col_lower.get(k, "DESCRIPTIVE"))).strip().upper()
        weight_share = max(0.0, discriminative_by_col.get(k_lookup, discriminative_by_col.get(k, 0.0))) / total_weight
        sim = float(field_similarities.get(k, 1.0))

        va = str(_attrs_a.get(k, "") or "").strip()
        vb = str(_attrs_b.get(k, "") or "").strip()
        if not va and not vb:
            continue  # no information either way — treat as neutral, not conflicting

        if sim >= IDENTITY_FIELD_SIM_MIN and sem in {"IDENTIFIER", "QUASI_IDENTIFIER"}:
            evidence_mass += weight_share
            identity_hits += 1
        elif sim >= DESCRIPTIVE_FIELD_SIM_MIN and sem in ("DESCRIPTIVE", "STREET_ADDRESS"):
            evidence_mass += weight_share
            descriptive_hits += 1
            if sem == "STREET_ADDRESS":
                street_address_hits += 1
        elif sim >= DESCRIPTIVE_FIELD_SIM_MIN and sem == "NAME":
            evidence_mass += weight_share
        elif sim < CONFLICT_FIELD_SIM_MAX:
            conflict_mass += weight_share

    if identity_hits >= 1 and conflict_mass <= profile.max_conflict_mass:
        return EvidenceDecision(
            passes=True,
            match_type="full_signal",
            evidence_mass=evidence_mass,
            conflict_mass=conflict_mass,
            detail="identity evidence pass",
        )

    if (
        descriptive_hits >= profile.min_descriptive_fields
        and evidence_mass >= profile.min_evidence_mass
        and conflict_mass <= profile.max_conflict_mass
        and (not profile.require_street_address_hit or street_address_hits >= 1)
    ):
        return EvidenceDecision(
            passes=True,
            match_type="weak_signal",
            evidence_mass=evidence_mass,
            conflict_mass=conflict_mass,
            detail="composite descriptive evidence pass",
        )

    if (
        cosine_similarity_score >= max(SEMANTIC_ONLY_COSINE_MIN, effective_threshold)
        and evidence_mass >= profile.min_evidence_mass * 0.75
        and conflict_mass <= profile.max_conflict_mass
        and (not profile.require_street_address_hit or street_address_hits >= 1)
    ):
        return EvidenceDecision(
                passes=True,
                match_type="weak_signal",
                evidence_mass=evidence_mass,
                conflict_mass=conflict_mass,
                detail="semantic embedding plus field evidence pass",
        )

    return EvidenceDecision(
        passes=False,
        match_type=None,
        evidence_mass=evidence_mass,
        conflict_mass=conflict_mass,
        detail="insufficient schema-agnostic evidence",
    )


@dataclass(frozen=True)
class MinimumEvidenceColumns:
    """Column groups for minimum-evidence evaluation (A/B/C conditions)."""

    identifier_fields: tuple[str, ...]
    name_fields: tuple[str, ...]
    location_fields: tuple[str, ...]


@dataclass(frozen=True)
class MatchWeights:
    learned_threshold: float
    effective_threshold: float
    discriminative_by_col: dict[str, float]
    discriminative_share: dict[str, float]
    w_embed_blend: float
    w_field_blend: float


@dataclass(frozen=True)
class EvidenceProfile:
    """Semantic evidence groups derived from learned column classification."""

    identity_fields: tuple[str, ...]
    descriptive_fields: tuple[str, ...]
    date_fields: tuple[str, ...]
    all_content_fields: tuple[str, ...]
    top_discriminative_fields: tuple[str, ...]
    min_descriptive_fields: int
    min_evidence_mass: float
    max_conflict_mass: float
    # True when this schema has at least one STREET_ADDRESS-classified
    # field available. When True, the weak_signal descriptive-evidence
    # path requires one of the matched descriptive_hits to specifically be
    # a STREET_ADDRESS field — city/postalcode agreement alone is never
    # enough, since those values are shared by every record in the same
    # town regardless of entity identity. When False (no address field in
    # this schema), the gate falls back to the prior behavior so non-
    # location domains (e.g. product or vendor data with no address
    # column) aren't penalized for a field type they don't have.
    require_street_address_hit: bool = False


def load_match_weights(weights_path: Path, threshold_path: Path) -> MatchWeights:
    if not weights_path.is_file():
        raise SystemExit(
            f"Missing required file: {weights_path}. "
            "Run src/rule_learner.py on your CSV first."
        )
    if not threshold_path.is_file():
        raise SystemExit(
            f"Missing required file: {threshold_path}. "
            "Run src/rule_learner.py on your CSV first."
        )

    with weights_path.open(encoding="utf-8") as f:
        weights_json: dict[str, Any] = json.load(f)
    with threshold_path.open(encoding="utf-8") as f:
        threshold_json: dict[str, Any] = json.load(f)

    learned_threshold = float(threshold_json["threshold"])
    effective_threshold = max(learned_threshold, LEARNED_THRESHOLD_MIN_FLOOR)

    columns_meta: dict[str, dict[str, Any]] = weights_json.get("columns") or {}
    discriminative_by_col: dict[str, float] = {}
    for col_name, meta in columns_meta.items():
        key = col_key(col_name)
        if not key:
            continue
        raw = float(meta.get("discriminative_score", 0.0) or 0.0)
        discriminative_by_col[key] = max(0.0, raw)

    sum_disc = sum(discriminative_by_col.values())
    if sum_disc <= 0:
        n = max(1, len(discriminative_by_col))
        discriminative_share = {k: 1.0 / n for k in discriminative_by_col}
    else:
        discriminative_share = {
            k: v / sum_disc for k, v in discriminative_by_col.items()
        }

    disc_vals = np.array(list(discriminative_by_col.values()), dtype=np.float64)
    if disc_vals.size > 0 and disc_vals.sum() > 0:
        p = disc_vals / disc_vals.sum()
        h = float(-np.sum(p * np.log(p + 1e-12)))
        hmax = float(np.log(len(p)))
        w_cos = h / hmax if hmax > 1e-12 else 0.5
    else:
        w_cos = 0.5
    w_embed_blend = float(min(EMBED_BLEND_MAX, max(EMBED_BLEND_MIN, w_cos)))
    w_field_blend = 1.0 - w_embed_blend

    print("=" * 72)
    print("  match.py — loaded configuration (data-driven)")
    print("=" * 72)
    print(f"  attribute_weights: {weights_path}")
    print(f"  learned_threshold: {threshold_path}")
    print(
        f"  Learned threshold: {learned_threshold:.3f} "
        f"(effective = max(learned, {LEARNED_THRESHOLD_MIN_FLOOR:.2f}))"
    )
    print(
        "  applied cosine vs field blend "
        f"(embedding≤{EMBED_BLEND_MAX:.2f}, field≥{FIELD_BLEND_MIN:.2f}): "
        f"embed={w_embed_blend:.4f}, field={w_field_blend:.4f}"
    )
    print("  discriminative_score (raw from JSON; share = raw / sum(raw)):")
    for k in sorted(discriminative_by_col.keys()):
        print(
            f"    {k:24}  raw={discriminative_by_col[k]:.6f}  "
            f"share={discriminative_share.get(k, 0.0):.6f}"
        )
    print("=" * 72 + "\n")

    return MatchWeights(
        learned_threshold=learned_threshold,
        effective_threshold=effective_threshold,
        discriminative_by_col=discriminative_by_col,
        discriminative_share=discriminative_share,
        w_embed_blend=w_embed_blend,
        w_field_blend=w_field_blend,
    )


def evidence_profile_from_semantic_map(
    semantic_by_col_lower: dict[str, str],
    allowed_cols_lower: frozenset[str],
    record_id_column_lower: str,
    discriminative_by_col: dict[str, float],
) -> EvidenceProfile:
    """Build schema-agnostic evidence groups from semantic type and learned weights."""
    rid = col_key(record_id_column_lower)

    identity: list[str] = []
    descriptive: list[str] = []
    date: list[str] = []
    content: list[str] = []
    has_street_address_field = False

    for key, sem_raw in semantic_by_col_lower.items():
        key = col_key(key)
        if not key or key == rid or key not in allowed_cols_lower:
            continue

        sem = str(sem_raw).strip().upper()
        content.append(key)

        if sem in {"IDENTIFIER", "QUASI_IDENTIFIER"}:
            identity.append(key)
        elif sem in ("DESCRIPTIVE", "STREET_ADDRESS"):
            descriptive.append(key)
            if sem == "STREET_ADDRESS":
                has_street_address_field = True
        elif sem == "DATE":
            date.append(key)

    ranked = sorted(
        set(content),
        key=lambda k: discriminative_by_col.get(k, 0.0),
        reverse=True,
    )

    min_desc = 2 if len(set(descriptive)) >= 2 else len(set(descriptive))
    # STREET_ADDRESS fields are the only DESCRIPTIVE-bucket fields with
    # enough cardinality to discriminate between different entities at the
    # same general location. City and postal code are shared by every
    # record in that town/zip regardless of which specific organization it
    # is, so two records agreeing only on city+postalcode carry almost no
    # evidence that they're the same entity — that combination alone
    # should never be sufficient grounds for a weak_signal merge.
    # has_street_address_field is set inside the loop above (not re-looked-up
    # here) because semantic_by_col_lower is keyed by the ORIGINAL-case column
    # names loaded from column_classification.json, while `key` has already
    # been lowercased by this point — re-querying the dict with the
    # lowercased key would silently miss every entry and always return False.

    return EvidenceProfile(
        identity_fields=tuple(sorted(set(identity))),
        descriptive_fields=tuple(sorted(set(descriptive))),
        date_fields=tuple(sorted(set(date))),
        all_content_fields=tuple(sorted(set(content))),
        top_discriminative_fields=tuple(ranked[: max(1, min(5, len(ranked)))]),
        min_descriptive_fields=min_desc,
        min_evidence_mass=MIN_EVIDENCE_MASS,
        max_conflict_mass=MAX_CONFLICT_MASS,
        require_street_address_hit=has_street_address_field,
    )


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        raw = pickle.load(f)
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        sk = str(k).strip()
        if not sk or sk == EMBEDDINGS_STORE_META_KEY:
            continue
        if isinstance(v, dict):
            continue
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        out[sk] = arr
    return out


def load_embeddings_file_registry(path: Path) -> dict[str, str] | None:
    """
    Optional mapping of embedding key (customer / record id) → human-readable source label.

    Supported shapes:
      - ``{"files": {"my.csv": ["id1", "id2"], "other.csv": ["x"]}}``
      - Flat ``{"id1": "my.csv", "id2": "my.csv"}`` (values must be strings).
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    id_to_bucket: dict[str, str] = {}
    files_obj = data.get("files")
    if isinstance(files_obj, dict):
        for bucket, ids in files_obj.items():
            label = str(bucket).strip() or "unknown"
            if not isinstance(ids, list):
                continue
            for rid in ids:
                sk = str(rid).strip()
                if sk:
                    id_to_bucket[sk] = label
        if id_to_bucket:
            return id_to_bucket
    for k, v in data.items():
        if k == "files" or isinstance(v, (dict, list)):
            continue
        if isinstance(v, str):
            sk = str(k).strip()
            if sk:
                id_to_bucket[sk] = str(v).strip() or "unknown"
    return id_to_bucket if id_to_bucket else None


def infer_id_source_bucket(record_id: str) -> str:
    """Coarse grouping when ``embeddings_file_registry.json`` is absent."""
    s = str(record_id).strip()
    if not s:
        return "(empty_id)"
    if _ID_BUCKET_UUID.match(s):
        return f"uuid:{s[:8].lower()}"
    m = re.match(r"^([A-Za-z]+)", s)
    if m:
        return f"prefix:{m.group(1)}"
    if "_" in s:
        return f"prefix:{s.split('_', 1)[0]}"
    if "-" in s:
        return f"prefix:{s.split('-', 1)[0]}"
    return "prefix:(other)"


def print_global_embedding_source_breakdown(global_ids: list[str], registry_path: Path) -> None:
    reg = load_embeddings_file_registry(registry_path)
    if reg:
        print(
            f"  Source breakdown (from {registry_path.name} — id → file/source label):"
        )
        counts = Counter(reg.get(gid, "(not_in_registry)") for gid in global_ids)
    else:
        print(
            f"  Source breakdown (id-prefix heuristic; for per-file counts add "
            f"{registry_path.name} with a top-level \"files\" object mapping "
            "labels to lists of record ids):"
        )
        counts = Counter(infer_id_source_bucket(gid) for gid in global_ids)
    for label, n in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
        print(f"    {label}: {n}")


def global_embedding_matrix_and_ids(
    global_emb: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """
    Rows of ``X`` align with ``global_ids`` (sorted keys). Skips non-finite vectors or
    lengths that disagree with the first valid embedding.
    """
    if not global_emb:
        raise SystemExit("embeddings_global.pkl is empty")
    ids = sorted(global_emb.keys(), key=str)
    dim: int | None = None
    good_ids: list[str] = []
    rows: list[np.ndarray] = []
    skipped = 0
    for rid in ids:
        arr = np.asarray(global_emb[rid], dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            skipped += 1
            continue
        if dim is None:
            dim = int(arr.size)
        elif int(arr.size) != dim:
            skipped += 1
            continue
        rows.append(arr)
        good_ids.append(str(rid))
    if not good_ids:
        raise SystemExit(
            "No usable embedding vectors in embeddings_global.pkl "
            "(missing or non-uniform dimensions)."
        )
    if skipped:
        print(
            f"Warning: skipped {skipped} embedding(s) in global store "
            "(wrong length or non-finite values)."
        )
    return np.vstack(rows), good_ids


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or math.isnan(na) or math.isnan(nb):
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    if math.isnan(c):
        return 0.0
    return max(-1.0, min(1.0, c))


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower()), set(b.lower())
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 1.0


def sequence_similarity(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_name_match(
    rid_a: str,
    rid_b: str,
    rows_by_id: dict[str, dict[str, str]],
    name_fields: tuple[str, ...],
    threshold: float = 0.80,
) -> bool:
    """True if any name field has sequence_similarity >= threshold between the two rows."""
    ra = rows_by_id.get(rid_a) or {}
    rb = rows_by_id.get(rid_b) or {}
    for f in name_fields:
        va = str(ra.get(f, "") or "").strip().lower()
        vb = str(rb.get(f, "") or "").strip().lower()
        if va and vb:
            if difflib.SequenceMatcher(None, va, vb).ratio() >= threshold:
                return True
    return False


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word tokens (letters/digits)."""
    wa = set(re.findall(r"[\w]+", str(a).lower()))
    wb = set(re.findall(r"[\w]+", str(b).lower()))
    if not wa and not wb:
        return 1.0
    inter = len(wa & wb)
    uni = len(wa | wb)
    return inter / uni if uni else 1.0


def normalize_fingerprint(fp: str) -> str:
    s = str(fp).strip()
    if _ATTR_SEP not in s:
        return s.lower()
    t, v = s.split(_ATTR_SEP, 1)
    return f"{str(t).strip().lower()}{_ATTR_SEP}{str(v).strip().lower()}"


def load_customer_rows_by_id(
    path: Path,
    record_id_column_lower: str,
) -> dict[str, dict[str, str]]:
    """Map record id (``record_id_column`` from classification JSON) -> row with lowercase keys."""
    out: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return out
    rid_key = col_key(record_id_column_lower)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        keys = list(reader.fieldnames)
        key_map = {k: col_key(k) for k in keys}
        for row in reader:
            norm_row: dict[str, str] = {}
            for k in keys:
                lk = key_map[k]
                raw = row.get(k)
                norm_row[lk] = "" if raw is None else str(raw).strip()
            rid = norm_row.get(rid_key, "").strip()
            if rid:
                out[rid] = norm_row
    return out


def load_customer_rows_union_sources(
    primary_csv: Path,
    record_id_column_lower: str,
    registry_path: Path,
) -> tuple[dict[str, dict[str, str]], tuple[Path, ...]]:
    """
    Rows keyed by record id for minimum-evidence and CSV fallback across pipeline runs.

    Prefer ``merged_input.csv`` in the data folder when present; otherwise merge the
    primary CSV with any paths listed under ``files`` in ``embeddings_file_registry.json``
    (values resolved under the data folder when relative).
    """
    rid_key = col_key(record_id_column_lower)
    if MERGED_INPUT_CSV.is_file():
        merged = load_customer_rows_by_id(MERGED_INPUT_CSV, rid_key)
        return merged, (MERGED_INPUT_CSV,)
    paths_ordered: list[Path] = []
    seen_resolved: set[str] = set()

    def _add_path(p: Path) -> None:
        if not p.is_file():
            return
        key = str(p.resolve())
        if key not in seen_resolved:
            seen_resolved.add(key)
            paths_ordered.append(p)

    _add_path(primary_csv.resolve())
    reg: dict[str, Any] | None = None
    if registry_path.is_file():
        try:
            reg = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reg = None
    if isinstance(reg, dict):
        files_obj = reg.get("files")
        if isinstance(files_obj, dict):
            for fname in files_obj.keys():
                fn = str(fname).strip()
                if not fn:
                    continue
                pth = Path(fn)
                if not pth.is_absolute():
                    pth = data_dir() / fn
                _add_path(pth)

    merged_rows: dict[str, dict[str, str]] = {}
    for p in paths_ordered:
        merged_rows.update(load_customer_rows_by_id(p, rid_key))
    return merged_rows, tuple(paths_ordered)


def exact_duplicate_pairs(
    rows_by_id: dict[str, dict[str, str]],
    content_columns_lower: frozenset[str],
    semantic_by_col_lower: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Group records by a normalized composite key built from name + address fields.
    Any group with 2+ records emits all-pairs combinations as near-certain matches.
    Returns (record_id_a, record_id_b) pairs, a < b lexicographically.
    """
    def norm(s: str) -> str:
        # lowercase, strip punctuation, collapse whitespace
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]", "", s)
        return re.sub(r" +", " ", s)

    name_fields = tuple(
        sorted(
            k for k, v in semantic_by_col_lower.items()
            if str(v).strip().upper() == "NAME" and k in content_columns_lower
        )
    )
    location_fields = tuple(
        sorted(
            k for k, v in semantic_by_col_lower.items()
            if str(v).strip().upper() in ("DESCRIPTIVE", "STREET_ADDRESS")
            and k in content_columns_lower
        )
    )
    key_fields = tuple(f for f in (*name_fields, *location_fields) if f in content_columns_lower)
    if not key_fields:
        return []

    buckets: dict[str, list[str]] = defaultdict(list)
    for rid, row in rows_by_id.items():
        parts = []
        for f in key_fields:
            val = norm(str(row.get(f, "") or ""))
            parts.append(val)
        # Build key only from non-empty fields present in this record
        non_empty_parts = [(f, v) for f, v in zip(key_fields, parts) if v]
        if len(non_empty_parts) < 2:
            continue
        # Hash only the fields that have values, keyed by field name
        key_str = "|".join(f"{f}:{v}" for f, v in non_empty_parts)
        key = hashlib.md5(key_str.encode()).hexdigest()
        buckets[key].append(rid)

    pairs: list[tuple[str, str]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        members_sorted = sorted(members)
        for i, a in enumerate(members_sorted):
            for b in members_sorted[i + 1:]:
                pairs.append((a, b))
    return pairs


def record_id_lineage_c_cc(record_id: str) -> str:
    """Coarse C vs CC bucket from record id prefix (``CC*`` before ``C*``)."""
    s = str(record_id).strip()
    sup = s.upper()
    if sup.startswith("CC"):
        return "CC"
    if sup.startswith("C"):
        return "C"
    return ""


def pair_lineage_bucket(
    record_id_a: str,
    record_id_b: str,
    origin_map: dict[str, str] | None = None,
) -> str:
    """
    Classify pair lineage using optional registry-backed id -> source mapping.

    Returns:
      - ``cross_file`` when both ids are known and source labels differ
      - ``same_file`` when both ids are known and source labels are equal
      - ``unknown`` when no map is provided or either id is missing
    """
    if origin_map is None:
        return "unknown"
    src_a = origin_map.get(str(record_id_a).strip())
    src_b = origin_map.get(str(record_id_b).strip())
    if not src_a or not src_b:
        return "unknown"
    return "same_file" if src_a == src_b else "cross_file"


def fetch_shared_attribute_values_for_pair(
    store, id_a: str, id_b: str, record_label: str
) -> list[dict[str, Any]]:
    rows = store.shared_attribute_rows(id_a, id_b, record_label)
    return [{"attr_type": r["attr_type"], "value": r["value"]} for r in rows]


def neo_shared_av_rows_to_fingerprints(rows: list[dict[str, Any]]) -> set[str]:
    fps: set[str] = set()
    for row in rows:
        t_raw = row.get("attr_type")
        v_raw = row.get("value")
        if t_raw is None or v_raw is None:
            continue
        t_s = str(t_raw).strip().lower()
        v_s = str(v_raw).strip().lower()
        if not t_s or not v_s:
            continue
        fps.add(f"{t_s}{_ATTR_SEP}{v_s}")
    return fps


def graph_overlap_labels_for_pair(
    store,
    sa: set[str],
    sb: set[str],
    graph_id_a: str,
    graph_id_b: str,
    metadata_cols_lower: frozenset[str],
    rows_merged: dict[str, dict[str, str]],
    csv_compare_cols_lower: frozenset[str],
    record_label: str,
) -> tuple[list[str], frozenset[str], str | None]:
    """
    Shared field labels: fingerprint intersection, then (for C vs CC) stored shared
    ``AttributeValue`` nodes by record-node ``id``, then CSV fallback on merged rows.
    """
    graph_labels = drop_metadata_shared_labels(
        graph_shared_attribute_labels(sa, sb),
        metadata_cols_lower,
    )
    if graph_labels:
        sk = shared_content_attribute_keys_graph(graph_labels)
        return graph_labels, sk, "|".join(graph_labels)

    cross_c_cc = (
        record_id_lineage_c_cc(graph_id_a) == "C"
        and record_id_lineage_c_cc(graph_id_b) == "CC"
        or record_id_lineage_c_cc(graph_id_a) == "CC"
        and record_id_lineage_c_cc(graph_id_b) == "C"
    )
    if cross_c_cc:
        neo_rows = fetch_shared_attribute_values_for_pair(
            store, graph_id_a, graph_id_b, record_label
        )
        neo_fps = neo_shared_av_rows_to_fingerprints(neo_rows)
        if neo_fps:
            raw_labels = sorted({fingerprint_to_label(fp) for fp in neo_fps})
            graph_labels = drop_metadata_shared_labels(raw_labels, metadata_cols_lower)
            if graph_labels:
                sk = shared_content_attribute_keys_graph(graph_labels)
                return graph_labels, sk, "|".join(graph_labels)

    csv_fields = drop_metadata_csv_columns(
        csv_fallback_similar_fields(
            graph_id_a, graph_id_b, rows_merged, csv_compare_cols_lower
        ),
        metadata_cols_lower,
    )
    if csv_fields:
        sk = shared_content_attribute_keys_csv(csv_fields)
        return csv_fields, sk, "|".join(csv_fields)

    return [], frozenset(), None


def _csv_row_field_norm(row: dict[str, str], logical_field_lower: str) -> str:
    return str(row.get(logical_field_lower, "") or "").strip().lower()


def _pair_csv_field_exact_match(
    ra: dict[str, str],
    rb: dict[str, str],
    logical_field_lower: str,
) -> bool:
    va = _csv_row_field_norm(ra, logical_field_lower)
    vb = _csv_row_field_norm(rb, logical_field_lower)
    if not va or not vb:
        return False
    return va == vb


@dataclass(frozen=True)
class MinimumEvidenceResult:
    """Outcome of minimum-evidence rules (shared_attributes-first, then CSV fallback)."""

    passes: bool
    source: str
    shared_field_names: frozenset[str]
    cond_a: bool
    cond_b: bool
    cond_c: bool
    detail: str


def parse_shared_attributes_field_names(shared_attributes: str) -> frozenset[str]:
    """
    Field names from pipe-separated graph/CSV overlap labels ``field:value`` segments.
    Strips the value part (first ``:`` splits type from value); ignores ``embedding_only``.
    """
    s = str(shared_attributes).strip()
    if not s or s == "embedding_only":
        return frozenset()
    out: set[str] = set()
    for part in s.split("|"):
        key = graph_label_attribute_key(part)
        if key:
            out.add(key)
    return frozenset(out)


def _minimum_evidence_abc_on_field_set(
    field_names: frozenset[str],
    mev: MinimumEvidenceColumns,
) -> tuple[bool, bool, bool]:
    """Conditions A/B/C from a set of attribute keys (lowercase) overlapping the pair."""
    cond_a = any(f in field_names for f in mev.identifier_fields)
    cond_b = any(f in field_names for f in mev.name_fields) and any(
        f in field_names for f in mev.location_fields
    )
    cond_c = bool(mev.name_fields) and all(f in field_names for f in mev.name_fields)
    return cond_a, cond_b, cond_c


def _minimum_evidence_detail_shared(
    field_names: frozenset[str],
    cond_a: bool,
    cond_b: bool,
    cond_c: bool,
    mev: MinimumEvidenceColumns,
) -> str:
    if cond_a:
        hit = next((f for f in mev.identifier_fields if f in field_names), None)
        return f"A pass (identifier_field {hit!r} in shared attributes)"
    if cond_b:
        nh = next((f for f in mev.name_fields if f in field_names), None)
        lh = next((f for f in mev.location_fields if f in field_names), None)
        return f"B pass (name_field {nh!r} and location_field {lh!r} in shared attributes)"
    if cond_c:
        return f"C pass (all name_fields {list(mev.name_fields)!r} present in shared attributes)"
    reasons: list[str] = []
    if not any(f in field_names for f in mev.identifier_fields):
        reasons.append("A fail (no identifier_field in shared attributes)")
    if not (
        any(f in field_names for f in mev.name_fields)
        and any(f in field_names for f in mev.location_fields)
    ):
        reasons.append(
            "B fail (need ≥1 name_field and ≥1 location_field in shared attributes)"
        )
    if mev.name_fields:
        missing = [f for f in mev.name_fields if f not in field_names]
        if missing:
            reasons.append(f"C fail (name_fields missing from shared: {missing!r})")
        elif not cond_c:
            reasons.append("C fail")
    parts = "; ".join(reasons) if reasons else "no A/B/C satisfied"
    return parts


def evaluate_minimum_evidence(
    shared_attributes: str,
    rid_a: str,
    rid_b: str,
    rows_by_id: dict[str, dict[str, str]],
    mev: MinimumEvidenceColumns,
    discriminative_by_col: dict[str, float] | None = None,
) -> MinimumEvidenceResult:
    """
    Prefer evidence from ``shared_attributes`` (stored attribute overlap ``field:value|…``).
    When there are no parsed field names (e.g. ``embedding_only``), fall back to CSV
    row equality (same rules as before for the CSV path).
    """
    shared_field_names = parse_shared_attributes_field_names(shared_attributes)
    if shared_field_names:
        cond_a, cond_b, cond_c = _minimum_evidence_abc_on_field_set(shared_field_names, mev)
        passes = cond_a or cond_b or cond_c

        # Fuzzy-name + location corroboration: allow pass if name is similar and
        # 2+ location fields overlap, even without exact name match in shared fields.
        if not passes and mev.name_fields:
            ra = rows_by_id.get(rid_a) or {}
            rb = rows_by_id.get(rid_b) or {}
            name_sim = max(
                (difflib.SequenceMatcher(None,
                    str(ra.get(f, "") or "").strip().lower(),
                    str(rb.get(f, "") or "").strip().lower()).ratio()
                 for f in mev.name_fields
                 if ra.get(f) and rb.get(f)),
                default=0.0,
            )
            descriptive_overlap = [f for f in shared_field_names if f in mev.location_fields]
            if name_sim >= 0.80 and len(descriptive_overlap) >= 2:
                quasi_id_overlap = [
                    f for f in shared_field_names
                    if f in mev.identifier_fields
                ]
                if name_sim >= 0.99 and not quasi_id_overlap:
                    pass  # exact chain name, no quasi-id match — different branches
                else:
                    disc = discriminative_by_col or {}
                    top_key = max(descriptive_overlap, key=lambda k: disc.get(k, 0.0))
                    top_sim = difflib.SequenceMatcher(
                        None,
                        str(ra.get(top_key, "") or "").strip().lower(),
                        str(rb.get(top_key, "") or "").strip().lower(),
                    ).ratio()
                    if top_sim >= 0.80:
                        passes = True
                        cond_b = True

        detail = _minimum_evidence_detail_shared(
            shared_field_names, cond_a, cond_b, cond_c, mev
        )
        return MinimumEvidenceResult(
            passes=passes,
            source="shared_attributes",
            shared_field_names=shared_field_names,
            cond_a=cond_a,
            cond_b=cond_b,
            cond_c=cond_c,
            detail=detail,
        )

    ra = rows_by_id.get(rid_a) or {}
    rb = rows_by_id.get(rid_b) or {}
    cond_a = any(_pair_csv_field_exact_match(ra, rb, f) for f in mev.identifier_fields)
    cond_b = (
        any(_pair_csv_field_exact_match(ra, rb, f) for f in mev.name_fields)
        and any(_pair_csv_field_exact_match(ra, rb, f) for f in mev.location_fields)
    )
    mutual_name_matches = 0
    cond_c = True
    for f in mev.name_fields:
        va = _csv_row_field_norm(ra, f)
        vb = _csv_row_field_norm(rb, f)
        if va and vb:
            if va == vb:
                mutual_name_matches += 1
            else:
                cond_c = False
                break
    cond_c = bool(cond_c and mutual_name_matches >= 2)

    # Fuzzy-name + location corroboration fallback
    if not (cond_a or cond_b or cond_c) and mev.name_fields:
        fuzzy_name = fuzzy_name_match(rid_a, rid_b, rows_by_id, mev.name_fields)
        loc_exact = [f for f in mev.location_fields if _pair_csv_field_exact_match(ra, rb, f)]
        if fuzzy_name and len(loc_exact) >= 2:
            disc = discriminative_by_col or {}
            top_key = max(loc_exact, key=lambda k: disc.get(k, 0.0))
            top_disc = disc.get(top_key, 0.0)
            # Require the most discriminative location field to have meaningful
            # uniqueness — low discriminative score means it's a low-cardinality
            # field (state, country) that can't anchor a fuzzy-name match alone.
            all_disc = [v for v in disc.values() if v > 0]
            median_disc = float(np.median(all_disc)) if all_disc else 0.0
            if top_disc >= median_disc:
                cond_b = True

    passes = bool(cond_a or cond_b or cond_c)
    if cond_a:
        hit = next(
            (f for f in mev.identifier_fields if _pair_csv_field_exact_match(ra, rb, f)),
            None,
        )
        detail = f"csv_fallback A pass (identifier_field {hit!r} exact match in CSV)"
    elif cond_b:
        nh = next(
            (f for f in mev.name_fields if _pair_csv_field_exact_match(ra, rb, f)), None
        )
        lh = next(
            (f for f in mev.location_fields if _pair_csv_field_exact_match(ra, rb, f)),
            None,
        )
        if nh:
            detail = f"csv_fallback B pass (name {nh!r} and location {lh!r} exact in CSV)"
        else:
            detail = "csv_fallback B pass (fuzzy name match + 2+ location fields)"
    elif cond_c:
        detail = "csv_fallback C pass (≥2 agreeing name_fields in CSV)"
    else:
        detail = (
            "csv_fallback all fail (no shared field names from graph; "
            "CSV has no A/B/C evidence)"
        )
    return MinimumEvidenceResult(
        passes=passes,
        source="csv_fallback",
        shared_field_names=frozenset(),
        cond_a=cond_a,
        cond_b=cond_b,
        cond_c=cond_c,
        detail=detail,
    )


def csv_fallback_similar_fields(
    record_id_a: str,
    record_id_b: str,
    rows_by_id: dict[str, dict[str, str]],
    content_columns_lower: frozenset[str],
    min_jaccard: float = CSV_FALLBACK_JACCARD_MIN,
) -> list[str]:
    """
    Column names where values match (case-insensitive) or token Jaccard >= min_jaccard.
    Only compares columns in ``content_columns_lower`` (caller should omit
    ``metadata_columns``). Metadata must not appear in the returned list.
    """
    if not content_columns_lower:
        return []
    ra = rows_by_id.get(str(record_id_a).strip())
    rb = rows_by_id.get(str(record_id_b).strip())
    if not ra or not rb:
        return []
    common = set(ra.keys()) & set(rb.keys()) & content_columns_lower
    hits: list[str] = []
    for col in sorted(common):
        va = str(ra.get(col, "")).strip().lower()
        vb = str(rb.get(col, "")).strip().lower()
        if not va or not vb:
            continue
        if va == vb:
            hits.append(col)
        elif token_jaccard(va, vb) >= min_jaccard:
            hits.append(col)
    return hits


def graph_shared_attribute_labels(sa: set[str], sb: set[str]) -> list[str]:
    """Shared AttributeValue fingerprints after normalizing type and value for matching."""
    na = {normalize_fingerprint(fp) for fp in sa}
    nb = {normalize_fingerprint(fp) for fp in sb}
    common_norm = na & nb
    if not common_norm:
        return []
    rep: dict[str, str] = {}
    for fp in sorted(sa | sb):
        nn = normalize_fingerprint(fp)
        if nn in common_norm and nn not in rep:
            rep[nn] = fp
    return sorted(fingerprint_to_label(rep[k]) for k in sorted(rep.keys()))


def fingerprints_to_attr_map(fps: set[str]) -> dict[str, str]:
    """Map normalized column key -> raw value string (no fixed schema)."""
    out: dict[str, str] = {}
    for fp in fps:
        if _ATTR_SEP not in fp:
            continue
        t, v = fp.split(_ATTR_SEP, 1)
        key = str(t).strip().lower()
        if key:
            out[key] = v
    return out


def code_normalize(value: Any) -> str:
    """Casefold and drop spaces and punctuation."""
    return _CODE_DROP.sub("", str(value).casefold())


def code_like_columns(
    semantic_by_col: dict[str, str],
    rows_by_id: dict[str, dict[str, str]],
    content_cols: frozenset[str],
) -> frozenset[str]:
    """
    Code-like columns, decided from this run's data: semantic type IDENTIFIER or
    QUASI_IDENTIFIER, or any content column where at least CODE_NOSPACE_SHARE_MIN of the
    non-empty values contain no space.
    """
    out: set[str] = set()
    for col in sorted(content_cols):
        if str(semantic_by_col.get(col, "")).strip().upper() in ("IDENTIFIER", "QUASI_IDENTIFIER"):
            out.add(col)
            continue
        n = n_nospace = 0
        for row in rows_by_id.values():
            v = str(row.get(col, "") or "").strip()
            if v:
                n += 1
                if " " not in v:
                    n_nospace += 1
        if n and n_nospace >= CODE_NOSPACE_SHARE_MIN * n:
            out.add(col)
    return frozenset(out)


def identifying_columns(
    code_cols: frozenset[str], normalized_weight: dict[str, float], n_content_cols: int
) -> frozenset[str]:
    """Identifying fields: code-like columns whose learned normalized weight is at least 1/(2n), n = content columns used for scoring."""
    if n_content_cols <= 0:
        return frozenset()
    cut = 1.0 / (2 * n_content_cols)
    return frozenset(c for c in code_cols if float(normalized_weight.get(c, 0.0)) >= cut - 1e-12)


def code_path_label(
    score: float, threshold: float, evidence_ok: bool, conflict: bool
) -> tuple[str, bool]:
    """
    match_type and evidence_passes for a code-path pair. A pair passes when its score is strictly
    greater than the threshold (as on the text path) and its evidence mass is enough. A conflict
    (an identifying field differs) gives code_conflict only when it blocks a pair that would
    otherwise have passed or fallen in the review band; for those rows evidence_passes is true
    exactly when the pair would otherwise have passed. Any other pair keeps its ordinary label.
    """
    would_pass = score > threshold and evidence_ok
    in_band = abs(_micro(score) - _micro(threshold)) <= _micro(REVIEW_MARGIN)
    if conflict and (would_pass or in_band):
        return "code_conflict", would_pass
    if would_pass:
        return "code_signal", True
    if score <= threshold:
        return "below_threshold", evidence_ok
    return "insufficient_evidence", False


def code_path_pair(
    attrs_a: dict[str, str],
    attrs_b: dict[str, str],
    code_cols: frozenset[str],
    content_cols: frozenset[str],
    disc: dict[str, float],
    identifying: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """
    Field-only score for a pair, over every content column both records have a value in
    (None when the two records share no value in a code-like column). Values are normalized
    (casefold, no spaces or punctuation); equal values score 1.0, otherwise the order-aware
    ratio of difflib.SequenceMatcher on the normalized values. Weights are the learned
    discriminative weights renormalized over the shared fields (equal weights when all are
    zero). Evidence mass: weight share of fields equal after normalization. Conflict rule: any
    identifying field (a code-like column of high learned weight, see identifying_columns)
    present in both records that differs after normalization; conflict_mass is the weight
    share of those fields.
    """
    fields = []
    for c in sorted(content_cols | code_cols):
        na, nb = code_normalize(attrs_a.get(c, "")), code_normalize(attrs_b.get(c, ""))
        if na and nb:
            fields.append((c, na, nb))
    if not any(c in code_cols for c, _, _ in fields):
        return None
    raw = {c: max(0.0, float(disc.get(c, 0.0))) for c, _, _ in fields}
    total = sum(raw.values())
    weights = {c: (raw[c] / total if total > 1e-12 else 1.0 / len(fields)) for c in raw}
    sims: dict[str, float] = {}
    contributions: dict[str, float] = {}
    equal_values: dict[str, str] = {}
    evidence_mass = conflict_mass = 0.0
    conflict = False
    for c, na, nb in fields:
        equal = na == nb
        sim = 1.0 if equal else difflib.SequenceMatcher(None, na, nb).ratio()
        sims[c] = sim
        contributions[c] = weights[c] * sim
        if equal:
            evidence_mass += weights[c]
            equal_values[c] = na
        elif c in identifying and c in code_cols:
            conflict = True
            conflict_mass += weights[c]
    return {
        "score_uncapped": sum(contributions.values()),
        "sims": sims,
        "weights": weights,
        "contributions": contributions,
        "equal_values": equal_values,
        "evidence_mass": evidence_mass,
        "conflict_mass": conflict_mass,
        "conflict": conflict,
        "passes": evidence_mass >= MIN_EVIDENCE_MASS,
    }


def weighted_field_similarity(
    attrs_a: dict[str, str],
    attrs_b: dict[str, str],
    discriminative_by_col: dict[str, float],
    semantic_by_col: dict[str, str] | None = None,
    record_emb_a: np.ndarray | None = None,
    record_emb_b: np.ndarray | None = None,
) -> tuple[
    float, float, dict[str, float], dict[str, float], dict[str, float], dict[str, float]
]:
    """
    Returns (similarity, mass, per_field_contributions, per_field_display_similarities,
    per_field_char_jaccard, per_field_weight).
    Uses discriminative_score-derived weights, renormalized over overlapping keys.

    ``per_field_char_jaccard`` and ``per_field_weight`` (the renormalized weight ``wn``)
    are the field-specific inputs the similarity is built from; the embedding part of a
    hybrid field is the whole-record cosine, the same for every hybrid field.

    For columns whose semantic type is NAME or DESCRIPTIVE, when record-level
    embeddings are available, blends cosine similarity of record vectors with
    per-field ``char_jaccard`` using ``HYBRID_EMBED_WEIGHT`` / ``HYBRID_CHAR_WEIGHT``.

    per_field_display_similarities uses sequence_similarity for display (not the hybrid
    blend), so reviewers see actual string match quality per field.
    """
    common = set(attrs_a.keys()) & set(attrs_b.keys())
    weights: list[tuple[str, float]] = []
    for k in common:
        w = float(discriminative_by_col.get(k, 0.0))
        if w > 0:
            weights.append((k, w))
    if not weights:
        return 0.0, 0.0, {}, {}, {}, {}
    sem = semantic_by_col or {}
    hybrid_ok = record_emb_a is not None and record_emb_b is not None
    s_w = sum(w for _, w in weights)
    num = 0.0
    per_field: dict[str, float] = {}
    per_field_sim: dict[str, float] = {}
    per_field_jaccard: dict[str, float] = {}
    per_field_weight: dict[str, float] = {}
    for k, w in weights:
        wn = w / s_w
        va = str(attrs_a[k])
        vb = str(attrs_b[k])
        field_sem = str(sem.get(k, "")).strip().upper()
        raw_sim = char_jaccard(va, vb)
        per_field_jaccard[k] = raw_sim
        per_field_weight[k] = wn
        if hybrid_ok and field_sem in HYBRID_SEMANTIC_TYPES:
            sim = (
                HYBRID_EMBED_WEIGHT * cosine_similarity(record_emb_a, record_emb_b)
                + HYBRID_CHAR_WEIGHT * raw_sim
            )
        else:
            sim = raw_sim
        contribution = wn * raw_sim      # line 1154: was wn * sim
        per_field[k] = contribution

        num += wn * sim                  # separate accumulator for the score   
        # Display similarity: token jaccard for single-token DESCRIPTIVE (city/state/zip),
        # sequence similarity for everything else (addresses, names).
        per_field_sim[k] = (
            token_jaccard(va, vb)
            if field_sem == "DESCRIPTIVE" and len(va.split()) == 1
            else sequence_similarity(va, vb)
        )
    return float(num), float(s_w), per_field, per_field_sim, per_field_jaccard, per_field_weight


def fingerprint_to_label(fp: str) -> str:
    if _ATTR_SEP in fp:
        t, v = fp.split(_ATTR_SEP, 1)
        return f"{t}:{v}"
    return fp



def fetch_record_context(store, record_labels: list[str]):
    labels = [str(lb).strip() for lb in record_labels if str(lb).strip()]
    if not labels:
        return {}
    rows = store.record_context_rows(labels, _ATTR_SEP)
    meta: dict[str, dict] = {}
    for r in rows:
        cid_raw = r["cid"]
        if cid_raw is None or str(cid_raw).strip() == "":
            continue
        ck = str(cid_raw).strip()
        fps = r["fps"] or []
        fp_set = {str(x) for x in fps if x is not None and str(x) != ""}
        rid = r["record_id"]
        meta[ck] = {
            "record_id": str(rid) if rid is not None else ck,
            "input_file": str(r.get("input_file") or "").strip(),
            "fingerprints": fp_set,
            "attrs_by_col": fingerprints_to_attr_map(fp_set),
        }
    return meta


def neighbor_pairs(
    X: np.ndarray,
    node_ids: list[str],
    top_k: int,
) -> set[tuple[str, str]]:
    n = X.shape[0]
    out: set[tuple[str, str]] = set()
    if n < 2:
        return out

    k_query = min(n, max(2, top_k + 1))
    mode, mode_source = nn_mode()
    if mode == "approximate":
        if not _ANNOY_AVAILABLE:
            raise SystemExit(
                f"Neighbor search mode 'approximate' (from {mode_source}) needs Annoy, "
                "which is not installed. Install verity/requirements-ann.txt or use "
                "--nn-mode exact."
            )
        use_annoy = True
    elif mode == "exact":
        use_annoy = False
    else:
        use_annoy = n >= APPROXIMATE_NN_THRESHOLD and _ANNOY_AVAILABLE
    update_run_info(
        nn_mode_resolved="approximate" if use_annoy else "exact",
        n_records_in_index=n,
    )

    if use_annoy:
        assert AnnoyIndex is not None
        print(
            "Neighbor search: approximate (Annoy, angular distance, n_trees=50, seed=0), "
            f"n={n} (mode approximate)"
        )
        dim = int(X.shape[1])
        t0 = time.perf_counter()
        index = AnnoyIndex(dim, "angular")
        index.set_seed(0)
        x_f32 = np.ascontiguousarray(X, dtype=np.float32)
        for i in range(n):
            index.add_item(i, x_f32[i].tolist())
        index.build(50)
        build_s = time.perf_counter() - t0
        print(f"Neighbor index build time: {build_s:.3f}s")
        record_timing("verity.neighbors.build", build_s, parent="verity.match",
                      items=n, item="record", note="Annoy add_item + build(50)")

        t_q = time.perf_counter()
        for i in range(n):
            idx_list, dist_list = index.get_nns_by_item(
                i, k_query, include_distances=True
            )
            ranked = [
                (int(j), float(d))
                for j, d in zip(idx_list, dist_list)
                if int(j) != i
            ]
            eid_i = node_ids[i]
            for j, _ in ranked[:top_k]:
                eid_j = node_ids[int(j)]
                a, b = (eid_i, eid_j) if eid_i < eid_j else (eid_j, eid_i)
                out.add((a, b))
        record_timing("verity.neighbors.query", time.perf_counter() - t_q, parent="verity.match",
                      items=n, item="record", note=f"Annoy top-{top_k}")
        return out

    if mode == "auto" and n >= APPROXIMATE_NN_THRESHOLD and not _ANNOY_AVAILABLE:
        warnings.warn(
            "annoy is not installed; using exact sklearn NearestNeighbors, which will "
            "be slow for large datasets. Install with: "
            "pip install -r verity/requirements-ann.txt",
            UserWarning,
            stacklevel=2,
        )
    print(
        f"Neighbor search: exact (sklearn NearestNeighbors, cosine, brute), n={n} "
        "(mode exact)"
    )
    nn = NearestNeighbors(
        n_neighbors=k_query,
        metric="cosine",
        algorithm="brute",
    )
    t0 = time.perf_counter()
    nn.fit(X)
    build_s = time.perf_counter() - t0
    print(f"Neighbor index build time: {build_s:.3f}s")
    record_timing("verity.neighbors.build", build_s, parent="verity.match",
                  items=n, item="record", note="sklearn NearestNeighbors fit (cosine, brute)")
    t_q = time.perf_counter()
    dist, idx = nn.kneighbors(X, return_distance=True)

    for i in range(n):
        eid_i = node_ids[i]
        ranked = [
            (int(j), float(d))
            for j, d in zip(idx[i], dist[i])
            if int(j) != i
        ]
        for j, _ in ranked[:top_k]:
            eid_j = node_ids[int(j)]
            a, b = (eid_i, eid_j) if eid_i < eid_j else (eid_j, eid_i)
            out.add((a, b))
    record_timing("verity.neighbors.query", time.perf_counter() - t_q, parent="verity.match",
                  items=n, item="record", note=f"kneighbors top-{top_k} + pair set")

    return out


def canonical_match_candidate_row(r: dict[str, Any]) -> dict[str, Any]:
    """Return a row dict with ``record_a`` < ``record_b`` lexicographically (undirected)."""
    ra = str(r["record_a"])
    rb = str(r["record_b"])
    if ra < rb:
        return r
    if ra > rb:
        out = dict(r)
        out["record_a"] = rb
        out["record_b"] = ra
        return out
    return r


def dedupe_match_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    At most one row per undirected pair, with ``record_a`` < ``record_b``.
    When a pair appears more than once, keep the row that ranks highest on:
    exact duplicate first, then stronger match type (full_signal, weak_signal,
    embedding_only), then higher score. Ties keep the earlier row.
    """
    type_rank = {"full_signal": 2, "weak_signal": 1, "code_signal": 1}

    def rank(r: dict[str, Any]) -> tuple[int, int, float]:
        is_exact = int(r.get("shared_attributes") == EXACT_DUPLICATE_SHARED_ATTRIBUTES)
        return (
            is_exact,
            type_rank.get(str(r.get("match_type", "")), 0),
            float(r["score"]),
        )

    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        c = canonical_match_candidate_row(r)
        key = (str(c["record_a"]), str(c["record_b"]))
        prev = best.get(key)
        if prev is None or rank(c) > rank(prev):
            best[key] = c
    return list(best.values())


def _fmt6(x: Any) -> str:
    return "" if x is None else f"{float(x):.6f}"


def _json6(d: dict[str, float] | None) -> str:
    if d is None:
        return ""
    return json.dumps(
        {str(k): round(float(v), 6) for k, v in sorted(d.items())}, ensure_ascii=False
    )


def build_scored_pair_rows(
    exact_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    One row per unordered pair over exact-duplicate pairs and every scored neighbor pair,
    with the same dedupe as match_candidates.csv. score and match_type come from the
    winning row; the scoring columns come from the neighbor scoring of the pair (blank for
    an exact-duplicate pair that neighbor search did not find).
    """
    neighbor_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in scored_rows:
        c = canonical_match_candidate_row(r)
        neighbor_by_key[(str(c["record_a"]), str(c["record_b"]))] = c
    out: list[dict[str, Any]] = []
    for r in dedupe_match_candidate_rows(exact_rows + scored_rows):
        key = (str(r["record_a"]), str(r["record_b"]))
        is_exact = r.get("shared_attributes") == EXACT_DUPLICATE_SHARED_ATTRIBUTES
        nb = neighbor_by_key.get(key)
        if is_exact:
            source = "both" if nb is not None else "exact"
        else:
            source = "neighbor"
        out.append(
            {
                "record_a": key[0],
                "record_b": key[1],
                "score": float(r["score"]),
                "match_type": r["match_type"],
                "pair_source": source,
                "nb": nb,
            }
        )
    return out


def _micro(x: float) -> int:
    """A value as an integer number of millionths, as written to the CSV (%.6f)."""
    return int(round(float(f"{float(x):.6f}") * 1_000_000))


def decision_status_for(score: float, match_type: str, effective_threshold: float) -> str:
    """
    review when the written score is within REVIEW_MARGIN of the written effective
    threshold, inclusive; otherwise Verity's own decision: match when the pair is written
    to match_candidates.csv, else non_match. Compared on the 6-decimal values written to
    the file, in integer millionths, so the band edges are exact.
    """
    if abs(_micro(score) - _micro(effective_threshold)) <= _micro(REVIEW_MARGIN):
        return "review"
    return "match" if match_type in MATCH_CANDIDATES_WRITE_MATCH_TYPES else "non_match"


def write_scored_pairs(
    rows: list[dict[str, Any]], effective_threshold: float
) -> Counter:
    """Append rows to scored_pairs.csv; return decision_status counts."""
    status_counts: Counter = Counter()
    write_header = (not SCORED_PAIRS_PATH.is_file()) or SCORED_PAIRS_PATH.stat().st_size == 0
    SCORED_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCORED_PAIRS_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORED_PAIRS_FIELDNAMES)
        if write_header:
            w.writeheader()
        for r in rows:
            nb = r["nb"]
            status = decision_status_for(r["score"], r["match_type"], effective_threshold)
            if r["match_type"] == "code_conflict" and nb and nb["evidence_passes"]:
                status = "review"                  # would have matched but for a conflicting code-like field
            status_counts[status] += 1
            w.writerow(
                {
                    "record_a": r["record_a"],
                    "record_b": r["record_b"],
                    "score": _fmt6(r["score"]),
                    "score_uncapped": _fmt6(nb["score_uncapped"]) if nb else "",
                    "cosine": _fmt6(nb["cosine"]) if nb else "",
                    "match_type": r["match_type"],
                    "decision_status": status,
                    "pair_source": r["pair_source"],
                    "applied_threshold": _fmt6(nb["applied_threshold"]) if nb else "",
                    "effective_threshold": _fmt6(effective_threshold),
                    "evidence_passes": (
                        ("true" if nb["evidence_passes"] else "false") if nb else ""
                    ),
                    "evidence_mass": _fmt6(nb["evidence_mass"]) if nb else "",
                    "conflict_mass": _fmt6(nb["conflict_mass"]) if nb else "",
                    "field_similarities": _json6(nb["field_jaccard"]) if nb else "",
                    "field_weights": _json6(nb["field_weights"]) if nb else "",
                    "evidence_field_similarities": (
                        _json6(nb["field_similarities"]) if nb else ""
                    ),
                }
            )
    return status_counts


def write_scored_pairs_columns(
    semantic_by_col_lower: dict[str, str], mw: Any
) -> None:
    """Column -> semantic type, and the constants needed to rebuild score_uncapped."""
    doc = {
        "field_semantic_types": {
            str(k): str(v).strip().upper() for k, v in sorted(semantic_by_col_lower.items())
        },
        "score_formula": {
            "w_embed_blend": mw.w_embed_blend,
            "w_field_blend": mw.w_field_blend,
            "w_when_field_weights_empty": {"embed": 1.0, "field": 0.0},
            "hybrid_embed_weight": HYBRID_EMBED_WEIGHT,
            "hybrid_char_weight": HYBRID_CHAR_WEIGHT,
            "hybrid_semantic_types": list(HYBRID_SEMANTIC_TYPES),
            "score_cap": 1.0,
            "rule": (
                "score_uncapped = w_e*cosine + w_f*sum_k(field_weights[k]*s_k), "
                "where s_k = hybrid_embed_weight*cosine + "
                "hybrid_char_weight*field_similarities[k] if field_semantic_types[k] is "
                "in hybrid_semantic_types, else field_similarities[k]; (w_e, w_f) = "
                "(w_embed_blend, w_field_blend), or (1, 0) when field_weights is empty; "
                "score = min(score_uncapped, score_cap). Exact-duplicate rows "
                "(pair_source exact or both) have score 1.0 by rule."
            ),
        },
        "learned_threshold": mw.learned_threshold,
        "review_margin": REVIEW_MARGIN,
        "decision_status_rule": (
            "review when |score - effective_threshold| <= review_margin (inclusive, on the "
            "6-decimal values); otherwise match when match_type is full_signal, weak_signal "
            "or embedding_only (the rows of match_candidates.csv), else non_match"
        ),
        "effective_threshold": mw.effective_threshold,
        "column_notes": {
            "field_similarities": "per-field char Jaccard: the field-only part the score uses",
            "evidence_field_similarities": (
                "per-field display similarity read by the evidence gate, not by the score"
            ),
            "applied_threshold": (
                "threshold Verity applied: learned threshold for pairs with no shared "
                "keys, effective threshold otherwise"
            ),
        },
    }
    SCORED_PAIRS_COLUMNS_PATH.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def summarize_scores(
    scores: list[float],
    threshold: float,
    match_type_counts: dict[str, int] | None = None,
) -> None:
    arr = np.array(scores, dtype=np.float64)
    print(
        f"Scores for merge-quality pairs only (match_type full_signal or weak_signal): "
        f"{len(scores)} (effective threshold {threshold:.6f})"
    )
    if match_type_counts is not None:
        print("Match type counts (unique undirected pairs):")
        for key in sorted(match_type_counts.keys()):
            print(f"  {key}: {match_type_counts.get(key, 0)}")
    if len(scores) == 0:
        return
    print(
        "Score stats — min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}".format(
            float(arr.min()),
            float(arr.max()),
            float(arr.mean()),
            float(arr.std(ddof=0)),
        )
    )
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p}: {float(np.percentile(arr, p)):.4f}")

    lo = max(0.0, float(threshold) * 0.9)
    hi = min(1.0, float(arr.max()) + 0.02)
    if hi <= lo:
        hi = lo + 0.05
    bins = np.linspace(lo, hi, num=9)
    hist, _ = np.histogram(arr, bins=bins)
    print("Histogram (effective-threshold-aware bins):")
    for i in range(len(bins) - 1):
        print(f"  [{bins[i]:.3f}, {bins[i + 1]:.3f}): {int(hist[i])}")


def print_score_distribution_summary(scores: list[float]) -> None:
    """Summary for all scored neighbor pairs (clipped scores)."""
    print("=" * 72)
    print("  Match score distribution (all neighbor pairs)")
    print("=" * 72)
    n = len(scores)
    print(f"  Total pairs found: {n}")
    if n == 0:
        print("=" * 72 + "\n")
        return
    arr = np.array(scores, dtype=np.float64)
    print(f"  Min score: {float(arr.min()):.6f}")
    print(f"  Max score: {float(arr.max()):.6f}")
    print(f"  Average score: {float(arr.mean()):.6f}")
    c_65_70 = int(np.sum((arr >= 0.65) & (arr < 0.70)))
    c_70_80 = int(np.sum((arr >= 0.70) & (arr < 0.80)))
    c_80_90 = int(np.sum((arr >= 0.80) & (arr < 0.90)))
    c_90_100 = int(np.sum((arr >= 0.90) & (arr <= 1.0)))
    print("  Pairs in score bands:")
    print(f"    [0.65, 0.70): {c_65_70}")
    print(f"    [0.70, 0.80): {c_70_80}")
    print(f"    [0.80, 0.90): {c_80_90}")
    print(f"    [0.90, 1.00]: {c_90_100}")
    print("=" * 72 + "\n")


def main(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    record_label_from_csv_path: str | Path | None = None,
    new_record_ids: set[str] | None = None,
) -> None:
    """
    Loads ``data/{{csv_stem}}/column_classification.json`` when present, else
    ``data/column_classification.json`` (with a warning), then unions ``content_columns``
    with ``data/column_classification_global.json`` when present.
    Also loads ``attribute_weights.json`` (for weights and semantic minimum-evidence field groups).
    Embeddings come from ``data/embeddings_global.pkl`` (cumulative across pipeline runs).
    The neighbour index is built from **every** vector in that store so records from
    different ingestion files can appear as mutual top-k neighbours.
    ``match_candidates.csv`` lists undirected neighbour pairs that pass merge graph
    criteria (``match_type`` ``full_signal`` or ``weak_signal`` only). A **full** match
    run (``new_record_ids`` empty/omitted, or equal to the union of known CSV row ids)
    deletes any existing ``match_candidates.csv`` then overwrites it. A **delta** run
    (non-empty ``new_record_ids`` strictly smaller than that union) keeps prior rows,
    appends new rows (header only if the file is missing or empty), and only writes pairs
    where ``record_a`` or ``record_b`` is in ``new_record_ids``. Scoring always uses the
    full neighbour index.
    Other classified pairs (below threshold, insufficient evidence, embedding-only, etc.)
    are scored but omitted from the CSV so ``merge.py`` only sees true merge candidates.
    Other artifacts (``learned_threshold.json``, ``attribute_weights.json``)
    are read from the same directory as the resolved CSV.
    """
    csv_path = resolve_input_csv(input_path)
    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        if not f.readline().strip():
            raise SystemExit(f"Input CSV has no header row: {csv_path}")

    label_path = (
        resolve_input_csv(record_label_from_csv_path)
        if record_label_from_csv_path is not None
        else csv_path
    )
    record_label = record_label_from_csv(label_path)
    cls_path = resolve_match_column_classification_path(csv_path)
    classification = apply_global_content_overlay(load_column_classification(cls_path))
    content_raw = classification.get("content_columns") or []
    content_cols_lower = frozenset(
        col_key(c)
        for c in content_raw
        if str(c).strip()
    )
    meta_raw = classification.get("metadata_columns") or []
    metadata_cols_lower = frozenset(
        col_key(c) for c in meta_raw if str(c).strip()
    )
    if not content_cols_lower:
        raise SystemExit(
            f"{cls_path} has no content_columns — run rule_learner on this dataset first."
        )
    csv_compare_cols_lower = frozenset(
        c for c in content_cols_lower if c not in metadata_cols_lower
    )
    rid_col_l = col_key(classification.get("record_id_column") or "")
    if not rid_col_l:
        raise SystemExit(f"{cls_path} is missing record_id_column.")
    rows_by_id, row_sources = load_customer_rows_union_sources(
        csv_path, rid_col_l, EMBEDDINGS_FILE_REGISTRY_PATH
    )
    new_id_set = {str(x) for x in new_record_ids} if new_record_ids else set()
    all_union_ids = {str(k) for k in rows_by_id.keys()}
    is_full_match_run = not new_id_set or new_id_set == all_union_ids
    if is_full_match_run:
        OUTPUT_PATH.unlink(missing_ok=True)
        SCORED_PAIRS_PATH.unlink(missing_ok=True)
        print(f"Run mode: FULL — {len(all_union_ids)} records")
    else:
        print(f"Run mode: DELTA — {len(new_id_set)} new records")

    src_msg = ", ".join(str(p) for p in row_sources)
    print(
        f"Loaded column classification: {cls_path}\n"
        f"  content_columns ({len(content_cols_lower)}): "
        f"{', '.join(sorted(content_cols_lower))}\n"
        f"  metadata_columns ({len(metadata_cols_lower)}): "
        f"{', '.join(sorted(metadata_cols_lower))}\n"
        f"  CSV row union for evidence / fallback ({len(rows_by_id)} ids): {src_msg}"
    )

    weights_path = data_dir() / "attribute_weights.json"
    threshold_path = data_dir() / "learned_threshold.json"

    mw = load_match_weights(weights_path, threshold_path)
    sem_by_lower = load_column_semantic_types(classification, weights_path)

    evidence_profile = evidence_profile_from_semantic_map(
        sem_by_lower,
        csv_compare_cols_lower,
        rid_col_l,
        mw.discriminative_by_col,
    )

    id_f = ", ".join(evidence_profile.identity_fields)
    desc_f = ", ".join(evidence_profile.descriptive_fields)
    top_f = ", ".join(evidence_profile.top_discriminative_fields)

    print(f"identity_fields: [{id_f}]")
    print(f"descriptive_fields: [{desc_f}]")
    print(f"top_discriminative_fields: [{top_f}]")
    print(
        "evidence gates: "
        f"min_descriptive_fields={evidence_profile.min_descriptive_fields}, "
        f"min_evidence_mass={evidence_profile.min_evidence_mass:.3f}, "
        f"max_conflict_mass={evidence_profile.max_conflict_mass:.3f}"
    )

    exact_pairs = exact_duplicate_pairs(
        rows_by_id=rows_by_id,
        content_columns_lower=csv_compare_cols_lower,
        semantic_by_col_lower=sem_by_lower,
    )
    if new_id_set and not is_full_match_run:
        exact_pairs = [
            (a, b) for a, b in exact_pairs
            if a in new_id_set or b in new_id_set
        ]
    print(f"Exact-duplicate pairs (composite name+location key): {len(exact_pairs)}")
    # Written together with the neighbor rows below, through one dedupe, so a pair
    # found both ways appears once in match_candidates.csv.
    exact_rows: list[dict[str, Any]] = [
        {
            "record_a": _rid_a,
            "record_b": _rid_b,
            "score": 1.0,
            "shared_attributes": EXACT_DUPLICATE_SHARED_ATTRIBUTES,
            "match_type": "full_signal",
            "evidence_mass": 1.0,
            "conflict_mass": 0.0,
            "evidence_detail": "exact duplicate (normalized name+location composite key)",
            "field_contributions": {},
            "field_similarities": {},
        }
        for _rid_a, _rid_b in exact_pairs
    ]

    load_dotenv(PROJECT_ROOT / ".env")

    if not GLOBAL_EMBEDDINGS_PATH.is_file():
        raise SystemExit(f"Missing global embeddings file: {GLOBAL_EMBEDDINGS_PATH}")

    global_emb = load_embeddings(GLOBAL_EMBEDDINGS_PATH)
    if not global_emb:
        raise SystemExit("embeddings_global.pkl is empty")

    X, global_nn_ids = global_embedding_matrix_and_ids(global_emb)
    n_global = len(global_nn_ids)
    print(f"Building neighbour index from {n_global} global records")
    print_global_embedding_source_breakdown(global_nn_ids, EMBEDDINGS_FILE_REGISTRY_PATH)
    origin_map = load_embeddings_file_registry(EMBEDDINGS_FILE_REGISTRY_PATH)

    pairs = neighbor_pairs(X, global_nn_ids, TOP_K)
    print(
        f"Neighbor graph: {n_global} global records, unique undirected pairs "
        f"from top-{TOP_K} NN: {len(pairs)}"
    )

    from record_store import open_store

    store = open_store()

    emb_map: dict[str, np.ndarray] = {
        k: np.asarray(global_emb[k], dtype=np.float64) for k in global_emb
    }

    pending: list[dict[str, Any]] = []
    disc = mw.discriminative_by_col
    skipped_missing_store_context = 0
    skipped_missing_embeddings = 0
    skipped_no_semantic_field = 0
    skipped_single_token_semantic_fields = 0
    # Code path (pairs that fail the multi-word test): counters and per-run context, built on first use.
    text_path_pairs = 0
    code_path_pairs = 0
    skipped_no_code_field = 0
    code_ctx: dict[str, Any] = {}
    # Pairs that would have been embedding_only (no shared keys, high cosine, score above
    # the embedding-only threshold) but were blocked because both records share an input file.
    embedding_only_blocked_same_file = 0

    record_labels = store.record_labels()
    context = fetch_record_context(store, record_labels)
    if origin_map is None:
        # No registry: use each record's input file from the store for lineage counts.
        origin_map = {
            str(v["record_id"]): v["input_file"] for v in context.values() if v["input_file"]
        } or None
    neo_ids = frozenset(context.keys())
    missing_in_pickle = sorted(neo_ids - frozenset(global_emb.keys()))
    if missing_in_pickle:
        sample = ", ".join(missing_in_pickle[:5])
        print(
            f"Warning: {len(missing_in_pickle)} stored {record_label} id(s) have no row in "
            f"{GLOBAL_EMBEDDINGS_PATH.name} (not in neighbour index). Example: {sample!r}. "
            "Run embed for those sources."
        )

    evidence_debug_cross_remaining = 5
    shared_keys_sem_debug_remaining = 5
    # Timing: verity.score covers this loop (features, score, evidence per pair) and the
    # decision pass below; per-pair latency is measured for each pair that is scored.
    t_score = time.perf_counter()
    pair_seconds: list[float] = []
    for ea, eb in sorted(pairs):
        t_pair = time.perf_counter()
        ca = context.get(ea)
        cb = context.get(eb)
        if ca is None or cb is None:
            skipped_missing_store_context += 1
            continue

        if ea not in emb_map or eb not in emb_map:
            skipped_missing_embeddings += 1
            continue

        va = np.asarray(emb_map[ea], dtype=np.float64)
        vb = np.asarray(emb_map[eb], dtype=np.float64)
        cos_part = cosine_similarity(va, vb)

        sa = ca["fingerprints"]
        sb = cb["fingerprints"]
        aa = ca["attrs_by_col"]
        ab = cb["attrs_by_col"]

        # Correctness guard: records with blank DESCRIPTIVE/NAME fields embed only
        # identifier tokens. Those vectors cluster artificially and yield spuriously
        # high cosine vs other blank-description records. Skip such pairs.
        sem_fields = frozenset(
            k for k, s in sem_by_lower.items()
            if str(s).strip().upper() in ("DESCRIPTIVE", "NAME", "STREET_ADDRESS")
        )
        if sem_fields:
            if not any(k in aa for k in sem_fields) or not any(k in ab for k in sem_fields):
                skipped_no_semantic_field += 1
                continue
            has_a = any(" " in str(aa.get(k, "")) for k in sem_fields)
            has_b = any(" " in str(ab.get(k, "")) for k in sem_fields)
            if not has_a or not has_b:
                skipped_single_token_semantic_fields += 1
                if not code_ctx:
                    code_ctx["cols"] = code_like_columns(sem_by_lower, rows_by_id, csv_compare_cols_lower)
                    code_ctx["ident"] = identifying_columns(code_ctx["cols"], mw.discriminative_share, len(csv_compare_cols_lower))
                cp = code_path_pair(aa, ab, code_ctx["cols"], csv_compare_cols_lower, disc, code_ctx["ident"])
                if cp is None:
                    skipped_no_code_field += 1
                    continue
                code_path_pairs += 1
                eq_labels = "|".join(f"{c}:{v}" for c, v in sorted(cp["equal_values"].items()))
                pending.append(
                    {
                        "score": min(cp["score_uncapped"], 1.0),
                        "cos_part": cos_part,
                        "rid_a": str(ca["record_id"]),
                        "rid_b": str(cb["record_id"]),
                        "shared_attributes": eq_labels or "code_path",
                        "resolved_type": "code_signal",
                        "code_path": True,
                        "code_conflict": cp["conflict"],
                        "has_shared_keys": True,
                        "evidence_passes": cp["passes"],
                        "evidence_mass": cp["evidence_mass"],
                        "conflict_mass": cp["conflict_mass"],
                        "evidence_detail": "code path: " + ("code conflict" if cp["conflict"] else "evidence pass" if cp["passes"] else "insufficient evidence"),
                        "field_contributions": cp["contributions"],
                        "field_similarities": cp["sims"],
                        "score_uncapped": cp["score_uncapped"],
                        "field_jaccard": cp["sims"],
                        "field_weights": cp["weights"],
                    }
                )
                pair_seconds.append(time.perf_counter() - t_pair)
                continue
        text_path_pairs += 1

        (
            field_sim,
            mass,
            field_contributions,
            field_similarities,
            field_jaccard,
            field_weights,
        ) = weighted_field_similarity(
            aa,
            ab,
            disc,
            semantic_by_col=sem_by_lower,
            record_emb_a=emb_map.get(ea),
            record_emb_b=emb_map.get(eb),
        )

        if mass <= 1e-12:
            w_e, w_f = 1.0, 0.0
        else:
            w_e, w_f = mw.w_embed_blend, mw.w_field_blend

        score_uncapped = w_e * cos_part + w_f * field_sim
        score = min(score_uncapped, 1.0)

        rid_a = str(ca["record_id"])
        rid_b = str(cb["record_id"])
        file_a, file_b = ca["input_file"], cb["input_file"]
        cross_file = bool(file_a) and bool(file_b) and file_a != file_b

        _overlap_labels, shared_keys, candidate_join = graph_overlap_labels_for_pair(
            store,
            sa,
            sb,
            rid_a,
            rid_b,
            metadata_cols_lower,
            rows_by_id,
            csv_compare_cols_lower,
            record_label,
        )

        if not shared_keys:
            shared_attributes = "embedding_only"
        else:
            shared_attributes = candidate_join or "embedding_only"
            if shared_keys_sem_debug_remaining > 0:
                types_dbg = {
                    k: sem_by_lower.get(k, "MISSING") for k in shared_keys
                }
                print(f"DEBUG shared_keys={shared_keys}, types={types_dbg}")
                shared_keys_sem_debug_remaining -= 1

        decision = evaluate_schema_agnostic_evidence(
            shared_keys=shared_keys,
            field_similarities=field_similarities,
            semantic_by_col_lower=sem_by_lower,
            discriminative_by_col=mw.discriminative_by_col,
            profile=evidence_profile,
            cosine_similarity_score=cos_part,
            effective_threshold=mw.effective_threshold,
            attrs_a=aa,
            attrs_b=ab,
            cross_file=cross_file,
        )
        if decision.detail.startswith("embedding-only blocked") and (
            score > mw.learned_threshold
        ):
            embedding_only_blocked_same_file += 1

        resolved_type = decision.match_type

        if (
            evidence_debug_cross_remaining > 0
            and pair_lineage_bucket(rid_a, rid_b, origin_map) == "cross_file"
        ):
            evidence_debug_cross_remaining -= 1
            flds = ", ".join(sorted(shared_keys))
            print(
                "[EVIDENCE DEBUG] "
                f"pair {rid_a!r} ↔ {rid_b!r} | "
                f"shared_attributes={shared_attributes!r} | "
                f"parsed_field_names=[{flds}] | "
                f"evidence_mass={decision.evidence_mass:.4f} | "
                f"conflict_mass={decision.conflict_mass:.4f} | "
                f"passes={decision.passes} | "
                f"{decision.detail}"
            )

        pending.append(
            {
                "score": score,
                "cos_part": cos_part,
                "rid_a": rid_a,
                "rid_b": rid_b,
                "shared_attributes": shared_attributes,
                "resolved_type": resolved_type,
                "has_shared_keys": bool(shared_keys),
                "evidence_passes": decision.passes,
                "evidence_mass": decision.evidence_mass,
                "conflict_mass": decision.conflict_mass,
                "evidence_detail": decision.detail,
                "field_contributions": field_contributions,
                "field_similarities": field_similarities,
                "score_uncapped": score_uncapped,
                "field_jaccard": field_jaccard,
                "field_weights": field_weights,
            }
        )
        pair_seconds.append(time.perf_counter() - t_pair)

    threshold = mw.effective_threshold
    embedding_only_threshold = mw.learned_threshold
    print(
        f"Effective score threshold: {threshold:.4f} "
        f"(max(learned={mw.learned_threshold:.4f}, floor={LEARNED_THRESHOLD_MIN_FLOOR:.2f}))"
    )
    print(
        f"Embedding-only score threshold: {embedding_only_threshold:.4f} "
        f"(learned threshold from {threshold_path.name}, no floor)"
    )
    all_pair_scores = [p["score"] for p in pending]
    rows_out: list[dict] = []
    dropped_no_minimum_evidence = 0
    dropped_overlap_signal = 0
    for item in pending:
        score = item["score"]
        cos_part = item["cos_part"]
        rid_a = item["rid_a"]
        rid_b = item["rid_b"]
        shared_attributes = item["shared_attributes"]
        resolved_type = item["resolved_type"]
        has_shared = item["has_shared_keys"]
        evidence_passes = item["evidence_passes"]

        effective_score_threshold = (
            embedding_only_threshold if not has_shared else threshold
        )

        if item.get("code_path"):
            # Code path: same strict comparison as the text path; an identifying-field conflict
            # that blocks a pair which would otherwise pass or sit in the band is code_conflict.
            match_type, evidence_passes = code_path_label(
                score, effective_score_threshold, evidence_passes, item["code_conflict"]
            )
            if match_type == "insufficient_evidence":
                dropped_no_minimum_evidence += 1
        elif score <= effective_score_threshold:
            match_type = "below_threshold"
        elif not evidence_passes:
            dropped_no_minimum_evidence += 1
            match_type = "insufficient_evidence"
        elif resolved_type is None:
            dropped_overlap_signal += 1
            match_type = "overlap_rejected"
        else:
            match_type = resolved_type

        rows_out.append(
            {
                "record_a": rid_a,
                "record_b": rid_b,
                "score": score,
                "shared_attributes": shared_attributes,
                "match_type": match_type,
                "evidence_mass": item.get("evidence_mass", 0.0),
                "conflict_mass": item.get("conflict_mass", 0.0),
                "evidence_detail": item.get("evidence_detail", ""),
                "field_contributions": item.get("field_contributions") or {},
                "field_similarities": item.get("field_similarities") or {},
                # Extra fields for scored_pairs.csv only; match_candidates.csv ignores them.
                "cosine": cos_part,
                "score_uncapped": item["score_uncapped"],
                "applied_threshold": effective_score_threshold,
                "evidence_passes": evidence_passes,
                "field_jaccard": item["field_jaccard"],
                "field_weights": item["field_weights"],
            }
        )

    record_timing(
        "verity.score", time.perf_counter() - t_score, parent="verity.match",
        items=len(pair_seconds), item="pair", per_item_s=pair_seconds,
        note="per pair: field similarity, score, shared keys, evidence; plus the decision pass",
    )
    print_score_distribution_summary(all_pair_scores)

    id_disp = ", ".join(evidence_profile.identity_fields)
    desc_disp = ", ".join(evidence_profile.descriptive_fields)
    top_disp = ", ".join(evidence_profile.top_discriminative_fields)

    print(
        f"Dropped {dropped_no_minimum_evidence} pairs — insufficient schema-agnostic evidence\n"
        f" (identity_fields used: [{id_disp}]\n"
        f"  descriptive_fields used: [{desc_disp}]\n"
        f"  top_discriminative_fields: [{top_disp}])"
    )

    if dropped_overlap_signal:
        print(
            f"Dropped {dropped_overlap_signal} pairs — overlap is not "
            "full_signal (need 2+ fields, or a single QUASI_IDENTIFIER only) "
            "and not pure embedding-only"
        )
    if skipped_missing_store_context:
        print(
            f"Skipped {skipped_missing_store_context} neighbour pair(s) — "
            "one or both endpoints not present in the record store "
            f"(embeddings exist but no {record_label} match for scoring)."
        )
    if skipped_missing_embeddings:
        print(
            f"Skipped {skipped_missing_embeddings} neighbour pair(s) — "
            "missing embedding vector for one or both endpoints in the global store."
        )

    rows_out.sort(key=lambda r: (-r["score"], r["record_a"], r["record_b"]))

    n_cross_file = n_same_file = n_unknown = 0
    for r in rows_out:
        b = pair_lineage_bucket(str(r["record_a"]), str(r["record_b"]), origin_map)
        if b == "cross_file":
            n_cross_file += 1
        elif b == "same_file":
            n_same_file += 1
        else:
            n_unknown += 1
    print(
        f"Cross-file pairs (registry origin): {n_cross_file}\n"
        f"Same-file pairs (registry origin): {n_same_file}\n"
        f"Unknown-lineage pairs (missing registry ids or map): {n_unknown}"
    )

    n_dropped_insufficient = sum(
        1 for r in rows_out if r["match_type"] == "insufficient_evidence"
    )
    n_dropped_below = sum(1 for r in rows_out if r["match_type"] == "below_threshold")

    neighbor_rows = [
        r for r in rows_out if r["match_type"] in MATCH_CANDIDATES_WRITE_MATCH_TYPES
    ]
    rows_to_write = dedupe_match_candidate_rows(exact_rows + neighbor_rows)
    exact_keys = {tuple(sorted((str(a), str(b)))) for a, b in exact_pairs}
    n_exact_also_neighbor = sum(
        1
        for r in neighbor_rows
        if tuple(sorted((str(r["record_a"]), str(r["record_b"])))) in exact_keys
    )
    print(
        f"Exact-duplicate pairs also found by neighbor search: {n_exact_also_neighbor} "
        "(one row kept per pair)"
    )
    rows_to_write.sort(key=lambda r: (-r["score"], r["record_a"], r["record_b"]))

    if new_id_set and not is_full_match_run:
        total_pairs_before_delta = len(rows_to_write)
        new_ids = frozenset(new_id_set)
        rows_to_write = [
            r
            for r in rows_to_write
            if str(r["record_a"]) in new_ids or str(r["record_b"]) in new_ids
        ]
        print(
            f"Delta mode: {len(rows_to_write)} pairs involving new records "
            f"(from {total_pairs_before_delta} total pairs)"
        )

    n_full_signal_kept = sum(
        1 for r in rows_to_write if r["match_type"] == "full_signal"
    )
    n_weak_signal_kept = sum(
        1 for r in rows_to_write if r["match_type"] == "weak_signal"
    )

    t_write = time.perf_counter()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not OUTPUT_PATH.is_file()) or OUTPUT_PATH.stat().st_size == 0

    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_CANDIDATES_FIELDNAMES)
        if write_header:
            w.writeheader()
        for r in rows_to_write:
            fc = r.get("field_contributions") or {}
            fs = r.get("field_similarities") or {}
            row_base = {
                "score": f"{r['score']:.6f}",
                "shared_attributes": r["shared_attributes"],
                "match_type": r["match_type"],
                "evidence_mass": f"{float(r.get('evidence_mass', 0.0)):.6f}",
                "conflict_mass": f"{float(r.get('conflict_mass', 0.0)):.6f}",
                "evidence_detail": r.get("evidence_detail", ""),
                "field_contributions": json.dumps(fc, ensure_ascii=False) if fc else "",
                "field_similarities": json.dumps(fs, ensure_ascii=False) if fs else "",
            }
            w.writerow(
                {
                    "record_a": r["record_a"],
                    "record_b": r["record_b"],
                    **row_base,
                }
            )

    store.close()

    scored_pair_rows = build_scored_pair_rows(exact_rows, rows_out)
    if new_id_set and not is_full_match_run:
        new_ids_sp = frozenset(new_id_set)
        scored_pair_rows = [
            r
            for r in scored_pair_rows
            if r["record_a"] in new_ids_sp or r["record_b"] in new_ids_sp
        ]
    scored_pair_rows.sort(key=lambda r: (-r["score"], r["record_a"], r["record_b"]))
    decision_status_counts = write_scored_pairs(scored_pair_rows, threshold)
    write_scored_pairs_columns(sem_by_lower, mw)
    record_timing(
        "verity.write_outputs", time.perf_counter() - t_write, parent="verity.match",
        items=len(rows_to_write) + len(scored_pair_rows), item="row",
        note="match_candidates.csv, scored_pairs.csv, scored_pairs_columns.json",
    )
    n_exact_only = sum(1 for r in scored_pair_rows if r["pair_source"] == "exact")
    if skipped_no_semantic_field or skipped_single_token_semantic_fields:
        print(
            f"Skipped {skipped_no_semantic_field} neighbour pair(s) — a record has no "
            "NAME/DESCRIPTIVE/STREET_ADDRESS value; "
            f"{skipped_single_token_semantic_fields} — no such value contains a space."
        )
    update_run_info(
        neighbor_pairs=len(pairs),
        skipped_missing_store_context=skipped_missing_store_context,
        skipped_missing_embeddings=skipped_missing_embeddings,
        skipped_no_semantic_field=skipped_no_semantic_field,
        skipped_single_token_semantic_fields=skipped_single_token_semantic_fields,
        text_path_pairs=text_path_pairs,
        code_path_pairs=code_path_pairs,
        skipped_no_code_field=skipped_no_code_field,
        code_like_columns=sorted(code_ctx["cols"]) if code_ctx else [],
        identifying_columns=sorted(code_ctx["ident"]) if code_ctx else [],
        embedding_only_blocked_same_file=embedding_only_blocked_same_file,
        scored_neighbor_pairs=len(rows_out),
        exact_duplicate_pairs=len(exact_pairs),
        exact_only_pairs=n_exact_only,
        scored_pairs_rows=len(scored_pair_rows),
        match_candidates_rows=len(rows_to_write),
        review_margin=REVIEW_MARGIN,
        decision_status_counts={
            s: decision_status_counts.get(s, 0) for s in ("match", "non_match", "review")
        },
    )

    type_counts: dict[str, int] = {}
    for r in rows_out:
        t = str(r["match_type"])
        type_counts[t] = type_counts.get(t, 0) + 1
    merge_quality_scores = [
        float(r["score"])
        for r in rows_out
        if r["match_type"] in MATCH_CANDIDATES_WRITE_MATCH_TYPES
    ]
    summarize_scores(merge_quality_scores, threshold, match_type_counts=type_counts)
    print(f"Kept {n_full_signal_kept} full_signal pairs")
    print(f"Kept {n_weak_signal_kept} weak_signal pairs")
    print(f"Dropped {n_dropped_insufficient} insufficient_evidence pairs")
    print(f"Dropped {n_dropped_below} below_threshold pairs")
    print(f"Wrote {len(rows_to_write)} row(s) to match_candidates.csv")
    print(
        f"Wrote {len(scored_pair_rows)} row(s) to scored_pairs.csv "
        f"({n_exact_only} exact-duplicate pair(s) not found by neighbor search)"
    )
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Score candidate record-node pairs from embeddings + graph "
            "(label from CSV stem, same as build_graph)."
        )
    )
    ap.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "Source customer CSV path (absolute, or relative to project root); "
            "artifacts are read from its directory. "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    args = ap.parse_args()
    main(
        input_path=args.csv_path if args.csv_path is not None else DEFAULT_INPUT_PATH
    )

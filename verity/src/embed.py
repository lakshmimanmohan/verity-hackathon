"""
Embed record texts from their stored attribute values using
sentence-transformers and IDF-weighted combination; persist to the embeddings store,
``data/embeddings_global.pkl`` (with a ``content_columns`` fingerprint in a reserved
meta entry), and ``data/<csv_stem>/embeddings.pkl``. The record node label matches
``build_graph`` (derived from the input CSV stem).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import data_dir, record_timing, update_run_info


def model_in_local_cache(model_name: str) -> bool | None:
    """True if the model's files are in the local Hugging Face cache (None if unknown)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)

MODEL_NAME = "all-MiniLM-L6-v2"
# The embedding model is loaded from this exact Hub revision, from model.safetensors only (no pickle weights), with no custom
# code (trust_remote_code off). Changing the revision is a model change: it changes the embeddings.
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WRITE_BATCH_SIZE = 50
STORE_FETCH_PAGE_SIZE = 5000
ENCODE_SEGMENT_BATCH = 512
PICKLE_SAVE_EVERY_NEW_RECORDS = 5000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = None
GLOBAL_EMBEDDINGS_PATH = data_dir() / "embeddings_global.pkl"
# Reserved pickle key for ``content_columns`` fingerprint and related metadata (not a record id).
EMBEDDINGS_STORE_META_KEY = "__embeddings_store_meta__"
COLUMN_CLASSIFICATION_NAME = "column_classification.json"
DATA_COLUMN_CLASSIFICATION_PATH = data_dir() / COLUMN_CLASSIFICATION_NAME


def slug_for(attr_type: str) -> str:
    return attr_type.lower()


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


def segment_text(attr_type: str, value: str) -> str:
    return f"{slug_for(attr_type)}:{value}"


def _rows_into_buckets(rows: Iterable[Any]) -> defaultdict[str, set[tuple[str, str]]]:
    buckets: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        cid = r["record_id"]
        if cid is None or str(cid).strip() == "":
            continue
        record_key = str(cid).strip()
        bucket = buckets[record_key]
        attr_type = r["attr_type"]
        value = r["value"]
        if attr_type is None or value is None:
            continue
        bucket.add((str(attr_type), str(value)))
    return buckets


def fetch_records_page(store, record_label: str, skip: int, limit: int) -> dict[str, list[tuple[str, str]]]:
    """
    One page of ``record_label`` records (ordered by string id) with their
    (attr_type, value) pairs, same shape as legacy ``fetch_records``.
    """
    result = store.records_page_rows(record_label, skip, limit)
    buckets = _rows_into_buckets(result)
    return {ck: sorted(s) for ck, s in buckets.items()}


def memory_usage_mb() -> tuple[float, str]:
    """Return (megabytes, label) for current process RSS when possible."""
    try:
        import psutil

        rss = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
        return rss, "RSS MB (psutil)"
    except ImportError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                h, ctypes.byref(counters), counters.cb
            ):
                ws = float(counters.WorkingSetSize) / (1024.0 * 1024.0)
                return ws, "working set MB (Windows)"
        except Exception:
            pass
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kilobytes; macOS often bytes — treat large values as bytes.
        if ru > 10**9:
            mb = float(ru) / (1024.0 * 1024.0)
        else:
            mb = float(ru) / 1024.0
        return mb, "max RSS MB (resource, approximate)"
    except Exception:
        return -1.0, "unavailable (install psutil for accurate RSS on Windows)"


def _stored_vector_ok(vectors_store: dict[str, list[float]], sk: str, dim: int) -> bool:
    prev = vectors_store.get(sk)
    if prev is None:
        return False
    arr = np.asarray(prev, dtype=np.float64).reshape(-1)
    return int(arr.size) == dim


def encode_segments_in_batches(
    model: SentenceTransformer,
    segments: list[str],
    batch_size: int = ENCODE_SEGMENT_BATCH,
) -> dict[str, np.ndarray]:
    """Encode unique segment strings in fixed-size chunks (no single giant encode list)."""
    segment_embeddings: dict[str, np.ndarray] = {}
    n = len(segments)
    for start in range(0, n, batch_size):
        chunk = segments[start : start + batch_size]
        vecs = model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=False,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for text, vec in zip(chunk, vecs):
            segment_embeddings[text] = np.asarray(vec, dtype=np.float64)
    return segment_embeddings


def document_frequency(groups: dict[str, list[tuple[str, str]]]) -> dict[tuple[str, str], int]:
    df: dict[tuple[str, str], int] = defaultdict(int)
    for attrs in groups.values():
        for key in set(attrs):
            df[key] += 1
    return dict(df)


def inverse_frequency_weights(
    attrs: list[tuple[str, str]],
    df: dict[tuple[str, str], int],
    num_records: int,
) -> list[float]:
    out: list[float] = []
    for key in attrs:
        freq = df.get(key, 1)
        w = math.log((num_records + 1.0) / (freq + 1e-6)) + 1.0
        out.append(w)
    return out


def resolve_input_csv(input_path: str | Path) -> Path:
    """Resolve CSV path; relative paths are taken from ``PROJECT_ROOT``."""
    p = Path(input_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def resolve_column_classification_path(csv_path: Path) -> Path:
    """``column_classification.json`` in the data folder."""
    if DATA_COLUMN_CLASSIFICATION_PATH.is_file():
        return DATA_COLUMN_CLASSIFICATION_PATH
    raise SystemExit(
        f"Missing column classification file. Expected:\n"
        f"  {DATA_COLUMN_CLASSIFICATION_PATH}\n"
        "Run src/rule_learner.py on your CSV first."
    )


def load_column_classification(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def content_columns_fingerprint(classification: dict[str, Any]) -> str:
    """Stable hash of normalized ``content_columns`` for embedding-invalidation."""
    cols = classification.get("content_columns") or []
    norm = sorted({str(c).strip().lower() for c in cols if str(c).strip()})
    blob = json.dumps(norm, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_csv_record_ids(csv_path: Path, classification: dict[str, Any]) -> set[str]:
    """Record ids from the current CSV (``record_id_column``), normalized as strings."""
    rid_col = str(classification.get("record_id_column") or "").strip().lower()
    if not rid_col:
        raise SystemExit("column_classification.json is missing record_id_column.")
    out: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        keys = list(reader.fieldnames)
        key_map = {k: str(k).strip().lower() for k in keys}
        for row in reader:
            rid = ""
            for k in keys:
                if key_map[k] == rid_col:
                    raw = row.get(k)
                    rid = "" if raw is None else str(raw).strip()
                    break
            if rid:
                out.add(rid)
    return out


def load_global_embeddings_store(
    path: Path,
) -> tuple[dict[str, list[float]], dict[str, Any] | None]:
    """
    Load cumulative embeddings from previous runs. Returns ``(vectors_by_id, meta)``.
    ``meta`` holds ``content_columns_fingerprint`` when the store was last saved by
    this pipeline; ``None`` if missing (legacy flat pickle).
    """
    if not path.is_file():
        return {}, None
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        return {}, None
    meta: dict[str, Any] | None = None
    out: dict[str, list[float]] = {}
    for k, v in raw.items():
        sk = str(k).strip()
        if not sk:
            continue
        if sk == EMBEDDINGS_STORE_META_KEY and isinstance(v, dict):
            meta = dict(v)
            continue
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            out[sk] = [float(x) for x in v]
        elif hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            try:
                out[sk] = [float(x) for x in v]  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return out, meta


def save_global_embeddings_store(
    path: Path,
    vectors: dict[str, list[float]],
    embed_meta: dict[str, Any],
) -> None:
    """Persist vectors plus metadata under ``EMBEDDINGS_STORE_META_KEY``."""
    payload: dict[str, Any] = {
        EMBEDDINGS_STORE_META_KEY: dict(embed_meta),
        **vectors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def file_scoped_embeddings_path(csv_path: Path) -> Path:
    """``{csv_stem}/embeddings.pkl`` in the data folder."""
    return data_dir() / csv_path.stem / "embeddings.pkl"


def _embedding_to_plain_list(
    emb: object, record_id: str, expected_dim: int
) -> list[float] | None:
    """
    Convert an embedding to a plain Python list for pickle.
    Returns None if the value is missing, scalar-like, empty, or wrong length.
    """
    if emb is None:
        print(f"Warning: embedding is None for record id={record_id!r}, skipping.")
        return None
    if isinstance(emb, np.ndarray):
        arr = np.asarray(emb, dtype=np.float64).reshape(-1)
    elif isinstance(emb, (list, tuple)):
        arr = np.asarray(emb, dtype=np.float64).reshape(-1)
    else:
        print(
            f"Warning: embedding has unexpected type {type(emb).__name__} "
            f"for record id={record_id!r}, skipping."
        )
        return None
    if arr.size <= 1:
        print(
            f"Warning: embedding is scalar or empty (size={arr.size}) "
            f"for record id={record_id!r}, skipping."
        )
        return None
    if int(arr.size) != expected_dim:
        print(
            f"Warning: embedding length {arr.size} != model dimension {expected_dim} "
            f"for record id={record_id!r}, skipping."
        )
        return None
    return arr.tolist()


def main(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    record_label_from_csv_path: str | Path | None = None,
) -> set[str]:
    """
    Embeddings are computed from stored records (label matches
    ``record_label_from_csv`` for ``record_label_from_csv_path`` if set, else
    ``input_path``; graph loaded separately).
    ``input_path`` is the source CSV on disk; it must exist and is read once to validate
    the pipeline input before embedding.

    ``data/embeddings_global.pkl`` is loaded first (cumulative across runs). New
    vectors are computed only for record ids not yet in that store unless the file-scoped
    ``content_columns`` fingerprint changed, in which case every record is
    recomputed. Records are read in pages; the global pickle is
    flushed periodically during embedding so interrupted runs retain progress. The merged
    store is written to disk; only embeddings computed in this run are written to the record store
    (on a schema / fingerprint change, that is every record). A per-file copy is
    written to ``data/{{csv_stem}}/embeddings.pkl``.

    Returns:
        Set of record ids whose embeddings were computed in this run (same scope as
        ``to_compute`` / write batch), for match-phase delta vs full detection.
    """
    vectors_store, store_meta = load_global_embeddings_store(GLOBAL_EMBEDDINGS_PATH)
    n_existing = len(vectors_store)
    print(f"Global store: {n_existing} existing records")
    m0, m0_label = memory_usage_mb()
    if m0 >= 0:
        print(f"Memory at start: {m0:.1f} {m0_label}")
    else:
        print(f"Memory at start: {m0_label}")

    csv_path = resolve_input_csv(input_path)
    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path}")
    label_path = (
        resolve_input_csv(record_label_from_csv_path)
        if record_label_from_csv_path is not None
        else csv_path
    )
    record_label = record_label_from_csv(label_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        if not f.readline().strip():
            raise SystemExit(f"Input CSV has no header row: {csv_path}")

    cls_path = resolve_column_classification_path(csv_path)
    classification = load_column_classification(cls_path)
    current_fp = content_columns_fingerprint(classification)
    file_ids = load_csv_record_ids(csv_path, classification)

    from record_store import open_store

    store = open_store()

    print(
        f"Pass 1: paginated record store scan ({STORE_FETCH_PAGE_SIZE} rows per page) "
        f"for {record_label}…"
    )
    df_acc: defaultdict[tuple[str, str], int] = defaultdict(int)
    neo_ids: set[str] = set()
    skip = 0
    while True:
        page = fetch_records_page(store, record_label, skip, STORE_FETCH_PAGE_SIZE)
        if not page:
            break
        for rid, attrs in page.items():
            sk = str(rid).strip()
            if not sk:
                continue
            neo_ids.add(sk)
            for key in set(attrs):
                df_acc[key] += 1
        if len(page) < STORE_FETCH_PAGE_SIZE:
            break
        skip += STORE_FETCH_PAGE_SIZE

    num_records = len(neo_ids)
    if num_records == 0:
        store.close()
        raise SystemExit(f"No {record_label} nodes found.")

    df: dict[tuple[str, str], int] = dict(df_acc)
    print(f"  … indexed {num_records} {record_label} nodes")
    new_vs_global = sorted(neo_ids - set(vectors_store.keys()))

    stored_fp = (store_meta or {}).get("content_columns_fingerprint")
    schema_changed = store_meta is None or stored_fp != current_fp
    if schema_changed:
        print("Schema change detected — recomputing all embeddings")
    else:
        print("Schema unchanged — embedding new records only")
        print(f"New records to embed: {len(new_vs_global)} from {csv_path.name}")

    print(f"  … document frequency over {len(df)} distinct (attribute, value) pairs")

    print(f"Loading sentence-transformers model '{MODEL_NAME}'…")
    cache_hit = model_in_local_cache(MODEL_NAME)
    t_load = time.perf_counter()
    model = SentenceTransformer(MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=False,
                                model_kwargs={"use_safetensors": True})
    dim = int(model.get_sentence_embedding_dimension())
    load_s = time.perf_counter() - t_load
    record_timing(
        "verity.embed.model_load", load_s, parent="verity.embed",
        note=f"model {MODEL_NAME}; model_cache_hit={cache_hit}",
    )
    update_run_info(model_cache_hit=cache_hit, embedding_model_revision=MODEL_REVISION, embedding_weights="model.safetensors")
    encode_s = 0.0
    n_segments = 0

    if schema_changed:
        to_compute: set[str] = set(neo_ids)
        n_reused = 0
    else:
        to_compute = {sk for sk in neo_ids if not _stored_vector_ok(vectors_store, sk, dim)}
        n_reused = num_records - len(to_compute)

    if schema_changed:
        print(f"Recomputing embeddings for {len(to_compute)} records")
    elif to_compute:
        print(f"Embedding {len(to_compute)} record(s) not yet in global store")

    embed_meta = {
        "content_columns_fingerprint": current_fp,
        "model_name": MODEL_NAME,
    }

    n_new = len(to_compute)
    new_since_last_save = 0
    fused_count = 0
    if to_compute:
        print("Computing IDF-weighted record embeddings (paginated)…")
        skip = 0
        page_num = 0
        while True:
            page = fetch_records_page(store, record_label, skip, STORE_FETCH_PAGE_SIZE)
            if not page:
                break
            page_num += 1
            page_compute_keys = [sk for sk in page if sk in to_compute]
            if page_compute_keys:
                uniq_segments = sorted(
                    {
                        segment_text(t, v)
                        for sk in page_compute_keys
                        for t, v in page[sk]
                    }
                )
                segment_embeddings: dict[str, np.ndarray] = {}
                if uniq_segments:
                    print(
                        f"  … page {page_num}: encode {len(uniq_segments)} segment(s) "
                        f"for {len(page_compute_keys)} record(s)"
                    )
                    t_enc = time.perf_counter()
                    segment_embeddings = encode_segments_in_batches(model, uniq_segments)
                    encode_s += time.perf_counter() - t_enc
                    n_segments += len(uniq_segments)

                for sk in sorted(page_compute_keys):
                    attrs = page[sk]
                    if not attrs:
                        combined = np.zeros(dim, dtype=np.float64)
                    else:
                        ordered = sorted(attrs)
                        weights = inverse_frequency_weights(ordered, df, num_records)
                        weighted_sum = np.zeros(dim, dtype=np.float64)
                        w_tot = 0.0
                        for (atype, aval), wt in zip(ordered, weights):
                            seg_key = segment_text(atype, aval)
                            weighted_sum += wt * segment_embeddings[seg_key]
                            w_tot += wt
                        combined = weighted_sum / w_tot if w_tot > 0 else weighted_sum
                    as_list = _embedding_to_plain_list(combined, sk, dim)
                    if as_list is not None:
                        vectors_store[sk] = as_list
                        new_since_last_save += 1
                        fused_count += 1
                        while new_since_last_save >= PICKLE_SAVE_EVERY_NEW_RECORDS:
                            save_global_embeddings_store(
                                GLOBAL_EMBEDDINGS_PATH, vectors_store, embed_meta
                            )
                            new_since_last_save -= PICKLE_SAVE_EVERY_NEW_RECORDS
                    if fused_count % 100 == 0 or fused_count == n_new:
                        print(f"  … fused {fused_count} / {n_new} records")

            if len(page) < STORE_FETCH_PAGE_SIZE:
                break
            skip += STORE_FETCH_PAGE_SIZE

    save_global_embeddings_store(GLOBAL_EMBEDDINGS_PATH, vectors_store, embed_meta)
    record_timing(
        "verity.embed.encode", encode_s, parent="verity.embed",
        items=n_segments, item="segment",
        note=f"sentence-transformers encode only, {n_new} record(s) embedded",
    )

    m1, m1_label = memory_usage_mb()
    if m1 >= 0:
        print(f"Memory after embedding: {m1:.1f} {m1_label}")
    else:
        print(f"Memory after embedding: {m1_label}")

    n_total = len(vectors_store)
    print(f"Global store updated: {n_total} total records")

    file_scoped_path = file_scoped_embeddings_path(csv_path)
    file_scoped_path.parent.mkdir(parents=True, exist_ok=True)
    file_scoped: dict[str, list[float]] = {
        k: vectors_store[k] for k in file_ids if k in vectors_store
    }
    with file_scoped_path.open("wb") as f:
        pickle.dump(file_scoped, f, protocol=pickle.HIGHEST_PROTOCOL)

    neo_write_ids = sorted(to_compute)
    rows_out: list[dict[str, Any]] = []
    missing_vec: list[str] = []
    for cid in neo_write_ids:
        emb = vectors_store.get(cid)
        if emb is None:
            missing_vec.append(cid)
            continue
        rows_out.append({"record_id": cid, "embedding": emb})
    if missing_vec:
        sample = ", ".join(missing_vec[:5])
        print(
            f"Warning: {len(missing_vec)} store id(s) have no vector in the global store "
            f"after merge (e.g. {sample!r})."
        )

    store.close()
    if schema_changed:
        print(f"Summary: full recompute of {n_new} record(s); {len(rows_out)} vectors saved.")
    else:
        print(
            f"Summary: reused {n_reused} from disk, computed {n_new}; "
            f"{len(rows_out)} new vectors saved."
        )
    print("Done.")
    return set(to_compute)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Embed stored records (label from CSV stem, same as build_graph; "
            "validates source CSV exists)."
        )
    )
    ap.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "Source customer CSV path (absolute, or relative to project root). "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    args = ap.parse_args()
    main(
        input_path=args.csv_path if args.csv_path is not None else DEFAULT_INPUT_PATH
    )

"""
Load a customer CSV (default ``data/customers_raw.csv``) into the local record store:
record nodes (label derived from the CSV stem), AttributeValue, SourceSystem nodes
and HAS_ATTRIBUTE, SOURCED_FROM, CO_OCCURS_WITH relationships.

Column roles come from ``data/column_classification.json`` when present; otherwise
``column_classification.json`` next to the CSV or inferred classification applies.
``metadata_columns`` are never turned into AttributeValue nodes or HAS_ATTRIBUTE
edges (record id, lineage, and similar fields still populate record nodes / edges).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from column_classification import (
    classification_or_infer,
    column_to_attr_type,
    neo_property_key,
)
from config import data_dir

BATCH_SIZE = 2000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = None
DEFAULT_CLASSIFICATION_JSON = data_dir() / "column_classification.json"

# Loaded at import; merged into inferred classification when the default JSON is absent.
_DISK_COLUMN_CLASSIFICATION: dict[str, Any] = {}
if DEFAULT_CLASSIFICATION_JSON.is_file():
    with DEFAULT_CLASSIFICATION_JSON.open(encoding="utf-8") as _ccf:
        _DISK_COLUMN_CLASSIFICATION = json.load(_ccf)


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


def _col_ci(name: str) -> str:
    return str(name).strip().lower()


def load_classification_at_start(csv_path: Path) -> dict[str, Any]:
    """Prefer ``column_classification.json`` in the data folder; else infer and merge disk metadata_columns."""
    if DEFAULT_CLASSIFICATION_JSON.is_file():
        with DEFAULT_CLASSIFICATION_JSON.open(encoding="utf-8") as f:
            return json.load(f)
    base = classification_or_infer(csv_path)
    disk_meta = _DISK_COLUMN_CLASSIFICATION.get("metadata_columns") or []
    if disk_meta:
        merged = {str(x) for x in (base.get("metadata_columns") or [])}
        merged.update(str(x) for x in disk_meta)
        base["metadata_columns"] = sorted(merged, key=str.lower)
    return base


def normalize_attribute_value(raw: str) -> str:
    """
    Normalize before AttributeValue merge: value-pattern driven, no column names.

    - Strip leading tab and whitespace.
    - Try removing a leading alphabetic word (e.g. product code labels), then
      formatting separators (hyphens, dots, slashes, spaces). If the remainder
      is all digits with length >= 6, that digit string is the canonical form.
    - Else try removing formatting from the whole value; same digit rule.
    - Else lowercase + collapse internal spaces.
    """
    s = str(raw).lstrip("\t").strip()
    if not s:
        return ""

    # Leading word + formatting strip (before lowercasing — preserves alpha prefixes)
    stripped_prefix = re.sub(r"^[A-Za-z]+\s*", "", s)
    digits_after_prefix = re.sub(r"[\s\-\.\/]", "", stripped_prefix)
    if re.fullmatch(r"\d{6,}", digits_after_prefix):
        return digits_after_prefix

    # Formatting-only strip on full value
    digits_full = re.sub(r"[\s\-\.\/\t]", "", s)
    if re.fullmatch(r"\d{6,}", digits_full):
        return digits_full

    sl = s.lower()
    return re.sub(r" +", " ", sl)


def row_get_ci(row: dict[str, Any], col: str) -> str:
    target = str(col).strip().lower()
    for k, v in row.items():
        if str(k).strip().lower() == target:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return str(v).strip()
    return ""


def neo_scalar(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+\.\d+", s):
        try:
            return float(s)
        except ValueError:
            pass
    return s


def row_to_payload(
    row: dict[str, Any],
    classification: dict[str, Any],
    *,
    source_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``row`` should omit metadata_columns (attrs and customer_props). Id / lineage fields
    are read from ``source_row`` when provided, else from ``row``."""
    rid_col = classification["record_id_column"]
    id_row = source_row if source_row is not None else row
    record_id = row_get_ci(id_row, rid_col)
    if not record_id:
        raise ValueError(f"Empty record id for column {rid_col!r}")

    ss = row_get_ci(id_row, "source_system")
    te_raw = row_get_ci(id_row, "true_entity_id")
    true_entity_id: int | str | None
    if te_raw == "":
        true_entity_id = None
    else:
        try:
            true_entity_id = int(te_raw)
        except ValueError:
            true_entity_id = te_raw

    meta_skip = {_col_ci(c) for c in (classification.get("metadata_columns") or [])}
    content_cols = list(classification.get("content_columns") or [])
    attrs: list[dict[str, str]] = []
    for col in content_cols:
        if _col_ci(col) in meta_skip:
            continue
        raw = row_get_ci(row, col)
        val = normalize_attribute_value(raw)
        if val == "":
            continue
        attrs.append({"type": column_to_attr_type(col), "value": val})

    customer_props: dict[str, Any] = {}
    for k, raw in row.items():
        nk = neo_property_key(str(k))
        if not nk or nk == "embedding":
            continue
        if nk == "id":
            continue
        sc = neo_scalar(raw)
        if sc is not None:
            customer_props[nk] = sc

    return {
        "id": record_id,
        "source_system": ss,
        "true_entity_id": true_entity_id,
        "attrs": attrs,
        "customer_props": customer_props,
    }


def resolve_input_csv(input_path: str | Path) -> Path:
    """Resolve CSV path; relative paths are taken from ``PROJECT_ROOT``."""
    p = Path(input_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def main(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    record_label_from_csv_path: str | Path | None = None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    path = resolve_input_csv(input_path)
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")

    label_path = (
        resolve_input_csv(record_label_from_csv_path)
        if record_label_from_csv_path is not None
        else path
    )
    record_label = record_label_from_csv(label_path)

    classification = load_classification_at_start(path)
    metadata_columns = list(classification.get("metadata_columns") or [])
    metadata_columns_ci = {_col_ci(c) for c in metadata_columns}
    skipped_display = sorted(metadata_columns, key=str.lower)
    print(f"Skipping metadata columns: {skipped_display}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_rows = [dict(r) for r in reader]

    all_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        try:
            filtered_row = {
                k: v
                for k, v in r.items()
                if _col_ci(k) not in metadata_columns_ci
            }
            payload = row_to_payload(filtered_row, classification, source_row=r)
            # Provenance: the input file name (not the path). The store keeps the first
            # file a record id was loaded from.
            payload["input_file"] = path.name
            all_rows.append(payload)
        except ValueError as e:
            raise SystemExit(f"Invalid row (missing id?): {e}") from e

    total = len(all_rows)
    if total == 0:
        raise SystemExit("No rows in CSV.")

    from record_store import open_store

    store = open_store()
    try:
        print(f"Loading {total} {record_label} rows into the record store ({store.path.name})…")
        store.load_records(all_rows, record_label)
    finally:
        store.close()

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load customer CSV into the record store.")
    ap.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "Input CSV path (absolute, or relative to project root). "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    args = ap.parse_args()
    main(
        input_path=args.csv_path if args.csv_path is not None else DEFAULT_INPUT_PATH
    )

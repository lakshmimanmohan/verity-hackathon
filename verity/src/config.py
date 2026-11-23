# src/config.py
from __future__ import annotations
import re
from pathlib import Path

def record_label_from_csv(csv_path: Path) -> str:
    """
    Derive the record label from the CSV filename stem (same rules as build_graph).
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


def record_id_pref_from_classification(classification: dict) -> str:
    """
    Return the record_id_column already resolved by column_classification.
    Falls back to 'id' if missing.
    """
    return str(classification.get("record_id_column") or "id").strip()
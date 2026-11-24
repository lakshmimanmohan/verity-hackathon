"""
Local record store for the pipeline: a single SQLite file, Python standard library only.

The file is ``verity_store.sqlite`` in the data folder (``config.data_dir()``).

What the pipeline stores per record:
  - a property map (record id, source system, every non-metadata CSV field)
  - a set of (attr_type, value) pairs, the normalized content values
Golden records are also written here, alongside ``golden_records.csv``.
Embeddings live in ``embeddings_global.pkl`` in the same folder.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from config import data_dir

STORE_FILENAME = "verity_store.sqlite"
GOLDEN_LABEL = "GoldenRecord"


def _attr_value_key(raw: Any) -> str:
    """Trim and lowercase a value the way stored attribute values are keyed."""
    return str(raw).strip().lower()


# --------------------------------------------------------------------------- SQLite

class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                label TEXT NOT NULL,
                id    TEXT NOT NULL,
                props TEXT NOT NULL,
                PRIMARY KEY (label, id)
            );
            CREATE TABLE IF NOT EXISTS attrs (
                label     TEXT NOT NULL,
                id        TEXT NOT NULL,
                attr_type TEXT NOT NULL,
                value     TEXT NOT NULL,
                PRIMARY KEY (label, id, attr_type, value)
            );
            CREATE INDEX IF NOT EXISTS attrs_by_value ON attrs (attr_type, value);
            CREATE TABLE IF NOT EXISTS golden (
                golden_id    TEXT PRIMARY KEY,
                confidence   REAL,
                member_count INTEGER,
                props        TEXT NOT NULL,
                member_ids   TEXT NOT NULL,
                record_label TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- build_graph
    def load_records(self, rows: list[dict[str, Any]], record_label: str) -> None:
        """
        Upsert records and their (attr_type, value) pairs. Re-loading an id merges its
        properties and adds new attribute values; nothing is removed.
        """
        cur = self.conn.cursor()
        for row in rows:
            rid = str(row["id"])
            existing = cur.execute(
                "SELECT props FROM records WHERE label = ? AND id = ?", (record_label, rid)
            ).fetchone()
            props: dict[str, Any] = json.loads(existing[0]) if existing else {}
            props["id"] = rid
            props["source_system"] = row.get("source_system")
            if row.get("true_entity_id") is None:
                props.pop("true_entity_id", None)
            else:
                props["true_entity_id"] = row["true_entity_id"]
            if props.get("source_system") is None:
                props.pop("source_system", None)
            props.update(row.get("customer_props") or {})
            if row.get("input_file") and not props.get("_input_file"):
                props["_input_file"] = str(row["input_file"])
            cur.execute(
                "INSERT OR REPLACE INTO records (label, id, props) VALUES (?, ?, ?)",
                (record_label, rid, json.dumps(props, ensure_ascii=False)),
            )
            attr_rows = []
            for a in row.get("attrs") or []:
                v = _attr_value_key(a["value"])
                if v == "":
                    continue
                attr_rows.append((record_label, rid, a["type"], v))
            cur.executemany(
                "INSERT OR IGNORE INTO attrs (label, id, attr_type, value) VALUES (?, ?, ?, ?)",
                attr_rows,
            )
        self.conn.commit()

    # ---- main
    def record_count(self, record_label: str) -> int:
        return int(self.conn.execute(
            "SELECT count(*) FROM records WHERE label = ?", (record_label,)
        ).fetchone()[0])

    def missing_record_ids(self, record_ids: list[str], record_label: str) -> int:
        present = {r[0] for r in self.conn.execute(
            "SELECT id FROM records WHERE label = ?", (record_label,)
        )}
        return sum(1 for rid in record_ids if str(rid) not in present)

    def distinct_attr_types(self, record_label: str) -> set[str]:
        return {
            str(r[0]).lower()
            for r in self.conn.execute(
                "SELECT DISTINCT attr_type FROM attrs WHERE label = ?", (record_label,)
            )
            if r[0] is not None and str(r[0]).strip() != ""
        }

    # ---- embed
    def records_page_rows(self, record_label: str, skip: int, limit: int) -> list[dict[str, Any]]:
        """
        One page of records ordered by id, one row per (attr_type, value).
        A record without attributes still returns one row with null attr_type / value.
        """
        ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM records WHERE label = ? ORDER BY id LIMIT ? OFFSET ?",
            (record_label, int(limit), int(skip)),
        )]
        if not ids:
            return []
        by_id: dict[str, list[tuple[str, str]]] = {i: [] for i in ids}
        for chunk in _chunks(ids, 900):
            q = ",".join("?" * len(chunk))
            for rid, t, v in self.conn.execute(
                f"SELECT id, attr_type, value FROM attrs WHERE label = ? AND id IN ({q})",
                (record_label, *chunk),
            ):
                by_id[rid].append((t, v))
        out: list[dict[str, Any]] = []
        for rid in ids:
            pairs = by_id[rid]
            if not pairs:
                out.append({"record_id": rid, "attr_type": None, "value": None})
            for t, v in pairs:
                out.append({"record_id": rid, "attr_type": t, "value": v})
        return out

    # ---- match
    def record_labels(self) -> list[str]:
        """Record labels present in the store (one label per input CSV stem)."""
        return [
            r[0] for r in self.conn.execute("SELECT DISTINCT label FROM records ORDER BY label")
            if str(r[0]).endswith("Record") and r[0] != GOLDEN_LABEL
        ]

    def record_context_rows(self, record_labels: list[str], sep: str) -> list[dict[str, Any]]:
        """Per record: id, source system and its "type<sep>value" fingerprints."""
        out: list[dict[str, Any]] = []
        for label in record_labels:
            fps: dict[str, set[str]] = {}
            for rid, t, v in self.conn.execute(
                "SELECT id, attr_type, value FROM attrs WHERE label = ?", (label,)
            ):
                fps.setdefault(rid, set()).add(f"{_attr_value_key(t)}{sep}{_attr_value_key(v)}")
            for rid, props_json in self.conn.execute(
                "SELECT id, props FROM records WHERE label = ?", (label,)
            ):
                props = json.loads(props_json)
                out.append({
                    "cid": str(rid),
                    "record_id": props.get("id", rid),
                    "source_system": props.get("source_system"),
                    "input_file": props.get("_input_file"),
                    "fps": sorted(fps.get(rid, set())),
                })
        return out

    def shared_attribute_rows(self, id_a: str, id_b: str, record_label: str) -> list[dict[str, Any]]:
        """(attr_type, value) pairs that both records hold."""
        return [
            {"attr_type": t, "value": v}
            for t, v in self.conn.execute(
                """
                SELECT a.attr_type, a.value FROM attrs a
                JOIN attrs b ON b.label = a.label AND b.attr_type = a.attr_type AND b.value = a.value
                WHERE a.label = ? AND a.id = ? AND b.id = ?
                """,
                (record_label, str(id_a), str(id_b)),
            )
        ]

    # ---- merge
    def all_record_ids(self, record_label: str) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT id FROM records WHERE label = ? ORDER BY id", (record_label,)
        )]

    def record_props_rows(self, ids: list[str], record_label: str) -> list[dict[str, Any]]:
        """Property maps for the given record ids."""
        out: list[dict[str, Any]] = []
        for chunk in _chunks([str(i) for i in ids], 900):
            q = ",".join("?" * len(chunk))
            for rid, props_json in self.conn.execute(
                f"SELECT id, props FROM records WHERE label = ? AND id IN ({q})",
                (record_label, *chunk),
            ):
                out.append({"id": rid, "props": json.loads(props_json)})
        return out

    def prepare_golden(self, clear: bool) -> None:
        if clear:
            self.conn.execute("DELETE FROM golden")
            self.conn.commit()

    def write_golden(self, payload: dict[str, Any], record_label: str) -> None:
        """Upsert one golden record and its member ids."""
        gid = str(payload["golden_id"])
        row = self.conn.execute("SELECT props FROM golden WHERE golden_id = ?", (gid,)).fetchone()
        props = json.loads(row[0]) if row else {}
        props.update(payload.get("dyn_props") or {})
        self.conn.execute(
            "INSERT OR REPLACE INTO golden VALUES (?, ?, ?, ?, ?, ?)",
            (gid, float(payload["confidence"]), int(payload["member_count"]),
             json.dumps(props, ensure_ascii=False),
             json.dumps(list(payload.get("member_ids") or [])), record_label),
        )
        self.conn.commit()


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --------------------------------------------------------------------------- factory

def open_store() -> SQLiteStore:
    """Open the record store in the data folder."""
    return SQLiteStore(data_dir() / STORE_FILENAME)

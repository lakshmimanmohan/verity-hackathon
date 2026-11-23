# src/config.py
from __future__ import annotations
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_ENV = "VERITY_DATA_DIR"
DATA_DIR_MARKER = ".verity_data_dir"
_RETIRED_STORE_ENV = "VERITY_STORE_PATH"
NN_MODE_ENV = "VERITY_NN_MODE"
NN_MODES = ("exact", "approximate", "auto")
DEFAULT_NN_MODE = "auto"
RUN_INFO_NAME = "run_info.json"


def _check_env_sources() -> None:
    """
    The data folder is set from the process environment only. A ``.env`` line
    could change it partway through a run (modules call ``load_dotenv`` at
    different times), so either key in ``.env`` is an error.
    """
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        values = dotenv_values(env_file)
        for key in (DATA_DIR_ENV, NN_MODE_ENV, _RETIRED_STORE_ENV):
            if key in values:
                raise SystemExit(
                    f"{key} is set in {env_file}. Set it in the shell "
                    "environment instead; it is not read from .env."
                )
    if os.environ.get(_RETIRED_STORE_ENV):
        raise SystemExit(
            f"{_RETIRED_STORE_ENV} is no longer supported: the record store lives in "
            f"the data folder. Unset it and set {DATA_DIR_ENV} instead."
        )


def data_dir() -> Path:
    """
    Folder for every file Verity reads or writes besides the input CSV:
    ``VERITY_DATA_DIR`` from the process environment if set (relative paths
    resolve from the current directory), else ``data/`` under the project root.
    """
    _check_env_sources()
    raw = (os.environ.get(DATA_DIR_ENV) or "").strip()
    if not raw:
        return DEFAULT_DATA_DIR
    return Path(raw).expanduser().resolve()


def nn_mode() -> tuple[str, str]:
    """
    Neighbor search mode and where it came from: ``VERITY_NN_MODE`` from the process
    environment (``main.py --nn-mode`` sets it), else ``auto``.

    - ``exact``: sklearn NearestNeighbors (cosine, brute) at any size
    - ``approximate``: Annoy at any size; stops if Annoy is not installed
    - ``auto``: Annoy at 10,000 records or more when installed, else exact
    """
    _check_env_sources()
    raw = (os.environ.get(NN_MODE_ENV) or "").strip().lower()
    if not raw:
        return DEFAULT_NN_MODE, "default"
    if raw not in NN_MODES:
        raise SystemExit(
            f"{NN_MODE_ENV}={raw!r} is not valid. Use one of: {', '.join(NN_MODES)}."
        )
    return raw, NN_MODE_ENV


def annoy_installed() -> bool:
    return importlib.util.find_spec("annoy") is not None


def update_run_info(**fields: Any) -> None:
    """
    Merge ``fields`` into the latest entry of ``run_info.json`` in the data folder
    (one entry per pipeline invocation, appended by ``start_run_info``).
    """
    path = data_dir() / RUN_INFO_NAME
    doc: dict[str, Any] = {"runs": []}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
    runs = doc.setdefault("runs", [])
    if not runs:
        runs.append({})
    runs[-1].update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def start_run_info(**fields: Any) -> None:
    """Append a new entry to ``run_info.json`` for this pipeline invocation."""
    path = data_dir() / RUN_INFO_NAME
    doc: dict[str, Any] = {"runs": []}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
    doc.setdefault("runs", []).append(dict(fields))
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


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
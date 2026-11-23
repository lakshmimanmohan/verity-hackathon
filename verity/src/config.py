# src/config.py
from __future__ import annotations
import os
import re
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_ENV = "VERITY_DATA_DIR"
DATA_DIR_MARKER = ".verity_data_dir"
_RETIRED_STORE_ENV = "VERITY_STORE_PATH"


def _check_env_sources() -> None:
    """
    The data folder is set from the process environment only. A ``.env`` line
    could change it partway through a run (modules call ``load_dotenv`` at
    different times), so either key in ``.env`` is an error.
    """
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        values = dotenv_values(env_file)
        for key in (DATA_DIR_ENV, _RETIRED_STORE_ENV):
            if key in values:
                raise SystemExit(
                    f"{key} is set in {env_file}. Set {DATA_DIR_ENV} in the shell "
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
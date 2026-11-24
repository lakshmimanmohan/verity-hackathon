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


# ---------------------------------------------------------------- timings
# One record per timed stage, shared with the experiment's option scripts:
#   {"stage": "verity.score", "parent": "verity.match", "seconds": 61.3,
#    "items": 216458, "item": "pair", "per_item_ms": {"p50": .., "p95": .., "max": ..},
#    "threads": 1, "note": ""}
# "parent" is null for top-level stages; children of a parent add up to it (a
# "<parent>.other" record holds the untimed remainder).
_TIMINGS: list[dict[str, Any]] = []


def record_timing(
    stage: str,
    seconds: float,
    *,
    parent: str | None = None,
    items: int | None = None,
    item: str | None = None,
    per_item_s: list[float] | None = None,
    note: str = "",
) -> None:
    rec: dict[str, Any] = {
        "stage": stage,
        "parent": parent,
        "seconds": round(float(seconds), 6),
        "items": items,
        "item": item,
        "note": note,
    }
    if per_item_s:
        xs = sorted(per_item_s)

        def pct(q: float) -> float:
            i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
            return xs[i] * 1000.0

        rec["per_item_ms"] = {
            "p50": round(pct(0.50), 4),
            "p95": round(pct(0.95), 4),
            "max": round(xs[-1] * 1000.0, 4),
        }
    _TIMINGS.append(rec)


def recorded_timings() -> list[dict[str, Any]]:
    return list(_TIMINGS)


def _cpu_model() -> str | None:
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or None


def host_info() -> dict[str, Any]:
    """Hardware and runtime facts that timings depend on."""
    import platform

    ram_gb = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    torch_threads = None
    try:
        import torch

        torch_threads = int(torch.get_num_threads())
    except Exception:
        pass
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch_threads": torch_threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


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
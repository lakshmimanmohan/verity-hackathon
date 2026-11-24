"""
Spending ledger for LLM calls: one append-only file that adds up every call of every run and
provider, and refuses any call that would push the total past the lifetime cap.

  python options/llm_ledger.py [--ledger results/llm_ledger.jsonl]      # print the report

How it works
  - Before a call (or a batch) the caller reserves an upper bound of its cost. The reservation
    is refused (LedgerRefused) if lifetime total + this reservation would exceed the lifetime cap
    (LIFETIME_CAP_USD) or if the run's total + this reservation would exceed the run's cap.
  - After the call the caller settles the reservation with the actual cost, or releases it.
  - Cache hits cost nothing and are recorded with usd 0 (counted, never charged).
  - Every operation takes an OS file lock (msvcrt on Windows, fcntl elsewhere) around
    read-total / check / append, so parallel runs and threads cannot jointly exceed a cap. The
    OS releases the lock if a process dies.
  - A crash between reserve and settle leaves the reservation open. It keeps counting against
    the totals (the safe direction) until someone settles or releases it by id.
  - Entries are fsynced. A torn last line from a crash is ignored when reading: it can only be
    an operation that never completed (an unsettled reservation stays open; a lost cache-hit
    note costs nothing).
All money is compared in whole micro-dollars, so floating point cannot sneak past a cap.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LIFETIME_CAP_USD = 45.0
LOCK_TIMEOUT_S = 120.0
DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "results" / "llm_ledger.jsonl"


class LedgerRefused(RuntimeError):
    """A reservation would push a total past its cap."""


def micro(usd: float) -> int:
    return int(round(float(usd) * 1_000_000))


@contextlib.contextmanager
def file_lock(path: Path, timeout_s: float = LOCK_TIMEOUT_S):
    """Exclusive cross-process lock on `path` (created if missing); released on exit or process death."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+b")
    deadline = time.monotonic() + timeout_s
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                fh.seek(0)
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"could not lock {path} within {timeout_s}s")
                    time.sleep(0.005)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"could not lock {path} within {timeout_s}s")
                    time.sleep(0.005)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


@dataclass
class State:
    settled_micro: int = 0
    open_reserved_micro: int = 0
    open: dict = field(default_factory=dict)          # reservation id -> reserve entry
    by_run_micro: dict = field(default_factory=dict)  # run -> settled + open reserved
    cache_hits: int = 0
    skipped_lines: int = 0
    entries: int = 0

    @property
    def total_micro(self) -> int:
        return self.settled_micro + self.open_reserved_micro

    @property
    def total_usd(self) -> float:
        return self.total_micro / 1_000_000


class Ledger:
    def __init__(self, path: Path | str = DEFAULT_LEDGER, lifetime_cap_usd: float = LIFETIME_CAP_USD):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.cap_micro = micro(lifetime_cap_usd)

    # ---- reading (call with the lock held when the result feeds a decision)
    def _read(self) -> State:
        st = State()
        if not self.path.exists():
            return st
        raw = self.path.read_bytes()
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                t = e["type"]
            except (ValueError, KeyError, TypeError):
                st.skipped_lines += 1
                continue
            st.entries += 1
            if t == "reserve":
                st.open[e["id"]] = e
                u = micro(e["usd"])
                st.open_reserved_micro += u
                st.by_run_micro[e["run"]] = st.by_run_micro.get(e["run"], 0) + u
            elif t in ("settle", "release"):
                r = st.open.pop(e["id"], None)
                if r is None:
                    continue
                ru = micro(r["usd"])
                st.open_reserved_micro -= ru
                actual = micro(e["usd"]) if t == "settle" else 0
                st.settled_micro += actual
                st.by_run_micro[r["run"]] = st.by_run_micro.get(r["run"], 0) - ru + actual
            elif t == "cache_hits":
                st.cache_hits += int(e.get("n", 1))
        return st

    def _append(self, entry: dict) -> None:
        entry = {"t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), **entry}
        data = (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as f:
            if f.tell() > 0:
                with open(self.path, "rb") as r:
                    r.seek(-1, os.SEEK_END)
                    if r.read(1) != b"\n":      # torn last line from a crash: start a fresh line
                        f.write(b"\n")
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    # ---- public API
    def state(self) -> State:
        with file_lock(self.lock_path):
            return self._read()

    def reserve(self, usd: float, *, run: str, provider: str, model: str,
                run_cap_usd: float, note: str = "") -> str:
        """Reserve an upper bound; returns the reservation id, or raises LedgerRefused."""
        if usd < 0:
            raise ValueError("reservation must be >= 0")
        u = micro(usd)
        with file_lock(self.lock_path):
            st = self._read()
            if st.total_micro + u > self.cap_micro:
                raise LedgerRefused(
                    f"lifetime cap: total ${st.total_usd:.4f} (settled + open reservations) + "
                    f"${usd:.4f} would exceed ${self.cap_micro / 1e6:.2f}")
            run_total = st.by_run_micro.get(run, 0)
            if run_total + u > micro(run_cap_usd):
                raise LedgerRefused(
                    f"run cap: run {run!r} total ${run_total / 1e6:.4f} + ${usd:.4f} would exceed "
                    f"--max-usd ${run_cap_usd:.2f}")
            rid = uuid.uuid4().hex
            self._append({"type": "reserve", "id": rid, "run": run, "provider": provider,
                          "model": model, "usd": round(usd, 6), "note": note})
            return rid

    def settle(self, res_id: str, actual_usd: float, **details) -> None:
        with file_lock(self.lock_path):
            st = self._read()
            r = st.open.get(res_id)
            if r is None:
                raise KeyError(f"reservation {res_id} is not open")
            over = micro(actual_usd) > micro(r["usd"])
            self._append({"type": "settle", "id": res_id, "run": r["run"], "provider": r["provider"],
                          "model": r["model"], "usd": round(actual_usd, 6),
                          "reserved_usd": r["usd"], "over_reservation": over, **details})

    def release(self, res_id: str, note: str = "") -> None:
        with file_lock(self.lock_path):
            st = self._read()
            r = st.open.get(res_id)
            if r is None:
                raise KeyError(f"reservation {res_id} is not open")
            self._append({"type": "release", "id": res_id, "run": r["run"], "provider": r["provider"],
                          "model": r["model"], "usd": 0.0, "note": note})

    def cache_hits(self, n: int, *, run: str, provider: str, model: str) -> None:
        """Record n cache hits: counted, cost $0, never charged."""
        if n <= 0:
            return
        with file_lock(self.lock_path):
            self._append({"type": "cache_hits", "n": int(n), "run": run, "provider": provider,
                          "model": model, "usd": 0.0})


def report(ledger: Ledger) -> str:
    st = ledger.state()
    lines = [f"Ledger: {ledger.path}",
             f"Lifetime cap ${ledger.cap_micro / 1e6:.2f}; total ${st.total_usd:.4f} "
             f"(settled ${st.settled_micro / 1e6:.4f} + open reservations ${st.open_reserved_micro / 1e6:.4f}); "
             f"remaining ${(ledger.cap_micro - st.total_micro) / 1e6:.4f}",
             f"Entries {st.entries}, cache hits {st.cache_hits}, unreadable lines skipped {st.skipped_lines}"]
    if st.by_run_micro:
        lines.append("By run:")
        lines += [f"  {r:<40} ${u / 1e6:.4f}" for r, u in sorted(st.by_run_micro.items())]
    if st.open:
        lines.append("Open reservations (a crashed run leaves these; settle or release by id):")
        lines += [f"  {i}  run {e['run']}  ${e['usd']:.4f}  {e['provider']} {e['model']}  {e.get('note', '')}"
                  for i, e in st.open.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the LLM spending ledger report.")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    print(report(Ledger(ap.parse_args().ledger)))

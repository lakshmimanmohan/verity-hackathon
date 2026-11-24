"""
Cache of LLM answers keyed by the pair's content.

The key is the SHA-256 of the provider, the model string the request names, a hash of the
request settings, a hash of the system prompt, and the exact user turn (both records' content
fields in presentation order, no record ids). The same content under the same prompt and
settings is never paid for twice, across runs and providers' reruns.

One file per provider, append-only JSON lines {"key", "value"}. Reads tolerate a torn last
line; writes take a file lock and are fsynced, so parallel runs can share a file; a miss
re-reads whatever other processes appended since the last read.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from llm_ledger import file_lock


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pair_key(provider: str, model: str, settings_sha: str, system_sha: str, user_text: str) -> str:
    return sha256_text(json.dumps([provider, model, settings_sha, system_sha, user_text],
                                  ensure_ascii=False, separators=(",", ":")))


class PairCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._data: dict[str, dict] = {}
        self._offset = 0
        self.refresh()

    def refresh(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        if not chunk:
            return
        # Only consume complete lines; a partial last line is read again next time.
        end = chunk.rfind(b"\n") + 1
        for line in chunk[:end].split(b"\n"):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                self._data[e["key"]] = e["value"]
            except (ValueError, KeyError, TypeError):
                continue
        self._offset += end

    def get(self, key: str) -> dict | None:
        if key not in self._data:
            self.refresh()
        return self._data.get(key)

    def put(self, key: str, value: dict) -> None:
        line = (json.dumps({"key": key, "value": value}, ensure_ascii=False, separators=(",", ":"))
                + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.lock_path):
            with open(self.path, "ab") as f:
                if f.tell() > 0:
                    with open(self.path, "rb") as r:
                        r.seek(-1, os.SEEK_END)
                        if r.read(1) != b"\n":
                            f.write(b"\n")
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        self._data[key] = value

    def __len__(self) -> int:
        self.refresh()
        return len(self._data)

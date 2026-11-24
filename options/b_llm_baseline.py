"""
Option B: frontier-LLM baseline. For each test band pair the LLM answers same / different from
the two records' content fields; outside the band Verity's decision_status stands
(eval/score_option.py combines them).

  python options/b_llm_baseline.py results/<run> --provider anthropic|openai <mode>

Modes (exactly one)
  --build-fewshot       pick the few-shot examples from TRAIN band pairs (labels read for
                        train pairs only); writes <run>/options/b_llm/fewshot.json
  --show-prompt         print the full system prompt and two example user turns (train pairs)
  --dry-run             build every prompt for a split (default: test band) and estimate
                        tokens and cost, standard and Batch; no API call, no key needed
  --freeze-prompt       record the SHA-256 of the system prompt + model + settings
  --eval-train N        live calls on N stratified TRAIN band pairs (N = 1000 is the cost and
                        latency pilot); reports quality on that sample, tokens, cost, latency
                        and projects the full test band
  --final               the full test band (needs --freeze-prompt done, an unchanged prompt and
                        --approve-final-run); batch by default, --mode live for live calls

No test band pair is sent to any LLM except by --final. Live calls and batches need an
approved entry in options/prices.json, a key in the repo-root .env (ANTHROPIC_API_KEY or
OPENAI_API_KEY; never printed, logged or written anywhere) and room in the spending ledger
(lifetime cap $45 across all runs; --max-usd per run, default 10). Answers are cached by the
pair's content, so a rerun or a resume never pays twice.

Outputs go to <run>/options/b_llm/<provider>/<label>/: decisions.csv (record_a, record_b,
decision, confidence, model) written only when every pair has an answer, calls.jsonl,
run_meta.json, timings.json (the shared timing format), report.txt.

The OpenAI request, usage and batch code is written from memory of that API and has not run
against it; the Anthropic code follows the SDK documentation and has not run against the live
API either. Both are exercised only through a fake provider in options/test_llm_ledger.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "verity" / "src"))
from llm_cache import PairCache, pair_key, sha256_text  # noqa: E402
from llm_ledger import DEFAULT_LEDGER, LIFETIME_CAP_USD, Ledger, LedgerRefused, micro  # noqa: E402

PRICES_PATH = HERE / "prices.json"
PROMPT_PATH = HERE / "b_system_prompt.txt"
CACHE_DIR = REPO / "results" / "llm_cache"
SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["same", "different"]},
                   "confidence": {"type": "number"}},
    "required": ["decision", "confidence"],
    "additionalProperties": False,
}
TYPE_RANK = {"NAME": 0, "STREET_ADDRESS": 1, "DESCRIPTIVE": 2, "QUASI_IDENTIFIER": 3, "IDENTIFIER": 4, "DATE": 5}
HARD_TAGS = ("acronym", "truncation", "po_box")
CHARS_PER_TOKEN_ESTIMATE = 3.5      # rough: replaced by measured tokens in the pilot
CHARS_PER_TOKEN_UPPER = 2.5         # for reservations: deliberately pessimistic
OUTPUT_TOKENS_ESTIMATE = 25         # {"decision": "different", "confidence": 0.95} is ~18 tokens
BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 200_000_000       # Anthropic's limit is 256 MB per batch; stay clear of it
FEWSHOT_CONFIDENCE = {"same": 0.95, "different": 0.05}


# ------------------------------------------------------------------ data

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class RunData:
    """Record content fields (classification's content columns, ordered by semantic type) and the split."""

    def __init__(self, run: Path):
        csv.field_size_limit(10**8)
        self.run = run
        data = run / "verity_data"
        cls = json.loads((data / "column_classification.json").read_text(encoding="utf-8"))
        sem = {k.lower(): str(v).upper() for k, v in json.loads(
            (data / "scored_pairs_columns.json").read_text(encoding="utf-8"))["field_semantic_types"].items()}
        raw = read_csv(run / "orgs_raw.csv")
        keymap = {k.lower(): k for k in raw[0]}
        rid = keymap[str(cls["record_id_column"]).lower()]
        cols = [keymap[c.lower()] for c in cls["content_columns"] if c.lower() in keymap]
        cols.sort(key=lambda c: (TYPE_RANK.get(sem.get(c.lower(), ""), 9), c.lower()))
        self.cols = cols
        self.records = {r[rid]: {c: (r.get(c) or "").strip() for c in cols} for r in raw}
        self.side = {r["record_id"]: r["side"] for r in read_csv(run / "split_records.csv")}

    def band(self, side_name: str) -> list[tuple[str, str, float, str]]:
        """(record_a, record_b, score, match_type) of band pairs (decision_status review) wholly on one side."""
        out = []
        with (self.run / "verity_data" / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["decision_status"] != "review":
                    continue
                a, b = r["record_a"], r["record_b"]
                if self.side.get(a) == side_name and self.side.get(b) == side_name:
                    out.append((a, b, float(r["score"]), r["match_type"]))
        return out


class TrainTruth:
    """Ground truth for TRAIN pairs only; asking about anything else raises."""

    def __init__(self, run: Path, side: dict[str, str]):
        self._ent, self._tags, self._decoy = {}, {}, {}
        for r in read_csv(run / "orgs_raw_truth.csv"):
            if side.get(r["record_id"]) == "train":
                self._ent[r["record_id"]] = r["true_entity_id"]
                self._tags[r["record_id"]] = set(r["variants"].split("+"))
                self._decoy[r["record_id"]] = r["decoy_group"]

    def _check(self, *ids: str) -> None:
        for i in ids:
            if i not in self._ent:
                raise RuntimeError(f"truth requested for a record outside the train group: {i}")

    def same(self, a: str, b: str) -> bool:
        self._check(a, b)
        return self._ent[a] == self._ent[b]

    def tags(self, a: str, b: str) -> set[str]:
        self._check(a, b)
        return self._tags[a] | self._tags[b]

    def is_decoy_pair(self, a: str, b: str) -> bool:
        self._check(a, b)
        return bool(self._decoy[a]) and self._decoy[a] == self._decoy[b] and self._ent[a] != self._ent[b]


# ------------------------------------------------------------------ prompt

def render_record(rec: dict[str, str], cols: list[str]) -> str:
    return "\n".join(f"  {c}: {rec[c] if rec[c] else '(missing)'}" for c in cols)


def user_turn(rec_a: dict[str, str], rec_b: dict[str, str], cols: list[str]) -> str:
    return (f"Record A:\n{render_record(rec_a, cols)}\n\nRecord B:\n{render_record(rec_b, cols)}\n\n"
            "Are these two records the same organization? Reply with the JSON object only.")


def build_fewshot(rd: RunData, truth: TrainTruth, train_band) -> list[dict]:
    """8 examples from train band pairs, deterministic (seed 0): 4 same (acronym, truncation,
    po_box, plain) and 4 different (2 decoys, the highest-scoring other pair, 1 random)."""
    rng = random.Random(0)
    pairs = sorted(train_band)
    true = [p for p in pairs if truth.same(p[0], p[1])]
    false = [p for p in pairs if not truth.same(p[0], p[1])]
    chosen: list[tuple[tuple, str, str]] = []
    used: set[tuple[str, str]] = set()

    def take(pool, stratum, label):
        pool = [p for p in pool if (p[0], p[1]) not in used]
        if not pool:
            return
        p = rng.choice(pool)
        used.add((p[0], p[1]))
        chosen.append((p, stratum, label))

    for tag in HARD_TAGS:
        take([p for p in true if tag in truth.tags(p[0], p[1])
              and len(truth.tags(p[0], p[1]) & set(HARD_TAGS)) == 1], f"same, {tag}", "same")
    take([p for p in true if not truth.tags(p[0], p[1]) & set(HARD_TAGS)] or true, "same, plain variation", "same")
    decoys = [p for p in false if truth.is_decoy_pair(p[0], p[1])]
    take(decoys, "different, same name (decoy)", "different")
    take(decoys, "different, same name (decoy)", "different")
    non_decoy = [p for p in false if not truth.is_decoy_pair(p[0], p[1])]
    hardest = sorted(non_decoy, key=lambda p: (-p[2], p[0], p[1]))[:1]
    if hardest:
        used.add((hardest[0][0], hardest[0][1]))
        chosen.append((hardest[0], "different, highest score", "different"))
    take(non_decoy, "different, random", "different")
    rng.shuffle(chosen)
    return [{"stratum": s, "label": lab, "ids": [p[0], p[1]],
             "a": rd.records[p[0]], "b": rd.records[p[1]]} for p, s, lab in chosen]


def render_examples(fewshot: list[dict], cols: list[str]) -> str:
    blocks = []
    for i, ex in enumerate(fewshot, start=1):
        ans = json.dumps({"decision": ex["label"], "confidence": FEWSHOT_CONFIDENCE[ex["label"]]})
        blocks.append(f"Example {i}\nRecord A:\n{render_record(ex['a'], cols)}\n\nRecord B:\n"
                      f"{render_record(ex['b'], cols)}\n\nAnswer: {ans}")
    return "\n\n".join(blocks)


def build_system_prompt(fewshot: list[dict], cols: list[str]) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{EXAMPLES}", render_examples(fewshot, cols)).rstrip() + "\n"


def parse_answer(text: str) -> tuple[str, float]:
    d = json.loads(text)
    dec = d["decision"]
    conf = float(d["confidence"])
    if dec not in ("same", "different"):
        raise ValueError(f"decision {dec!r}")
    return dec, min(1.0, max(0.0, conf))


# ------------------------------------------------------------------ providers

@dataclass
class CallResult:
    decision: str | None = None
    confidence: float | None = None
    refused: bool = False
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    model: str | None = None
    request_id: str | None = None
    latency_s: float | None = None
    error: str | None = None

    @property
    def answered(self) -> bool:
        """Has an answer worth caching: a parsed decision or a refusal."""
        return self.error is None and (self.decision is not None or self.refused)

    def to_cache(self) -> dict:
        return {"decision": self.decision, "confidence": self.confidence, "refused": self.refused,
                "stop_reason": self.stop_reason, "usage": self.usage, "model": self.model,
                "request_id": self.request_id}

    @staticmethod
    def from_cache(d: dict) -> "CallResult":
        return CallResult(decision=d["decision"], confidence=d["confidence"], refused=d["refused"],
                          stop_reason=d["stop_reason"], usage=d["usage"], model=d["model"],
                          request_id=d["request_id"])


class Provider:
    name = "base"
    cache_write_fee = True      # does the provider charge extra for writing a prompt cache entry?

    def __init__(self, model: str, cfg: dict):
        self.model, self.cfg = model, cfg
        self.settings = cfg["request_settings"]
        self.transport = cfg.get("transport_settings", {})   # settings that cannot change an answer

    # settings that can change an answer (cache TTL cannot)
    def answer_settings(self) -> dict:
        return {k: v for k, v in self.settings.items() if not k.startswith("cache_ttl") and k != "temperature"}

    def settings_label(self) -> str:
        """Compact, human-readable answer settings, written to every result file."""
        return ";".join(f"{k}={json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v}"
                        for k, v in sorted(self.answer_settings().items()))

    def batch_limits(self) -> tuple[int, int]:
        lim = self.cfg.get("batch_limits") or {}
        return int(lim.get("max_requests", BATCH_MAX_REQUESTS)), int(lim.get("max_bytes", BATCH_MAX_BYTES))

    def settings_sha(self) -> str:
        return sha256_text(json.dumps(self.answer_settings(), sort_keys=True))

    def require_approved(self) -> None:
        if not self.cfg.get("approved"):
            raise SystemExit(f"{self.model}: options/prices.json has approved: false. Check the prices and "
                             "settings against the provider's documentation, then set approved to true.")
        for k in ("input", "output"):
            if self.cfg.get(k) is None:
                raise SystemExit(f"{self.model}: options/prices.json has no {k} price.")

    def price(self, k: str) -> float:
        v = self.cfg.get(k)
        if v is None:
            raise SystemExit(f"{self.model}: options/prices.json has no {k} price.")
        return float(v)

    def have_prices(self) -> bool:
        return all(self.cfg.get(k) is not None for k in ("input", "output"))

    # to be provided by subclasses
    def request(self, system: str, user: str, mode: str) -> dict: raise NotImplementedError
    def call(self, params: dict) -> CallResult: raise NotImplementedError
    def cost_of(self, usage: dict, mode: str) -> float: raise NotImplementedError
    def submit_batch(self, items: list[tuple[str, dict]]) -> str: raise NotImplementedError
    def batch_done(self, batch_id: str) -> bool: raise NotImplementedError
    def batch_results(self, batch_id: str) -> dict[str, CallResult]: raise NotImplementedError

    def usd(self, *, uncached_in: float, cache_read_in: float, cache_write_in: float, out: float,
            batch: bool, ttl: str = "5m") -> float | None:
        """Cost of a token mix, or None when prices are missing."""
        if not self.have_prices():
            return None
        c = self.cfg
        read = c.get("cache_read") if c.get("cache_read") is not None else c["input"]
        write = c.get("cache_write_" + ttl) if c.get("cache_write_" + ttl) is not None else c["input"]
        if not batch:
            return (uncached_in * c["input"] + cache_read_in * read + cache_write_in * write + out * c["output"]) / 1e6
        mult = c.get("batch_multiplier")
        if mult is None:
            return None
        # Batch: everything at the batch multiplier, except cached input when the provider lists its own price.
        read_b = c["batch_cache_read"] if c.get("batch_cache_read") is not None else read * mult
        return ((uncached_in * c["input"] + cache_write_in * write + out * c["output"]) * mult
                + cache_read_in * read_b) / 1e6

    def upper_bound_usd(self, system: str, user: str, mode: str) -> float:
        """Pessimistic cost of one request: the system prompt billed as a cache write, tokens at
        2.5 characters each, and the full max output."""
        ttl = self.ttl(mode)
        sys_t = math.ceil(len(system) / CHARS_PER_TOKEN_UPPER)
        usr_t = math.ceil(len(user) / CHARS_PER_TOKEN_UPPER)
        v = self.usd(uncached_in=usr_t, cache_read_in=0, cache_write_in=sys_t,
                     out=self.max_output_tokens(), batch=(mode == "batch"), ttl=ttl)
        if v is None:
            raise SystemExit(f"{self.model}: prices missing in options/prices.json; cannot reserve.")
        return v

    def ttl(self, mode: str) -> str:
        return self.settings.get("cache_ttl_batch" if mode == "batch" else "cache_ttl_live", "5m")

    def max_output_tokens(self) -> int:
        return int(self.settings.get("max_tokens", self.settings.get("max_output_tokens", 64)))

    def cacheable_prefix_tokens(self, sys_tokens: int) -> int:
        """How many tokens of the system prompt can be served from the prompt cache (estimate)."""
        return sys_tokens if sys_tokens >= 512 else 0       # Anthropic: 512-token minimum on this model


def _scrub(msg: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            msg = msg.replace(s, "[redacted]")
    return msg[:300]


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, cfg: dict, client=None):
        super().__init__(model, cfg)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise SystemExit("ANTHROPIC_API_KEY is not set (put it in the repo-root .env).")
            self._client = anthropic.Anthropic(api_key=key, max_retries=5)
        return self._client

    def request(self, system: str, user: str, mode: str) -> dict:
        s = self.settings
        cc: dict = {"type": "ephemeral"}
        if self.ttl(mode) == "1h":
            cc["ttl"] = "1h"
        return {"model": self.model, "max_tokens": int(s["max_tokens"]),
                "system": [{"type": "text", "text": system, "cache_control": cc}],
                "messages": [{"role": "user", "content": user}],
                "thinking": s["thinking"],
                "output_config": {"effort": s["effort"], "format": {"type": "json_schema", "schema": SCHEMA}}}

    @staticmethod
    def _usage(u) -> dict:
        cc = getattr(u, "cache_creation", None)
        return {"input_tokens": getattr(u, "input_tokens", 0) or 0, "output_tokens": getattr(u, "output_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                "cache_write_5m": getattr(cc, "ephemeral_5m_input_tokens", None) if cc else None,
                "cache_write_1h": getattr(cc, "ephemeral_1h_input_tokens", None) if cc else None}

    def _to_result(self, msg, latency: float | None) -> CallResult:
        res = CallResult(stop_reason=msg.stop_reason, usage=self._usage(msg.usage), model=msg.model,
                         request_id=getattr(msg, "_request_id", None), latency_s=latency)
        if msg.stop_reason == "refusal":
            res.refused = True
            return res
        text = next((b.text for b in msg.content if getattr(b, "type", "") == "text"), "")
        try:
            res.decision, res.confidence = parse_answer(text)
        except (ValueError, KeyError, TypeError) as e:
            res.error = f"unparseable answer ({type(e).__name__}); stop_reason {msg.stop_reason}"
        return res

    def call(self, params: dict) -> CallResult:
        t0 = time.perf_counter()
        msg = self.client.messages.create(**params)
        return self._to_result(msg, time.perf_counter() - t0)

    def cost_of(self, usage: dict, mode: str) -> float:
        w5, w1 = usage.get("cache_write_5m"), usage.get("cache_write_1h")
        cc = usage.get("cache_creation_input_tokens", 0) or 0
        if w5 is None and w1 is None:
            w5, w1 = (cc, 0) if self.ttl(mode) == "5m" else (0, cc)
        c = self.cfg
        v = ((usage.get("input_tokens", 0) or 0) * self.price("input")
             + (usage.get("cache_read_input_tokens", 0) or 0) * self.price("cache_read")
             + (w5 or 0) * self.price("cache_write_5m") + (w1 or 0) * self.price("cache_write_1h")
             + (usage.get("output_tokens", 0) or 0) * self.price("output")) / 1e6
        return v * (self.price("batch_multiplier") if mode == "batch" else 1.0)

    def submit_batch(self, items: list[tuple[str, dict]]) -> str:
        b = self.client.messages.batches.create(requests=[{"custom_id": cid, "params": p} for cid, p in items])
        return b.id

    def batch_done(self, batch_id: str) -> bool:
        return self.client.messages.batches.retrieve(batch_id).processing_status == "ended"

    def batch_results(self, batch_id: str) -> dict[str, CallResult]:
        out = {}
        for r in self.client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                out[r.custom_id] = self._to_result(r.result.message, None)
            else:
                out[r.custom_id] = CallResult(error=f"batch result {r.result.type}")
        return out


class OpenAIProvider(Provider):
    """OpenAI Responses API with a strict JSON-schema structured output, no tools.

    Written from the documentation read on 2026-11-24 (see options/prices.json notes) and
    exercised only through fake responses in options/test_b_openai.py; it has not run against
    the live API. Temperature is not set (the docs do not say it is accepted at reasoning
    effort none)."""
    name = "openai"
    cache_write_fee = False      # no write fee listed for gpt-5.4

    def cacheable_prefix_tokens(self, sys_tokens: int) -> int:
        # automatic caching; for models before GPT-5.6 cached_tokens is reported rounded down to a
        # multiple of 128, and the minimum length is not documented (1,024 is stated for GPT-5.6+)
        return (sys_tokens // 128) * 128 if sys_tokens >= 1024 else 0

    def __init__(self, model: str, cfg: dict, client=None):
        super().__init__(model, cfg)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import openai
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit("OPENAI_API_KEY is not set (put it in the repo-root .env).")
            self._client = openai.OpenAI(api_key=key, max_retries=5)
        return self._client

    def require_approved(self) -> None:
        super().require_approved()
        for k in ("cache_read", "batch_multiplier"):
            if self.cfg.get(k) is None:
                raise SystemExit(f"{self.model}: options/prices.json has no {k} price.")
        if not self.settings.get("reasoning_effort"):
            raise SystemExit(f"{self.model}: request_settings.reasoning_effort is not set in options/prices.json.")

    def _cache_key(self, system: str, user: str) -> str | None:
        k = int(self.transport.get("prompt_cache_key_buckets", 0) or 0)
        if k <= 0:
            return None
        return f"{sha256_text(system)[:16]}-{int(sha256_text(user)[:8], 16) % k}"

    def request(self, system: str, user: str, mode: str) -> dict:
        s = self.settings
        body = {"model": self.model,
                # reusable instructions go in a developer message at the start of input so they are
                # part of the cacheable prefix; the pair comes last
                "input": [{"role": "developer", "content": system}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "decision", "strict": True, "schema": SCHEMA}},
                "reasoning": {"effort": s["reasoning_effort"]},
                "max_output_tokens": int(s["max_output_tokens"]),
                "store": bool(self.transport.get("store", False))}
        key = self._cache_key(system, user)
        if key is not None:
            body["prompt_cache_key"] = key
        return body

    @staticmethod
    def from_body(body: dict, latency: float | None = None, request_id: str | None = None) -> CallResult:
        """Turn a Responses API response object (as a dict) into a CallResult."""
        u = body.get("usage") or {}
        cached = ((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0
        reasoning = ((u.get("output_tokens_details") or {}).get("reasoning_tokens")) or 0
        usage = {"input_tokens": u.get("input_tokens", 0) or 0, "cached_tokens": cached,
                 "output_tokens": u.get("output_tokens", 0) or 0, "reasoning_tokens": reasoning}
        status = body.get("status")
        reason = (body.get("incomplete_details") or {}).get("reason")
        res = CallResult(stop_reason=f"{status}:{reason}" if reason else status, usage=usage,
                         model=body.get("model"), request_id=request_id or body.get("id"), latency_s=latency)
        text, refusal = "", False
        for item in body.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    refusal = True
                elif part.get("type") == "output_text":
                    text += part.get("text", "")
        if refusal:
            res.refused = True
            return res
        if status == "failed":
            res.error = f"response failed: {(body.get('error') or {}).get('code')}"
            return res
        if status == "incomplete":
            res.error = f"incomplete response ({reason})"
            return res
        try:
            res.decision, res.confidence = parse_answer(text)
        except (ValueError, KeyError, TypeError) as e:
            res.error = f"unparseable answer ({type(e).__name__}); status {status}"
        return res

    def call(self, params: dict) -> CallResult:
        t0 = time.perf_counter()
        resp = self.client.responses.create(**params)
        return self.from_body(resp.model_dump(), time.perf_counter() - t0, getattr(resp, "_request_id", None))

    def cost_of(self, usage: dict, mode: str) -> float:
        cached = usage.get("cached_tokens", 0) or 0
        fresh = (usage.get("input_tokens", 0) or 0) - cached
        out = usage.get("output_tokens", 0) or 0      # reasoning tokens are billed as output
        if mode == "batch":
            mult = self.price("batch_multiplier")
            read = self.cfg.get("batch_cache_read")
            read = self.price("cache_read") * mult if read is None else read
            return ((fresh * self.price("input") + out * self.price("output")) * mult + cached * read) / 1e6
        return (fresh * self.price("input") + cached * self.price("cache_read") + out * self.price("output")) / 1e6

    def submit_batch(self, items: list[tuple[str, dict]]) -> str:
        import io
        lines = "\n".join(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": p},
                                     ensure_ascii=False) for cid, p in items)
        f = self.client.files.create(file=("batch.jsonl", io.BytesIO(lines.encode("utf-8"))), purpose="batch")
        return self.client.batches.create(input_file_id=f.id, endpoint="/v1/responses",
                                          completion_window="24h").id

    def batch_done(self, batch_id: str) -> bool:
        st = self.client.batches.retrieve(batch_id).status
        if st == "failed":
            raise RuntimeError(f"batch {batch_id} failed validation")
        return st in ("completed", "expired", "cancelled")   # expired / cancelled may hold partial results

    def batch_results(self, batch_id: str) -> dict[str, CallResult]:
        b = self.client.batches.retrieve(batch_id)
        out: dict[str, CallResult] = {}
        if not getattr(b, "output_file_id", None):
            return out
        for line in self.client.files.content(b.output_file_id).text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            resp = r.get("response")
            if r.get("error") or not resp or resp.get("status_code") != 200:
                out[r["custom_id"]] = CallResult(error="batch request failed")
                continue
            out[r["custom_id"]] = self.from_body(resp["body"], None, resp.get("request_id"))
        return out


def load_prices() -> dict:
    return json.loads(PRICES_PATH.read_text(encoding="utf-8"))["models"]


def make_provider(name: str, model: str | None, live: bool) -> Provider:
    models = load_prices()
    entries = [(m, c) for m, c in models.items() if c["provider"] == name and (model is None or m == model)]
    if not entries:
        raise SystemExit(f"no {name} model {model or ''} in options/prices.json")
    m, cfg = entries[0]
    cls = AnthropicProvider if name == "anthropic" else OpenAIProvider
    p = cls(m, cfg)
    if live:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
        p.require_approved()
    return p


# ------------------------------------------------------------------ pipeline

@dataclass
class Item:
    key: str
    a: str
    b: str
    user: str


@dataclass
class Outcome:
    results: dict = field(default_factory=dict)      # key -> CallResult
    pending: list = field(default_factory=list)      # keys without an answer
    cache_hits: int = 0
    spent_usd: float = 0.0
    stopped: str | None = None                       # why the run stopped early
    latencies: list = field(default_factory=list)
    batch_walls: list = field(default_factory=list)
    n_calls: int = 0


def est_request_bytes(system: str, user: str) -> int:
    return len(system.encode("utf-8")) + len(user.encode("utf-8")) + 600


def run_items(provider: Provider, items: list[Item], system: str, *, ledger: Ledger, cache: PairCache,
              run_id: str, run_cap_usd: float, mode: str, concurrency: int, out_dir: Path,
              batch_max_requests: int | None = None, batch_max_bytes: int | None = None,
              poll_s: float = 30.0, sleep=time.sleep, log=print) -> Outcome:
    """Answer every distinct item: from the cache when possible, otherwise by live calls or batches,
    each reserved in the ledger first. Stops early (outcome.stopped) if the ledger refuses."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lim_req, lim_bytes = provider.batch_limits()
    batch_max_requests = batch_max_requests or lim_req
    batch_max_bytes = batch_max_bytes or lim_bytes
    oc = Outcome()
    calls_log = out_dir / "calls.jsonl"
    log_lock = threading.Lock()

    def log_call(item: Item, res: CallResult, usd: float, m: str) -> None:
        rec = {"key": item.key[:16], "a": item.a, "b": item.b, "mode": m, "usd": round(usd, 6),
               "latency_s": res.latency_s, "usage": res.usage, "stop_reason": res.stop_reason,
               "model": res.model, "request_id": res.request_id, "refused": res.refused, "error": res.error}
        with log_lock, open(calls_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    seen: dict[str, Item] = {}
    for it in items:
        seen.setdefault(it.key, it)
    todo: list[Item] = []
    for key, it in seen.items():
        hit = cache.get(key)
        if hit is not None:
            oc.results[key] = CallResult.from_cache(hit)
            oc.cache_hits += 1
        else:
            todo.append(it)
    ledger.cache_hits(oc.cache_hits, run=run_id, provider=provider.name, model=provider.model)

    if mode == "live":
        stop = threading.Event()
        lock = threading.Lock()

        def one(it: Item) -> None:
            if stop.is_set():
                return
            ub = provider.upper_bound_usd(system, it.user, "live")
            try:
                rid = ledger.reserve(ub, run=run_id, provider=provider.name, model=provider.model,
                                     run_cap_usd=run_cap_usd, note="live")
            except LedgerRefused as e:
                with lock:
                    oc.stopped = oc.stopped or str(e)
                stop.set()
                return
            try:
                res = provider.call(provider.request(system, it.user, "live"))
            except (Exception, SystemExit) as e:  # noqa: BLE001 - provider errors are logged without payloads
                ledger.release(rid, note=f"error {type(e).__name__}")
                if isinstance(e, SystemExit):         # e.g. a missing API key: stop the whole run
                    raise
                res = CallResult(error=_scrub(f"{type(e).__name__}: {e}", [os.environ.get("ANTHROPIC_API_KEY", ""),
                                                                              os.environ.get("OPENAI_API_KEY", "")]))
                with lock:
                    oc.n_calls += 1
                log_call(it, res, 0.0, "live")
                return
            usd = provider.cost_of(res.usage, "live")
            ledger.settle(rid, usd, input_tokens=res.usage.get("input_tokens", res.usage.get("prompt_tokens")),
                          output_tokens=res.usage.get("output_tokens", res.usage.get("completion_tokens")))
            if res.answered:
                cache.put(it.key, res.to_cache())
            with lock:
                oc.n_calls += 1
                oc.spent_usd += usd
                if res.latency_s is not None:
                    oc.latencies.append(res.latency_s)
                if res.answered:
                    oc.results[it.key] = res
            log_call(it, res, usd, "live")

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            list(ex.map(one, todo))
    else:
        i = 0
        ub_cache: dict[str, float] = {}
        while i < len(todo):
            st = ledger.state()
            headroom = min(ledger.cap_micro - st.total_micro,
                           micro(run_cap_usd) - st.by_run_micro.get(run_id, 0)) / 1e6
            chunk: list[Item] = []
            tot, size = 0.0, 0
            for it in todo[i:]:
                ub = ub_cache.setdefault(it.key, provider.upper_bound_usd(system, it.user, "batch"))
                nb = est_request_bytes(system, it.user)
                if len(chunk) >= batch_max_requests or size + nb > batch_max_bytes or tot + ub > headroom:
                    break
                chunk.append(it)
                tot += ub
                size += nb
            if not chunk:
                oc.stopped = f"no room in the caps for the next batch request (headroom ${headroom:.4f})"
                break
            try:
                rid = ledger.reserve(tot, run=run_id, provider=provider.name, model=provider.model,
                                     run_cap_usd=run_cap_usd, note=f"batch of {len(chunk)}")
            except LedgerRefused as e:
                oc.stopped = str(e)
                break
            t0 = time.perf_counter()
            cid_to_item = {it.key[:32]: it for it in chunk}
            try:
                bid = provider.submit_batch([(cid, provider.request(system, it.user, "batch"))
                                             for cid, it in cid_to_item.items()])
            except Exception as e:  # noqa: BLE001
                ledger.release(rid, note=f"submit failed {type(e).__name__}")
                raise
            with open(out_dir / "batches.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"batch_id": bid, "reservation": rid, "requests": len(chunk),
                                    "reserved_usd": round(tot, 6)}) + "\n")
            log(f"batch {bid}: {len(chunk)} requests, reserved ${tot:.4f}")
            while not provider.batch_done(bid):
                sleep(poll_s)
            results = provider.batch_results(bid)
            oc.batch_walls.append(time.perf_counter() - t0)
            spent = 0.0
            for cid, it in cid_to_item.items():
                res = results.get(cid) or CallResult(error="missing from batch results")
                usd = provider.cost_of(res.usage, "batch") if res.usage else 0.0
                spent += usd
                if res.answered:
                    cache.put(it.key, res.to_cache())
                    oc.results[it.key] = res
                log_call(it, res, usd, "batch")
            ledger.settle(rid, spent, batch_id=bid, requests=len(chunk))
            oc.spent_usd += spent
            oc.n_calls += len(chunk)
            i += len(chunk)
    oc.pending = [k for k in seen if k not in oc.results]
    return oc


def write_decisions(path: Path, items: list[Item], oc: Outcome, requested_model: str, settings: str = "") -> None:
    """model = the model string the response reported; settings = the answer settings of the request."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_a", "record_b", "decision", "confidence", "model", "settings"])
        for it in items:
            r = oc.results[it.key]
            if r.refused:
                w.writerow([it.a, it.b, "different", "", r.model or requested_model, settings])
            else:
                w.writerow([it.a, it.b, r.decision, f"{r.confidence:.4f}", r.model or requested_model, settings])


def write_timings(out_dir: Path, provider: Provider, oc: Outcome, n_pairs: int, mode: str) -> None:
    import config as vcfg
    vcfg._TIMINGS.clear()
    stage = f"optionB.{provider.name}"
    if oc.latencies:
        vcfg.record_timing(f"{stage}.llm_call", sum(oc.latencies), items=len(oc.latencies), item="call",
                           per_item_s=oc.latencies, note=f"live calls, {provider.model}")
    for i, w in enumerate(oc.batch_walls):
        vcfg.record_timing(f"{stage}.batch_wall", w, items=None, note=f"batch {i + 1} wall time")
    doc = {"option": "B", "provider": provider.name, "model": provider.model, "host": vcfg.host_info(),
           "timings": vcfg.recorded_timings(),
           "timing_summary": {"mode": mode, "pairs": n_pairs, "calls": oc.n_calls, "cache_hits": oc.cache_hits,
                              "spent_usd": round(oc.spent_usd, 6), "settings": provider.settings_label()}}
    (out_dir / "timings.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ estimates and modes

def make_items(rd: RunData, pairs, provider: Provider, system: str, settings_sha: str, system_sha: str) -> list[Item]:
    items = []
    for a, b, *_ in pairs:
        user = user_turn(rd.records[a], rd.records[b], rd.cols)
        items.append(Item(pair_key(provider.name, provider.model, settings_sha, system_sha, user), a, b, user))
    return items


def estimate(provider: Provider, system: str, items: list[Item], label: str, extra: list[str] | None = None) -> str:
    n = len(items)
    sys_t = math.ceil(len(system) / CHARS_PER_TOKEN_ESTIMATE)
    user_ts = [math.ceil(len(it.user) / CHARS_PER_TOKEN_ESTIMATE) for it in items]
    user_sum = sum(user_ts)
    out_t = OUTPUT_TOKENS_ESTIMATE
    payload = sum(est_request_bytes(system, it.user) for it in items)
    max_req, max_bytes = provider.batch_limits()
    chunks = max(1, math.ceil(payload / max_bytes), math.ceil(n / max_req))
    cached_prefix = provider.cacheable_prefix_tokens(sys_t)
    L = [f"Dry run: {label}, {n:,} pairs, {provider.name} / {provider.model} ({provider.settings_label()})"]
    L += list(extra or [])
    L += [f"  tokens are ESTIMATES at {CHARS_PER_TOKEN_ESTIMATE} characters per token (no tokenizer, no API call); the pilot measures them",
          f"  system prompt: {len(system):,} characters, ~{sys_t:,} tokens, of which ~{cached_prefix:,} cacheable"
          + ("" if cached_prefix else "  (below the caching minimum: it would NOT cache)"),
          f"  user turn: mean ~{user_sum / max(n, 1):.0f} tokens, max ~{max(user_ts, default=0)}",
          f"  output: ~{out_t} tokens per answer (max {provider.max_output_tokens()})",
          f"  tokens for the whole set: input ~{(sys_t + user_sum / max(n, 1)) * n / 1e6:.2f}M, output ~{out_t * n / 1e3:.0f}K",
          f"  as batch files the payload is ~{payload / 1e6:.0f} MB (the prompt travels with every request): "
          f"{chunks} batch(es) of at most {max_req:,} requests / {max_bytes / 1e6:.0f} MB"]
    if not provider.have_prices():
        L.append(f"  cost: NOT COMPUTED, no confirmed prices for {provider.model} in options/prices.json")
        return "\n".join(L)
    c = provider.cfg
    keys = max(1, int(provider.transport.get("prompt_cache_key_buckets", 0) or 1))
    ttl_l, ttl_b = provider.ttl("live"), provider.ttl("batch")

    def total(*, batch: bool, hits: bool) -> float:
        out = n * out_t
        if not hits or not cached_prefix:
            return provider.usd(uncached_in=user_sum + n * sys_t, cache_read_in=0, cache_write_in=0, out=out, batch=batch)
        misses = (chunks if batch else 1) * (1 if provider.cache_write_fee else keys)   # first request(s) fill the cache
        fresh = user_sum + n * (sys_t - cached_prefix)
        reads = max(n - misses, 0) * cached_prefix
        if provider.cache_write_fee:
            return provider.usd(uncached_in=fresh, cache_read_in=reads, cache_write_in=misses * cached_prefix,
                                out=out, batch=batch, ttl=ttl_b if batch else ttl_l)
        return provider.usd(uncached_in=fresh + misses * cached_prefix, cache_read_in=reads, cache_write_in=0,
                            out=out, batch=batch)

    L.append(f"  prices (USD per million tokens, {c.get('price_date')}): input {c['input']}, cached input {c.get('cache_read')}, "
             f"output {c['output']}; batch multiplier {c.get('batch_multiplier')}, batch cached input {c.get('batch_cache_read')}; "
             f"approved: {c.get('approved')}")
    if not provider.cache_write_fee:
        L.append("  cache-write fee: none listed for this model (assumed 0); batch cache hits are NOT documented, so the no-hit batch case is shown")
    rows = [("STANDARD, live, prompt cached", total(batch=False, hits=True)),
            ("STANDARD, live, no caching", total(batch=False, hits=False))]
    if c.get("batch_multiplier") is not None:
        rows += [("BATCH, with cache hits", total(batch=True, hits=True)),
                 ("BATCH, no cache hits at all", total(batch=True, hits=False))]
    for name, v in rows:
        L.append(f"  {name + ':':<34} ${v:>9,.2f}   (${v / n * 1000:.3f} per 1,000 pairs)")
    ub = sum(provider.upper_bound_usd(system, it.user, "batch") for it in items)
    L.append(f"  ledger reservations if run as batches: ${ub:,.2f} in all (pessimistic upper bounds, reserved batch by batch "
             f"and settled to the actual cost; lifetime cap ${LIFETIME_CAP_USD:.0f}). A live run needs --max-usd of at least "
             f"${rows[0][1]:,.0f}.")
    return "\n".join(L)


def sampler_pool(train_band, exclude_pairs: set, exclude_records: set) -> list:
    """Train band pairs a tuning or pilot sample may draw from: not a few-shot pair and touching no
    record that appears in a few-shot example."""
    return [p for p in sorted(train_band)
            if (p[0], p[1]) not in exclude_pairs and p[0] not in exclude_records and p[1] not in exclude_records]


def stratified_train_sample(rd: RunData, truth: TrainTruth, train_band, exclude: set, n: int,
                            exclude_records: frozenset | set = frozenset()) -> list:
    """About 25% same, 25% decoy different (as many as exist), the rest other different; seed 0.
    Excludes the few-shot pairs and every pair touching a record used in a few-shot example."""
    rng = random.Random(0)
    pool = sampler_pool(train_band, exclude, set(exclude_records))
    true = [p for p in pool if truth.same(p[0], p[1])]
    decoy = [p for p in pool if truth.is_decoy_pair(p[0], p[1])]
    other = [p for p in pool if not truth.same(p[0], p[1]) and not truth.is_decoy_pair(p[0], p[1])]
    n_true, n_dec = round(n * 0.25), round(n * 0.25)
    pick = (rng.sample(true, min(n_true, len(true))) + rng.sample(decoy, min(n_dec, len(decoy))))
    pick += rng.sample(other, min(n - len(pick), len(other)))
    return sorted(pick)


def example_turns(rd: RunData, truth: TrainTruth, train_band, exclude: set,
                  exclude_records: frozenset | set = frozenset()) -> list[tuple[str, str]]:
    rng = random.Random(0)
    pool = sampler_pool(train_band, exclude, set(exclude_records))
    t = rng.choice([p for p in pool if truth.same(p[0], p[1])])
    f = rng.choice([p for p in pool if truth.is_decoy_pair(p[0], p[1])])
    return [("same (train pair, label from the truth file)", user_turn(rd.records[t[0]], rd.records[t[1]], rd.cols)),
            ("different (train pair, a decoy: same name, different organization)",
             user_turn(rd.records[f[0]], rd.records[f[1]], rd.cols))]


def weighted_test_sample(rd: RunData, run: Path, size: int, seed: int = 0):
    """A weighted sample of the TEST band for a cheaper second-provider run: every true pair (weight 1)
    plus random non-matches (weight = inverse sampling rate). The truth file is read for TEST records
    only to tell true from false when drawing the sample; labels never enter a prompt. Used by
    --dry-run now; sending these pairs to an LLM still needs --final."""
    ent = {r["record_id"]: r["true_entity_id"] for r in read_csv(run / "orgs_raw_truth.csv")
           if rd.side.get(r["record_id"]) == "test"}
    band = sorted(rd.band("test"))
    true = [p for p in band if ent[p[0]] == ent[p[1]]]
    false = [p for p in band if ent[p[0]] != ent[p[1]]]
    n_false = size - len(true)
    if n_false < 0 or n_false > len(false):
        raise SystemExit(f"cannot draw {size}: {len(true)} true pairs and {len(false)} non-matches in the test band")
    picked = sorted(random.Random(seed).sample(false, n_false))
    w_false = len(false) / n_false
    weights = {(p[0], p[1]): 1.0 for p in true}
    weights.update({(p[0], p[1]): w_false for p in picked})
    return sorted(true + picked), weights, len(true), len(false), n_false, w_false


def main() -> int:
    ap = argparse.ArgumentParser(description="Option B: frontier-LLM baseline.")
    ap.add_argument("run", type=Path)
    ap.add_argument("--provider", choices=("anthropic", "openai"), required=True)
    ap.add_argument("--model", default=None, help="Entry in options/prices.json (default: the provider's first)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-fewshot", action="store_true")
    g.add_argument("--show-prompt", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--freeze-prompt", action="store_true")
    g.add_argument("--eval-train", type=int, metavar="N")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--split", choices=("test", "train", "test-sample"), default="test",
                    help="--dry-run: which pairs to price (test-sample: all true test band pairs plus random non-matches)")
    ap.add_argument("--sample-size", type=int, default=5000, help="--dry-run --split test-sample: sample size")
    ap.add_argument("--mode", choices=("live", "batch"), default=None, help="--final: default batch")
    ap.add_argument("--approve-final-run", action="store_true", help="--final: confirm the test band may be sent")
    ap.add_argument("--max-usd", type=float, default=10.0, help="Cap for this run (default 10)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label", default=None)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = ap.parse_args()

    run = args.run
    live = bool(args.eval_train or args.final)
    provider = make_provider(args.provider, args.model, live=live)
    bdir = run / "options" / "b_llm"
    fewshot_path = bdir / "fewshot.json"
    rd = RunData(run)

    if args.build_fewshot or not fewshot_path.exists():
        truth = TrainTruth(run, rd.side)
        tb = rd.band("train")
        fs = build_fewshot(rd, truth, tb)
        bdir.mkdir(parents=True, exist_ok=True)
        fewshot_path.write_text(json.dumps(fs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"few-shot examples ({len(fs)}) from train band pairs -> {fewshot_path}")
        for e in fs:
            print(f"  {e['label']:<10} {e['stratum']}")
        if args.build_fewshot:
            return 0
    fewshot = json.loads(fewshot_path.read_text(encoding="utf-8"))
    fs_ids = {(e["ids"][0], e["ids"][1]) for e in fewshot}
    fs_records = {r for pr in fs_ids for r in pr}
    system = build_system_prompt(fewshot, rd.cols)
    system_sha, settings_sha = sha256_text(system), provider.settings_sha()
    pdir = bdir / provider.name
    pdir.mkdir(parents=True, exist_ok=True)

    if args.show_prompt:
        truth = TrainTruth(run, rd.side)
        print("=" * 30, "SYSTEM PROMPT", "=" * 30)
        print(system)
        for title, turn in example_turns(rd, truth, rd.band("train"), fs_ids, fs_records):
            print("=" * 30, f"EXAMPLE USER TURN: {title}", "=" * 30)
            print(turn)
        print("=" * 72)
        print(f"system prompt sha256 {system_sha}\nrequest settings {json.dumps(provider.answer_settings())}")
        return 0

    if args.dry_run:
        extra, label = [], f"{args.split} band"
        if args.split == "test-sample":
            pairs, weights, n_true, n_false_all, n_false, w_false = weighted_test_sample(rd, run, args.sample_size)
            label = f"test band sample of {args.sample_size:,}"
            extra = [f"  sample: all {n_true} true test band pairs (weight 1) + {n_false:,} random non-matches of {n_false_all:,} "
                     f"(seed 0, weight {w_false:.4f} = inverse sampling rate); labels used only to draw the sample",
                     f"  weights sum to {sum(weights.values()):,.1f} (the full test band has {n_true + n_false_all:,} pairs)"]
            with (bdir / f"test_sample{args.sample_size}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["record_a", "record_b", "weight"])
                for a_, b_, *_ in pairs:
                    w.writerow([a_, b_, f"{weights[(a_, b_)]:.6f}"])
        else:
            pairs = rd.band(args.split)
        items = make_items(rd, pairs, provider, system, settings_sha, system_sha)
        text = estimate(provider, system, items, label, extra)
        print(text)
        (pdir / f"dry_run_{args.split}.txt").write_text(text + "\n", encoding="utf-8")
        return 0

    freeze_path = pdir / "prompt_freeze.json"
    if args.freeze_prompt:
        doc = {"provider": provider.name, "model": provider.model, "system_prompt_sha256": system_sha,
               "settings_sha256": settings_sha, "answer_settings": provider.answer_settings(),
               "fewshot_pairs": sorted(f"{a}|{b}" for a, b in fs_ids),
               "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        freeze_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"froze prompt sha256 {system_sha[:16]}… for {provider.model} -> {freeze_path}")
        return 0

    ledger = Ledger(args.ledger)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = PairCache(CACHE_DIR / f"{provider.name}.jsonl")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    if args.eval_train:
        truth = TrainTruth(run, rd.side)
        pairs = stratified_train_sample(rd, truth, rd.band("train"), fs_ids, args.eval_train, fs_records)
        label = args.label or f"train{args.eval_train}-{stamp}"
        mode = "live"
        split_note = f"{len(pairs)} stratified TRAIN band pairs"
    else:
        if not freeze_path.exists():
            raise SystemExit("--final needs --freeze-prompt first.")
        fz = json.loads(freeze_path.read_text(encoding="utf-8"))
        if fz["system_prompt_sha256"] != system_sha or fz["settings_sha256"] != settings_sha or fz["model"] != provider.model:
            raise SystemExit("the prompt, settings or model changed since --freeze-prompt; refusing the test band.")
        if not args.approve_final_run:
            raise SystemExit("--final sends the whole test band: add --approve-final-run to confirm.")
        pairs = rd.band("test")
        label = args.label or f"final-{stamp}"
        mode = args.mode or "batch"
        truth = None
        split_note = f"{len(pairs)} TEST band pairs"
    out_dir = pdir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    items = make_items(rd, pairs, provider, system, settings_sha, system_sha)
    run_id = f"{run.name}/b_llm/{provider.name}/{label}"
    print(f"{provider.name} / {provider.model}: {split_note}, mode {mode}, run cap ${args.max_usd:.2f}")
    oc = run_items(provider, items, system, ledger=ledger, cache=cache, run_id=run_id, run_cap_usd=args.max_usd,
                   mode=mode, concurrency=args.concurrency, out_dir=out_dir)
    write_timings(out_dir, provider, oc, len(items), mode)
    models_seen = Counter(r.model for r in oc.results.values())
    meta = {"provider": provider.name, "requested_model": provider.model, "models_reported": dict(models_seen),
            "settings": provider.settings_label(),
            "system_prompt_sha256": system_sha, "answer_settings": provider.answer_settings(), "mode": mode,
            "pairs": len(items), "distinct_prompts": len({i.key for i in items}), "cache_hits": oc.cache_hits,
            "calls": oc.n_calls, "spent_usd": round(oc.spent_usd, 6), "pending": len(oc.pending),
            "stopped": oc.stopped, "refusals": sum(r.refused for r in oc.results.values())}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    lines = [f"Option B {provider.name} / {provider.model} ({provider.settings_label()}): {split_note}",
             f"calls {oc.n_calls}, cache hits {oc.cache_hits}, spent ${oc.spent_usd:.4f}, pending {len(oc.pending)}, "
             f"refusals {meta['refusals']}, models reported {dict(models_seen)}"]
    if oc.stopped:
        lines.append(f"STOPPED EARLY: {oc.stopped}")
    if oc.latencies:
        xs = sorted(oc.latencies)
        lines.append(f"latency per call: p50 {xs[len(xs) // 2]:.2f}s, p95 {xs[int(0.95 * (len(xs) - 1))]:.2f}s, max {xs[-1]:.2f}s")
    if not oc.pending:
        write_decisions(out_dir / "decisions.csv", items, oc, provider.model, provider.settings_label())
        lines.append(f"decisions: {out_dir / 'decisions.csv'}")
        if truth is not None:
            tp = fp = fn = tn = 0
            for it in items:
                r = oc.results[it.key]
                pred = r.decision == "same" and not r.refused
                act = truth.same(it.a, it.b)
                tp, fp, fn, tn = tp + (pred and act), fp + (pred and not act), fn + (not pred and act), tn + (not pred and not act)
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            lines.append(f"quality on this stratified TRAIN sample (not representative: ~25% same): "
                         f"precision {p:.4f} recall {rc:.4f} F1 {2 * p * rc / (p + rc) if p + rc else 0:.4f} "
                         f"TP {tp} FP {fp} FN {fn} TN {tn}")
    else:
        lines.append("no decisions.csv: some pairs have no answer yet; rerun to resume (answered pairs are cached)")
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0 if not oc.pending else 2


if __name__ == "__main__":
    sys.exit(main())

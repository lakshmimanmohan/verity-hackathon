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
  --final-sample FILE   the fixed weighted sample of test band pairs (record_a, record_b, weight; from
                        eval/make_sample.py), live: needs <run>/options/b_llm/frozen_prompt.json matching
                        the current prompt hash, model, reasoning effort and settings hash, plus
                        --approve-final-run, and passes a cost gate first (pairs to call x the pilot's
                        cost per call x 1.3 must fit under the ledger's remaining cap and under
                        --platform-limit minus the ledger total). Keys, workers and the per-key rate come
                        from frozen_prompt.json. The weights never reach the model and the truth file is
                        never read. With --estimate-only: the checks, the projection and the gate, no call.

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
sys.path.insert(0, str(REPO / "eval"))
import make_sample  # noqa: E402
from llm_cache import PairCache, pair_key, sha256_text  # noqa: E402
from llm_ledger import DEFAULT_LEDGER, LIFETIME_CAP_USD, Ledger, LedgerRefused, micro  # noqa: E402

PRICES_PATH = HERE / "prices.json"
PROMPT_PATH = HERE / "b_system_prompt.txt"
APPENDIX_PATH = HERE / "b_prompt_appendix.txt"    # fixed generic padding, appended last (part of the prompt hash)
CACHE_DIR = REPO / "results" / "llm_cache"
SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["same", "different"]},
                   "p_same": {"type": "number"}},
    "required": ["decision", "p_same"],
    "additionalProperties": False,
}
TYPE_RANK = {"NAME": 0, "STREET_ADDRESS": 1, "DESCRIPTIVE": 2, "QUASI_IDENTIFIER": 3, "IDENTIFIER": 4, "DATE": 5}
HARD_TAGS = ("acronym", "truncation", "po_box")
CHARS_PER_TOKEN_ESTIMATE = 3.5      # rough: replaced by measured tokens in the pilot
CHARS_PER_TOKEN_UPPER = 2.5         # for reservations: deliberately pessimistic
OUTPUT_TOKENS_ESTIMATE = 25         # {"decision": "different", "confidence": 0.95} is ~18 tokens
BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 200_000_000       # Anthropic's limit is 256 MB per batch; stay clear of it
FEWSHOT_P_SAME = {"same": 0.95, "different": 0.05}      # the p_same shown in the few-shot answers
# Shared-prefix length window (OpenAI gpt-5.4 caching, measured 2026-11-24 in cache_diagnostic.txt): a visible
# prefix of about 1,670 tokens cached 0; about 1,870 and about 2,730 both cached 1,792. Tokens are estimated at
# 3.5 characters each (round 3: 6,579 characters predicted 1,880, observed about 1,872).
PREFIX_TARGET = (1900, 1950)
PREFIX_BLOCK_BELOW = 1870
PREFIX_WARN_ABOVE = 2000
HIT_RATIO_WARN = 0.80
HIT_RATIO_STOP = 0.50
HIT_RATIO_STOP_AFTER = 100          # calls past warm-up (each key's first call) before the stop rule applies


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
        ans = json.dumps({"decision": ex["label"], "p_same": FEWSHOT_P_SAME[ex["label"]]})
        blocks.append(f"Example {i}\nRecord A:\n{render_record(ex['a'], cols)}\n\nRecord B:\n"
                      f"{render_record(ex['b'], cols)}\n\nAnswer: {ans}")
    return "\n\n".join(blocks)


def build_system_prompt(fewshot: list[dict], cols: list[str]) -> str:
    """The system prompt file with the few-shot examples, then the fixed appendix (if present) after a
    blank line. Everything here is part of the system prompt hash."""
    base = PROMPT_PATH.read_text(encoding="utf-8").replace("{EXAMPLES}", render_examples(fewshot, cols)).rstrip() + "\n"
    if APPENDIX_PATH.exists():
        base = base.rstrip("\n") + "\n\n" + APPENDIX_PATH.read_text(encoding="utf-8").rstrip("\n") + "\n"
    return base


def predicted_prefix_tokens(system: str) -> int:
    return round(len(system) / CHARS_PER_TOKEN_ESTIMATE)


def observed_prefix_tokens(input_tokens: int, user: str) -> int:
    """input_tokens of a call minus the estimated tokens of that call's pair (user turn)."""
    return int(input_tokens) - round(len(user) / CHARS_PER_TOKEN_ESTIMATE)


def prefix_verdict(tokens: int) -> tuple[str, str]:
    """(status, message) for a shared-prefix length: block / warn / outside-target / ok."""
    lo, hi = PREFIX_TARGET
    if tokens < PREFIX_BLOCK_BELOW:
        return "block", f"{tokens} tokens is below {PREFIX_BLOCK_BELOW}, the shortest length shown to cache: BLOCKED"
    if tokens > PREFIX_WARN_ABOVE:
        return "warn", f"{tokens} tokens is above {PREFIX_WARN_ABOVE}: WARNING (tokens past the cached 1,792 are billed in full)"
    if lo <= tokens <= hi:
        return "ok", f"{tokens} tokens, inside the target {lo} to {hi}"
    return "outside", f"{tokens} tokens, outside the target {lo} to {hi} (not blocked)"


def prefix_check_text(system: str) -> tuple[str, str]:
    t = predicted_prefix_tokens(system)
    status, msg = prefix_verdict(t)
    app = APPENDIX_PATH.read_text(encoding="utf-8") if APPENDIX_PATH.exists() else ""
    return status, (f"prefix-length check: system prompt {len(system):,} characters (appendix {len(app):,} characters, "
                    f"sha256 {sha256_text(system)[:12]}); predicted shared prefix {msg}")


def parse_answer(text: str) -> tuple[str, float]:
    d = json.loads(text)
    dec = d["decision"]
    conf = float(d["p_same"])          # P(same); kept in CallResult.confidence and the decisions file's confidence column
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
    def request(self, system: str, user: str, mode: str, slot: int | None = None) -> dict: raise NotImplementedError

    def n_cache_keys(self) -> int:
        return 0

    def cache_key_for(self, system: str, slot: int | None) -> str | None:
        return None
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
        # Measured on gpt-5.4 (cache_diagnostic.txt): visible prefixes of about 1,870 and 2,730 tokens both cached
        # 1,792; about 1,670 cached 0. Working hypothesis (untested past the first): breakpoints every 2,048 tokens
        # counted from a hidden system message of about 256, i.e. cached = 2048 k - 256 for the largest k that fits.
        k = (sys_tokens + 256) // 2048
        return 2048 * k - 256 if k >= 1 else 0

    def n_cache_keys(self) -> int:
        return int(self.transport.get("prompt_cache_keys", 0) or 0)

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
            self._client = openai.OpenAI(api_key=key, max_retries=2)
        return self._client

    def require_approved(self) -> None:
        super().require_approved()
        for k in ("cache_read", "batch_multiplier"):
            if self.cfg.get(k) is None:
                raise SystemExit(f"{self.model}: options/prices.json has no {k} price.")
        if not self.settings.get("reasoning_effort"):
            raise SystemExit(f"{self.model}: request_settings.reasoning_effort is not set in options/prices.json.")

    def cache_key_for(self, system: str, slot: int | None) -> str | None:
        """verity-b-<prompt hash>-<slot>; slots are assigned round-robin in sorted pair order (make_items)."""
        if slot is None or self.n_cache_keys() <= 0:
            return None
        return f"verity-b-{sha256_text(system)[:12]}-{slot}"

    def request(self, system: str, user: str, mode: str, slot: int | None = None) -> dict:
        s = self.settings
        body = {"model": self.model,
                # reusable instructions go in a developer message at the start of input so they are
                # part of the cacheable prefix; the pair comes last
                "input": [{"role": "developer", "content": system}, {"role": "user", "content": user}],
                "text": {"format": {"type": "json_schema", "name": "decision", "strict": True, "schema": SCHEMA}},
                "reasoning": {"effort": s["reasoning_effort"]},
                "max_output_tokens": int(s["max_output_tokens"]),
                "store": bool(self.transport.get("store", False))}
        key = self.cache_key_for(system, slot)
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
    slot: int | None = None      # prompt-cache key slot (round-robin in sorted pair order), None = no key


class KeyRateLimiter:
    """At most per_minute requests per key: a key's request starts at least 60 / per_minute seconds after
    that key's previous request started. Thread-safe; clock and sleep are injectable for tests."""

    def __init__(self, per_minute: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / float(per_minute)
        self.clock, self.sleep = clock, sleep
        self._next: dict = {}
        self._lock = threading.Lock()

    def acquire(self, key) -> float:
        """Wait until key may be used; returns the seconds waited."""
        with self._lock:
            now = self.clock()
            start = max(now, self._next.get(key, now))
            self._next[key] = start + self.interval
        wait = start - now
        if wait > 0:
            self.sleep(wait)
        return wait


class HitStats:
    """Prompt-cache hit ratio = cached input tokens / input tokens, overall and excluding each key's first
    call in the run (the warm-up)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.seen_slots: set = set()
        self.n = self.n_ex = 0
        self.inp = self.cached = self.inp_ex = self.cached_ex = 0
        self.per_slot: dict = {}

    def record(self, slot, usage: dict) -> dict:
        i = int(usage.get("input_tokens", 0) or 0)
        c = int(usage.get("cached_tokens", 0) or 0)
        with self._lock:
            first = slot not in self.seen_slots
            self.seen_slots.add(slot)
            self.n += 1
            self.inp += i
            self.cached += c
            ps = self.per_slot.setdefault(str(slot), {"calls": 0, "input_tokens": 0, "cached_tokens": 0})
            ps["calls"] += 1
            ps["input_tokens"] += i
            ps["cached_tokens"] += c
            if not first:
                self.n_ex += 1
                self.inp_ex += i
                self.cached_ex += c
            return self.snapshot_locked()

    def snapshot_locked(self) -> dict:
        return {"calls": self.n, "input_tokens": self.inp, "cached_tokens": self.cached,
                "hit_ratio": self.cached / self.inp if self.inp else None,
                "calls_excl_first_per_key": self.n_ex, "input_tokens_excl_first": self.inp_ex,
                "cached_tokens_excl_first": self.cached_ex,
                "hit_ratio_excl_first": self.cached_ex / self.inp_ex if self.inp_ex else None,
                "keys_used": len(self.seen_slots)}

    def snapshot(self) -> dict:
        with self._lock:
            d = self.snapshot_locked()
            d["per_key"] = {k: dict(v) for k, v in sorted(self.per_slot.items())}
            return d


def hit_line(h: dict) -> str:
    f = (lambda x: "n/a" if x is None else f"{x:.3f}")
    return (f"hit ratio after {h['calls']} calls: overall {f(h['hit_ratio'])} ({h['cached_tokens']:,} / "
            f"{h['input_tokens']:,} input tokens); excluding each key's first call {f(h['hit_ratio_excl_first'])} "
            f"({h['calls_excl_first_per_key']} calls)")


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
    hits: dict = field(default_factory=dict)            # HitStats snapshot (live mode)
    hit_log: list = field(default_factory=list)         # the every-100-calls lines
    first_prefix: dict = field(default_factory=dict)    # observed prefix from the first live call
    errors: int = 0                                     # API errors (exceptions or failed responses)
    unusable: int = 0                                   # answers that could not be used (parse, incomplete)


def est_request_bytes(system: str, user: str) -> int:
    return len(system.encode("utf-8")) + len(user.encode("utf-8")) + 600


def run_items(provider: Provider, items: list[Item], system: str, *, ledger: Ledger, cache: PairCache,
              run_id: str, run_cap_usd: float, mode: str, concurrency: int, out_dir: Path,
              batch_max_requests: int | None = None, batch_max_bytes: int | None = None,
              poll_s: float = 30.0, sleep=time.sleep, log=print,
              call_spacing_s: float = 0.0, stop_on_error: bool = False,
              rate_limiter: KeyRateLimiter | None = None, hit_report_every: int = 100,
              check_prefix: bool = False) -> Outcome:
    """Answer every distinct item: from the cache when possible, otherwise by live calls or batches,
    each reserved in the ledger first. Stops early (outcome.stopped) if the ledger refuses. Live calls with
    call_spacing_s > 0 run one after another with that pause between them; stop_on_error stops the run at
    the first API error (an exception after the SDK's retries, such as a 429, or a failed response); an
    unparseable or incomplete answer is counted and left pending, not a stop. With rate_limiter, every live
    call waits for its key slot. The first live call runs alone; with check_prefix its observed prefix is
    checked against the window and the run stops if it is below PREFIX_BLOCK_BELOW. The prompt-cache hit
    ratio is logged every hit_report_every calls; below HIT_RATIO_WARN (excluding each key's first call) a
    warning, and below HIT_RATIO_STOP once HIT_RATIO_STOP_AFTER calls past warm-up are in, the run stops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lim_req, lim_bytes = provider.batch_limits()
    batch_max_requests = batch_max_requests or lim_req
    batch_max_bytes = batch_max_bytes or lim_bytes
    oc = Outcome()
    calls_log = out_dir / "calls.jsonl"
    log_lock = threading.Lock()

    def log_call(item: Item, res: CallResult, usd: float, m: str) -> None:
        rec = {"key": item.key[:16], "a": item.a, "b": item.b, "mode": m, "slot": item.slot, "usd": round(usd, 6),
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
        hits = HitStats()
        t_run0 = time.monotonic()

        def halt(reason: str) -> None:
            with lock:
                oc.stopped = oc.stopped or reason
            stop.set()

        def one(it: Item) -> None:
            if stop.is_set():
                return
            if rate_limiter is not None:
                rate_limiter.acquire(it.slot)
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
                res = provider.call(provider.request(system, it.user, "live", slot=it.slot))
            except (Exception, SystemExit) as e:  # noqa: BLE001 - provider errors are logged without payloads
                ledger.release(rid, note=f"error {type(e).__name__}")
                if isinstance(e, SystemExit):         # e.g. a missing API key: stop the whole run
                    raise
                res = CallResult(error=_scrub(f"{type(e).__name__}: {e}", [os.environ.get("ANTHROPIC_API_KEY", ""),
                                                                              os.environ.get("OPENAI_API_KEY", "")]))
                with lock:
                    oc.n_calls += 1
                    oc.errors += 1
                log_call(it, res, 0.0, "live")
                if stop_on_error:
                    halt(f"API error, stopped at once: {res.error}")
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
            failed = bool(res.error) and res.error.startswith("response failed")
            with lock:
                if failed:
                    oc.errors += 1
                elif not res.answered:
                    oc.unusable += 1
            if stop_on_error and failed:
                halt(f"API error (failed response), stopped at once: {res.error}")
            if res.usage:
                h = hits.record(it.slot, res.usage)
                if check_prefix and not oc.first_prefix:
                    obs = observed_prefix_tokens(res.usage.get("input_tokens", 0), it.user)
                    status, msg = prefix_verdict(obs)
                    with lock:
                        oc.first_prefix = {"input_tokens": res.usage.get("input_tokens"),
                                           "cached_tokens": res.usage.get("cached_tokens"),
                                           "pair_tokens_estimate": round(len(it.user) / CHARS_PER_TOKEN_ESTIMATE),
                                           "observed_prefix_tokens": obs, "status": status}
                    log(f"first live call: input_tokens {res.usage.get('input_tokens')}, cached "
                        f"{res.usage.get('cached_tokens')}; observed shared prefix {msg}")
                    if status == "block":
                        halt(f"observed prefix {obs} tokens below {PREFIX_BLOCK_BELOW}")
                if hit_report_every and h["calls"] % hit_report_every == 0:
                    mins = max(time.monotonic() - t_run0, 1e-9) / 60
                    with lock:
                        spent_now = oc.spent_usd
                    line = hit_line(h) + f"; cost so far ${spent_now:.4f}; {h['calls'] / mins:.1f} calls per minute"
                    if h["hit_ratio_excl_first"] is not None and h["hit_ratio_excl_first"] < HIT_RATIO_WARN:
                        line += f"  WARNING: below {HIT_RATIO_WARN:.2f}"
                    with lock:
                        oc.hit_log.append(line)
                    log(line)
                if (h["calls_excl_first_per_key"] >= HIT_RATIO_STOP_AFTER and h["hit_ratio_excl_first"] is not None
                        and h["hit_ratio_excl_first"] < HIT_RATIO_STOP):
                    halt(f"hit ratio excluding each key's first call {h['hit_ratio_excl_first']:.3f} < "
                         f"{HIT_RATIO_STOP} after {h['calls_excl_first_per_key']} calls past warm-up")

        if call_spacing_s > 0:
            for n_done, it in enumerate(todo):
                if stop.is_set():
                    break
                if n_done:
                    sleep(call_spacing_s)
                one(it)
        elif todo:
            one(todo[0])                                 # the first call alone: prefix check, cache warm-up
            if not stop.is_set() and len(todo) > 1:
                with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
                    list(ex.map(one, todo[1:]))
        oc.hits = hits.snapshot()
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
                bid = provider.submit_batch([(cid, provider.request(system, it.user, "batch", slot=it.slot))
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
                              "spent_usd": round(oc.spent_usd, 6), "settings": provider.settings_label(),
                              "prompt_cache": {k: v for k, v in oc.hits.items() if k != "per_key"}}}
    (out_dir / "timings.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ estimates and modes

def make_items(rd: RunData, pairs, provider: Provider, system: str, settings_sha: str, system_sha: str) -> list[Item]:
    """One item per pair, in sorted pair order; prompt-cache key slots round-robin over that order."""
    items = []
    k = provider.n_cache_keys()
    for i, (a, b, *_) in enumerate(sorted(pairs, key=lambda p: (p[0], p[1]))):
        user = user_turn(rd.records[a], rd.records[b], rd.cols)
        items.append(Item(pair_key(provider.name, provider.model, settings_sha, system_sha, user), a, b, user,
                          slot=(i % k) if k > 0 else None))
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
    keys = max(1, provider.n_cache_keys())
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
                            exclude_records: frozenset | set = frozenset(), seed: int = 0,
                            fixed_decoys: list | None = None) -> list:
    """About 25% same, 25% decoy different (as many as exist), the rest other different.
    Excludes the pairs in exclude and every pair touching a record in exclude_records.
    fixed_decoys: decoy pairs to use as the decoy stratum instead of drawing decoys from the pool (they
    bypass exclude, which is how a later round reuses an earlier round's decoys on purpose)."""
    rng = random.Random(seed)
    pool = sampler_pool(train_band, exclude, set(exclude_records))
    true = [p for p in pool if truth.same(p[0], p[1])]
    decoy = [p for p in pool if truth.is_decoy_pair(p[0], p[1])]
    other = [p for p in pool if not truth.same(p[0], p[1]) and not truth.is_decoy_pair(p[0], p[1])]
    n_true, n_dec = round(n * 0.25), round(n * 0.25)
    if fixed_decoys is not None:
        pick = rng.sample(true, min(n_true, len(true))) + list(fixed_decoys)
    else:
        pick = (rng.sample(true, min(n_true, len(true))) + rng.sample(decoy, min(n_dec, len(decoy))))
    pick += rng.sample(other, min(n - len(pick), len(other)))
    return sorted(pick)


def earlier_live_pairs(rd: RunData, truth: TrainTruth, train_band, fs_ids: set, fs_records: set, bdir: Path) -> set:
    """Every train pair an earlier live run may have sent: the pairs of every run folder under
    <provider>/*/calls.jsonl (smoke test, tuning rounds) and the whole 60-pair seed-0 pool the caching
    diagnostic drew from (a superset of what it used)."""
    used = set()
    for calls in sorted(bdir.glob("*/*/calls.jsonl")):
        for line in calls.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                used.add((c["a"], c["b"]))
    used |= {(p[0], p[1]) for p in stratified_train_sample(rd, truth, train_band, fs_ids, 60, fs_records, seed=0)}
    used |= {(p[0], p[1]) for p in stratified_train_sample(rd, truth, train_band, fs_ids, 5, fs_records, seed=0)}
    return used


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
    """The fixed weighted sample of the TEST band (eval/make_sample.py holds the one sampler): every true
    pair (weight 1) plus random non-matches (weight = inverse sampling rate). Labels are read for TEST
    records only, to draw the sample; they never enter a prompt. Used by --dry-run now; sending these
    pairs to an LLM still needs --final."""
    return make_sample.draw_sample(make_sample.test_band_pairs(run), make_sample.test_entities(run), size, seed)


def write_run_outputs(out_dir: Path, provider: Provider, oc: Outcome, items: list[Item], system: str, *, mode: str,
                      split_note: str, ptext: str, pstatus: str, key_rpm: float, concurrency: int, truth=None,
                      extra_meta: dict | None = None) -> int:
    """timings.json, run_meta.json, report.txt and (only when every pair has an answer) decisions.csv.
    Returns 0 when complete, 2 when some pairs still have no answer."""
    system_sha = sha256_text(system)
    write_timings(out_dir, provider, oc, len(items), mode)
    models_seen = Counter(r.model for r in oc.results.values())
    meta = {"provider": provider.name, "requested_model": provider.model, "models_reported": dict(models_seen),
            "settings": provider.settings_label(),
            "system_prompt_sha256": system_sha, "answer_settings": provider.answer_settings(), "mode": mode,
            "pairs": len(items), "distinct_prompts": len({i.key for i in items}), "cache_hits": oc.cache_hits,
            "calls": oc.n_calls, "spent_usd": round(oc.spent_usd, 6), "pending": len(oc.pending),
            "stopped": oc.stopped, "refusals": sum(r.refused for r in oc.results.values()),
            "api_errors": oc.errors, "unusable_answers": oc.unusable,
            "prefix_check": {"predicted_tokens": predicted_prefix_tokens(system), "predicted_status": pstatus,
                             "first_call": oc.first_prefix},
            "prompt_cache_keys": provider.n_cache_keys(), "key_rpm": key_rpm, "concurrency": concurrency,
            "prompt_cache": oc.hits, "hit_log": oc.hit_log}
    meta.update(extra_meta or {})
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    lines = [f"Option B {provider.name} / {provider.model} ({provider.settings_label()}): {split_note}",
             f"calls {oc.n_calls}, cache hits {oc.cache_hits}, spent ${oc.spent_usd:.4f}, pending {len(oc.pending)}, "
             f"refusals {meta['refusals']}, models reported {dict(models_seen)}"]
    lines.append(ptext)
    if oc.first_prefix:
        fp = oc.first_prefix
        lines.append(f"first live call: input_tokens {fp['input_tokens']}, cached {fp['cached_tokens']}, pair about "
                     f"{fp['pair_tokens_estimate']} tokens: observed shared prefix {fp['observed_prefix_tokens']} ({fp['status']})")
    if oc.hits:
        lines += oc.hit_log
        lines.append("final " + hit_line(oc.hits))
        lines.append(f"api errors {oc.errors}, unusable answers {oc.unusable}; prompt-cache keys {provider.n_cache_keys()}, "
                     f"{key_rpm:g} requests per minute per key, {concurrency} workers")
    if oc.stopped:
        lines.append(f"STOPPED EARLY: {oc.stopped}")
    if oc.latencies:
        xs = sorted(oc.latencies)
        lines.append(f"latency per call: p50 {xs[len(xs) // 2]:.2f}s, p95 {xs[int(0.95 * (len(xs) - 1))]:.2f}s, max {xs[-1]:.2f}s")
    for k in ("gate", "power", "started_utc", "ended_utc", "run_time_minutes"):
        if k in (extra_meta or {}):
            lines.append(f"{k.replace('_', ' ')}: {extra_meta[k]}")
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


# ------------------------------------------------------------------ final run on the fixed sample

PILOT_COST_PER_CALL = 0.001545     # tune2, the pilot with the frozen prompt (frozen_prompt.json), measured 2026-11-24
GATE_MARGIN = 1.3
PLATFORM_LIMIT_USD = 30.0
SAMPLE_COLUMNS = ["record_a", "record_b", "weight"]


def power_status() -> str:
    """'ac', 'battery' or 'unknown' (Windows GetSystemPowerStatus; other systems report 'unknown')."""
    if sys.platform != "win32":
        return "unknown"
    import ctypes

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    st = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return "unknown"
    return {1: "ac", 0: "battery"}.get(st.ACLineStatus, "unknown")


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while the run is going (released when on is False or the process ends)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def check_frozen(frozen_path: Path, system: str, provider: Provider) -> dict:
    """The frozen prompt record must exist and match the current prompt hash, model string, reasoning effort and
    settings hash; returns it. Raises SystemExit naming what differs."""
    if not frozen_path.exists():
        raise SystemExit(f"refusing: {frozen_path} does not exist (freeze the prompt after a passing tuning round first).")
    fz = json.loads(frozen_path.read_text(encoding="utf-8"))
    current = {"system_prompt_sha256": sha256_text(system), "model": provider.model,
               "reasoning_effort": provider.settings.get("reasoning_effort"), "settings_sha256": provider.settings_sha()}
    diff = [f"{k}: frozen {fz.get(k)!r}, now {v!r}" for k, v in current.items() if fz.get(k) != v]
    if diff:
        raise SystemExit("refusing: the current prompt or settings do not match frozen_prompt.json: " + "; ".join(diff))
    slots = (fz.get("cache_key_scheme") or {}).get("slots")
    if slots is not None and slots != provider.n_cache_keys():
        raise SystemExit(f"refusing: frozen_prompt.json uses {slots} prompt-cache keys, options/prices.json {provider.n_cache_keys()}.")
    return fz


def load_sample_pairs(path: Path, test_band: set | None) -> list[tuple[str, str]]:
    """The sample's (record_a, record_b) pairs, sorted. Only the pair columns are used; the weight column is read
    only to check the file's layout and never reaches a request. Refuses extra columns (so no label column can
    ride along), repeated pairs, and pairs that are not test band pairs."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        cols = list(rdr.fieldnames or [])
        if cols != SAMPLE_COLUMNS:
            raise SystemExit(f"refusing: {path} must have exactly the columns {SAMPLE_COLUMNS}, found {cols}.")
        pairs = [(r["record_a"].strip(), r["record_b"].strip()) for r in rdr]
    if len(set(pairs)) != len(pairs):
        raise SystemExit(f"refusing: {path} repeats a pair.")
    if test_band is not None:
        bad = [q for q in pairs if q not in test_band]
        if bad:
            raise SystemExit(f"refusing: {len(bad)} sample pair(s) are not test band pairs, e.g. {bad[:3]}.")
    return sorted(pairs)


def final_sample_gate(n_to_call: int, ledger: Ledger, platform_limit: float, pilot_cost: float = PILOT_COST_PER_CALL,
                      margin: float = GATE_MARGIN) -> tuple[bool, str]:
    st = ledger.state()
    total = st.total_micro / 1e6
    remaining = (ledger.cap_micro - st.total_micro) / 1e6
    headroom = platform_limit - total
    need = n_to_call * pilot_cost * margin
    ok = need <= remaining and need <= headroom
    why = [] if ok else [w for w, bad in ((f"exceeds the ledger's remaining ${remaining:.2f}", need > remaining),
                                          (f"exceeds the platform headroom ${headroom:.2f}", need > headroom)) if bad]
    text = (f"cost gate: {n_to_call:,} pairs to call x ${pilot_cost:.6f} (pilot cost per call) x {margin:g} = ${need:.2f}; "
            f"ledger total ${total:.4f}, remaining under the ${ledger.cap_micro / 1e6:.0f} cap ${remaining:.2f}; platform limit "
            f"${platform_limit:.0f} - ledger total = ${headroom:.2f} -> " + ("PASS" if ok else "REFUSED: " + "; ".join(why)))
    return ok, text


def run_final_sample(*, provider: Provider, system: str, rd, sample_path: Path, frozen_path: Path, test_band: set | None,
                     ledger: Ledger, cache: PairCache, out_dir: Path, run_id: str, max_usd: float, approve: bool,
                     platform_limit: float = PLATFORM_LIMIT_USD, pilot_cost: float = PILOT_COST_PER_CALL,
                     estimate_only: bool = False, require_ac: bool = True, key_rpm: float | None = None,
                     concurrency: int | None = None, log=print) -> int:
    """The final option B run on the fixed sample of test band pairs (see --final-sample)."""
    fz = check_frozen(frozen_path, system, provider)
    scheme = fz.get("cache_key_scheme") or {}
    key_rpm = key_rpm if key_rpm is not None else float(scheme.get("requests_per_minute_per_key", 12))
    concurrency = concurrency if concurrency is not None else int(scheme.get("workers", 2))
    if not approve and not estimate_only:
        raise SystemExit("refusing: --final-sample sends test band pairs to the model: add --approve-final-run to confirm.")
    pairs = load_sample_pairs(sample_path, test_band)
    settings_sha, system_sha = provider.settings_sha(), sha256_text(system)
    items = make_items(rd, pairs, provider, system, settings_sha, system_sha)
    assert [(it.a, it.b) for it in items] == pairs
    pstatus, ptext = prefix_check_text(system)
    log(ptext)
    distinct = {it.key for it in items}
    cached = {k for k in distinct if cache.get(k) is not None}
    to_call = len(distinct) - len(cached)
    per_min = min(provider.n_cache_keys() * key_rpm, 95.3) if provider.n_cache_keys() else 95.3
    log(f"pairs: {len(items):,} (sample file {sample_path.name}); distinct prompts {len(distinct):,} "
        f"({len(items) - len(distinct):,} pairs repeat another pair's exact record content and share its answer); "
        f"already answered in the local cache: {len(cached):,}; calls to make: {to_call:,}")
    log(f"projection: about ${to_call * pilot_cost:.2f} at the pilot's ${pilot_cost:.6f} per call; about "
        f"{to_call / per_min:.0f} minutes at {per_min:.1f} calls per minute ({provider.n_cache_keys()} keys x {key_rpm:g} per "
        f"minute, {concurrency} workers; the pilot measured 95.3); run cap --max-usd ${max_usd:.2f}")
    ok, gate_text = final_sample_gate(to_call, ledger, platform_limit, pilot_cost)
    log(gate_text)
    if pstatus == "block":
        raise SystemExit("refusing: prefix below the shortest length shown to cache.")
    if not ok:
        raise SystemExit("refusing: " + gate_text)
    if estimate_only:
        log(f"estimate only: no call made; --approve-final-run {'given' if approve else 'NOT given'}")
        return 0
    power = power_status()
    log(f"power: {power}")
    if require_ac and power == "battery":
        raise SystemExit("refusing: the machine is on battery; plug it in (a sleep would stall the run).")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))} local "
        f"({time.strftime('%H:%M:%SZ', time.gmtime(t_start))})")
    keep_awake(True)
    try:
        oc = run_items(provider, items, system, ledger=ledger, cache=cache, run_id=run_id, run_cap_usd=max_usd,
                       mode="live", concurrency=concurrency, out_dir=out_dir, stop_on_error=True,
                       rate_limiter=KeyRateLimiter(key_rpm) if provider.n_cache_keys() else None,
                       hit_report_every=500, check_prefix=True, log=log)
    finally:
        keep_awake(False)
    t_end = time.time()
    log(f"end: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_end))} local "
        f"({time.strftime('%H:%M:%SZ', time.gmtime(t_end))}); run time {(t_end - t_start) / 60:.1f} minutes")
    extra = {"final_sample": str(sample_path), "frozen_prompt": str(frozen_path), "gate": gate_text, "power": power,
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
             "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
             "run_time_minutes": round((t_end - t_start) / 60, 2), "test_pairs": len(items)}
    return write_run_outputs(out_dir, provider, oc, items, system, mode="live",
                             split_note=f"{len(items)} TEST band sample pairs ({sample_path.name})", ptext=ptext,
                             pstatus=pstatus, key_rpm=key_rpm, concurrency=concurrency, truth=None, extra_meta=extra)


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
    g.add_argument("--final-sample", type=Path, default=None, metavar="SAMPLE_CSV")
    ap.add_argument("--split", choices=("test", "train", "test-sample"), default="test",
                    help="--dry-run: which pairs to price (test-sample: all true test band pairs plus random non-matches)")
    ap.add_argument("--sample-size", type=int, default=5000, help="--dry-run --split test-sample: sample size")
    ap.add_argument("--mode", choices=("live", "batch"), default=None, help="--final: default batch")
    ap.add_argument("--approve-final-run", action="store_true", help="--final: confirm the test band may be sent")
    ap.add_argument("--max-usd", type=float, default=10.0, help="Cap for this run (default 10)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--call-spacing-s", type=float, default=0.0,
                    help="live calls one after another with this pause between them (default: concurrent)")
    ap.add_argument("--stop-on-error", action="store_true", help="stop at the first API error (after the SDK's retries)")
    ap.add_argument("--key-rpm", type=float, default=12.0, help="live: requests per minute per prompt-cache key")
    ap.add_argument("--seed", type=int, default=0, help="--eval-train: sampler seed")
    ap.add_argument("--reuse-decoys-from", default=None, metavar="LABEL",
                    help="--eval-train: use the decoy pairs of an earlier run (e.g. tune1) as the decoy stratum")
    ap.add_argument("--exclude-earlier", action="store_true",
                    help="--eval-train: also exclude every pair of the smoke test and the caching diagnostic")
    ap.add_argument("--estimate-only", action="store_true",
                    help="--eval-train: print the prefix-length check and the cost estimate, then stop (no call)")
    ap.add_argument("--platform-limit", type=float, default=PLATFORM_LIMIT_USD,
                    help="--final-sample: the provider account's spending limit in USD (default 30)")
    ap.add_argument("--pilot-cost-per-call", type=float, default=PILOT_COST_PER_CALL,
                    help="--final-sample: measured cost per call of the pilot (default: tune2's)")
    ap.add_argument("--allow-battery", action="store_true", help="--final-sample: run even when on battery")
    ap.add_argument("--label", default=None)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = ap.parse_args()

    run = args.run
    live = bool(args.eval_train or args.final or (args.final_sample is not None and not args.estimate_only))
    provider = make_provider(args.provider, args.model, live=live)
    bdir = run / "options" / "b_llm"
    fewshot_path = bdir / "fewshot.json"
    rd = RunData(run)

    if args.final_sample is not None and not fewshot_path.exists():
        raise SystemExit("refusing: fewshot.json is missing (the final run never builds it: that would read the truth file).")
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
            make_sample.write_sample(bdir / f"test_sample{args.sample_size}.csv", pairs, weights)
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

    if args.final_sample is not None:
        label = args.label or f"sample-{stamp}"
        return run_final_sample(provider=provider, system=system, rd=rd, sample_path=args.final_sample,
                                frozen_path=bdir / "frozen_prompt.json", test_band=set(make_sample.test_band_pairs(run)),
                                ledger=ledger, cache=cache, out_dir=pdir / label,
                                run_id=f"{run.name}/b_llm/{provider.name}/{label}", max_usd=args.max_usd,
                                approve=args.approve_final_run, platform_limit=args.platform_limit,
                                pilot_cost=args.pilot_cost_per_call, estimate_only=args.estimate_only,
                                require_ac=not args.allow_battery)

    if args.eval_train:
        truth = TrainTruth(run, rd.side)
        tb = rd.band("train")
        excl = set(fs_ids)
        if args.exclude_earlier:
            earlier = earlier_live_pairs(rd, truth, tb, fs_ids, fs_records, bdir)
            excl |= earlier
            print(f"excluding {len(earlier)} pairs of earlier live tests (smoke test, caching diagnostic pool)")
        fixed = None
        if args.reuse_decoys_from:
            src = [json.loads(x) for x in (bdir / "openai" / args.reuse_decoys_from / "calls.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            fixed = sorted({(c["a"], c["b"]) for c in src if truth.is_decoy_pair(c["a"], c["b"])})
            assert not any(a in fs_records or b in fs_records or (a, b) in fs_ids for a, b in fixed)
            print(f"reusing the {len(fixed)} decoy pairs of {args.reuse_decoys_from} as the decoy stratum")
        pairs = stratified_train_sample(rd, truth, tb, excl, args.eval_train, fs_records, seed=args.seed, fixed_decoys=fixed)
        fixed_set = set(fixed or [])
        assert not any(((p[0], p[1]) in excl and (p[0], p[1]) not in fixed_set) or p[0] in fs_records or p[1] in fs_records
                       for p in pairs)
        label = args.label or f"train{args.eval_train}-{stamp}"
        mode = "live"
        split_note = f"{len(pairs)} stratified TRAIN band pairs (seed {args.seed})"
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
    items = make_items(rd, pairs, provider, system, settings_sha, system_sha)
    pstatus, ptext = prefix_check_text(system)
    print(ptext)
    if args.eval_train:
        n_same = sum(truth.same(it.a, it.b) for it in items)
        n_decoy = sum(truth.is_decoy_pair(it.a, it.b) for it in items)
        print(f"sample composition: {n_same} same, {n_decoy} decoy different, {len(items) - n_same - n_decoy} other different "
              f"(the sampler asks for 25% same and 25% decoy and takes as many as exist)")
        print(estimate(provider, system, items, f"{len(items)} TRAIN band pairs (live)"))
        if pstatus == "block":
            raise SystemExit("prefix below the shortest length shown to cache: refusing to send.")
        if args.estimate_only:
            return 0
    out_dir = pdir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run.name}/b_llm/{provider.name}/{label}"
    print(f"{provider.name} / {provider.model}: {split_note}, mode {mode}, run cap ${args.max_usd:.2f}")
    oc = run_items(provider, items, system, ledger=ledger, cache=cache, run_id=run_id, run_cap_usd=args.max_usd,
                   mode=mode, concurrency=args.concurrency, out_dir=out_dir,
                   call_spacing_s=args.call_spacing_s, stop_on_error=args.stop_on_error,
                   rate_limiter=KeyRateLimiter(args.key_rpm) if mode == "live" and provider.n_cache_keys() else None,
                   check_prefix=(mode == "live"))
    return write_run_outputs(out_dir, provider, oc, items, system, mode=mode, split_note=split_note, ptext=ptext,
                             pstatus=pstatus, key_rpm=args.key_rpm, concurrency=args.concurrency, truth=truth,
                             extra_meta={"reused_decoys_from": args.reuse_decoys_from if args.eval_train else None})


if __name__ == "__main__":
    sys.exit(main())

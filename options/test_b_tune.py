"""
Tests for the option B tuning-run machinery (fake provider, no API, no key): the per-key rate limiter,
round-robin prompt-cache key assignment in sorted pair order, the first call running alone with the
prefix-length check, the hit-ratio report and stop rule, the stop on the first API error, and the
appendix being part of the system prompt (and so of its hash).

  python options/test_b_tune.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b_llm_baseline as B  # noqa: E402
from b_llm_baseline import CallResult, HitStats, Item, KeyRateLimiter, Provider, make_items, run_items  # noqa: E402
from llm_cache import PairCache, pair_key  # noqa: E402
from llm_ledger import Ledger  # noqa: E402

SYSTEM = "fixed system prompt " * 20


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class SlotProvider(Provider):
    """Records (slot, start time, user) of every call; usage with a configurable cached share."""
    name = "fake"

    def __init__(self, n_keys=8, cached=1792, input_tokens=2000, fail_on=None, raise_on=None, call_s=0.0):
        super().__init__("fake-model-1", {
            "provider": "fake", "approved": True, "input": 2.5, "output": 15.0, "cache_read": 0.25,
            "batch_multiplier": 0.5, "request_settings": {"max_output_tokens": 64},
            "transport_settings": {"prompt_cache_keys": n_keys}})
        self.cache_write_fee = False
        self.calls = []
        self.lock = threading.Lock()
        self.cached, self.input_tokens = cached, input_tokens
        self.fail_on, self.raise_on, self.call_s = fail_on, raise_on, call_s

    def n_cache_keys(self):
        return int(self.transport.get("prompt_cache_keys", 0))

    def request(self, system, user, mode, slot=None):
        return {"user": user, "slot": slot}

    def call(self, params):
        with self.lock:
            self.calls.append((params["slot"], time.monotonic(), params["user"]))
            n = len(self.calls)
        if self.call_s:
            time.sleep(self.call_s)
        if self.raise_on is not None and n == self.raise_on:
            raise RuntimeError("Error code: 429 - rate limit (simulated)")
        if self.fail_on is not None and n == self.fail_on:
            return CallResult(error="response failed: server_error", usage={"input_tokens": 10, "cached_tokens": 0,
                                                                              "output_tokens": 0})
        cached = self.cached(n) if callable(self.cached) else self.cached
        return CallResult(decision="different", confidence=0.9, stop_reason="completed", model="fake-model-1",
                          usage={"input_tokens": self.input_tokens, "cached_tokens": cached, "output_tokens": 18},
                          latency_s=0.001)

    def cost_of(self, usage, mode):
        c = usage.get("cached_tokens", 0)
        return ((usage["input_tokens"] - c) * 2.5 + c * 0.25 + usage["output_tokens"] * 15.0) / 1e6


class FakeRD:
    """Just enough of RunData for make_items."""
    cols = ["name"]

    def __init__(self, ids):
        self.records = {i: {"name": f"Org {i}"} for i in ids}


def items_for(n, provider, shuffled=True):
    ids = [f"R{i:04d}" for i in range(2 * n)]
    pairs = [(ids[2 * i], ids[2 * i + 1], 0.8, "x") for i in range(n)]
    if shuffled:
        pairs = pairs[::-1]                                # make_items must sort them
    return make_items(FakeRD(ids), pairs, provider, SYSTEM, "s", "sys")


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="tune_test_"))
        self.ledger = Ledger(self.dir / "l.jsonl", lifetime_cap_usd=100.0)
        self.cache = PairCache(self.dir / "c.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_live(self, prov, items, **kw):
        logs = []
        oc = run_items(prov, items, SYSTEM, ledger=self.ledger, cache=self.cache, run_id="r", run_cap_usd=50.0,
                       mode="live", concurrency=kw.pop("concurrency", 2), out_dir=self.dir / "o",
                       log=logs.append, stop_on_error=kw.pop("stop_on_error", True), **kw)
        return oc, logs


class LimiterTests(unittest.TestCase):
    def test_spaces_requests_per_key_and_keys_are_independent(self):
        fc = FakeClock()
        lim = KeyRateLimiter(12, clock=fc.clock, sleep=fc.sleep)     # 5 s per key
        self.assertEqual(lim.acquire(0), 0)
        self.assertEqual(lim.acquire(1), 0)                        # another key: no wait
        self.assertEqual(lim.acquire(0), 5.0)                      # same key: waits the interval
        self.assertEqual(fc.t, 5.0)
        self.assertEqual(lim.acquire(0), 5.0)
        self.assertEqual(lim.acquire(1), 0)                        # key 1 was last used at t=0; now t=10
        fc.t += 20
        self.assertEqual(lim.acquire(0), 0)                        # long idle: no wait, no burst credit kept
        self.assertEqual(lim.acquire(0), 5.0)

    def test_is_thread_safe(self):
        lim = KeyRateLimiter(600)                                  # 0.1 s per key, real clock
        starts = []
        lk = threading.Lock()

        def worker():
            lim.acquire("k")
            with lk:
                starts.append(time.monotonic())

        ts = [threading.Thread(target=worker) for _ in range(5)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        starts.sort()
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertTrue(all(g >= 0.09 for g in gaps), gaps)


class AssignmentTests(Base):
    def test_slots_are_round_robin_in_sorted_pair_order(self):
        prov = SlotProvider(n_keys=8)
        items = items_for(20, prov)
        self.assertEqual([(it.a, it.b) for it in items], sorted((it.a, it.b) for it in items))
        self.assertEqual([it.slot for it in items], [i % 8 for i in range(20)])
        self.assertEqual(set(it.slot for it in items_for(20, SlotProvider(n_keys=0))), {None})

    def test_pipeline_sends_each_items_slot_and_the_limiter_spaces_each_key(self):
        prov = SlotProvider(n_keys=4)
        items = items_for(12, prov)
        oc, _ = self.run_live(prov, items, rate_limiter=KeyRateLimiter(600))     # 0.1 s per key
        self.assertEqual(len(oc.results), 12)
        sent = {u: s for s, _, u in prov.calls}
        self.assertEqual({it.user: it.slot for it in items}, sent)               # each call carried its slot
        by_slot = {}
        for s, t, _ in prov.calls:
            by_slot.setdefault(s, []).append(t)
        for s, ts in by_slot.items():
            ts.sort()
            self.assertEqual(len(ts), 3)
            self.assertTrue(all(b - a >= 0.09 for a, b in zip(ts, ts[1:])), (s, ts))

    def test_the_first_call_runs_alone(self):
        prov = SlotProvider(n_keys=8, call_s=0.05)
        items = items_for(6, prov)
        self.run_live(prov, items, concurrency=4)
        t_first = prov.calls[0][1]
        self.assertTrue(all(t >= t_first + 0.045 for _, t, _ in prov.calls[1:]))  # nothing started during call 1


class HitRatioTests(Base):
    def test_hit_stats_exclude_each_keys_first_call(self):
        h = HitStats()
        for slot in (0, 1, 0, 1):
            h.record(slot, {"input_tokens": 2000, "cached_tokens": 0 if slot not in h.seen_slots else 1792})
        snap = h.snapshot()
        self.assertEqual((snap["calls"], snap["calls_excl_first_per_key"], snap["keys_used"]), (4, 2, 2))
        self.assertAlmostEqual(snap["hit_ratio"], 2 * 1792 / 8000)
        self.assertAlmostEqual(snap["hit_ratio_excl_first"], 1792 / 2000)
        self.assertEqual(snap["per_key"]["0"]["calls"], 2)

    def test_reports_every_100_calls_and_warns_below_0_80(self):
        prov = SlotProvider(n_keys=8, cached=1500)                  # 0.75 of 2,000
        oc, logs = self.run_live(prov, items_for(250, prov))
        hit_lines = [x for x in logs if x.startswith("hit ratio")]
        self.assertEqual(len(hit_lines), 2)                         # at 100 and 200 calls
        self.assertTrue(all("WARNING" in x for x in hit_lines))
        self.assertIsNone(oc.stopped)                               # 0.75 is above the 0.5 stop
        self.assertEqual(oc.hits["calls"], 250)
        self.assertEqual(oc.hit_log, hit_lines)

    def test_stops_when_the_ratio_is_below_0_5_after_100_calls_past_warm_up(self):
        prov = SlotProvider(n_keys=8, cached=500)                   # 0.25
        oc, _ = self.run_live(prov, items_for(300, prov), concurrency=1)
        self.assertIsNotNone(oc.stopped)
        self.assertIn("hit ratio", oc.stopped)
        self.assertEqual(len(prov.calls), 8 + 100)                  # 8 warm-up calls, then 100: stop
        self.assertEqual(len(oc.pending), 300 - 108)

    def test_good_ratio_runs_to_the_end(self):
        prov = SlotProvider(n_keys=8, cached=lambda n: 0 if n <= 8 else 1792)
        oc, logs = self.run_live(prov, items_for(120, prov))
        self.assertIsNone(oc.stopped)
        self.assertEqual(len(prov.calls), 120)
        self.assertFalse(any("WARNING" in x for x in logs))
        self.assertAlmostEqual(oc.hits["hit_ratio_excl_first"], 1792 / 2000)


class ErrorAndPrefixTests(Base):
    def test_stops_at_the_first_api_exception(self):
        prov = SlotProvider(n_keys=8, raise_on=5)
        oc, _ = self.run_live(prov, items_for(40, prov), concurrency=1)
        self.assertIn("API error", oc.stopped)
        self.assertIn("429", oc.stopped)
        self.assertEqual(len(prov.calls), 5)
        self.assertEqual(self.ledger.state().open_reserved_micro, 0)  # the failed call's reservation was released
        self.assertEqual(oc.errors, 1)

    def test_stops_at_a_failed_response_but_not_at_an_unparseable_one(self):
        prov = SlotProvider(n_keys=8, fail_on=3)
        oc, _ = self.run_live(prov, items_for(40, prov), concurrency=1)
        self.assertIn("failed response", oc.stopped)
        self.assertEqual(len(prov.calls), 3)

        class Unparseable(SlotProvider):
            def call(self, params):
                r = super().call(params)
                if len(self.calls) == 2:
                    r = CallResult(error="unparseable answer (JSONDecodeError); status completed", usage=r.usage)
                return r
        prov2 = Unparseable(n_keys=8)
        self.cache = PairCache(self.dir / "c2.jsonl")
        oc2, _ = self.run_live(prov2, items_for(10, prov2), concurrency=1)
        self.assertIsNone(oc2.stopped)
        self.assertEqual((len(prov2.calls), oc2.unusable, len(oc2.pending)), (10, 1, 1))

    def test_first_call_prefix_below_the_window_blocks_the_run(self):
        prov = SlotProvider(n_keys=8, input_tokens=1700)            # prefix about 1,700 - pair tokens < 1,870
        oc, logs = self.run_live(prov, items_for(30, prov), check_prefix=True, concurrency=4)
        self.assertIn("observed prefix", oc.stopped)
        self.assertEqual(len(prov.calls), 1)                        # the first call ran alone; nothing after it
        self.assertEqual(oc.first_prefix["status"], "block")
        self.assertTrue(any(x.startswith("first live call") for x in logs))

    def test_first_call_prefix_inside_the_window_runs_on(self):
        prov = SlotProvider(n_keys=8, input_tokens=1935)
        items = items_for(10, prov)
        oc, _ = self.run_live(prov, items, check_prefix=True)
        self.assertIsNone(oc.stopped)
        pair_t = round(len(items[0].user) / B.CHARS_PER_TOKEN_ESTIMATE)
        self.assertEqual(oc.first_prefix["observed_prefix_tokens"], 1935 - pair_t)


class PrefixWindowTests(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(B.prefix_verdict(1869)[0], "block")
        self.assertEqual(B.prefix_verdict(1870)[0], "outside")
        self.assertEqual(B.prefix_verdict(1900)[0], "ok")
        self.assertEqual(B.prefix_verdict(1950)[0], "ok")
        self.assertEqual(B.prefix_verdict(1990)[0], "outside")
        self.assertEqual(B.prefix_verdict(2001)[0], "warn")

    def test_appendix_is_last_in_the_system_prompt_and_changes_its_hash(self):
        tmp = Path(tempfile.mkdtemp(prefix="tune_app_"))
        try:
            prompt = tmp / "p.txt"
            prompt.write_text("Instructions.\n\n{EXAMPLES}\n", encoding="utf-8")
            app = tmp / "a.txt"
            app.write_text("APPENDIX LINE 1\nAPPENDIX LINE 2\n", encoding="utf-8")
            with mock.patch.object(B, "PROMPT_PATH", prompt), mock.patch.object(B, "APPENDIX_PATH", app):
                with_app = B.build_system_prompt([], ["name"])
            with mock.patch.object(B, "PROMPT_PATH", prompt), mock.patch.object(B, "APPENDIX_PATH", tmp / "none.txt"):
                without = B.build_system_prompt([], ["name"])
            self.assertTrue(with_app.endswith("\n\nAPPENDIX LINE 1\nAPPENDIX LINE 2\n"))
            self.assertTrue(with_app.startswith(without.rstrip("\n")))
            self.assertNotEqual(B.sha256_text(with_app), B.sha256_text(without))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_real_prompt_is_inside_the_target_window(self):
        import json
        run = HERE.parent / "results" / "dev_2026_run11"
        if not (run / "options" / "b_llm" / "fewshot.json").exists():
            self.skipTest("run11 not present")
        fewshot = json.loads((run / "options" / "b_llm" / "fewshot.json").read_text(encoding="utf-8"))
        cols = sorted({c for e in fewshot for c in e["a"]}, key=lambda c: list(fewshot[0]["a"]).index(c))
        system = B.build_system_prompt(fewshot, cols)
        self.assertEqual(B.prefix_verdict(B.predicted_prefix_tokens(system))[0], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)

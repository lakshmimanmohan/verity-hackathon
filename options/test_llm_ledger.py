"""
Tests for the LLM spending ledger, the pair cache and the option B pipeline (fake provider,
no API, no key).

  python options/test_llm_ledger.py          (or: python -m unittest options/test_llm_ledger.py)
  LLM_TEST_TMP=<dir> to keep temporary files under a chosen folder.

Ledger: refuses a reservation past the lifetime cap or the run cap (compared in whole
micro-dollars), counts cache hits as $0, survives a crash between reserve and settle and a torn
last line, and two or more parallel processes cannot jointly exceed the cap.
Pipeline: live and batch runs through a fake provider; a second run is answered entirely from the
cache at $0; a ledger refusal stops the run with answered pairs kept; failed calls are released.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b_llm_baseline import CallResult, Item, Provider, run_items  # noqa: E402
from llm_cache import PairCache, pair_key  # noqa: E402
from llm_ledger import Ledger, LedgerRefused  # noqa: E402


def tmpdir() -> Path:
    base = os.environ.get("LLM_TEST_TMP")
    return Path(tempfile.mkdtemp(prefix="llm_test_", dir=base))


class FakeProvider(Provider):
    """Deterministic provider: answers 'same' when the user turn contains 'ACME'."""
    name = "fake"

    def __init__(self, fail_keys=()):
        super().__init__("fake-model-1", {
            "provider": "fake", "approved": True, "input": 2.0, "output": 10.0, "cache_read": 0.1,
            "cache_write_5m": 2.5, "cache_write_1h": 4.0, "batch_multiplier": 0.5,
            "request_settings": {"max_tokens": 64, "cache_ttl_live": "5m", "cache_ttl_batch": "1h"}})
        self.calls = 0
        self.batches = []
        self.fail = set(fail_keys)

    def request(self, system, user, mode, slot=None):
        return {"user": user, "mode": mode, "slot": slot}

    def _answer(self, params) -> CallResult:
        user = params["user"]
        if any(f in user for f in self.fail):
            raise RuntimeError("simulated provider failure")
        n_in = len(user) // 4
        return CallResult(decision="same" if "ACME" in user else "different", confidence=0.9,
                          stop_reason="end_turn", usage={"input_tokens": n_in, "output_tokens": 20},
                          model="fake-model-1-reported", request_id="req_fake", latency_s=0.001)

    def call(self, params):
        self.calls += 1
        return self._answer(params)

    def cost_of(self, usage, mode):
        v = (usage["input_tokens"] * 2.0 + usage["output_tokens"] * 10.0) / 1e6
        return v * (0.5 if mode == "batch" else 1.0)

    def submit_batch(self, items):
        self.batches.append(items)
        return f"batch_{len(self.batches)}"

    def batch_done(self, batch_id):
        return True

    def batch_results(self, batch_id):
        out = {}
        for cid, params in self.batches[int(batch_id.split("_")[1]) - 1]:
            try:
                out[cid] = self._answer(params)
            except RuntimeError:
                out[cid] = CallResult(error="batch request errored")
        return out


def make_items(n, prefix="Record") -> list[Item]:
    items = []
    for i in range(n):
        user = f"{prefix} {i}: " + ("ACME Corp " if i % 2 == 0 else "Other Inc ") + "x" * 40
        items.append(Item(pair_key("fake", "fake-model-1", "s", "sys", user), f"a{i}", f"b{i}", user))
    return items


SYSTEM = "fixed system prompt " * 20


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir()
        self.path = self.dir / "ledger.jsonl"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def reserve(self, led, usd, run="r", cap=1000.0):
        return led.reserve(usd, run=run, provider="fake", model="m", run_cap_usd=cap)

    def test_refuses_a_reservation_past_the_lifetime_cap(self):
        led = Ledger(self.path, lifetime_cap_usd=1.0)
        rid = self.reserve(led, 0.6)
        led.settle(rid, 0.6)
        with self.assertRaises(LedgerRefused):
            self.reserve(led, 0.5)                      # 0.6 + 0.5 > 1.0
        self.assertEqual(led.state().total_micro, 600_000)   # a refusal writes nothing
        self.reserve(led, 0.4)                           # exactly reaches the cap: allowed
        with self.assertRaises(LedgerRefused):
            self.reserve(led, 0.000001)                  # one micro-dollar more: refused

    def test_float_sums_cannot_sneak_past_the_cap(self):
        led = Ledger(self.path, lifetime_cap_usd=1.0)
        for usd in (0.1, 0.2, 0.7):                      # 0.1 + 0.2 + 0.7 is 0.9999999999999999 in floats
            self.reserve(led, usd)
        self.assertEqual(led.state().total_micro, 1_000_000)
        with self.assertRaises(LedgerRefused):
            self.reserve(led, 0.000001)

    def test_refuses_past_the_run_cap_but_other_runs_are_unaffected(self):
        led = Ledger(self.path, lifetime_cap_usd=100.0)
        self.reserve(led, 0.4, run="run1", cap=0.5)
        with self.assertRaises(LedgerRefused) as cm:
            self.reserve(led, 0.2, run="run1", cap=0.5)
        self.assertIn("run cap", str(cm.exception))
        self.reserve(led, 0.4, run="run2", cap=0.5)      # a different run has its own cap

    def test_settle_replaces_the_reservation_with_the_actual_cost(self):
        led = Ledger(self.path, lifetime_cap_usd=1.0)
        rid = self.reserve(led, 0.5)
        led.settle(rid, 0.1)                             # cheaper than reserved
        st = led.state()
        self.assertEqual((st.settled_micro, st.open_reserved_micro), (100_000, 0))
        self.reserve(led, 0.9)                           # the freed room is usable
        rid2 = self.reserve(led, 0.0)
        led.release(rid2)
        with self.assertRaises(KeyError):
            led.settle(rid, 0.1)                         # already settled

    def test_cache_hits_cost_nothing(self):
        led = Ledger(self.path, lifetime_cap_usd=1.0)
        self.reserve(led, 0.3)
        before = led.state().total_micro
        led.cache_hits(500, run="r", provider="fake", model="m")
        st = led.state()
        self.assertEqual(st.total_micro, before)
        self.assertEqual(st.cache_hits, 500)
        self.reserve(led, 0.7)                           # hits used none of the cap

    def test_a_second_pipeline_run_is_answered_from_the_cache_at_zero_cost(self):
        led = Ledger(self.path, lifetime_cap_usd=10.0)
        cache = PairCache(self.dir / "cache.jsonl")
        items = make_items(20)
        p1 = FakeProvider()
        oc1 = run_items(p1, items, SYSTEM, ledger=led, cache=cache, run_id="r1", run_cap_usd=5.0,
                        mode="live", concurrency=4, out_dir=self.dir / "o1")
        self.assertEqual((p1.calls, oc1.cache_hits, len(oc1.pending)), (20, 0, 0))
        spent = led.state().total_micro
        self.assertGreater(spent, 0)
        p2 = FakeProvider()
        oc2 = run_items(p2, items, SYSTEM, ledger=led, cache=PairCache(self.dir / "cache.jsonl"), run_id="r2",
                        run_cap_usd=5.0, mode="live", concurrency=4, out_dir=self.dir / "o2")
        self.assertEqual((p2.calls, oc2.cache_hits, len(oc2.pending)), (0, 20, 0))
        st = led.state()
        self.assertEqual(st.total_micro, spent)          # the rerun cost $0
        self.assertEqual(st.cache_hits, 20)
        self.assertEqual(st.by_run_micro.get("r2", 0), 0)

    def test_survives_a_crash_between_reserve_and_settle(self):
        code = (f"import os,sys; sys.path.insert(0, {str(HERE)!r}); from llm_ledger import Ledger; "
                f"l = Ledger({str(self.path)!r}, 1.0); l.reserve(0.7, run='crashy', provider='p', model='m', "
                "run_cap_usd=5.0); os._exit(3)")
        rc = subprocess.run([sys.executable, "-c", code]).returncode
        self.assertEqual(rc, 3)                          # died without settling
        led = Ledger(self.path, lifetime_cap_usd=1.0)    # "restart": a fresh object reads the file
        st = led.state()
        self.assertEqual(st.open_reserved_micro, 700_000)  # the orphan still counts (safe direction)
        self.assertEqual(len(st.open), 1)
        with self.assertRaises(LedgerRefused):
            self.reserve(led, 0.5)                       # 0.7 orphan + 0.5 > 1.0
        orphan = next(iter(st.open))
        led.settle(orphan, 0.2)                          # reconcile it with the real cost
        self.reserve(led, 0.5)                           # room again: 0.2 + 0.5 <= 1.0
        self.assertEqual(led.state().settled_micro, 200_000)

    def test_survives_a_torn_last_line(self):
        led = Ledger(self.path, lifetime_cap_usd=1.0)
        rid = self.reserve(led, 0.3)
        led.settle(rid, 0.3)
        with open(self.path, "ab") as f:                 # a crash mid-write: half a line, no newline
            f.write(b'{"type":"settle","id":"abc","usd":9')
        led2 = Ledger(self.path, lifetime_cap_usd=1.0)
        st = led2.state()
        self.assertEqual(st.settled_micro, 300_000)      # the half line is ignored
        self.assertEqual(st.skipped_lines, 1)
        self.reserve(led2, 0.2)                          # appending starts a fresh line
        st = led2.state()
        self.assertEqual((st.settled_micro, st.open_reserved_micro), (300_000, 200_000))
        self.assertEqual(st.skipped_lines, 1)            # only the torn fragment is skipped

    def test_parallel_processes_cannot_jointly_exceed_the_cap(self):
        n_procs, tries, cap = 4, 8, 10.0                 # 32 attempts of $1 against a $10 cap
        code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); from llm_ledger import Ledger, LedgerRefused; "
                f"l = Ledger({str(self.path)!r}, {cap}); ok = 0\n"
                f"for _ in range({tries}):\n"
                "    try:\n"
                "        rid = l.reserve(1.0, run='par', provider='p', model='m', run_cap_usd=1000.0)\n"
                "        l.settle(rid, 1.0); ok += 1\n"
                "    except LedgerRefused:\n"
                "        pass\n"
                "print(ok)")
        procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
                 for _ in range(n_procs)]
        wins = [int(p.communicate()[0].strip()) for p in procs]
        self.assertTrue(all(p.returncode == 0 for p in procs))
        self.assertEqual(sum(wins), 10)                  # exactly cap / $1 reservations succeeded
        st = Ledger(self.path, lifetime_cap_usd=cap).state()
        self.assertEqual(st.total_micro, 10_000_000)
        self.assertEqual(st.open_reserved_micro, 0)

    def test_parallel_threads_in_one_run_cannot_exceed_the_run_cap(self):
        led = Ledger(self.path, lifetime_cap_usd=100.0)
        cache = PairCache(self.dir / "cache.jsonl")
        items = make_items(60)
        prov = FakeProvider()
        ub = prov.upper_bound_usd(SYSTEM, items[0].user, "live")
        cap = ub * 10.5                                   # room for about 10 worst-case reservations at a time
        oc = run_items(prov, items, SYSTEM, ledger=led, cache=cache, run_id="r", run_cap_usd=cap,
                       mode="live", concurrency=8, out_dir=self.dir / "o")
        st = led.state()
        self.assertLessEqual(st.by_run_micro["r"], round(cap * 1e6))
        self.assertEqual(st.open_reserved_micro, 0)
        self.assertEqual(len(oc.results) + len(oc.pending), 60)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_ledger_refusal_stops_the_run_and_keeps_the_answers(self):
        led = Ledger(self.dir / "l.jsonl", lifetime_cap_usd=100.0)
        cache = PairCache(self.dir / "c.jsonl")
        items = make_items(30)
        prov = FakeProvider()
        ub = prov.upper_bound_usd(SYSTEM, items[0].user, "live")
        cap = ub * 5.5
        oc = run_items(prov, items, SYSTEM, ledger=led, cache=cache, run_id="r", run_cap_usd=cap,
                       mode="live", concurrency=1, out_dir=self.dir / "o")
        self.assertIsNotNone(oc.stopped)
        self.assertIn("run cap", oc.stopped)
        done = len(oc.results)
        # Reservations settle to the (much smaller) actual cost, so more than 5 calls fit; the run
        # must stop exactly when settled spend + one more worst-case reservation exceeds the cap.
        self.assertTrue(5 < done < 30)
        self.assertEqual(len(oc.pending), 30 - done)
        self.assertEqual(prov.calls, done)               # no call was made for a refused reservation
        self.assertEqual(len(cache), done)               # the answers were cached for the resume
        spent = led.state().by_run_micro["r"]
        self.assertLessEqual(spent, round(cap * 1e6))
        self.assertGreater(spent + round(ub * 1e6), round(cap * 1e6))   # the next reservation really did not fit

    def test_a_failed_call_is_released_not_charged_and_stays_pending(self):
        led = Ledger(self.dir / "l.jsonl", lifetime_cap_usd=100.0)
        cache = PairCache(self.dir / "c.jsonl")
        items = make_items(6)
        prov = FakeProvider(fail_keys=("Record 3",))
        oc = run_items(prov, items, SYSTEM, ledger=led, cache=cache, run_id="r", run_cap_usd=5.0,
                       mode="live", concurrency=2, out_dir=self.dir / "o")
        self.assertEqual(oc.pending, [items[3].key])
        st = led.state()
        self.assertEqual(st.open_reserved_micro, 0)      # the reservation was released
        self.assertEqual(len(oc.results), 5)

    def test_batch_mode_chunks_by_request_count_and_settles_each_batch(self):
        led = Ledger(self.dir / "l.jsonl", lifetime_cap_usd=100.0)
        cache = PairCache(self.dir / "c.jsonl")
        items = make_items(25)
        prov = FakeProvider()
        oc = run_items(prov, items, SYSTEM, ledger=led, cache=cache, run_id="r", run_cap_usd=50.0,
                       mode="batch", concurrency=1, out_dir=self.dir / "o", batch_max_requests=10,
                       sleep=lambda s: None, log=lambda *a: None)
        self.assertEqual([len(b) for b in prov.batches], [10, 10, 5])
        self.assertEqual(len(oc.results), 25)
        st = led.state()
        self.assertEqual(st.open_reserved_micro, 0)
        self.assertGreater(st.settled_micro, 0)
        self.assertLess(st.settled_micro / 1e6, 0.01)    # actual cost, far below the reservations
        self.assertTrue((self.dir / "o" / "batches.jsonl").exists())

    def test_batch_mode_stops_when_the_next_request_does_not_fit(self):
        led = Ledger(self.dir / "l.jsonl", lifetime_cap_usd=100.0)
        cache = PairCache(self.dir / "c.jsonl")
        items = make_items(25)
        prov = FakeProvider()
        ub = prov.upper_bound_usd(SYSTEM, items[0].user, "batch")
        oc = run_items(prov, items, SYSTEM, ledger=led, cache=cache, run_id="r", run_cap_usd=ub * 12,
                       mode="batch", concurrency=1, out_dir=self.dir / "o", sleep=lambda s: None,
                       log=lambda *a: None)
        # the first batch takes what fits (about 12); settled at actual cost, later batches continue
        self.assertEqual(len(oc.results) + len(oc.pending), 25)
        self.assertLessEqual(led.state().by_run_micro["r"], round(ub * 12 * 1e6) + 1)

    def test_cache_key_depends_on_content_not_on_record_ids(self):
        k1 = pair_key("p", "m", "s", "sys", "Record A:\n  name: X\n")
        k2 = pair_key("p", "m", "s", "sys", "Record A:\n  name: X\n")
        k3 = pair_key("p", "m", "s2", "sys", "Record A:\n  name: X\n")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)                      # different settings -> different key


if __name__ == "__main__":
    unittest.main(verbosity=2)

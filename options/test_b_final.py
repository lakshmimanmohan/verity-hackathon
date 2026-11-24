"""
Tests for --final-sample in options/b_llm_baseline.py (fake provider, no API, no key): refuses without
frozen_prompt.json, on any hash / model / effort mismatch, without --approve-final-run, and when the projected
cost does not fit; the output pair set equals the sample exactly; a run stopped part-way resumes from the cache
with the same decisions and no cost for answered pairs; the weights and the truth never appear in a request
(and the truth file is never read).

  python options/test_b_final.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b_llm_baseline as B  # noqa: E402
from b_llm_baseline import CallResult, Provider  # noqa: E402
from llm_cache import PairCache  # noqa: E402
from llm_ledger import Ledger  # noqa: E402

SYSTEM = "fixed system prompt text " * 268      # 6,700 characters: a predicted prefix of 1,914 tokens, inside the window
WEIGHT = "7.654321"                       # an unusual value: must never appear in a request


class FakeProvider(Provider):
    """Deterministic: 'same' when both names share their first word; records every request body."""
    name = "fake"

    def __init__(self, raise_on=None):
        super().__init__("fake-model-1", {
            "provider": "fake", "approved": True, "input": 2.5, "output": 15.0, "cache_read": 0.25,
            "batch_multiplier": 0.5,
            "request_settings": {"max_output_tokens": 64, "reasoning_effort": "none"},
            "transport_settings": {"prompt_cache_keys": 8}})
        self.cache_write_fee = False
        self.requests, self.lock, self.raise_on = [], threading.Lock(), raise_on

    def n_cache_keys(self):
        return 8

    def request(self, system, user, mode, slot=None):
        return {"model": self.model, "input": [{"role": "developer", "content": system}, {"role": "user", "content": user}],
                "prompt_cache_key": f"k-{slot}"}

    def call(self, params):
        with self.lock:
            self.requests.append(json.dumps(params))
            n = len(self.requests)
        if self.raise_on is not None and n == self.raise_on:
            raise RuntimeError("Error code: 429 - simulated")
        user = params["input"][1]["content"]
        names = [ln.split(":", 1)[1].strip() for ln in user.splitlines() if ln.strip().startswith("name:")]
        same = len(names) == 2 and names[0].split()[0] == names[1].split()[0]
        return CallResult(decision="same" if same else "different", confidence=0.97 if same else 0.02,
                          stop_reason="completed", model="fake-model-1", latency_s=0.001,
                          usage={"input_tokens": 2000, "cached_tokens": 1792, "output_tokens": 18})

    def cost_of(self, usage, mode):
        c = usage.get("cached_tokens", 0)
        return ((usage["input_tokens"] - c) * 2.5 + c * 0.25 + usage["output_tokens"] * 15.0) / 1e6


class FakeRD:
    cols = ["name", "city"]

    def __init__(self, n):
        self.records = {}
        for i in range(n):                       # every user turn distinct; odd pairs share the first name word
            self.records[f"A{i:04d}"] = {"name": f"Org{i} Holdings", "city": "Springfield"}
            self.records[f"B{i:04d}"] = {"name": f"Org{i} Group" if i % 2 else f"Other{i} Group", "city": "Springfield"}


class Base(unittest.TestCase):
    N = 40

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="final_test_"))
        self.rd = FakeRD(self.N)
        self.pairs = [(f"A{i:04d}", f"B{i:04d}") for i in range(self.N)]
        self.sample = self.dir / "sample.csv"
        with self.sample.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(B.SAMPLE_COLUMNS)
            for a, b in reversed(self.pairs):                 # unsorted on purpose
                w.writerow([a, b, WEIGHT])
        self.band = set(self.pairs) | {("A9998", "B9998")}
        self.ledger = Ledger(self.dir / "ledger.jsonl", lifetime_cap_usd=45.0)
        self.cache = PairCache(self.dir / "cache.jsonl")
        self.prov = FakeProvider()
        self.frozen = self.dir / "frozen_prompt.json"
        self.write_frozen()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_frozen(self, **changes):
        doc = {"system_prompt_sha256": B.sha256_text(SYSTEM), "model": self.prov.model, "reasoning_effort": "none",
               "settings_sha256": self.prov.settings_sha(),
               "cache_key_scheme": {"slots": 8, "requests_per_minute_per_key": 60000, "workers": 2}}
        doc.update(changes)
        self.frozen.write_text(json.dumps(doc), encoding="utf-8")

    def run_final(self, prov=None, out="o", **kw):
        logs = []
        args = dict(provider=prov or self.prov, system=SYSTEM, rd=self.rd, sample_path=self.sample, frozen_path=self.frozen,
                    test_band=self.band, ledger=self.ledger, cache=self.cache, out_dir=self.dir / out, run_id="r/final",
                    max_usd=10.0, approve=True, require_ac=False, log=logs.append)
        args.update(kw)
        return B.run_final_sample(**args), logs

    def decisions(self, out="o"):
        with (self.dir / out / "decisions.csv").open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


class RefusalTests(Base):
    def test_refuses_without_frozen_prompt_json(self):
        self.frozen.unlink()
        with self.assertRaises(SystemExit) as cm:
            self.run_final()
        self.assertIn("frozen_prompt.json", str(cm.exception))
        self.assertEqual(self.prov.requests, [])

    def test_refuses_on_any_mismatch(self):
        for field, bad in (("system_prompt_sha256", "0" * 64), ("model", "other-model"), ("reasoning_effort", "low"),
                           ("settings_sha256", "f" * 64)):
            self.write_frozen(**{field: bad})
            with self.assertRaises(SystemExit) as cm:
                self.run_final()
            self.assertIn(field, str(cm.exception))
        self.write_frozen(cache_key_scheme={"slots": 16})
        with self.assertRaises(SystemExit):
            self.run_final()
        self.assertEqual(self.prov.requests, [])

    def test_refuses_without_approval(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_final(approve=False)
        self.assertIn("--approve-final-run", str(cm.exception))
        self.assertEqual(self.run_final(approve=False, estimate_only=True)[0], 0)    # the estimate needs no approval
        self.assertEqual(self.prov.requests, [])

    def test_refuses_when_the_projected_cost_does_not_fit(self):
        # 40 pairs x $0.001545 x 1.3 = $0.0803
        with self.assertRaises(SystemExit) as cm:
            self.run_final(platform_limit=0.05)                                       # platform headroom $0.05
        self.assertIn("platform headroom", str(cm.exception))
        small = Ledger(self.dir / "small.jsonl", lifetime_cap_usd=0.07)               # ledger remaining $0.07
        with self.assertRaises(SystemExit) as cm:
            self.run_final(ledger=small)
        self.assertIn("ledger's remaining", str(cm.exception))
        self.assertEqual(self.prov.requests, [])
        ok, text = B.final_sample_gate(40, self.ledger, 30.0)
        self.assertTrue(ok)
        self.assertIn("PASS", text)

    def test_refuses_pairs_outside_the_test_band_extra_columns_and_repeats(self):
        with self.assertRaises(SystemExit):
            self.run_final(test_band=set(self.pairs[:-1]))
        bad = self.dir / "bad.csv"
        bad.write_text("record_a,record_b,weight,same\nA0000,B0000,1.0,1\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            self.run_final(sample_path=bad)
        self.assertIn("columns", str(cm.exception))
        bad.write_text("record_a,record_b,weight\nA0000,B0000,1.0\nA0000,B0000,1.0\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_final(sample_path=bad)
        self.assertEqual(self.prov.requests, [])


class RunTests(Base):
    def test_output_pair_set_equals_the_sample_exactly(self):
        rc, _ = self.run_final()
        self.assertEqual(rc, 0)
        rows = self.decisions()
        self.assertEqual([(r["record_a"], r["record_b"]) for r in rows], sorted(self.pairs))   # exact set, sorted order
        self.assertEqual(list(rows[0]), ["record_a", "record_b", "decision", "confidence", "model", "settings"])
        self.assertEqual(len(self.prov.requests), self.N)
        for f in ("report.txt", "run_meta.json", "timings.json"):
            self.assertTrue((self.dir / "o" / f).exists())
        meta = json.loads((self.dir / "o" / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["calls"], meta["cache_hits"], meta["pairs"]), (self.N, 0, self.N))
        keys = [json.loads(r)["prompt_cache_key"] for r in self.prov.requests]
        self.assertEqual(set(keys), {f"k-{i}" for i in range(8)})                            # round-robin over 8 keys

    def test_resume_after_a_stop_gives_the_same_decisions_and_costs_nothing_for_answered_pairs(self):
        clean_cache = PairCache(self.dir / "clean_cache.jsonl")
        self.run_final(out="clean", cache=clean_cache, ledger=Ledger(self.dir / "clean_ledger.jsonl", 45.0))
        clean = self.decisions("clean")

        killed = FakeProvider(raise_on=15)                       # dies on the 15th call (calls already in flight on the other worker still finish)
        rc, _ = self.run_final(prov=killed, out="o")
        self.assertEqual(rc, 2)
        answered = len(PairCache(self.dir / "cache.jsonl"))
        self.assertTrue(14 <= answered <= 16, answered)     # 14 before the error, plus calls already in flight on the other worker
        self.assertFalse((self.dir / "o" / "decisions.csv").exists())       # not written while pairs are missing
        spent_first = self.ledger.state().total_micro
        self.assertEqual(self.ledger.state().open_reserved_micro, 0)

        resumed = FakeProvider()
        rc, _ = self.run_final(prov=resumed, out="o")
        self.assertEqual(rc, 0)
        self.assertEqual(len(resumed.requests), self.N - answered)          # only the unanswered pairs were sent
        spent_second = self.ledger.state().total_micro - spent_first
        per_call = round(resumed.cost_of({"input_tokens": 2000, "cached_tokens": 1792, "output_tokens": 18}, "live") * 1e6)
        self.assertEqual(spent_second, per_call * (self.N - answered))     # answered pairs cost nothing
        self.assertEqual(self.decisions("o"), clean)

    def test_weights_and_truth_never_reach_a_request(self):
        with mock.patch.object(B.TrainTruth, "__init__", side_effect=AssertionError("truth read")), \
             mock.patch.object(B.make_sample, "test_entities", side_effect=AssertionError("truth read")):
            rc, _ = self.run_final()
        self.assertEqual(rc, 0)
        blob = "\n".join(self.prov.requests)
        self.assertNotIn(WEIGHT, blob)
        self.assertNotIn("weight", blob)
        for word in ("true_entity_id", "decoy_group", "variants"):
            self.assertNotIn(word, blob)

    def test_logs_the_gate_and_the_start_and_end_times(self):
        _, logs = self.run_final()
        self.assertTrue(any(x.startswith("cost gate:") and "PASS" in x for x in logs))
        self.assertTrue(any(x.startswith("start:") for x in logs) and any(x.startswith("end:") for x in logs))

    def test_progress_lines_carry_calls_hit_ratio_cost_and_rate(self):
        items = B.make_items(self.rd, self.pairs, self.prov, SYSTEM, "s", "sys")
        logs = []
        B.run_items(self.prov, items, SYSTEM, ledger=self.ledger, cache=self.cache, run_id="r", run_cap_usd=5.0,
                    mode="live", concurrency=2, out_dir=self.dir / "p", log=logs.append, hit_report_every=10)
        lines = [x for x in logs if x.startswith("hit ratio after")]
        self.assertEqual(len(lines), 4)
        self.assertTrue(all("cost so far $" in x and "calls per minute" in x for x in lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for the OpenAI path of options/b_llm_baseline.py, against fake objects (no API, no key):
the request body (Responses API, developer message first, strict JSON schema, reasoning effort
none, no temperature, no tools), response parsing (cached tokens, refusal, incomplete, failed),
cost arithmetic (standard and batch with the listed cached-input price), batch input lines and
output parsing, and the approval gates in options/prices.json.

  python options/test_b_openai.py
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b_llm_baseline import OpenAIProvider, SCHEMA, load_prices, make_provider  # noqa: E402

MODEL = "gpt-5.4-2026-03-05"
SYSTEM = "static instructions " * 90          # about 1,800 characters
USER = "Record A:\n  company_name: ACME\n\nRecord B:\n  company_name: ACME Inc\n\nAre these ...?"


def body(text='{"decision": "same", "confidence": 0.9}', status="completed", cached=1664, refusal=False,
         incomplete=None):
    part = {"type": "refusal", "refusal": "I can't help with that."} if refusal else {"type": "output_text", "text": text}
    b = {"id": "resp_1", "model": MODEL, "status": status,
         "output": [{"type": "reasoning", "summary": []}, {"type": "message", "content": [part]}],
         "usage": {"input_tokens": 1800, "input_tokens_details": {"cached_tokens": cached},
                   "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0}}}
    if incomplete:
        b["incomplete_details"] = {"reason": incomplete}
    return b


class OpenAITests(unittest.TestCase):
    def provider(self, client=None, **cfg_changes) -> OpenAIProvider:
        cfg = copy.deepcopy(load_prices()[MODEL])
        cfg.update(cfg_changes)
        return OpenAIProvider(MODEL, cfg, client=client)

    # ---- configuration
    def test_prices_json_has_the_dated_snapshot_effort_none_and_the_documented_prices(self):
        cfg = load_prices()[MODEL]
        self.assertEqual(cfg["provider"], "openai")
        self.assertEqual((cfg["input"], cfg["cache_read"], cfg["output"], cfg["batch_multiplier"]), (2.5, 0.25, 15.0, 0.5))
        self.assertEqual(cfg["request_settings"]["reasoning_effort"], "none")
        self.assertIs(cfg["approved"], False)
        self.assertEqual(cfg["batch_limits"]["max_requests"], 50000)
        self.assertLessEqual(cfg["batch_limits"]["max_bytes"], 200_000_000)
        self.assertIn("reasoning_effort=none", self.provider().settings_label())
        self.assertEqual(make_provider("openai", None, live=False).model, MODEL)

    def test_live_use_is_refused_until_approved(self):
        with self.assertRaises(SystemExit):
            self.provider().require_approved()
        self.provider(approved=True).require_approved()          # approved + prices present: passes
        with self.assertRaises(SystemExit):
            self.provider(approved=True, input=None).require_approved()

    # ---- request
    def test_request_body(self):
        p = self.provider()
        r = p.request(SYSTEM, USER, "live")
        self.assertEqual(r["model"], MODEL)
        self.assertEqual([m["role"] for m in r["input"]], ["developer", "user"])       # static text first, the pair last
        self.assertEqual((r["input"][0]["content"], r["input"][1]["content"]), (SYSTEM, USER))
        fmt = r["text"]["format"]
        self.assertEqual((fmt["type"], fmt["strict"], fmt["name"]), ("json_schema", True, "decision"))
        self.assertEqual(fmt["schema"], SCHEMA)
        self.assertEqual(fmt["schema"]["additionalProperties"], False)
        self.assertEqual(sorted(fmt["schema"]["required"]), ["confidence", "decision"])
        self.assertEqual(r["reasoning"], {"effort": "none"})
        self.assertEqual(r["max_output_tokens"], 64)
        self.assertIs(r["store"], False)
        for absent in ("temperature", "top_p", "tools", "tool_choice", "instructions", "previous_response_id"):
            self.assertNotIn(absent, r)

    def test_prompt_cache_key_is_stable_per_content_and_bounded(self):
        p = self.provider()
        k1 = p.request(SYSTEM, USER, "live")["prompt_cache_key"]
        self.assertEqual(k1, p.request(SYSTEM, USER, "batch")["prompt_cache_key"])
        keys = {p.request(SYSTEM, f"{USER} {i}", "live")["prompt_cache_key"] for i in range(2000)}
        self.assertLessEqual(len(keys), 32)
        self.assertGreater(len(keys), 16)
        self.assertEqual(len({k.rsplit("-", 1)[0] for k in keys}), 1)       # same prompt hash prefix
        q = self.provider(transport_settings={"store": False, "prompt_cache_key_buckets": 0})
        self.assertNotIn("prompt_cache_key", q.request(SYSTEM, USER, "live"))

    # ---- responses
    def test_parses_an_answer_and_cached_tokens(self):
        r = OpenAIProvider.from_body(body())
        self.assertEqual((r.decision, r.confidence, r.error, r.refused), ("same", 0.9, None, False))
        self.assertEqual(r.usage, {"input_tokens": 1800, "cached_tokens": 1664, "output_tokens": 20, "reasoning_tokens": 0})
        self.assertEqual((r.model, r.request_id, r.stop_reason), (MODEL, "resp_1", "completed"))
        self.assertTrue(r.answered)

    def test_clamps_confidence_and_rejects_bad_answers(self):
        self.assertEqual(OpenAIProvider.from_body(body('{"decision": "same", "confidence": 1.7}')).confidence, 1.0)
        for bad in ('not json', '{"decision": "maybe", "confidence": 0.5}', '{"decision": "same"}'):
            r = OpenAIProvider.from_body(body(bad))
            self.assertIsNotNone(r.error)
            self.assertFalse(r.answered)

    def test_refusal_incomplete_and_failed(self):
        r = OpenAIProvider.from_body(body(refusal=True))
        self.assertTrue(r.refused and r.answered and r.decision is None)
        r = OpenAIProvider.from_body(body(text="", status="incomplete", incomplete="max_output_tokens"))
        self.assertIn("incomplete", r.error)
        self.assertEqual(r.stop_reason, "incomplete:max_output_tokens")
        self.assertFalse(r.answered)                                   # not cached: it will be retried
        r = OpenAIProvider.from_body({**body(), "status": "failed", "error": {"code": "server_error"}})
        self.assertIn("server_error", r.error)

    # ---- cost
    def test_cost_standard_and_batch(self):
        p = self.provider()
        u = {"input_tokens": 1800, "cached_tokens": 1664, "output_tokens": 20, "reasoning_tokens": 0}
        std = ((1800 - 1664) * 2.5 + 1664 * 0.25 + 20 * 15.0) / 1e6
        bat = (((1800 - 1664) * 2.5 + 20 * 15.0) * 0.5 + 1664 * 0.13) / 1e6      # cached input at the listed $0.13
        self.assertAlmostEqual(p.cost_of(u, "live"), std, places=12)
        self.assertAlmostEqual(p.cost_of(u, "batch"), bat, places=12)
        nocache = {**u, "cached_tokens": 0}
        self.assertAlmostEqual(p.cost_of(nocache, "batch"), (1800 * 2.5 + 20 * 15.0) * 0.5 / 1e6, places=12)
        reasoning = {**u, "output_tokens": 120, "reasoning_tokens": 100}      # reasoning tokens bill as output
        self.assertAlmostEqual(p.cost_of(reasoning, "live"), ((1800 - 1664) * 2.5 + 1664 * 0.25 + 120 * 15.0) / 1e6, places=12)

    def test_upper_bound_covers_the_actual_cost(self):
        p = self.provider()
        ub = p.upper_bound_usd(SYSTEM, USER, "live")
        actual = p.cost_of({"input_tokens": 700, "cached_tokens": 0, "output_tokens": 64, "reasoning_tokens": 0}, "live")
        self.assertGreater(ub, actual)

    # ---- batch
    def fake_client(self, status="completed", output_lines=None):
        cap = SimpleNamespace(uploaded=None, created=None)

        def files_create(file, purpose):
            cap.uploaded = (file[0], file[1].read().decode("utf-8"), purpose)
            return SimpleNamespace(id="file_in")

        def batches_create(input_file_id, endpoint, completion_window):
            cap.created = (input_file_id, endpoint, completion_window)
            return SimpleNamespace(id="batch_1")

        text = "\n".join(output_lines or [])
        client = SimpleNamespace(
            files=SimpleNamespace(create=files_create, content=lambda fid: SimpleNamespace(text=text)),
            batches=SimpleNamespace(create=batches_create,
                                    retrieve=lambda bid: SimpleNamespace(status=status, output_file_id="file_out")))
        return client, cap

    def test_batch_input_lines_use_the_responses_endpoint(self):
        client, cap = self.fake_client()
        p = self.provider(client=client)
        bid = p.submit_batch([("c1", p.request(SYSTEM, USER, "batch")), ("c2", p.request(SYSTEM, USER + "x", "batch"))])
        self.assertEqual(bid, "batch_1")
        self.assertEqual(cap.created, ("file_in", "/v1/responses", "24h"))
        name, text, purpose = cap.uploaded
        self.assertEqual(purpose, "batch")
        lines = [json.loads(x) for x in text.splitlines()]
        self.assertEqual([x["custom_id"] for x in lines], ["c1", "c2"])
        self.assertTrue(all(x["method"] == "POST" and x["url"] == "/v1/responses" for x in lines))
        self.assertEqual({x["body"]["model"] for x in lines}, {MODEL})               # one model per file
        self.assertEqual(lines[0]["body"]["reasoning"], {"effort": "none"})

    def test_batch_statuses_and_results(self):
        ok = json.dumps({"id": "batch_req_1", "custom_id": "c1",
                         "response": {"status_code": 200, "request_id": "req_9", "body": body()}, "error": None})
        bad = json.dumps({"id": "batch_req_2", "custom_id": "c2", "response": None,
                          "error": {"code": "batch_expired", "message": "x"}})
        client, _ = self.fake_client(output_lines=[ok, bad])
        p = self.provider(client=client)
        self.assertTrue(p.batch_done("batch_1"))
        res = p.batch_results("batch_1")
        self.assertEqual((res["c1"].decision, res["c1"].request_id, res["c1"].usage["cached_tokens"]), ("same", "req_9", 1664))
        self.assertIsNotNone(res["c2"].error)
        for st, done in (("in_progress", False), ("finalizing", False), ("validating", False),
                         ("completed", True), ("expired", True), ("cancelled", True)):
            c, _ = self.fake_client(status=st)
            self.assertEqual(self.provider(client=c).batch_done("b"), done, st)
        c, _ = self.fake_client(status="failed")
        with self.assertRaises(RuntimeError):
            self.provider(client=c).batch_done("b")

    def test_cacheable_prefix_rounds_down_to_128_and_needs_1024(self):
        p = self.provider()
        self.assertEqual(p.cacheable_prefix_tokens(1671), 1664)
        self.assertEqual(p.cacheable_prefix_tokens(1024), 1024)
        self.assertEqual(p.cacheable_prefix_tokens(1000), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

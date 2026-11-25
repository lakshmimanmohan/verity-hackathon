"""
Tests for the pair-cache override VERITY_LLM_CACHE_DIR in options/b_llm_baseline.py (no API, no key, no live call):

  * unset: the request body, the cache key and the settings hash of 50 pairs are byte-identical to the code committed before the
    override existed (git commit BASELINE), and the cache folder is results/llm_cache;
  * set in the shell: the folder is the override; set in .env: the run stops; a blank value means the default;
  * with the override, new entries go to the override folder and the real cache file is untouched (sha256 before and after);
  * the CLI reads the override (--estimate-only counts answers in the override folder, not in the real cache);
  * the stage 7 archive (freeze_run.write_cache_entries) finds its entries in the override folder.

  python options/test_b_cache_override.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "eval"))
import b_llm_baseline as B  # noqa: E402
import freeze_run as F  # noqa: E402
from llm_cache import PairCache, sha256_text  # noqa: E402

BASELINE = "2786b73"                                           # the last commit before the override was added
DEV = REPO / "results" / "dev_2026_run11"
REAL_CACHE = REPO / "results" / "llm_cache" / "openai.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


def committed_module():
    """options/b_llm_baseline.py as committed at BASELINE, loaded as a module that thinks it lives in options/."""
    r = subprocess.run(["git", "show", f"{BASELINE}:options/b_llm_baseline.py"], cwd=REPO, capture_output=True)
    if r.returncode != 0:
        return None
    mod = types.ModuleType("b_llm_baseline_committed")
    mod.__file__ = str(HERE / "b_llm_baseline.py")
    sys.modules[mod.__name__] = mod
    exec(compile(r.stdout.decode("utf-8"), mod.__file__, "exec"), mod.__dict__)
    return mod


def sample_pairs(n: int) -> list[tuple[str, str]]:
    with (DEV / "options" / "b_llm" / "test_sample5000.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted((r["record_a"], r["record_b"]) for r in rows)[:n]


def requests_for(mod, pairs):
    """(settings hash, system hash, [(cache key, user text, serialized request body)]) the way --final-sample builds them."""
    provider = mod.make_provider("openai", None, live=False)
    rd = mod.RunData(DEV)
    fewshot = json.loads((DEV / "options" / "b_llm" / "fewshot.json").read_text(encoding="utf-8"))
    system = mod.build_system_prompt(fewshot, rd.cols)
    settings_sha, system_sha = provider.settings_sha(), sha256_text(system)
    items = mod.make_items(rd, pairs, provider, system, settings_sha, system_sha)
    out = [(it.key, it.user, json.dumps(provider.request(system, it.user, "live", slot=it.slot), sort_keys=True, ensure_ascii=False)) for it in items]
    return settings_sha, system_sha, out


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_when_unset_or_blank(self):
        self.assertEqual(B.resolve_cache_dir({}, self.tmp / "no.env"), REPO / "results" / "llm_cache")
        self.assertEqual(B.resolve_cache_dir({B.CACHE_DIR_ENV: "  "}, self.tmp / "no.env"), REPO / "results" / "llm_cache")

    def test_shell_value_is_used(self):
        self.assertEqual(B.resolve_cache_dir({B.CACHE_DIR_ENV: str(self.tmp / "c")}, self.tmp / "no.env"), (self.tmp / "c").resolve())

    def test_setting_it_in_the_env_file_stops_the_run(self):
        for line in (f"{B.CACHE_DIR_ENV}=/x", f"export {B.CACHE_DIR_ENV}=/x", f"  {B.CACHE_DIR_ENV} = /x"):
            env = self.tmp / ".env"
            env.write_text("OPENAI_API_KEY=placeholder\n" + line + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                B.resolve_cache_dir({}, env)
            self.assertIn("shell environment only", str(cm.exception))
            self.assertNotIn("/x", str(cm.exception))                             # the value is never echoed

    def test_an_env_file_without_it_is_fine(self):
        env = self.tmp / ".env"
        env.write_text("OPENAI_API_KEY=placeholder\n# VERITY_LLM_CACHE_DIR is a shell setting\n", encoding="utf-8")
        self.assertEqual(B.resolve_cache_dir({}, env), REPO / "results" / "llm_cache")

    def test_the_repo_env_file_does_not_set_it(self):
        env = REPO / ".env"
        if env.exists():
            names = [ln.strip().removeprefix("export ").split("=", 1)[0].strip() for ln in env.read_text(encoding="utf-8").splitlines()]
            self.assertNotIn(B.CACHE_DIR_ENV, names)


class UnsetIsUnchangedTests(unittest.TestCase):
    def test_request_body_cache_key_and_settings_hash_of_50_pairs_equal_the_committed_code(self):
        old = committed_module()
        if old is None:
            self.skipTest(f"commit {BASELINE} not available")
        pairs = sample_pairs(50)
        self.assertEqual(len(pairs), 50)
        new_settings, new_system, new_items = requests_for(B, pairs)
        old_settings, old_system, old_items = requests_for(old, pairs)
        self.assertEqual(new_settings, old_settings)
        self.assertEqual(new_system, old_system)
        self.assertEqual(len(new_items), 50)
        for (k1, u1, b1), (k2, u2, b2) in zip(new_items, old_items):
            self.assertEqual(k1, k2)
            self.assertEqual(u1, u2)
            self.assertEqual(b1.encode("utf-8"), b2.encode("utf-8"))
        self.assertGreater(len({k for k, _, _ in new_items}), 40)

    def test_the_default_cache_dir_is_unchanged_when_the_variable_is_unset(self):
        if B.CACHE_DIR_ENV in os.environ:
            self.skipTest("VERITY_LLM_CACHE_DIR is set in this shell")
        self.assertEqual(B.CACHE_DIR, REPO / "results" / "llm_cache")
        old = committed_module()
        if old is not None:
            self.assertEqual(B.CACHE_DIR, old.CACHE_DIR)


class OverrideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.override = self.tmp / "override_cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_entries_go_to_the_override_and_the_real_cache_is_untouched(self):
        before = sha(REAL_CACHE)
        with mock.patch.object(B, "CACHE_DIR", self.override):
            self.override.mkdir(parents=True)
            cache = PairCache(B.CACHE_DIR / "openai.jsonl")                  # as main() builds it
            cache.put("k" * 64, {"decision": "same", "p_same": 0.9})
        self.assertTrue((self.override / "openai.jsonl").exists())
        self.assertIn("k" * 64, (self.override / "openai.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(sha(REAL_CACHE), before)

    def test_main_builds_its_cache_from_cache_dir(self):
        src = (HERE / "b_llm_baseline.py").read_text(encoding="utf-8")
        self.assertIn('cache = PairCache(CACHE_DIR / f"{provider.name}.jsonl")', src)
        self.assertIn("CACHE_DIR.mkdir(parents=True, exist_ok=True)", src)
        self.assertEqual(src.count("results\" / \"llm_cache"), 1)             # the default lives in one place only

    def test_cli_estimate_counts_answers_in_the_override_not_in_the_real_cache(self):
        sample = DEV / "options" / "b_llm" / "test_sample5000.csv"
        base = [sys.executable, str(HERE / "b_llm_baseline.py"), str(DEV), "--provider", "openai", "--final-sample", str(sample),
                "--label", "override_test", "--max-usd", "0.10", "--estimate-only"]
        env = {k: v for k, v in os.environ.items() if k != B.CACHE_DIR_ENV}
        env.update({"PYTHONHASHSEED": "0", "PYTHONUTF8": "1"})
        default = subprocess.run(base, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8")
        env[B.CACHE_DIR_ENV] = str(self.override)
        over = subprocess.run(base, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(default.returncode, 0, default.stderr[-500:])
        self.assertEqual(over.returncode, 0, over.stderr[-500:])
        line = lambda r: next(x for x in r.stdout.splitlines() if x.startswith("pairs:"))
        self.assertIn("calls to make: 0", line(default))                       # the dev run's 5,000 answers are in the real cache
        self.assertIn("already answered in the local cache: 0", line(over))      # the override folder is empty
        self.assertNotIn("calls to make: 0", line(over))
        f = self.override / "openai.jsonl"
        self.assertTrue(not f.exists() or f.stat().st_size == 0)                # an estimate writes no entry

    def test_stage_7_archive_finds_its_entries_in_the_override(self):
        pairs = sample_pairs(20)
        sample = self.tmp / "sample.csv"
        with sample.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(B.SAMPLE_COLUMNS)
            for a, b in pairs:
                w.writerow([a, b, "1.0"])
        _, _, items = requests_for(B, pairs)
        R = SimpleNamespace(run_dir=DEV)
        with mock.patch.object(B, "CACHE_DIR", self.override):
            self.override.mkdir(parents=True)
            with self.assertRaises(F.StageFailed):                              # empty override: the entries are not found there
                F.write_cache_entries(R, sample, self.tmp / "none.gz")
            cache = PairCache(B.CACHE_DIR / "openai.jsonl")
            for key, _, _ in items:
                cache.put(key, {"decision": "different", "p_same": 0.1})
            n = F.write_cache_entries(R, sample, self.tmp / "archive.gz")
        self.assertEqual(n, len({k for k, _, _ in items}))
        import gzip
        with gzip.open(self.tmp / "archive.gz", "rt", encoding="utf-8") as f:
            keys = [json.loads(x)["key"] for x in f]
        self.assertEqual(set(keys), {k for k, _, _ in items})


if __name__ == "__main__":
    unittest.main(verbosity=2)

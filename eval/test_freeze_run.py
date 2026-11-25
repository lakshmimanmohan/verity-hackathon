"""
Tests for eval/freeze_run.py: the seed 7919 refusals, the manifest check, stage order and resume, and truth sealing.
Every test runs in a temporary folder with fake stages or a fake step runner: nothing is generated, no seed 7919 data
is created or read, and no LLM is called.

  python eval/test_freeze_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_run as F  # noqa: E402


def no_call(args, env, log_file):
    raise AssertionError(f"no step may run in this test: {args}")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "frozen.txt").write_text("v1\n", encoding="utf-8")
        self.logs: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, mode="rehearsal", seed=42, stages=None, call=None, approve=False, run_dir=None, freeze_dir=None) -> F.Runner:
        run_dir = run_dir or self.repo / "results" / "run"
        freeze_dir = freeze_dir or run_dir / "_freeze"
        R = F.Runner(mode, seed, run_dir, freeze_dir, approve=approve, allow_battery=True, repo=self.repo, frozen=["frozen.txt"],
                     stages=stages, call=call or no_call, log=self.logs.append, ledger_path=self.repo / "ledger.jsonl")
        R.pip_freeze = False
        return R

    def guards_fired(self, R: F.Runner) -> list[dict]:
        p = R.freeze_dir / "guards.jsonl"
        return [g for g in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if g["fired"]] if p.exists() else []


class DSettingsV2(Base):
    """The manifest hashes both D rule files and both settings files; v2 is D's setting, the original rule the secondary row."""

    def test_asset_list_has_both_rules_and_both_settings(self):
        self.assertEqual(sorted(Path(k).name for k in F.ASSETS_D),
                         ["frozen_settings.json", "frozen_settings_v2.json", "selection_rule.txt", "selection_rule_v2.txt"])
        for src in F.ASSETS_D.values():
            self.assertIn(src, F.FROZEN_FILES)
            self.assertTrue((F.REPO / src).exists(), src)

    def test_manifest_hashes_all_four_files_and_records_both_settings(self):
        R = F.Runner("rehearsal", 42, self.repo / "run", self.repo / "run" / "_freeze", allow_battery=True, repo=F.REPO, frozen=F.FROZEN_FILES,
                     stages=[(0, "zero", lambda R: {})], call=no_call, log=self.logs.append, ledger_path=self.repo / "ledger.jsonl")
        R.pip_freeze = False
        doc = R.write_manifest()
        for src in F.ASSETS_D.values():
            self.assertEqual(doc["files"][src], F.sha256_file(F.REPO / src))
        p = doc["params"]
        self.assertEqual((p["d_lo"], p["d_hi"]), (0.005, 0.9))                   # v2
        self.assertEqual((p["d_original_lo"], p["d_original_hi"]), (0.5, 0.9))   # the secondary row
        import hashlib
        rule = (F.REPO / F.ASSETS_D["options/d_hybrid/selection_rule_v2.txt"]).read_bytes().replace(bytes([13, 10]), bytes([10]))
        self.assertEqual(p["d_rule_v2_sha256_lf"], hashlib.sha256(rule).hexdigest())
        self.assertTrue(any("routing rule v2" in n for n in doc["notes"]))


class ManifestNoteAndReport(Base):
    def runner_real_repo(self, run_dir):
        R = F.Runner("rehearsal", 42, run_dir, self.repo / "_freeze", allow_battery=True, repo=F.REPO, frozen=F.FROZEN_FILES,
                     stages=[(0, "zero", lambda R: {})], call=no_call, log=self.logs.append, ledger_path=self.repo / "ledger.jsonl")
        R.pip_freeze = False
        return R

    def test_manifest_notes_the_literal_reading(self):
        notes = self.runner_real_repo(self.repo / "run").write_manifest()["notes"]
        self.assertIn("D decisions for pairs outside [lo, hi): C's own decision (literal v2 text); forced reading reported as a diagnostic", notes)

    def test_could_not_test_list_reads_the_v2_share_not_a_fixed_string(self):
        text = " | ".join(F.cannot_test(self.runner_real_repo(F.REPO / "results" / "dev_2026_run11")))
        self.assertNotIn("0.085%", text)
        self.assertIn("pooled 0.260%", text)                    # frozen_settings_v2.json: chosen_metrics.share 0.002595
        self.assertIn("51 of 44,882", text)                     # this run's own routed share (the dev run)
        self.assertIn("0.114%", text)

    def test_manifest_notes_the_cache_override_and_records_which_cache_is_used(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"VERITY_LLM_CACHE_DIR": ""}):
            doc = self.runner_real_repo(self.repo / "run").write_manifest()
        self.assertIn("B cache directory can be overridden by VERITY_LLM_CACHE_DIR; default, requests and keys unchanged", doc["notes"])
        self.assertEqual(doc["params"]["llm_cache_dir"], "results/llm_cache (default)")
        self.assertIn("options/b_llm_baseline.py", F.FROZEN_FILES)
        with mock.patch.dict(os.environ, {"VERITY_LLM_CACHE_DIR": str(self.repo / "c")}):
            doc = self.runner_real_repo(self.repo / "run2").write_manifest()
        self.assertEqual(doc["params"]["llm_cache_dir"], "VERITY_LLM_CACHE_DIR (shell override, set)")      # the path itself is not recorded

    def test_embedding_model_is_pinned_safetensors_and_noted_in_the_manifest(self):
        import re
        src = (F.REPO / "verity" / "src" / "embed.py").read_text(encoding="utf-8")
        rev = re.search(r'^MODEL_REVISION = "([0-9a-f]{40})"', src, re.M).group(1)
        self.assertEqual(F.embedding_revision(F.REPO), rev)
        self.assertIn("SentenceTransformer(MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=False", src)
        self.assertIn('"use_safetensors": True', src)
        self.assertNotIn("trust_remote_code=True", " ".join(p.read_text(encoding="utf-8") for p in (F.REPO / "verity" / "src").glob("*.py")))
        self.assertIn("verity/src/embed.py", F.FROZEN_FILES)
        notes = self.runner_real_repo(self.repo / "run").write_manifest()["notes"]
        self.assertIn(f"embedding model pinned to revision {rev}, safetensors; no output changed", notes)

    def test_stage_8_makes_the_forced_diagnostic_and_stage_10_compares_it(self):
        import inspect
        self.assertIn('"--reading", "forced"', inspect.getsource(F.s8_d))
        self.assertIn("compare_d_forced_vs_b.txt", inspect.getsource(F.s10_scoring))


class B_StageCaps(Base):
    """Per-run caps of the B stages: a rehearsal gets prior ledger spend for the run id + an allowance; the real run keeps absolute caps."""

    def spend(self, run_id: str, usd: float) -> None:
        import sys
        sys.path.insert(0, str(F.REPO / "options"))
        from llm_ledger import Ledger
        led = Ledger(self.repo / "ledger.jsonl", lifetime_cap_usd=45.0)
        rid = led.reserve(usd, run=run_id, provider="openai", model="m", run_cap_usd=10.0)
        led.settle(rid, usd)

    def test_rehearsal_cap_is_prior_spend_plus_the_allowance(self):
        R = self.runner()                                                  # rehearsal, run folder name "run"
        self.spend("run/b_llm/openai/sample5000", 0.74)
        self.spend("run/b_llm/openai/dfull", 0.03)
        self.assertEqual(F.b_run_id(R, "sample5000"), "run/b_llm/openai/sample5000")
        cap, prior = F.stage_cap(R, "sample5000")
        self.assertAlmostEqual(prior, 0.74)
        self.assertAlmostEqual(cap, 0.74 + 0.80)
        self.assertAlmostEqual(F.stage_cap(R, "dfull")[0], 0.03 + 0.20)

    def test_a_fresh_run_id_gets_exactly_the_allowance(self):
        R = self.runner()
        self.spend("another_run/b_llm/openai/sample5000", 0.9)                # spend under a different run id does not count
        for label, allowance in (("sample5000", 0.80), ("dfull", 0.20), ("decoycheck", 0.20)):
            self.assertEqual(F.stage_cap(R, label), (allowance, 0.0))

    def test_open_reservations_count_as_prior_spend(self):
        import sys
        sys.path.insert(0, str(F.REPO / "options"))
        from llm_ledger import Ledger
        Ledger(self.repo / "ledger.jsonl", lifetime_cap_usd=45.0).reserve(0.10, run="run/b_llm/openai/decoycheck", provider="openai", model="m", run_cap_usd=1.0)
        self.assertAlmostEqual(F.stage_cap(self.runner(), "decoycheck")[0], 0.30)

    def test_real_run_caps_are_absolute_and_unchanged(self):
        R = self.runner(mode="real", seed=F.EVAL_SEED, run_dir=self.repo / "results" / "eval_7919", freeze_dir=self.repo / "f")
        self.spend("eval_7919/b_llm/openai/sample5000", 3.0)                   # prior spend is ignored in the real run
        self.assertEqual(F.stage_cap(R, "sample5000"), (12.0, 0.0))
        self.assertEqual(F.stage_cap(R, "dfull"), (1.0, 0.0))
        self.assertEqual(F.stage_cap(R, "decoycheck"), (1.0, 0.0))
        self.assertEqual(F.B_CAPS["real"], {"sample5000": 12.0, "dfull": 1.0, "decoycheck": 1.0})
        self.assertEqual(F.B_CAPS["rehearsal"], {"sample5000": 0.80, "dfull": 0.20, "decoycheck": 0.20})      # rehearsal caps unchanged

    def test_manifest_notes_the_real_run_caps(self):
        notes = F.Runner("rehearsal", 42, self.repo / "run", self.repo / "run" / "_freeze", allow_battery=True, repo=F.REPO, frozen=[],
                         stages=[], call=no_call, log=self.logs.append, ledger_path=self.repo / "ledger.jsonl").write_manifest()["notes"]
        self.assertIn("real-run B stage caps: sample 12.00, routed 1.00, decoy 1.00 USD; rehearsal caps unchanged", notes)

    GATE = "cost gate: {n:,} pairs to call x $0.001545 (pilot cost per call) x 1.3 = ${x:.2f}; ledger total $13.9208, remaining under the $45 cap $31.08"

    def test_real_stage_refuses_to_start_when_its_cap_is_below_its_gate_projection(self):
        R = self.runner(mode="real", seed=F.EVAL_SEED, run_dir=self.repo / "results" / "eval_7919", freeze_dir=self.repo / "f")
        self.assertAlmostEqual(F.gate_projection(self.GATE.format(n=5000, x=10.0425)), 5000 * 0.001545 * 1.3)
        with self.assertRaises(F.Guard) as cm:
            F.check_cap_against_gate(R, 9, "decoycheck", 0.50, self.GATE.format(n=300, x=0.60))
        self.assertIn("cap $0.50", str(cm.exception))                       # both numbers are printed
        self.assertIn("gate projection $0.60", str(cm.exception))
        with self.assertRaises(F.Guard):                                      # a missing gate line also stops the stage
            F.check_cap_against_gate(R, 9, "decoycheck", 1.0, "no gate line here")

    def test_real_stage_starts_when_the_cap_covers_the_gate_projection(self):
        R = self.runner(mode="real", seed=F.EVAL_SEED, run_dir=self.repo / "results" / "eval_7919", freeze_dir=self.repo / "f")
        for label, n in (("sample5000", 5000), ("dfull", 450), ("decoycheck", 300)):
            F.check_cap_against_gate(R, 7, label, F.B_CAPS["real"][label], self.GATE.format(n=n, x=n * 0.001545 * 1.3))   # no exception

    def test_the_new_real_caps_cover_the_worst_case_gates(self):
        for label, n in (("sample5000", 5000), ("dfull", 450), ("decoycheck", 300)):
            self.assertGreaterEqual(F.B_CAPS["real"][label], n * 0.001545 * 1.3)

    def test_rehearsal_does_not_apply_the_gate_guard(self):
        R = self.runner()
        F.check_cap_against_gate(R, 9, "decoycheck", 0.01, self.GATE.format(n=300, x=0.60))                  # rehearsal: no exception

    def test_rehearsal_allowances_total_the_new_spend_cap(self):
        self.assertAlmostEqual(sum(F.B_CAPS["rehearsal"].values()), 1.20)


class FreezeDateNaming(Base):
    """The real freeze folder and tag are named from the manifest date; the rehearsal naming is unchanged."""

    def real(self, date_folder: str, freeze_date: str | None = None, stages=None):
        return F.Runner("real", F.EVAL_SEED, self.repo / "results" / "eval_7919", self.repo / "results" / date_folder, allow_battery=True,
                        repo=self.repo, frozen=["frozen.txt"], stages=stages or [(0, "zero", lambda R: {})], call=no_call, log=self.logs.append,
                        ledger_path=self.repo / "ledger.jsonl", freeze_date=freeze_date)

    def test_names_follow_the_date(self):
        self.assertEqual(F.freeze_name("2026-11-27"), "freeze_2026-11-27")
        self.assertEqual(F.tag_name("2026-11-27"), "freeze-2026-11-27")
        self.assertEqual(F.freeze_name("2026-12-03"), "freeze_2026-12-03")
        self.assertFalse(hasattr(F, "FREEZE_DATE") or hasattr(F, "TAG") or hasattr(F, "REAL_FREEZE_REL"))      # no fixed date left in the code

    def test_the_manifest_stores_the_date_and_the_tag_follows_it(self):
        R = self.real("freeze_2026-11-28", "2026-11-28")
        doc = R.write_manifest()
        self.assertEqual(doc["params"]["freeze_date"], "2026-11-28")
        self.assertEqual(doc["tag"], "freeze-2026-11-28")
        self.assertIn("freeze folder and tag are named from the manifest date", doc["notes"])
        self.assertEqual(R.manifest()["params"]["freeze_date"], "2026-11-28")        # later stages read it from the manifest

    def test_a_folder_that_does_not_match_the_manifest_date_stops_the_run(self):
        R = self.real("freeze_2026-11-28", "2026-11-28")
        R.write_manifest()
        wrong = self.repo / "results" / "freeze_2026-12-03"
        wrong.mkdir(parents=True)
        (wrong / "freeze_manifest.json").write_text((R.manifest_path).read_text(encoding="utf-8"), encoding="utf-8")      # manifest says 11-28
        R2 = self.real("freeze_2026-12-03", "2026-12-03")
        with self.assertRaises(F.Guard) as cm:
            R2.manifest()
        self.assertIn("does not match the manifest date 2026-11-28", str(cm.exception))
        with self.assertRaises(F.Guard):
            R2.run(only=0)
        self.assertEqual(self.guards_fired(R2) or [], [])           # stopped by the manifest check, before any stage ran
        self.assertFalse((wrong / "stages").exists())

    def test_the_rehearsal_naming_is_unchanged(self):
        R = self.runner()
        self.assertEqual(R.freeze_dir.name, "_freeze")
        self.assertIsNone(R.write_manifest()["tag"])
        self.assertEqual(self.runner().mode, "rehearsal")

    def test_find_real_freeze_reads_the_folder_by_its_manifest(self):
        self.assertIsNone(F.find_real_freeze(self.repo))
        (self.repo / "results").mkdir()
        self.real("freeze_2026-11-27", "2026-11-27").write_manifest()
        self.assertEqual(F.find_real_freeze(self.repo).name, "freeze_2026-11-27")
        (self.repo / "results" / "freeze_notadate").mkdir()                                      # not a date: ignored
        self.assertEqual(F.find_real_freeze(self.repo).name, "freeze_2026-11-27")

    def test_two_real_freeze_folders_stop_the_run(self):
        self.real("freeze_2026-11-27", "2026-11-27").write_manifest()
        self.real("freeze_2026-12-03", "2026-12-03").write_manifest()
        with self.assertRaises(F.Guard) as cm:
            F.find_real_freeze(self.repo)
        self.assertIn("happens once", str(cm.exception))

    def test_write_manifest_names_the_folder_from_the_day_it_is_written(self):
        import argparse
        from unittest import mock
        args = argparse.Namespace(rehearsal=False, approve_seed_7919=False, allow_battery=False, command="write-manifest", dev_seed=42)
        (self.repo / "results").mkdir()
        with mock.patch.object(F, "REPO", self.repo), mock.patch.object(F, "today", lambda: "2026-11-27"):
            R = F.runner_from_args(args)
        self.assertEqual(R.freeze_dir.name, "freeze_2026-11-27")
        self.assertEqual(R.freeze_date, "2026-11-27")

    def test_no_real_manifest_creates_nothing(self):
        import argparse
        args = argparse.Namespace(rehearsal=False, approve_seed_7919=False, allow_battery=False, command="run", dev_seed=42)
        with unittest_mock_patch(F, "REPO", self.repo):
            (self.repo / "results").mkdir()
            with self.assertRaises(F.Guard) as cm:
                F.runner_from_args(args)
        self.assertIn("write-manifest", str(cm.exception))
        self.assertEqual(sorted(p.name for p in (self.repo / "results").iterdir()), [])        # no folder, no log written


def unittest_mock_patch(obj, name, value):
    from unittest import mock
    return mock.patch.object(obj, name, value)


class PreflightTests(Base):
    """The read-only preflight: a pass case and a failing case per check. Runs against a temporary git repo with a local 'origin'."""

    GOOD_ENV = {"OPENAI_API_KEY": "sk-test-not-a-real-key-0123456789", "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}

    def git(self, *args, cwd=None):
        import subprocess
        r = subprocess.run(["git", *args], cwd=cwd or self.repo, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def setUp(self):
        super().setUp()
        origin = self.repo.parent / (self.repo.name + "_origin.git")
        self.git("init", "-q", "--bare", str(origin), cwd=self.repo.parent)
        self.git("init", "-q")
        self.git("config", "user.email", "t@example.org")
        self.git("config", "user.name", "t")
        (self.repo / ".gitignore").write_text("ledger.jsonl\nledger.jsonl.lock\n", encoding="utf-8")
        (self.repo / "results").mkdir()
        (self.repo / "results" / ".keep").write_text("", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "first")
        self.git("branch", "-M", "main")
        self.git("remote", "add", "origin", str(origin))
        self.git("push", "-q", "origin", "main")
        self.git("fetch", "-q", "origin")
        self.origin = origin

    def tearDown(self):
        import shutil
        shutil.rmtree(self.origin, ignore_errors=True)
        super().tearDown()

    def push_all(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        self.git("push", "-q", "origin", "main")
        self.git("fetch", "-q", "origin")

    def failed(self, **kw):
        args = {"env": dict(self.GOOD_ENV), "power": "ac", "ledger_path": self.repo / "ledger.jsonl", **kw}
        return [n for n, ok, _ in F.preflight_checks(self.repo, **args) if not ok]

    def test_pass_case(self):
        self.assertEqual(self.failed(), [])

    def test_dirty_tree_fails(self):
        (self.repo / "stray.txt").write_text("x", encoding="utf-8")                      # untracked, not ignored
        bad = self.failed()
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("clean tree"))
        (self.repo / "stray.txt").unlink()
        (self.repo / ".gitignore").write_text("ledger.jsonl\nledger.jsonl.lock\nchanged\n", encoding="utf-8")        # a modified tracked file
        self.assertTrue(self.failed()[0].startswith("clean tree"))

    def test_missing_key_fails_and_the_value_is_never_printed(self):
        env = {k: v for k, v in self.GOOD_ENV.items() if k != "OPENAI_API_KEY"}
        bad = self.failed(env=env)
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("OPENAI_API_KEY"))
        shown = " ".join(f"{n} {d}" for n, _, d in F.preflight_checks(self.repo, env=dict(self.GOOD_ENV), power="ac", ledger_path=self.repo / "ledger.jsonl"))
        self.assertNotIn("sk-test-not-a-real-key", shown)
        (self.repo / ".gitignore").write_text("ledger.jsonl\nledger.jsonl.lock\n.env\n", encoding="utf-8")
        (self.repo / ".env").write_text("OPENAI_API_KEY=sk-from-env-file-9999999\n", encoding="utf-8")       # a key in the repo-root .env counts
        self.push_all("ignore env")
        self.assertEqual(self.failed(env=env), [])
        shown = " ".join(f"{n} {d}" for n, _, d in F.preflight_checks(self.repo, env=env, power="ac", ledger_path=self.repo / "ledger.jsonl"))
        self.assertNotIn("sk-from-env-file", shown)

    def test_wrong_pythonhashseed_fails(self):
        for value in ("1", None):
            env = dict(self.GOOD_ENV)
            if value is None:
                del env["PYTHONHASHSEED"]
            else:
                env["PYTHONHASHSEED"] = value
            self.assertEqual(self.failed(env=env), ["PYTHONHASHSEED=0"])
        self.assertEqual(self.failed(env=dict(self.GOOD_ENV, PYTHONUTF8="0")), ["PYTHONUTF8=1"])

    def test_existing_freeze_folder_or_eval_folder_fails(self):
        (self.repo / "results" / "freeze_2026-11-27").mkdir()
        (self.repo / "results" / "freeze_2026-11-27" / ".keep").write_text("", encoding="utf-8")
        self.push_all("folder")
        self.assertEqual(self.failed(), ["no real freeze folder (results/freeze_<YYYY-MM-DD>) exists"])
        (self.repo / "results" / "eval_7919").mkdir()
        (self.repo / "results" / "eval_7919" / ".keep").write_text("", encoding="utf-8")
        self.push_all("eval")
        self.assertEqual(len(self.failed()), 2)

    def test_battery_fails(self):
        self.assertEqual(self.failed(power="battery"), ["AC power"])

    def test_ledger_headroom_fails_when_the_caps_do_not_fit_under_the_platform_limit(self):
        import sys
        sys.path.insert(0, str(F.REPO / "options"))
        from llm_ledger import Ledger
        led = Ledger(self.repo / "ledger.jsonl", lifetime_cap_usd=45.0)
        rid = led.reserve(17.0, run="x", provider="openai", model="m", run_cap_usd=20.0)
        led.settle(rid, 17.0)                                                          # 17 + caps 14 = 31 > 30
        bad = self.failed()
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith("ledger headroom"))

    def test_a_cap_below_a_worst_case_gate_fails(self):
        from unittest import mock
        with mock.patch.dict(F.B_CAPS["real"], {"decoycheck": 0.50}):
            bad = self.failed()
        self.assertEqual(bad, ["real-run cap covers the worst-case gate: decoycheck (300 pairs)"])

    def test_an_unpushed_head_fails(self):
        (self.repo / "more.txt").write_text("x", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "not pushed")
        self.assertEqual(self.failed(), ["HEAD is pushed (an ancestor of origin/main, as of the last fetch)"])

    def test_it_is_read_only(self):
        import hashlib

        def snapshot():
            kept = [p for p in sorted(self.repo.rglob("*")) if p.relative_to(self.repo).parts[0] != ".git"]
            files = {str(p.relative_to(self.repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in kept if p.is_file()}
            dirs = [str(p.relative_to(self.repo)) for p in kept if p.is_dir()]
            return files, dirs, hashlib.sha256((self.repo / ".git" / "index").read_bytes()).hexdigest()

        before = snapshot()
        F.preflight_checks(self.repo, env=dict(self.GOOD_ENV), power="ac", ledger_path=self.repo / "ledger.jsonl")
        self.assertEqual(snapshot(), before)                                           # no file, no folder, the git index untouched

    def test_the_command_exits_1_on_any_fail_and_0_when_all_pass(self):
        import contextlib
        import io
        from unittest import mock
        with mock.patch.object(F, "preflight_checks", return_value=[("a", True, "ok"), ("b", False, "bad")]), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(F.cmd_preflight(None), 1)
        self.assertIn("FAIL  b: bad", out.getvalue())
        with mock.patch.object(F, "preflight_checks", return_value=[("a", True, "ok")]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(F.cmd_preflight(None), 0)

    def test_manifest_notes_the_preflight_command(self):
        notes = F.Runner("rehearsal", 42, self.repo / "run", self.repo / "run" / "_freeze", allow_battery=True, repo=F.REPO, frozen=[],
                         stages=[], call=no_call, log=self.logs.append, ledger_path=self.repo / "ledger.jsonl").write_manifest()["notes"]
        self.assertIn("preflight command added (read-only)", notes)


class SeedRefusals(Base):
    def real(self, approve):
        return self.runner(mode="real", seed=F.EVAL_SEED, approve=approve, run_dir=self.repo / "results" / "eval_7919",
                           freeze_dir=self.repo / "results" / "freeze")

    def test_real_run_refused_without_approval(self):
        R = self.real(approve=False)
        with self.assertRaises(F.Guard) as cm:
            F.s1_generate(R)
        self.assertIn("--approve-seed-7919", str(cm.exception))
        self.assertFalse(R.run_dir.exists())
        self.assertEqual([g["guard"] for g in self.guards_fired(R)], ["--approve-seed-7919 given"])

    def test_real_run_refused_when_the_run_folder_exists(self):
        R = self.real(approve=True)
        R.run_dir.mkdir(parents=True)
        with self.assertRaises(F.Guard) as cm:
            F.s1_generate(R)
        self.assertIn("the run folder does not exist", str(cm.exception))
        self.assertEqual(list(R.run_dir.iterdir()), [])

    def test_real_run_refused_when_a_generation_record_exists(self):
        R = self.real(approve=True)
        R.freeze_dir.mkdir(parents=True)
        (R.freeze_dir / "generation_record.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(F.Guard) as cm:
            F.s1_generate(R)
        self.assertIn("no earlier generation record", str(cm.exception))

    def test_rehearsal_refuses_seed_7919(self):
        ns = argparse.Namespace(rehearsal=True, dev_seed=F.EVAL_SEED, approve_seed_7919=False, allow_battery=False)
        with self.assertRaises(F.Guard) as cm:
            F.runner_from_args(ns)
        self.assertIn("7919", str(cm.exception))
        R = self.runner(seed=F.EVAL_SEED)                       # and inside the stage, if a Runner were built by hand
        with self.assertRaises(F.Guard):
            F.s1_generate(R)
        self.assertFalse((R.run_dir / "orgs_raw.csv").exists())

    def test_rehearsal_refuses_the_approval_flag_and_other_seeds(self):
        with self.assertRaises(F.Guard):
            F.runner_from_args(argparse.Namespace(rehearsal=True, dev_seed=42, approve_seed_7919=True, allow_battery=False))
        with self.assertRaises(F.Guard):
            F.runner_from_args(argparse.Namespace(rehearsal=True, dev_seed=1234, approve_seed_7919=False, allow_battery=False))

    def test_rehearsal_refuses_a_run_folder_with_data(self):
        R = self.runner()
        R.run_dir.mkdir(parents=True)
        (R.run_dir / "orgs_raw.csv").write_text("record_id\n", encoding="utf-8")
        with self.assertRaises(F.Guard) as cm:
            F.s1_generate(R)
        self.assertIn("holds no data", str(cm.exception))

    def test_real_mode_refuses_a_development_seed(self):
        R = self.runner(mode="real", seed=42, approve=True, run_dir=self.repo / "r", freeze_dir=self.repo / "f")
        with self.assertRaises(F.Guard):
            F.s1_generate(R)


class ManifestCheck(Base):
    def test_hash_mismatch_stops_the_run_before_the_stage(self):
        called = []
        R = self.runner(stages=[(0, "zero", lambda R: called.append(0) or {}), (1, "one", lambda R: called.append(1) or {})])
        R.write_manifest()
        R.run(only=0)
        self.assertEqual(called, [0])
        (self.repo / "frozen.txt").write_text("v2\n", encoding="utf-8")
        with self.assertRaises(F.Guard) as cm:
            R.run(only=1)
        self.assertIn("frozen.txt", str(cm.exception))
        self.assertEqual(called, [0])                           # stage 1 never started
        self.assertFalse(R.done_file(1).exists())
        self.assertEqual(self.guards_fired(R)[-1]["guard"], "frozen files unchanged since the manifest")
        self.assertIn("guard fired", (R.freeze_dir / "deviations.md").read_text(encoding="utf-8"))

    def test_missing_frozen_file_stops_the_run(self):
        R = self.runner(stages=[(0, "zero", lambda R: {})])
        R.write_manifest()
        (self.repo / "frozen.txt").unlink()
        with self.assertRaises(F.Guard):
            R.run()

    def test_rehearsal_writes_its_manifest_and_records_the_hashes(self):
        R = self.runner(stages=[(0, "zero", lambda R: {})])
        R.run()
        m = json.loads(R.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(m["files"]["frozen.txt"], F.sha256_file(self.repo / "frozen.txt"))
        self.assertEqual(json.loads(R.done_file(0).read_text(encoding="utf-8"))["manifest_sha256"], F.sha256_file(R.manifest_path))

    def test_real_run_needs_a_manifest(self):
        R = self.runner(mode="real", seed=F.EVAL_SEED, run_dir=self.repo / "r", freeze_dir=self.repo / "f", stages=[(0, "zero", lambda R: {})])
        with self.assertRaises(F.Guard) as cm:
            R.run()
        self.assertIn("no manifest", str(cm.exception))


class StageOrderAndResume(Base):
    def make(self, fail_at=None):
        self.called: list[int] = []
        self.fail_at = fail_at

        def stage(n):
            def fn(R):
                self.called.append(n)
                if self.fail_at == n:
                    raise F.StageFailed(f"stage {n} broke")
                out = R.run_dir / f"out_{n}.txt"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(f"stage {n}\n", encoding="utf-8")
                return {"outputs": [out]}
            return fn
        return self.runner(stages=[(n, f"s{n}", stage(n)) for n in range(4)])

    def test_stages_run_in_order_and_write_done_files(self):
        R = self.make()
        R.run()
        self.assertEqual(self.called, [0, 1, 2, 3])
        for n in range(4):
            d = json.loads(R.done_file(n).read_text(encoding="utf-8"))
            self.assertEqual(d["exit"], 0)
            self.assertEqual(list(d["outputs"].values()), [F.sha256_file(R.run_dir / f"out_{n}.txt")])

    def test_a_stage_needs_every_earlier_stage_done(self):
        R = self.make()
        with self.assertRaises(F.Guard) as cm:
            R.run(only=2)
        self.assertIn("needs stages [0, 1]", str(cm.exception))
        self.assertEqual(self.called, [])

    def test_a_second_run_needs_resume(self):
        R = self.make()
        R.run()
        with self.assertRaises(F.Guard) as cm:
            R.run()
        self.assertIn("--resume", str(cm.exception))

    def test_resume_after_a_failure_skips_the_done_stages(self):
        R = self.make(fail_at=2)
        with self.assertRaises(F.StageFailed):
            R.run()
        self.assertEqual(self.called, [0, 1, 2])
        self.assertFalse(R.done_file(2).exists())
        self.called.clear()
        self.fail_at = None
        R.run(resume=True)
        self.assertEqual(self.called, [2, 3])
        dev = (R.freeze_dir / "deviations.md").read_text(encoding="utf-8")
        self.assertIn("failed: stage 2 broke", dev)
        self.assertIn("resume", dev)

    def test_resume_stops_if_a_done_stage_output_changed(self):
        R = self.make(fail_at=3)
        with self.assertRaises(F.StageFailed):
            R.run()
        (R.run_dir / "out_1.txt").write_text("edited\n", encoding="utf-8")
        self.called.clear()
        self.fail_at = None
        with self.assertRaises(F.Guard) as cm:
            R.run(resume=True)
        self.assertIn("out_1.txt", str(cm.exception))
        self.assertEqual(self.called, [])

    def test_one_stage_rerun_with_stage_flag(self):
        R = self.make()
        R.run()
        self.called.clear()
        R.run(only=3)
        self.assertEqual(self.called, [3])


class TruthSealing(Base):
    def setUp(self):
        super().setUp()
        self.run_dir = self.repo / "results" / "run"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "orgs_raw_truth.csv").write_text("record_id,true_entity_id\nR1,E1\n", encoding="utf-8")

    def test_truth_is_sealed_outside_truth_steps_and_restored_at_the_end(self):
        seen = []

        def call(args, env, log_file):
            seen.append((args[-1], (self.run_dir / "orgs_raw_truth.csv").exists(), (self.run_dir / "orgs_raw_truth.csv.sealed").exists()))
            return 0, ""

        def fn(R):
            R.step(0, [F.PY, "a.py", "no-truth"])
            R.step(0, [F.PY, "b.py", "truth"], truth=True)
            R.step(0, [F.PY, "c.py", "no-truth-again"])
            return {"outputs": [R.truth]}
        R = self.runner(stages=[(0, "zero", fn)], call=call)
        R.run()
        self.assertEqual(seen, [("no-truth", False, True), ("truth", True, False), ("no-truth-again", False, True)])
        self.assertTrue(R.truth.exists())
        self.assertFalse(R.sealed.exists())
        d = json.loads(R.done_file(0).read_text(encoding="utf-8"))
        self.assertEqual(d["reads_truth"], ["b.py"])
        self.assertIn("results/run/orgs_raw_truth.csv", d["outputs"])

    def test_restore_after_a_simulated_crash(self):
        R = self.runner(stages=[(0, "zero", lambda R: {}), (1, "one", lambda R: {})])
        R.run(only=0)
        R.seal()                                                # a crash while sealed: the process dies, no finally runs
        self.assertFalse(R.truth.exists())
        R2 = self.runner(stages=R.stages)
        R2.run(only=1)
        self.assertTrue(R2.truth.exists())
        self.assertFalse(R2.sealed.exists())
        self.assertIn("truth restored", (R2.freeze_dir / "deviations.md").read_text(encoding="utf-8"))

    def test_an_exception_inside_a_stage_still_leaves_the_truth_unsealed(self):
        def fn(R):
            R.seal()
            raise F.StageFailed("boom")
        R = self.runner(stages=[(0, "zero", fn)])
        with self.assertRaises(F.StageFailed):
            R.run()
        self.assertTrue(R.truth.exists())

    def test_resume_verifies_the_truth_hash_whichever_name_it_has(self):
        def fn(R):
            R.seal()
            return {"outputs": [R.truth]}
        R = self.runner(stages=[(0, "zero", fn), (1, "one", lambda R: (_ for _ in ()).throw(F.StageFailed("stop")))])
        with self.assertRaises(F.StageFailed):
            R.run()
        R.stages = [(0, "zero", fn), (1, "one", lambda R: {})]
        R.run(resume=True)                                      # no guard: the truth file matches under its unsealed name
        (R.truth).write_text("record_id,true_entity_id\nR1,E2\n", encoding="utf-8")
        R.done_file(1).unlink()
        with self.assertRaises(F.Guard):
            R.run(resume=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

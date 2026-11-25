"""
Tests for a B stage in which every answer comes from the pair cache (0 live calls, so no calls.jsonl and no prompt-cache
hit ratio): the decoy-check report and the sample report print "n/a (no live calls)" instead of crashing, and the freeze
runner's stage 7 archives the decisions and the stage's cache entries when there is no calls.jsonl.

The B run is the real run_final_sample with the fake provider of options/test_b_final.py (no API, no key): a first run
fills the cache, a second run over the same pairs is answered entirely from it.

  python eval/test_all_cached.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "options"))
sys.path.insert(0, str(HERE))
import b_llm_baseline as B  # noqa: E402
import freeze_run as F  # noqa: E402
import test_b_final as TB  # noqa: E402

NA = "n/a (no live calls)"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class AllCachedBase(TB.Base):
    N = 12
    LABEL = "cached"

    def setUp(self):
        super().setUp()
        first, _ = self.run_final(out="first")                       # live, fills the cache
        self.assertEqual(first, 0)
        self.assertTrue((self.dir / "first" / "calls.jsonl").exists())
        self.prov.requests.clear()
        second, _ = self.run_final(out="second")                     # every answer from the cache
        self.assertEqual(second, 0)
        self.assertEqual(self.prov.requests, [], "the second run must make no call")
        self.out = self.dir / "second"
        self.assertFalse((self.out / "calls.jsonl").exists(), "an all-cached run writes no calls.jsonl")
        meta = json.loads((self.out / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["calls"], meta["prompt_cache"]["hit_ratio_excl_first"]), (0, None))

    def make_run(self, decoy: bool) -> Path:
        """A run folder around the all-cached output: truth sidecar, split, records, classification, sample."""
        run = self.dir / "run"
        dest = run / "options" / "b_llm" / "fake" / self.LABEL
        shutil.copytree(self.out, dest)
        recs = list(self.rd.records.items())
        write_csv(run / "orgs_raw.csv", ["record_id", "name", "city"], [[k, v["name"], v["city"]] for k, v in recs])
        write_csv(run / "split_records.csv", ["record_id", "side"], [[k, "test"] for k, _ in recs])
        truth = []
        for i in range(self.N):
            a, b = f"A{i:04d}", f"B{i:04d}"
            same = (i % 2 == 1) and not decoy                      # odd pairs share the first word: true matches (not in the decoy set)
            truth += [[a, f"E{i:04d}", "clean", f"G{i}" if decoy else ""], [b, f"E{i:04d}" if same else f"F{i:04d}", "standard", f"G{i}" if decoy else ""]]
        write_csv(run / "orgs_raw_truth.csv", ["record_id", "true_entity_id", "variants", "decoy_group"], truth)
        (run / "verity_data").mkdir(parents=True, exist_ok=True)
        (run / "verity_data" / "column_classification.json").write_text(json.dumps({"record_id_column": "record_id", "content_columns": ["name", "city"]}), encoding="utf-8")
        shutil.copyfile(self.sample, run / "options" / "b_llm" / "test_sample5000.csv")
        return run

    def run_report(self, script: str, run: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(HERE / script), str(run), "--label", self.LABEL, "--provider", "fake"], capture_output=True, text=True,
                              encoding="utf-8", env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"})


class ReportTests(AllCachedBase):
    def test_decoy_check_report_prints_na_for_the_missing_hit_ratio(self):
        run = self.make_run(decoy=True)
        r = self.run_report("decoy_check_report.py", run)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
        report = (run / "options" / "b_llm" / "fake" / self.LABEL / "decoy_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"hit ratio excluding each key's first call {NA}", report)
        self.assertIn(f"calls 0, cache hits {self.N}", report)

    def test_sample_report_prints_na_for_the_missing_hit_ratio(self):
        run = self.make_run(decoy=False)
        r = self.run_report("b_sample_report.py", run)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
        report = (run / "options" / "b_llm" / "fake" / self.LABEL / "sample_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"hit ratio overall {NA}, excluding each key's first call {NA}", report)
        self.assertIn(f"calls 0 (log lines), cache hits {self.N}", report)

    def test_a_live_run_still_prints_numbers(self):
        run = self.make_run(decoy=False)
        shutil.copyfile(self.dir / "first" / "calls.jsonl", run / "options" / "b_llm" / "fake" / self.LABEL / "calls.jsonl")
        meta_p = run / "options" / "b_llm" / "fake" / self.LABEL / "run_meta.json"
        meta = json.loads((self.dir / "first" / "run_meta.json").read_text(encoding="utf-8"))
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        r = self.run_report("b_sample_report.py", run)
        self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
        text = (run / "options" / "b_llm" / "fake" / self.LABEL / "sample_report.txt").read_text(encoding="utf-8")
        self.assertNotIn(NA, text)
        self.assertRegex(text, r"hit ratio overall 0\.\d{3}, excluding each key's first call 0\.\d{3}")


class ArchiveTests(AllCachedBase):
    def setUp(self):
        super().setUp()
        self.repo = self.dir / "repo"
        self.run_dir = self.repo / "results" / "rehearsal_x"
        self.freeze = self.run_dir / "_freeze"
        fs = self.run_dir / "options" / "b_llm"
        fs.mkdir(parents=True)
        (fs / "fewshot.json").write_text("[]", encoding="utf-8")
        self.sample_copy = fs / "test_sample5000.csv"
        shutil.copyfile(self.sample, self.sample_copy)
        self.R = F.Runner("rehearsal", 42, self.run_dir, self.freeze, repo=self.repo, frozen=[], stages=[], call=lambda *a: (0, ""), log=lambda *_: None)

    def patches(self, bdir):
        fake_rd = mock.Mock(cols=self.rd.cols, records=self.rd.records)
        return [mock.patch.object(F, "install_assets", lambda R, stage, assets: []),
                mock.patch.object(F, "b_final", lambda R, stage, pairs, label: bdir),
                mock.patch.object(B, "make_provider", lambda name, model, live: self.prov),
                mock.patch.object(B, "RunData", lambda run: fake_rd),
                mock.patch.object(B, "build_system_prompt", lambda fewshot, cols: TB.SYSTEM),
                mock.patch.object(B, "CACHE_DIR", self.cache.path.parent),
                mock.patch.object(B, "load_sample_pairs", lambda path, band: self.pairs)]

    def run_stage7(self, bdir):
        # the fake provider's cache file is named after the provider: point PairCache at the test cache
        shutil.copyfile(self.cache.path, self.cache.path.parent / "fake.jsonl") if self.cache.path.name != "fake.jsonl" else None
        ps = self.patches(bdir)
        for p in ps:
            p.start()
            self.addCleanup(p.stop)
        return F.s7_b(self.R)

    def test_no_calls_jsonl_archives_decisions_and_cache_entries(self):
        res = self.run_stage7(self.out)
        arch = self.freeze / "b_archive"
        info = json.loads((arch / "archive.json").read_text(encoding="utf-8"))
        self.assertEqual(info["mode"], "cache_entries")
        self.assertEqual(info["archive_file"], "cache_entries.jsonl.gz")
        self.assertIn(f"{self.N} distinct prompts", info["what"])
        self.assertTrue((arch / "decisions.csv").exists())
        self.assertFalse((arch / "calls.jsonl.gz").exists())
        with gzip.open(arch / "cache_entries.jsonl.gz", "rt", encoding="utf-8") as f:
            entries = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(entries), self.N)
        self.assertEqual({e["value"]["decision"] for e in entries} <= {"same", "different"}, True)
        self.assertIn(arch / "cache_entries.jsonl.gz", res["outputs"])
        self.assertTrue(any("no calls.jsonl" in n and "cache entries" in n for n in self.R.current_notes), self.R.current_notes)
        self.assertTrue(any("n/a (no live calls)" in n for n in self.R.current_notes), self.R.current_notes)

    def test_with_calls_jsonl_the_archive_is_unchanged(self):
        res = self.run_stage7(self.dir / "first")
        arch = self.freeze / "b_archive"
        info = json.loads((arch / "archive.json").read_text(encoding="utf-8"))
        self.assertEqual((info["mode"], info["archive_file"]), ("calls.jsonl", "calls.jsonl.gz"))
        with gzip.open(arch / "calls.jsonl.gz", "rb") as f:
            self.assertEqual(f.read(), (self.dir / "first" / "calls.jsonl").read_bytes())
        self.assertFalse((arch / "cache_entries.jsonl.gz").exists())
        self.assertIn(arch / "calls.jsonl.gz", res["outputs"])

    def test_missing_cache_entries_stop_the_archive(self):
        (self.cache.path.parent / "fake.jsonl").write_text("", encoding="utf-8")
        for p in self.patches(self.out)[2:]:
            p.start()
            self.addCleanup(p.stop)
        with self.assertRaises(F.StageFailed) as cm:
            F.write_cache_entries(self.R, self.sample_copy, self.freeze / "x.gz")
        self.assertIn("cache entries missing", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

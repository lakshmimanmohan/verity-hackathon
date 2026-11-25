"""
Tests for eval/verify_freeze.py on a hand-built git repo: one file whose manifest hash is its LF blob, one whose manifest hash is the CRLF form of an
LF blob (a Windows checkout), one stored with CRLF; a changed file that must fail, a wrong tag, a wrong recorded HEAD, and a fresh clone with
different line-ending settings.

  python eval/test_verify_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_freeze as VF  # noqa: E402

MANIFEST = "results/freeze_2026-11-25/freeze_manifest.json"
A_BLOB = b"one\ntwo\n"                      # LF blob; the manifest holds the hash of this as it is
B_BLOB = b"x\ny\n"                          # LF blob; the manifest holds the hash of its CRLF form (a Windows working tree)
C_BLOB = b"p\r\nq\r\n"                      # a blob stored with CRLF; the manifest holds the hash of this as it is


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class RepoFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        for k, v in (("user.email", "t@example.org"), ("user.name", "t"), ("core.autocrlf", "false")):
            self.git("config", k, v)
        self.write("a.txt", A_BLOB)
        self.write("b.txt", B_BLOB)
        self.write("c.txt", C_BLOB)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "files")
        self.c1 = self.git("rev-parse", "HEAD").strip()
        self.files = {"a.txt": h(A_BLOB), "b.txt": h(VF.to_crlf(B_BLOB)), "c.txt": h(C_BLOB)}

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, cwd=None):
        r = subprocess.run(["git", *args], cwd=cwd or self.repo, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def write(self, rel: str, data: bytes) -> None:
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def commit_manifest(self, files=None, recorded_head=None, tag_at="manifest", tag="freeze-2026-11-25"):
        doc = {"mode": "real", "git_commit": recorded_head or self.c1, "tag": "freeze-2026-11-25", "params": {"freeze_date": "2026-11-25"}, "files": files or self.files}
        self.write(MANIFEST, json.dumps(doc).encode("utf-8"))
        self.git("add", MANIFEST)
        self.git("commit", "-q", "-m", "manifest")
        self.c2 = self.git("rev-parse", "HEAD").strip()
        if tag_at:
            self.git("tag", tag, self.c2 if tag_at == "manifest" else self.c1)

    def run_verify(self, repo=None):
        lines: list[str] = []
        ok, n, total = VF.verify(repo or self.repo, say=lines.append)
        return ok, n, total, "\n".join(lines)


class VerifyTests(RepoFixture):
    def test_lf_blob_crlf_form_and_crlf_blob_all_verify_and_say_which_matched(self):
        self.commit_manifest()
        ok, n, total, out = self.run_verify()
        self.assertTrue(ok)
        self.assertEqual((n, total), (3, 3))
        self.assertIn("a.txt: matches the committed blob as it is", out)
        self.assertIn("b.txt: matches the committed blob with LF converted to CRLF", out)
        self.assertIn("c.txt: matches the committed blob as it is", out)         # its blob already has CRLF: the blob itself matched
        self.assertIn("VERIFIED: 3 of 3", out)
        self.assertIn("= the manifest commit", out)
        self.assertIn("equals the first parent", out)

    def test_a_changed_file_fails(self):
        files = dict(self.files)
        files["a.txt"] = h(b"one\nCHANGED\n")                                           # the manifest hash is of other content than the committed blob
        self.commit_manifest(files=files)
        ok, n, total, out = self.run_verify()
        self.assertFalse(ok)
        self.assertEqual((n, total), (2, 3))
        self.assertIn("FAIL  a.txt: manifest", out)
        self.assertIn("FAILED: 2 of 3", out)

    def test_a_working_tree_edit_after_the_manifest_does_not_matter_but_a_committed_edit_of_the_blob_before_it_would(self):
        self.commit_manifest()
        self.write("a.txt", b"edited after the freeze\n")                              # the working tree is not what is verified
        self.assertTrue(self.run_verify()[0])
        self.git("add", "a.txt")
        self.git("commit", "-q", "-m", "later change")                                  # a later commit does not change the blob at the manifest commit
        self.assertTrue(self.run_verify()[0])

    def test_a_tag_on_the_wrong_commit_fails(self):
        self.commit_manifest(tag_at="parent")
        ok, _, _, out = self.run_verify()
        self.assertFalse(ok)
        self.assertIn("(NOT the manifest commit)", out)
        self.assertIn("FAILED", out)

    def test_a_missing_tag_fails(self):
        self.commit_manifest(tag_at=None)
        ok, _, _, out = self.run_verify()
        self.assertFalse(ok)
        self.assertIn("does not exist in this checkout", out)

    def test_a_recorded_head_that_is_not_the_parent_fails(self):
        self.commit_manifest(recorded_head="0" * 40)
        ok, _, _, out = self.run_verify()
        self.assertFalse(ok)
        self.assertIn("is NOT the first parent", out)

    def test_a_file_missing_at_the_manifest_commit_fails(self):
        files = dict(self.files)
        files["gone.txt"] = h(b"x")
        self.commit_manifest(files=files)
        ok, n, total, out = self.run_verify()
        self.assertFalse(ok)
        self.assertEqual((n, total), (3, 4))
        self.assertIn("gone.txt: not in the manifest commit", out)

    def test_no_manifest_fails(self):
        ok, n, total, out = self.run_verify()
        self.assertFalse(ok)
        self.assertIn("no single results/freeze_<date>/freeze_manifest.json", out)

    def test_a_fresh_clone_with_crlf_checkout_settings_verifies_too(self):
        self.commit_manifest()
        clone = self.root / "clone"
        subprocess.run(["git", "-c", "core.autocrlf=true", "clone", "-q", str(self.repo), str(clone)], check=True, capture_output=True)
        self.assertTrue((clone / "a.txt").exists())
        ok, n, total, out = self.run_verify(clone)
        self.assertTrue(ok, out)
        self.assertEqual((n, total), (3, 3))

    def test_the_command_line_exit_codes(self):
        self.commit_manifest()
        r = subprocess.run([sys.executable, str(HERE / "verify_freeze.py"), "--repo", str(self.repo)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("VERIFIED: 3 of 3", r.stdout)
        self.git("tag", "-f", "freeze-2026-11-25", self.c1)
        r = subprocess.run([sys.executable, str(HERE / "verify_freeze.py"), "--repo", str(self.repo)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAILED", r.stdout)


class HelperTests(unittest.TestCase):
    def test_to_crlf_and_groups(self):
        self.assertEqual(VF.to_crlf(b"a\nb\n"), b"a\r\nb\r\n")
        self.assertEqual(VF.to_crlf(b"a\r\nb\n"), b"a\r\nb\r\n")                         # no double conversion
        self.assertEqual(VF.group("protocol.md"), "protocol and requirements")
        self.assertEqual(VF.group("options/c_features.py"), "C recipe")
        self.assertEqual(VF.group("results/dev_2026_run11/options/d_hybrid/selection_rule_v2.txt"), "D rule and settings files (frozen copies)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

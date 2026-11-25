"""
Verify the freeze from the git history alone, so it works from a fresh checkout on another machine, line endings included.

  python eval/verify_freeze.py [--repo .] [--manifest results/freeze_<date>/freeze_manifest.json]

What it checks:
  1. The manifest commit: the commit that added the manifest file (git log --diff-filter=A on the manifest path). The manifest itself is read from that
     commit (git show), not from the working tree.
  2. The manifest's recorded HEAD (its "git_commit", the commit that was checked out when the manifest was written) is the first parent of the
     manifest commit.
  3. The tag named in the manifest (default freeze-<freeze_date>) points at the manifest commit.
  4. For each frozen file: take the committed blob at the manifest commit (git show <commit>:<path>), compute the sha256 of the blob and of the blob with
     LF converted to CRLF (a Windows checkout with autocrlf), and mark the file verified if either equals the manifest hash; it prints which one matched.
Prints VERIFIED or FAILED with a count; exit code 0 for VERIFIED, 1 for FAILED. It only reads git objects; it writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(repo: Path, *args: str) -> tuple[int, bytes]:
    r = subprocess.run(["git", "--no-optional-locks", *args], cwd=repo, capture_output=True)
    return r.returncode, r.stdout


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def to_crlf(blob: bytes) -> bytes:
    return blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def group(path: str) -> str:
    if path.startswith("results/dev_2026_run11/options/b_llm/"):
        return "B prompt assets (frozen copies)"
    if path.startswith("results/dev_2026_run11/options/d_hybrid/"):
        return "D rule and settings files (frozen copies)"
    if path in ("protocol.md", "requirements.txt", "verity/requirements.txt"):
        return "protocol and requirements"
    if path.startswith("verity/"):
        return "Verity source"
    if path.startswith("data/"):
        return "data generator"
    if path in ("options/c_decision_model.py", "options/c_features.py"):
        return "C recipe"
    if path == "options/d_hybrid.py":
        return "D"
    if path == "options/a_threshold.py":
        return "A"
    if path.startswith("options/"):
        return "B and shared option code"
    if path.startswith("eval/"):
        return "eval and runbook scripts"
    if path.startswith("cost/"):
        return "cost model"
    return "other"


def find_manifest(repo: Path) -> str | None:
    rc, out = git(repo, "ls-files", "results/freeze_????-??-??/freeze_manifest.json")
    paths = [p for p in out.decode("utf-8").splitlines() if p.strip()]
    return paths[0] if len(paths) == 1 else None


def verify(repo: Path, manifest_path: str | None = None, say=print) -> tuple[bool, int, int]:
    """Run all checks; returns (verified, files verified, files total)."""
    ok = True
    manifest_path = manifest_path or find_manifest(repo)
    if not manifest_path:
        say("FAIL  no single results/freeze_<date>/freeze_manifest.json is tracked in this checkout")
        return False, 0, 0
    # 1. the manifest commit: the commit that added the manifest file
    rc, out = git(repo, "log", "--diff-filter=A", "--format=%H", "--", manifest_path)
    added = out.decode("utf-8").split()
    if rc != 0 or len(added) != 1:
        say(f"FAIL  the manifest {manifest_path} was not added by exactly one commit (found {len(added)})")
        return False, 0, 0
    mc = added[0]
    say(f"manifest commit: {mc} (the commit that added {manifest_path})")
    rc, blob = git(repo, "show", f"{mc}:{manifest_path}")
    if rc != 0:
        say("FAIL  cannot read the manifest at its commit")
        return False, 0, 0
    manifest = json.loads(blob.decode("utf-8"))
    files = manifest["files"]
    # 2. the recorded HEAD is the first parent of the manifest commit
    rc, out = git(repo, "rev-parse", f"{mc}^")
    parent = out.decode("utf-8").strip() if rc == 0 else None
    rec = manifest.get("git_commit")
    good = parent is not None and parent == rec
    ok &= good
    say(f"{'OK  ' if good else 'FAIL'}  recorded HEAD {rec} {'equals' if good else 'is NOT'} the first parent of the manifest commit ({parent})")
    # 3. the tag points at the manifest commit
    tag = manifest.get("tag") or f"freeze-{manifest['params']['freeze_date']}"
    rc, out = git(repo, "rev-list", "-n", "1", tag)
    target = out.decode("utf-8").strip() if rc == 0 else None
    good = target == mc
    ok &= good
    say(f"{'OK  ' if good else 'FAIL'}  tag {tag} " + (f"points at {target}" + (" = the manifest commit" if good else " (NOT the manifest commit)") if target else "does not exist in this checkout"))
    # 4. every frozen file
    n_ok = 0
    for path, want in files.items():
        rc, content = git(repo, "show", f"{mc}:{path}")
        if rc != 0:
            ok = False
            say(f"FAIL  {path}: not in the manifest commit [{group(path)}]")
            continue
        h_lf, h_crlf = sha256(content), sha256(to_crlf(content))
        if h_lf == want:
            n_ok += 1
            say(f"OK    {path}: matches the committed blob as it is [{group(path)}]")
        elif h_crlf == want:
            n_ok += 1
            say(f"OK    {path}: matches the committed blob with LF converted to CRLF [{group(path)}]")
        else:
            ok = False
            say(f"FAIL  {path}: manifest {want[:16]}..., blob {h_lf[:16]}..., blob as CRLF {h_crlf[:16]}... [{group(path)}]")
    ok &= n_ok == len(files)
    say(f"{'VERIFIED' if ok else 'FAILED'}: {n_ok} of {len(files)} frozen files match the manifest; "
        f"manifest commit {mc}; tag {tag} {'at the manifest commit' if target == mc else 'NOT at the manifest commit'}")
    return ok, n_ok, len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the freeze from the git history alone.")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--manifest", default=None, help="repo-relative path of the manifest (default: the one tracked results/freeze_<date>/freeze_manifest.json)")
    args = ap.parse_args()
    ok, _, _ = verify(args.repo.resolve(), args.manifest)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

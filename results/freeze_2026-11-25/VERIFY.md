# Verifying the freeze

Run from any checkout of this repository that has the git history and the tag `freeze-2026-11-25` (a normal clone has both):

```
python eval/verify_freeze.py
```

It reads only git objects and writes nothing. Exit code 0 means VERIFIED, 1 means FAILED.

What it checks:

1. **The manifest commit.** It is the commit that added `results/freeze_2026-11-25/freeze_manifest.json`. The manifest itself is read from that commit, not from the working tree.
2. **The recorded HEAD.** The manifest's `git_commit` (the commit that was checked out when the manifest was written) must be the first parent of the manifest commit.
3. **The tag.** `freeze-2026-11-25` must point at the manifest commit.
4. **Every frozen file** (41 in the manifest). For each file it takes the committed blob at the manifest commit (`git show <commit>:<path>`), computes the sha256 of the blob and of the blob with LF converted to CRLF, and marks the file verified if either equals the hash in the manifest. The manifest hashes the working-tree bytes of the machine that wrote it, so a file can match as it is or in its CRLF form; the output says which one matched. This makes the check independent of the line-ending settings of the checkout.

The last line reads `VERIFIED: 41 of 41 frozen files match the manifest ...` or `FAILED: ...`.

Not frozen (and not checked): `README.md`, `CLAUDE.md` and everything outside the manifest's 41 files. `protocol.md` and `requirements.txt` are frozen.

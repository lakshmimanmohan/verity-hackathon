"""
The freeze run: one rehearsable runbook from "generate the data" to "results table", every parameter frozen in a
manifest beforehand.

  python eval/freeze_run.py write-manifest                       # real run: write results/freeze_2026-12-04/freeze_manifest.json
  (commit the manifest)                                          # then
  python eval/freeze_run.py tag                                  # create the local tag freeze-2026-12-04 at the manifest commit
  python eval/freeze_run.py run --approve-seed-7919              # Dec 4: stages 0 to 13 into results/eval_7919
  python eval/freeze_run.py run --approve-seed-7919 --resume     # after a stop: done stages are skipped
  python eval/freeze_run.py run --approve-seed-7919 --stage 9    # one stage (every earlier stage must be done)

  python eval/freeze_run.py run --rehearsal --dev-seed 42        # the rehearsal: same stages and code paths on a dev seed,
                                                                 # into results/rehearsal_42 (manifest written automatically)
  python eval/freeze_run.py hand-check --run results/eval_7919 [--score]    # after the freeze, not part of the run

Stages (each writes <freeze>/stages/stage_<n>.done.json with its outputs' sha256, time, ledger cost and the guards it
checked; the frozen files are re-hashed against the manifest at every stage start and the run stops on any mismatch):
   0 preflight      mode and seed, manifest present (real: committed), clean tree, AC power, key present, ledger
   1 generate       data/generate_data_org.py; data_manifest.json (seed, rows, sha256 of the CSV and the truth file)
   2 verity         exact search, PYTHONHASHSEED=0, PYTHONUTF8=1, fresh data folder
   3 data profile   eval/data_profile.py                                                         (reads truth)
   4 checks         check_scored_pairs, band.py (truth), evaluate_matches (truth), split.py, band_summary.json,
                    make_sample.py (truth, selection only)
   5 option A       a_threshold.py; scored on the full band and on the sample                    (scoring reads truth)
   6 option C       c_decision_model.py, the c_v1_lgbm recipe unchanged (train labels only); scored
   7 option B       the frozen prompt assets installed and verified; --final-sample; the run archived (gzipped calls.jsonl)
   8 option D       frozen v2 lo / hi (and the original rule as the secondary row); routed pairs; B on routed pairs not in the sample; exact full-band decisions; the
                    fallback-off case (C's cutoff inside or outside [lo, hi)); latency bench (one core)
   9 decoy check    every test band decoy pair (stops if more than 300)
  10 scoring        B on the sample; paired comparisons with B as reference; error reports        (reads truth)
  11 bars table     eval/bars_table.py (verdicts by the protocol rules)
  12 cost model     cost/cost_model.py --run <run>; the effort log read as it is (its hash printed); Verity timing from the
                    dev repeats (flagged) with this run's own pipeline time printed beside it as a spot check
  13 results        eval/results_table.py; freeze_report.txt
Truth sealing: outside the steps that read truth, the truth file is renamed orgs_raw_truth.csv.sealed, so an accidental
read fails; a run that starts and finds it sealed (a crash) restores it and records that in deviations.md.
Guards that fire stop the run and are written to <freeze>/guards.jsonl and deviations.md. Seed 7919 needs
--approve-seed-7919, is refused in rehearsal mode, and is refused when the run folder or a generation record exists.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
FREEZE_DATE = "2026-12-04"
TAG = f"freeze-{FREEZE_DATE}"
EVAL_SEED = 7919
DEV_SEEDS = (42, 2026)
DEV_RUN_REL = "results/dev_2026_run11"
REAL_RUN_REL = "results/eval_7919"
REAL_FREEZE_REL = f"results/freeze_{FREEZE_DATE}"
SAMPLE_SIZE = 5000
REHEARSAL_NONMATCHES = 60
DECOY_MAX = 300
B_CAPS = {"real": {"sample5000": 10.0, "dfull": 0.30, "decoycheck": 0.50},
          "rehearsal": {"sample5000": 0.75, "dfull": 0.10, "decoycheck": 0.15}}       # rehearsal total: $1.00
ARCHIVE_COMMIT_MAX_BYTES = 1_000_000
CHANGE_DATE = "2026-11-25"                       # written into the manifest's notes (all-cached B stage fix)
# frozen assets: run-relative destination -> repo-relative source (copied into the run folder and verified)
ASSETS_B = {"options/b_llm/fewshot.json": f"{DEV_RUN_REL}/options/b_llm/fewshot.json",
            "options/b_llm/frozen_prompt.json": f"{DEV_RUN_REL}/options/b_llm/frozen_prompt.json"}
# D has two frozen settings: v2 (the one that counts) and the original (lo 0.5, hi 0.9, the secondary row "D (original rule)").
# Both rule files and both settings files are frozen and hashed in the manifest.
ASSETS_D = {"options/d_hybrid/selection_rule.txt": f"{DEV_RUN_REL}/options/d_hybrid/selection_rule.txt",
            "options/d_hybrid/frozen_settings.json": f"{DEV_RUN_REL}/options/d_hybrid/frozen_settings.json",
            "options/d_hybrid/selection_rule_v2.txt": f"{DEV_RUN_REL}/options/d_hybrid/selection_rule_v2.txt",
            "options/d_hybrid/frozen_settings_v2.json": f"{DEV_RUN_REL}/options/d_hybrid/frozen_settings_v2.json"}
FROZEN_FILES = [
    "protocol.md", "requirements.txt", "verity/requirements.txt", "data/generate_data_org.py",
    "options/b_system_prompt.txt", "options/b_prompt_appendix.txt", "options/prices.json", "options/b_llm_baseline.py",
    "options/llm_cache.py", "options/llm_ledger.py", "options/a_threshold.py", "options/c_decision_model.py", "options/c_features.py",
    "options/d_hybrid.py", "options/latency_bench.py",
    "eval/freeze_run.py", "eval/data_profile.py", "eval/check_scored_pairs.py", "eval/band.py", "eval/evaluate_matches.py", "eval/split.py",
    "eval/make_sample.py", "eval/score_option.py", "eval/compare_options.py", "eval/error_bounds.py", "eval/b_sample_report.py",
    "eval/d_report.py", "eval/d_full_report.py", "eval/make_decoy_check.py", "eval/decoy_check_report.py", "eval/bars_table.py",
    "eval/results_table.py", "eval/hand_check.py", "cost/cost_model.py",
] + sorted(set(ASSETS_B.values()) | set(ASSETS_D.values()))
REHEARSAL_CANNOT_TEST = [
    "seed 7919's own data: its learned threshold, band size, true-pair count and decoy count",
    "whether C retrained on 7919 moves its calibrated probabilities so that D's routed share differs from the dev 0.085%",
    "the 52-minute B stage at 5,000 pairs: standby, network, rate limits and the cache hit ratio over that length",
    "the real --approve-seed-7919, run-folder and generation-record refusals (unit tests only) and the tag on a committed manifest",
    "C's training with the 7919 train band (a different number of true train pairs)",
    "the platform's monthly limit reset in December",
    "quality: seed 42 is a development seed, so the rehearsal numbers are not evidence",
]


class Guard(SystemExit):
    """A guard that fired: the run stops."""


class StageFailed(RuntimeError):
    """A step exited non-zero."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


class Runner:
    def __init__(self, mode: str, seed: int, run_dir: Path, freeze_dir: Path, *, approve: bool = False, allow_battery: bool = False,
                 repo: Path = REPO, frozen: list[str] | None = None, stages: list | None = None, call=None, log=print,
                 ledger_path: Path | None = None):
        assert mode in ("real", "rehearsal")
        self.mode, self.seed, self.run_dir, self.freeze_dir = mode, seed, Path(run_dir), Path(freeze_dir)
        self.approve, self.allow_battery, self.repo = approve, allow_battery, Path(repo)
        self.frozen = frozen if frozen is not None else FROZEN_FILES
        self.stages = stages if stages is not None else STAGES
        self.call = call or self._call
        self.log = log
        self.ledger_path = ledger_path
        self.pip_freeze = True
        self.truth =self.run_dir / "orgs_raw_truth.csv"
        self.sealed = self.run_dir / "orgs_raw_truth.csv.sealed"
        self.manifest_path = self.freeze_dir / "freeze_manifest.json"
        self.current_truth_steps: list[str] = []
        self.current_guards: list[dict] = []
        self.current_notes: list[str] = []

    # ------------------------------------------------------------------ records
    def _append(self, name: str, text: str) -> None:
        self.freeze_dir.mkdir(parents=True, exist_ok=True)
        with (self.freeze_dir / name).open("a", encoding="utf-8") as f:
            f.write(text)

    def deviation(self, stage, event: str, detail: str) -> None:
        path = self.freeze_dir / "deviations.md"
        if not path.exists():
            self._append("deviations.md", f"# Deviations ({self.mode}, seed {self.seed})\n\nEvery stage failure, guard that fired, restore after a "
                                          "crash, resume and rerun is listed here with its time. Nothing here is edited by hand afterwards.\n\n")
        self._append("deviations.md", f"- {utc()}  stage {stage}  {event}: {detail}\n")

    def guard(self, stage, name: str, ok: bool, detail: str) -> None:
        rec = {"time": utc(), "stage": stage, "guard": name, "fired": not ok, "detail": detail}
        self.current_guards.append(rec)
        self._append("guards.jsonl", json.dumps(rec) + "\n")
        if not ok:
            self.deviation(stage, f"guard fired ({name})", detail)
            raise Guard(f"stage {stage}: {name}: {detail}")

    def note(self, text: str) -> None:
        self.current_notes.append(text)
        self.log(f"    {text}")

    # ------------------------------------------------------------------ truth sealing
    def recover_seal(self) -> bool:
        if self.sealed.exists() and not self.truth.exists():
            self.sealed.rename(self.truth)
            self.deviation("start", "truth restored", "the truth file was left sealed (a crash or an interrupted run); restored on start")
            return True
        return False

    def seal(self) -> None:
        if self.truth.exists():
            self.truth.rename(self.sealed)

    def unseal(self) -> None:
        if self.sealed.exists() and not self.truth.exists():
            self.sealed.rename(self.truth)

    # ------------------------------------------------------------------ manifest
    def write_manifest(self) -> dict:
        files = {rel: sha256_file(self.repo / rel) for rel in self.frozen}
        try:
            head = git(self.repo, "rev-parse", "HEAD").strip()
        except Exception:                                    # not a git checkout (tests)
            head = None
        params = manifest_params(self)
        pf = ""
        if self.pip_freeze:
            try:
                pf = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
            except Exception:
                pass
        doc = {"created_utc": utc(), "mode": self.mode, "seed": self.seed, "run_dir": self.rel(self.run_dir), "freeze_dir": self.rel(self.freeze_dir),
               "git_commit": head, "files": files, "params": params, "python": sys.version.split()[0], "platform": platform.platform(),
               "pip_freeze_sha256": hashlib.sha256(pf.encode()).hexdigest(), "tag": TAG if self.mode == "real" else None,
               "not_frozen": ["cost/effort_log.csv (read as it is by the cost stage; its hash is printed)"],
               "notes": [f"reporting and driver scripts changed on {CHANGE_DATE} to handle an all-cached B stage; no model-side asset changed",
                         "cost model reads B per-call figures from the dev run when a stage made no live calls",
                         f"option D routing rule v2 (selection_rule_v2.txt, frozen_settings_v2.json) replaced the original rule as D's setting on {CHANGE_DATE}; "
                         "the original rule (lo 0.5, hi 0.9) is reported as the secondary row 'D (original rule)'; both rule files and both settings files are hashed here"]}
        self.freeze_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        return doc

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            raise Guard(f"no manifest at {self.manifest_path}: run 'python eval/freeze_run.py write-manifest' first")
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def check_manifest(self, stage) -> None:
        m = self.manifest()
        changed = [rel for rel, h in m["files"].items() if not (self.repo / rel).exists() or sha256_file(self.repo / rel) != h]
        self.guard(stage, "frozen files unchanged since the manifest", not changed,
                   "all match" if not changed else "changed: " + ", ".join(changed[:8]) + (" ..." if len(changed) > 8 else ""))

    def rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.repo.resolve()).as_posix()
        except ValueError:
            return str(p)

    # ------------------------------------------------------------------ steps
    def _call(self, args: list[str], env: dict, log_file: Path) -> tuple[int, str]:
        out_lines = []
        with log_file.open("a", encoding="utf-8") as lf, subprocess.Popen(args, cwd=self.repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                                      text=True, encoding="utf-8", errors="replace") as proc:
            for line in proc.stdout:
                lf.write(line)
                out_lines.append(line)
            rc = proc.wait()
        return rc, "".join(out_lines)

    def step(self, stage, args: list, truth: bool = False, env: dict | None = None, label: str | None = None, save: Path | None = None) -> str:
        if truth:
            self.unseal()
            self.current_truth_steps.append(label or Path(str(args[1])).name)
        else:
            self.seal()
        full = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"}
        for k, v in (env or {}).items():
            if v is None:
                full.pop(k, None)
            else:
                full[k] = str(v)
        args = [str(a) for a in args]
        logs = self.freeze_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_file = logs / f"stage_{stage:02d}.log" if isinstance(stage, int) else logs / f"{stage}.log"
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"\n$ {' '.join(args)}   [{'reads truth' if truth else 'truth sealed'}; {utc()}]\n")
        self.log(f"    $ {' '.join(Path(a).name if i == 1 else a for i, a in enumerate(args))}{'   (reads truth)' if truth else ''}")
        rc, out = self.call(args, full, log_file)
        if save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            save.write_text(out, encoding="utf-8")
        if rc != 0:
            raise StageFailed(f"{label or Path(args[1]).name} exited {rc}; see {self.rel(log_file)}")
        return out

    # ------------------------------------------------------------------ stage bookkeeping
    def done_file(self, n: int) -> Path:
        return self.freeze_dir / "stages" / f"stage_{n:02d}.done.json"

    def resolve(self, p: Path) -> Path:
        """The truth file under whichever name it has now (sealed or not); any other path unchanged."""
        p = Path(p)
        if p.resolve() in (self.truth.resolve(), self.sealed.resolve()):
            return self.truth if self.truth.exists() else self.sealed
        return p

    def is_done(self, n: int) -> bool:
        f = self.done_file(n)
        if not f.exists():
            return False
        doc = json.loads(f.read_text(encoding="utf-8"))
        bad = []
        for rel, h in doc["outputs"].items():
            p = self.resolve(self.repo / rel)
            if not p.exists() or sha256_file(p) != h:
                bad.append(rel)
        self.guard(n, "outputs unchanged since the stage finished", not bad, "all match" if not bad else "changed or missing: " + ", ".join(bad[:5]))
        return True

    def ledger_total(self) -> float:
        try:
            sys.path.insert(0, str(REPO / "options"))
            from llm_ledger import DEFAULT_LEDGER, Ledger
            return Ledger(self.ledger_path or DEFAULT_LEDGER).state().total_micro / 1e6
        except Exception:
            return 0.0

    def run(self, only: int | None = None, resume: bool = False) -> None:
        self.freeze_dir.mkdir(parents=True, exist_ok=True)
        self.recover_seal()
        if self.mode == "rehearsal" and not self.manifest_path.exists():
            self.write_manifest()
            self.log(f"rehearsal manifest written: {self.rel(self.manifest_path)}")
        done = [n for n, *_ in self.stages if self.done_file(n).exists()]
        if only is None:
            if done and not resume:
                raise Guard(f"stages {done} are already done in {self.rel(self.freeze_dir)}: use --resume to continue")
            todo = list(self.stages)
        else:
            missing = [n for n, *_ in self.stages if n < only and n not in done]
            if missing:
                self.guard(only, "stage order", False, f"stage {only} needs stages {missing} done first")
            todo = [s for s in self.stages if s[0] == only]
        if resume and done:
            self.deviation("start", "resume", f"resumed; stages already done: {done}")
        try:
            for n, name, fn in todo:
                if only is None and self.is_done(n):
                    self.log(f"stage {n:>2} {name}: done earlier, outputs verified, skipped")
                    continue
                self.current_truth_steps, self.current_guards, self.current_notes = [], [], []
                self.log(f"stage {n:>2} {name}: start {time.strftime('%H:%M:%S')}")
                self.check_manifest(n)
                t0, c0 = time.time(), self.ledger_total()
                try:
                    res = fn(self) or {}
                except (Guard, StageFailed) as e:
                    if isinstance(e, StageFailed):
                        self.deviation(n, "failed", str(e))
                    raise
                outputs = {}
                for pth in res.get("outputs", []):
                    p = self.resolve(Path(pth))
                    if p.exists():
                        key = self.truth if p == self.sealed else p          # recorded under the unsealed name
                        outputs[self.rel(key)] = sha256_file(p)
                doc = {"stage": n, "name": name, "mode": self.mode, "seed": self.seed, "ended_utc": utc(), "seconds": round(time.time() - t0, 1),
                       "ledger_cost_usd": round(self.ledger_total() - c0, 4), "exit": 0, "reads_truth": self.current_truth_steps,
                       "guards": self.current_guards, "notes": self.current_notes, "outputs": outputs,
                       "manifest_sha256": sha256_file(self.manifest_path)}
                self.done_file(n).parent.mkdir(parents=True, exist_ok=True)
                self.done_file(n).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
                self.log(f"stage {n:>2} {name}: done in {doc['seconds']:.0f} s, ledger cost ${doc['ledger_cost_usd']:.4f}, {len(outputs)} outputs")
        finally:
            self.unseal()                                      # the resting state: the truth file is back under its name


# ====================================================================== helpers for the stages

def manifest_params(R: Runner) -> dict:
    out = {"freeze_date": FREEZE_DATE, "seeds": {"data": R.seed, "split": 0, "sample": 0, "cv_and_model": 0, "bootstrap": 0},
           "sample": f"{SAMPLE_SIZE} pairs" if R.mode == "real" else f"every same-entity test band pair + {REHEARSAL_NONMATCHES} non-matches",
           "b_caps_usd": B_CAPS[R.mode], "decoy_max_pairs": DECOY_MAX, "c_recipe": "options/c_decision_model.py --model lgbm --version c_v1_lgbm "
           "(features, grid, 5-fold grouped CV, isotonic calibration and cutoff rule as in the code; no flags changed)",
           "verity": "--nn-mode exact, PYTHONHASHSEED=0, PYTHONUTF8=1, all threads"}
    try:
        prices = json.loads((R.repo / "options" / "prices.json").read_text(encoding="utf-8"))["models"]
        fz = json.loads((R.repo / ASSETS_B["options/b_llm/frozen_prompt.json"]).read_text(encoding="utf-8"))
        fs = json.loads((R.repo / ASSETS_D["options/d_hybrid/frozen_settings_v2.json"]).read_text(encoding="utf-8"))
        fo = json.loads((R.repo / ASSETS_D["options/d_hybrid/frozen_settings.json"]).read_text(encoding="utf-8"))
        out.update({"model": fz["model"], "reasoning_effort": fz["reasoning_effort"], "max_output_tokens": prices[fz["model"]]["request_settings"]["max_output_tokens"],
                    "system_prompt_sha256": fz["system_prompt_sha256"], "settings_sha256": fz["settings_sha256"], "cache_key_scheme": fz["cache_key_scheme"],
                    "d_lo": fs["lo"], "d_hi": fs["hi"], "d_rule_v2_sha256_lf": fs["rule_file_sha256_lf"],
                    "d_original_lo": fo["lo"], "d_original_hi": fo["hi"], "d_original_rule_sha256_lf": fo["rule_file_sha256_lf"], "pilot_cost_per_call_usd": 0.001545, "platform_limit_usd": 30.0})
    except (OSError, KeyError, json.JSONDecodeError):
        pass                                                   # tests without the real assets
    return out


def lines_minus_header(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def install_assets(R: Runner, stage: int, assets: dict) -> list[Path]:
    m = R.manifest()
    out = []
    for dest_rel, src_rel in assets.items():
        src, dest = R.repo / src_rel, R.run_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        ok = sha256_file(dest) == m["files"].get(src_rel)
        R.guard(stage, f"frozen asset {Path(dest_rel).name} matches the manifest", ok, f"{src_rel} -> {R.rel(dest)}")
        out.append(dest)
    return out


def b_final(R: Runner, stage: int, pairs_file: Path, label: str) -> Path:
    from b_llm_baseline import power_status  # noqa: E402
    if not R.allow_battery:
        R.guard(stage, "AC power", power_status() != "battery", f"power: {power_status()}")
    cap = B_CAPS[R.mode][label]
    base = [PY, "options/b_llm_baseline.py", R.run_dir, "--provider", "openai", "--final-sample", pairs_file, "--label", label, "--max-usd", cap]
    est = R.step(stage, base + ["--estimate-only"], label=f"B {label} estimate")
    for line in est.splitlines():
        if line.startswith(("cost gate", "pairs:", "projection")):
            R.note(line.strip()[:220])
    R.step(stage, base + ["--approve-final-run"], label=f"B {label}")
    return R.run_dir / "options" / "b_llm" / "openai" / label


# ====================================================================== stages

def check_mode(R: Runner, stage) -> None:
    if R.mode == "rehearsal":
        R.guard(stage, "rehearsal never uses the evaluation seed", R.seed != EVAL_SEED, f"seed {R.seed}")
        R.guard(stage, "rehearsal seed is a development seed", R.seed in DEV_SEEDS, f"seed {R.seed}; allowed {DEV_SEEDS}")
    else:
        R.guard(stage, "the real run uses the evaluation seed", R.seed == EVAL_SEED, f"seed {R.seed}")


def s0_preflight(R: Runner) -> dict:
    check_mode(R, 0)
    m = R.manifest()
    head = git(R.repo, "rev-parse", "HEAD").strip()
    if R.mode == "real":
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", R.rel(R.manifest_path)], cwd=R.repo, capture_output=True).returncode == 0
        R.guard(0, "the manifest is committed", tracked, R.rel(R.manifest_path))
        parent = git(R.repo, "rev-parse", "HEAD^").strip()
        R.guard(0, "HEAD is the manifest commit", m["git_commit"] in (head, parent), f"manifest {m['git_commit']}, HEAD {head}")
    else:
        R.guard(0, "HEAD is the commit the rehearsal manifest recorded", m["git_commit"] == head, f"manifest {m['git_commit']}, HEAD {head}")
    skip = (R.rel(R.run_dir).rstrip("/") + "/", R.rel(R.freeze_dir).rstrip("/") + "/")
    dirty = [ln for ln in git(R.repo, "status", "--porcelain", "--untracked-files=all").splitlines() if ln and not ln[3:].strip('"').startswith(skip)]
    R.guard(0, "clean tree (no modified tracked files, no untracked files outside .gitignore)", not dirty, "clean" if not dirty else "; ".join(dirty[:6]))
    sys.path.insert(0, str(REPO / "options"))
    from b_llm_baseline import power_status  # noqa: E402
    pw = power_status()
    R.guard(0, "AC power", R.allow_battery or pw != "battery", f"power: {pw}")
    key_set = False
    env_file = R.repo / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            k, _, v = line.strip().partition("=")
            if k == "OPENAI_API_KEY":
                key_set = bool(v.strip().strip('"\''))
    R.guard(0, "OPENAI_API_KEY is set (value never read out)", key_set, "set" if key_set else "empty or missing")
    total = R.ledger_total()
    R.note(f"ledger total ${total:.4f} before the run")
    eff = R.repo / "cost" / "effort_log.csv"
    R.note(f"effort log (not frozen) sha256 {sha256_file(eff)[:16]}...")
    pre = R.freeze_dir / "preflight.json"
    pre.write_text(json.dumps({"time": utc(), "head": head, "power": pw, "ledger_total_usd": total, "mode": R.mode, "seed": R.seed}, indent=1) + "\n", encoding="utf-8")
    return {"outputs": [pre, R.manifest_path]}


def s1_generate(R: Runner) -> dict:
    check_mode(R, 1)
    if R.mode == "real":
        R.guard(1, "--approve-seed-7919 given", R.approve, "seed 7919 is generated once, only with --approve-seed-7919")
        R.guard(1, "the run folder does not exist", not R.run_dir.exists(), R.rel(R.run_dir))
        R.guard(1, "no earlier generation record", not (R.freeze_dir / "generation_record.json").exists(), "seed 7919 is generated once")
    else:
        has_data = (R.run_dir / "orgs_raw.csv").exists() or (R.run_dir / "verity_data").exists() or R.truth.exists() or R.sealed.exists()
        R.guard(1, "the run folder holds no data", not has_data, R.rel(R.run_dir))
    R.run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = R.run_dir / "orgs_raw.csv"
    R.step(1, [PY, "data/generate_data_org.py", "--seed", R.seed, "--output", csv_path], truth=True, label="generate (writes the truth file)")
    rows = lines_minus_header(csv_path)
    dm = {"seed": R.seed, "rows": rows, "csv": R.rel(csv_path), "csv_sha256": sha256_file(csv_path), "truth": R.rel(R.truth),
          "truth_sha256": sha256_file(R.truth), "time": utc()}
    (R.freeze_dir / "data_manifest.json").write_text(json.dumps(dm, indent=1) + "\n", encoding="utf-8")
    (R.freeze_dir / "generation_record.json").write_text(json.dumps({**dm, "note": "written at generation; never regenerate this seed"}, indent=1) + "\n", encoding="utf-8")
    R.note(f"{rows:,} rows, csv sha256 {dm['csv_sha256'][:16]}..., truth sha256 {dm['truth_sha256'][:16]}...")
    R.seal()
    return {"outputs": [csv_path, R.truth, R.freeze_dir / "data_manifest.json", R.freeze_dir / "generation_record.json"]}


def s2_verity(R: Runner) -> dict:
    data = R.run_dir / "verity_data"
    R.step(2, [PY, "verity/src/main.py", "--nn-mode", "exact", "--input", (R.run_dir / "orgs_raw.csv").resolve()],
           env={"VERITY_DATA_DIR": data, "OMP_NUM_THREADS": None}, label="verity")
    ri = json.loads((data / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    R.guard(2, "exact neighbor search", ri.get("nn_mode_resolved") == "exact", f"resolved {ri.get('nn_mode_resolved')}")
    R.guard(2, "no uncommitted changes in verity/src", ri.get("git_src_uncommitted_changes") is False, f"{ri.get('git_src_uncommitted_changes')}")
    R.guard(2, "PYTHONHASHSEED=0", str(ri.get("pythonhashseed")) == "0", f"{ri.get('pythonhashseed')}")
    total = next((t["seconds"] for t in ri.get("timings", []) if t["stage"] == "verity.total"), None)
    R.note(f"verity.total {total} s; {ri.get('scored_pairs_rows')} scored pairs; commit {ri.get('git_commit', '')[:10]}")
    return {"outputs": [data / "run_info.json", data / "match_candidates.csv", data / "scored_pairs.csv", data / "scored_pairs_columns.json"]}


def s3_profile(R: Runner) -> dict:
    R.step(3, [PY, "eval/data_profile.py", R.run_dir], truth=True, label="data_profile")
    return {"outputs": [R.run_dir / "data_profile.txt", R.run_dir / "data_profile.csv"]}


def s4_checks(R: Runner) -> dict:
    run = R.run_dir
    R.step(4, [PY, "eval/check_scored_pairs.py", run], label="check_scored_pairs", save=run / "check_scored_pairs.txt")
    R.step(4, [PY, "eval/band.py", run, "--truth", R.truth], truth=True, label="band.py", save=run / "band_report.txt")
    R.step(4, [PY, "eval/evaluate_matches.py", "--truth", R.truth, "--matches", run / "verity_data" / "match_candidates.csv"], truth=True,
           label="evaluate_matches", save=run / "evaluator_output.txt")
    R.step(4, [PY, "eval/split.py", run], label="split")
    records = lines_minus_header(run / "orgs_raw.csv")
    csv.field_size_limit(10**8)
    with (run / "verity_data" / "scored_pairs.csv").open(newline="", encoding="utf-8") as f:
        band = sum(1 for r in csv.DictReader(f) if r["decision_status"] == "review")
    (run / "band_summary.json").write_text(json.dumps({"records": records, "band_pairs": band}, indent=1) + "\n", encoding="utf-8")
    R.note(f"band: {band:,} review pairs over {records:,} records ({band / records:.2f} per record)")
    sample = run / "options" / "b_llm" / "test_sample5000.csv"
    size = ["--size", SAMPLE_SIZE] if R.mode == "real" else ["--n-nonmatches", REHEARSAL_NONMATCHES]
    out = R.step(4, [PY, "eval/make_sample.py", run, "--out", sample, *size], truth=True, label="make_sample (selection only)")
    for line in out.splitlines():
        if line.startswith(("test band", "sample", "weights")):
            R.note(line.strip())
    if R.mode == "rehearsal":
        R.note(f"rehearsal: the sample keeps the standard file name test_sample5000.csv and holds {lines_minus_header(sample):,} pairs")
    return {"outputs": [run / "check_scored_pairs.txt", run / "band_report.txt", run / "evaluator_output.txt", run / "split_records.csv",
                        run / "band_summary.json", sample]}


def score(R: Runner, stage: int, decisions: Path, name: str, save: Path, sample: bool = False, json_out: Path | None = None) -> Path:
    args = [PY, "eval/score_option.py", R.run_dir, "--decisions", decisions, "--name", name]
    if sample:
        args += ["--sample", R.run_dir / "options" / "b_llm" / "test_sample5000.csv"]
    if json_out:
        args += ["--json-out", json_out]
    R.step(stage, args, truth=True, label=f"score {name}{' (sample)' if sample else ''}", save=save)
    return save


def s5_a(R: Runner) -> dict:
    d = R.run_dir / "options" / "a_threshold"
    R.step(5, [PY, "options/a_threshold.py", R.run_dir], label="a_threshold")
    outs = [d / "decisions.csv", score(R, 5, d / "decisions.csv", "A. threshold only", d / "score_report.txt"),
            score(R, 5, d / "decisions.csv", "A. threshold only", d / "score_report_sample5000.txt", sample=True)]
    return {"outputs": outs}


def s6_c(R: Runner) -> dict:
    d = R.run_dir / "options" / "c_model" / "c_v1_lgbm"
    R.step(6, [PY, "options/c_decision_model.py", R.run_dir, "--model", "lgbm", "--version", "c_v1_lgbm"], truth=True, label="c_decision_model (train labels only)")
    outs = [d / "decisions.csv", d / "cv_report.txt", d / "features.json", d / "model.txt", d / "calibrator.json",
            score(R, 6, d / "decisions.csv", "C v1 LightGBM", d / "score_report.txt"),
            score(R, 6, d / "decisions.csv", "C v1 LightGBM", d / "score_report_sample5000.txt", sample=True)]
    return {"outputs": outs}


def s7_b(R: Runner) -> dict:
    assets = install_assets(R, 7, ASSETS_B)
    bdir = b_final(R, 7, R.run_dir / "options" / "b_llm" / "test_sample5000.csv", "sample5000")
    arch = R.freeze_dir / "b_archive"
    arch.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bdir / "decisions.csv", arch / "decisions.csv")
    info = {"decisions_sha256": sha256_file(arch / "decisions.csv"), "rule": "commit the gzipped archive file if it is under 1 MB"}
    if (bdir / "calls.jsonl").exists():
        packed = arch / "calls.jsonl.gz"
        with (bdir / "calls.jsonl").open("rb") as src, gzip.open(packed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        info.update({"mode": "calls.jsonl", "what": "gzipped calls.jsonl of the stage"})
    else:                                                    # no live calls: every answer came from the cache
        packed = arch / "cache_entries.jsonl.gz"
        n = write_cache_entries(R, R.run_dir / "options" / "b_llm" / "test_sample5000.csv", packed)
        info.update({"mode": "cache_entries", "what": f"gzipped pair-cache entries of the stage's {n} distinct prompts (no calls.jsonl: 0 live calls)"})
    size = packed.stat().st_size
    commit = size < ARCHIVE_COMMIT_MAX_BYTES
    info.update({"archive_file": packed.name, "archive_gz_bytes": size, "commit_archive": commit})
    (arch / "archive.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    R.note(f"B archive: decisions.csv and {packed.name} ({size:,} bytes; {'under' if commit else 'over'} 1 MB: {'commit it' if commit else 'keep it local'})"
           + ("" if info["mode"] == "calls.jsonl" else "; no calls.jsonl, so the stage's cache entries were archived instead"))
    meta = json.loads((bdir / "run_meta.json").read_text(encoding="utf-8"))
    R.note(f"B: {meta['calls']} calls, cache hits {meta['cache_hits']}, spent ${meta['spent_usd']:.4f}, hit ratio excl. first "
           f"{(meta.get('prompt_cache') or {}).get('hit_ratio_excl_first') if meta['calls'] else 'n/a (no live calls)'}")
    return {"outputs": assets + [bdir / "decisions.csv", bdir / "run_meta.json", bdir / "report.txt", arch / "decisions.csv", packed, arch / "archive.json"]}


def write_cache_entries(R: Runner, sample: Path, out: Path) -> int:
    """The pair-cache entries (key, value) for the stage's distinct prompts, gzipped. Keys are rebuilt the way
    options/b_llm_baseline.py --final-sample builds them (same provider settings, same system prompt, same record text)."""
    sys.path.insert(0, str(REPO / "options"))
    import b_llm_baseline as B  # noqa: E402
    from llm_cache import PairCache, sha256_text  # noqa: E402
    provider = B.make_provider("openai", None, live=False)
    rd = B.RunData(R.run_dir)
    fewshot = json.loads((R.run_dir / "options" / "b_llm" / "fewshot.json").read_text(encoding="utf-8"))
    system = B.build_system_prompt(fewshot, rd.cols)
    pairs = B.load_sample_pairs(sample, None)
    items = B.make_items(rd, pairs, provider, system, provider.settings_sha(), sha256_text(system))
    cache = PairCache(B.CACHE_DIR / f"{provider.name}.jsonl")
    seen: dict = {}
    for it in items:
        seen.setdefault(it.key, cache.get(it.key))
    missing = [k[:12] for k, v in seen.items() if v is None]
    if missing:
        raise StageFailed(f"{len(missing)} cache entries missing for the archive, e.g. {missing[:3]}")
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for k, v in seen.items():
            f.write(json.dumps({"key": k, "value": v}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(seen)


def s8_d(R: Runner) -> dict:
    run = R.run_dir
    dd = run / "options" / "d_hybrid"
    assets = install_assets(R, 8, ASSETS_D)
    R.step(8, [PY, "options/d_hybrid.py", "list-routed", run, "--setting", "original"], label="d_hybrid list-routed (original rule)")
    n_orig = lines_minus_header(dd / "routed_fullband_pairs_original.csv")
    R.step(8, [PY, "options/d_hybrid.py", "list-routed", run, "--setting", "v2"], label="d_hybrid list-routed (v2)")
    routed = dd / "routed_fullband_pairs.csv"
    n_routed = lines_minus_header(routed)
    R.note(f"{n_routed} test band pairs routed to the LLM by v2; {n_orig} by the original rule")
    if n_routed:
        b_final(R, 8, routed, "dfull")
    else:
        R.note("no routed pairs: the B dfull call is skipped")
    R.step(8, [PY, "options/d_hybrid.py", "full", run, "--setting", "v2"], label="d_hybrid full (v2)")
    fs = json.loads((dd / "frozen_settings_v2.json").read_text(encoding="utf-8"))
    lo, hi = fs["lo"], fs["hi"]
    cutoff = float(json.loads((run / "options" / "c_model" / "c_v1_lgbm" / "features.json").read_text(encoding="utf-8"))["cutoff"])
    inside = lo <= cutoff < (hi if hi is not None else float("inf"))
    R.step(8, [PY, "options/d_hybrid.py", "full", run, "--setting", "v2", "--fallback", "off"], label="d_hybrid full --fallback off (v2)")
    chk_path = dd / "fallback_off_check.json"
    chk = json.loads(chk_path.read_text(encoding="utf-8"))
    if not inside and not chk.get("scored"):
        score(R, 8, dd / "decisions_fullband_fallback_off.csv", "D, LLM fallback off (full band)", dd / "score_report_fallback_off.txt",
              json_out=dd / "score_fallback_off.json")
        chk["scored"] = "score_report_fallback_off.txt"
    case = ("inside [lo, hi): identity with C verified" if inside and chk["identical_to_c"] else
            "inside [lo, hi) but not identical to C: scored" if inside else "outside [lo, hi): computed and scored")
    chk.update({"c_cutoff": cutoff, "lo": lo, "hi": hi, "cutoff_inside_lo_hi": inside, "case": case})
    chk_path.write_text(json.dumps(chk, indent=1) + "\n", encoding="utf-8")
    R.note(f"fallback off: C's cutoff {cutoff:.4f}, [lo, hi) = [{lo}, {hi}): {case}; {chk['differing_from_c']} pairs differ from C")
    R.step(8, [PY, "options/d_hybrid.py", "route", run, "--lo", lo, "--hi", "none" if hi is None else hi, "--out", dd / "decisions.csv"], label="d_hybrid route (sample)")
    outs = [score(R, 8, dd / "decisions_fullband.csv", "D. hybrid, exact full test band", dd / "score_report_fullband.txt", json_out=dd / "score_fullband.json"),
            score(R, 8, dd / "decisions.csv", "D. hybrid (C + LLM)", dd / "score_report_sample5000.txt", sample=True)]
    # the secondary row: the original rule, produced and scored the same way (its routed pairs lie inside v2's, so B has answered them all)
    fo = json.loads((dd / "frozen_settings.json").read_text(encoding="utf-8"))
    R.step(8, [PY, "options/d_hybrid.py", "full", run, "--setting", "original"], label="d_hybrid full (original rule)")
    R.step(8, [PY, "options/d_hybrid.py", "route", run, "--lo", fo["lo"], "--hi", "none" if fo["hi"] is None else fo["hi"], "--out", dd / "decisions_original.csv"],
           label="d_hybrid route (sample, original rule)")
    outs += [score(R, 8, dd / "decisions_fullband_original.csv", "D (original rule), exact full test band", dd / "score_report_fullband_original.txt",
                   json_out=dd / "score_fullband_original.json"),
             score(R, 8, dd / "decisions_original.csv", "D (original rule) (C + LLM)", dd / "score_report_sample5000_original.txt", sample=True)]
    R.note(f"original rule: lo {fo['lo']}, hi {fo['hi']}: {n_orig} routed pairs; scored as the secondary row")
    R.step(8, [PY, "options/latency_bench.py", run], env={"OMP_NUM_THREADS": "1"}, label="latency_bench (one core)")
    return {"outputs": assets + [routed, dd / "routed_fullband_pairs_original.csv", dd / "decisions_fullband.csv", dd / "decisions_fullband_original.csv", chk_path,
                                 dd / "decisions.csv", dd / "decisions_original.csv", run / "options" / "latency" / "latency_summary.json"] + outs}


def s9_decoy(R: Runner) -> dict:
    run = R.run_dir
    R.step(9, [PY, "eval/make_decoy_check.py", run], truth=True, label="make_decoy_check (selection only)")
    pairs = run / "options" / "b_llm" / "decoy_check_pairs.csv"
    n = lines_minus_header(pairs)
    R.guard(9, f"decoy pairs at most {DECOY_MAX}", n <= DECOY_MAX, f"{n} pairs")
    if n == 0:
        R.note("no decoy pairs in the test band: nothing to send")
        return {"outputs": [pairs]}
    bdir = b_final(R, 9, pairs, "decoycheck")
    R.step(9, [PY, "eval/decoy_check_report.py", run, "--label", "decoycheck"], truth=True, label="decoy_check_report")
    return {"outputs": [pairs, bdir / "decisions.csv", bdir / "decoy_report.txt"]}


def s10_scoring(R: Runner) -> dict:
    run = R.run_dir
    o = run / "options"
    bdec = o / "b_llm" / "openai" / "sample5000" / "decisions.csv"
    adec, cdec, ddec = o / "a_threshold" / "decisions.csv", o / "c_model" / "c_v1_lgbm" / "decisions.csv", o / "d_hybrid" / "decisions.csv"
    dodec = o / "d_hybrid" / "decisions_original.csv"
    outs = [score(R, 10, bdec, "B. frontier LLM (frozen prompt)", o / "b_llm" / "openai" / "sample5000" / "score_report_sample5000.txt", sample=True)]
    sample = o / "b_llm" / "test_sample5000.csv"
    for first, second, n1, n2, save in ((bdec, adec, "B (frontier LLM)", "A (threshold only)", o / "compare" / "b_vs_a.txt"),
                                        (bdec, cdec, "B (frontier LLM)", "C v1 LightGBM", o / "compare" / "b_vs_c_v1_lgbm.txt"),
                                        (cdec, bdec, "C v1 LightGBM", "B (frontier LLM)", o / "compare" / "c_v1_lgbm_vs_b.txt"),
                                        (ddec, bdec, "D (hybrid)", "B (frontier LLM)", o / "d_hybrid" / "compare_d_vs_b.txt"),
                                        (ddec, cdec, "D (hybrid)", "C v1 LightGBM", o / "d_hybrid" / "compare_d_vs_c.txt"),
                                        (dodec, bdec, "D (original rule)", "B (frontier LLM)", o / "d_hybrid" / "compare_d_original_vs_b.txt"),
                                        (ddec, dodec, "D (hybrid)", "D (original rule)", o / "d_hybrid" / "compare_d_vs_d_original.txt")):
        R.step(10, [PY, "eval/compare_options.py", run, "--sample", sample, "--first", first, "--second", second, "--first-name", n1, "--second-name", n2],
               truth=True, label=f"compare {n1} vs {n2}", save=save)
        outs.append(save)
    R.step(10, [PY, "eval/b_sample_report.py", run, "--label", "sample5000"], truth=True, label="b_sample_report")
    R.step(10, [PY, "eval/d_report.py", run], truth=True, label="d_report")
    R.step(10, [PY, "eval/d_full_report.py", run], truth=True, label="d_full_report")
    outs += [o / "b_llm" / "openai" / "sample5000" / "sample_report.txt", o / "d_hybrid" / "d_report.txt", o / "d_hybrid" / "d_fullband_report.txt"]
    return {"outputs": outs}


def s11_bars(R: Runner) -> dict:
    out = R.step(11, [PY, "eval/bars_table.py", R.run_dir], label="bars_table")
    for line in out.splitlines()[3:9]:
        R.note(line.rstrip()[:200])
    return {"outputs": [R.run_dir / "bars_table.csv", R.run_dir / "bars_table.txt"]}


def s12_cost(R: Runner) -> dict:
    eff = R.repo / "cost" / "effort_log.csv"
    R.note(f"effort log read as it is (not frozen): sha256 {sha256_file(eff)}")
    R.step(12, [PY, "cost/cost_model.py", "--run", R.run_dir, "--out", R.run_dir / "cost", "--dev-run", R.repo / DEV_RUN_REL], label="cost_model")
    ri = json.loads((R.run_dir / "verity_data" / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
    own = next((t["seconds"] for t in ri.get("timings", []) if t["stage"] == "verity.total"), None)
    vt = json.loads((R.repo / "results" / "dev_2026_timing_summary" / "verity_timing.json").read_text(encoding="utf-8"))
    dev_med = vt["settings"]["allthreads"]["stage"]["verity.total"]["median"]
    R.note(f"Verity pipeline time: the cost model uses the dev repeats (flagged); spot check only: this run's verity.total {own} s, "
           f"dev median {dev_med} s (all threads, 3 repeats)")
    cd = R.run_dir / "cost"
    return {"outputs": [cd / "summary.txt", cd / "assumptions.csv", cd / "cost_model.csv", cd / "options_vs_bars.csv", cd / "one_time_cost.csv"]}


def s13_results(R: Runner) -> dict:
    R.step(13, [PY, "eval/results_table.py", R.run_dir], label="results_table")
    rep = write_freeze_report(R)
    return {"outputs": [R.run_dir / "results_table.csv", R.run_dir / "results_table.txt", rep]}


def write_freeze_report(R: Runner) -> Path:
    m = R.manifest()
    L = [f"Freeze report ({R.mode}, seed {R.seed}), written {utc()}",
         f"  run folder {R.rel(R.run_dir)}; freeze folder {R.rel(R.freeze_dir)}; manifest sha256 {sha256_file(R.manifest_path)[:16]}...; git commit {m.get('git_commit')}",
         f"  frozen files: {len(m['files'])}; model {m['params'].get('model')}, reasoning effort {m['params'].get('reasoning_effort')}, "
         f"prompt {str(m['params'].get('system_prompt_sha256'))[:12]}, D lo {m['params'].get('d_lo')} hi {m['params'].get('d_hi')} "
         f"(original rule: lo {m['params'].get('d_original_lo')} hi {m['params'].get('d_original_hi')})",
         "", f"  {'stage':<20}{'seconds':>9}{'ledger $':>10}  exit  reads truth"]
    tot_s = tot_c = 0.0
    for n, name, _ in STAGES:
        f = R.done_file(n)
        if not f.exists():
            L.append(f"  {n:>2} {name:<17}   not done")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        tot_s += d["seconds"]
        tot_c += d["ledger_cost_usd"]
        L.append(f"  {n:>2} {name:<17}{d['seconds']:>9.0f}{d['ledger_cost_usd']:>10.4f}  {d['exit']:>4}  {', '.join(d['reads_truth']) or '-'}")
        for note in d["notes"]:
            L.append(f"        {note}")
    L.append(f"  {'total':<20}{tot_s:>9.0f}{tot_c:>10.4f}   (stage 13 itself not counted)")
    gpath = R.freeze_dir / "guards.jsonl"
    guards = [json.loads(x) for x in gpath.read_text(encoding="utf-8").splitlines()] if gpath.exists() else []
    fired = [g for g in guards if g["fired"]]
    L += ["", f"  guards checked: {len(guards)}; fired: {len(fired)}"]
    for g in fired:
        L.append(f"    FIRED stage {g['stage']}: {g['guard']}: {g['detail']}")
    seen = sorted({g["guard"] for g in guards})
    L.append("  guards that passed: " + "; ".join(seen))
    dev = R.freeze_dir / "deviations.md"
    L += ["", "  deviations: " + (str(sum(1 for x in dev.read_text(encoding="utf-8").splitlines() if x.startswith("- "))) + " (deviations.md)" if dev.exists() else "none")]
    arch = R.freeze_dir / "b_archive" / "archive.json"
    if arch.exists():
        a = json.loads(arch.read_text(encoding="utf-8"))
        L.append(f"  B archive: {a['archive_file']} {a['archive_gz_bytes']:,} bytes ({'commit it' if a['commit_archive'] else 'over 1 MB: keep it local'}); {a['what']}")
    if R.mode == "real":
        L.append(f"  tag: {TAG} at the manifest commit (python eval/freeze_run.py tag); to publish it: git push origin {TAG}")
    else:
        L.append("  tag: none in the rehearsal")
        L += ["", "  What the rehearsal could not test:"] + [f"    - {x}" for x in REHEARSAL_CANNOT_TEST]
    out = R.freeze_dir / "freeze_report.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    R.log("\n".join(L))
    return out


STAGES = [(0, "preflight", s0_preflight), (1, "generate", s1_generate), (2, "verity", s2_verity), (3, "data profile", s3_profile),
          (4, "checks", s4_checks), (5, "option A", s5_a), (6, "option C", s6_c), (7, "option B", s7_b), (8, "option D", s8_d),
          (9, "decoy check", s9_decoy), (10, "scoring", s10_scoring), (11, "bars table", s11_bars), (12, "cost model", s12_cost),
          (13, "results", s13_results)]


# ====================================================================== commands

def runner_from_args(args) -> Runner:
    if args.rehearsal:
        if args.approve_seed_7919:
            raise Guard("--approve-seed-7919 is never used with --rehearsal")
        if args.dev_seed == EVAL_SEED:
            raise Guard("seed 7919 is refused in rehearsal mode")
        if args.dev_seed not in DEV_SEEDS:
            raise Guard(f"--dev-seed must be one of {DEV_SEEDS}")
        root = REPO / "results" / f"rehearsal_{args.dev_seed}"
        return Runner("rehearsal", args.dev_seed, root, root / "_freeze", allow_battery=args.allow_battery)
    return Runner("real", EVAL_SEED, REPO / REAL_RUN_REL, REPO / REAL_FREEZE_REL, approve=args.approve_seed_7919, allow_battery=args.allow_battery)


def cmd_tag(args) -> int:
    if args.rehearsal:
        print("rehearsal: no tag is created")
        return 0
    manifest = REPO / REAL_FREEZE_REL / "freeze_manifest.json"
    rel = manifest.relative_to(REPO).as_posix()
    if subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True).returncode != 0:
        raise Guard(f"{rel} is not committed: commit the manifest first")
    if git(REPO, "status", "--porcelain", "--untracked-files=all").strip():
        raise Guard("the tree is not clean")
    if git(REPO, "tag", "--list", TAG).strip():
        print(f"tag {TAG} already exists at {git(REPO, 'rev-list', '-n', '1', TAG).strip()}")
    else:
        git(REPO, "tag", TAG, "HEAD")
        print(f"created local tag {TAG} at {git(REPO, 'rev-parse', 'HEAD').strip()}")
    print(f"to publish it (never --tags): git push origin {TAG}")
    return 0


def cmd_hand_check(args) -> int:
    run = Path(args.run)
    truth, sealed = run / "orgs_raw_truth.csv", run / "orgs_raw_truth.csv.sealed"
    if sealed.exists() and not truth.exists():
        sealed.rename(truth)
    flag = "--score" if args.score else "--sample"
    r = subprocess.run([PY, "eval/hand_check.py", flag, str(run)], cwd=REPO, env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"})
    if not args.score:
        print(f"\nLabel {run / 'hand_check_pairs.csv'} blind (do not open hand_check_key.csv), then: python eval/freeze_run.py hand-check --run {run} --score")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="The freeze run: generate, run, score, price, report.")
    ap.add_argument("command", nargs="?", default="run", choices=("run", "write-manifest", "tag", "hand-check"))
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--dev-seed", type=int, default=42)
    ap.add_argument("--approve-seed-7919", action="store_true")
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--run", default=None, help="hand-check: the run folder")
    ap.add_argument("--score", action="store_true", help="hand-check: score the labelled pairs")
    args = ap.parse_args()
    if args.command == "tag":
        return cmd_tag(args)
    if args.command == "hand-check":
        if not args.run:
            raise SystemExit("hand-check needs --run")
        return cmd_hand_check(args)
    R = runner_from_args(args)
    if args.command == "write-manifest":
        if R.manifest_path.exists():
            raise Guard(f"{R.rel(R.manifest_path)} exists; a manifest is written once")
        R.write_manifest()
        print(f"manifest written: {R.rel(R.manifest_path)}" + ("" if R.mode == "rehearsal" else
              f"\ncommit it, then create the tag: python eval/freeze_run.py tag"))
        return 0
    sys.path.insert(0, str(REPO / "options"))
    from b_llm_baseline import keep_awake  # noqa: E402
    keep_awake(True)
    t0 = time.time()
    try:
        R.run(only=args.stage, resume=args.resume)
    finally:
        keep_awake(False)
    print(f"\nfreeze run ({R.mode}) finished in {(time.time() - t0) / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

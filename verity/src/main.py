"""
Run the end-to-end MDM knowledge-graph pipeline:
learn rules from CSV → record store → embed → match → merge → match explorer HTML.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from column_classification import column_to_attr_type
from config import (
    DATA_DIR_ENV,
    DATA_DIR_MARKER,
    NN_MODE_ENV,
    NN_MODES,
    annoy_installed,
    data_dir,
    nn_mode,
    record_label_from_csv,
    start_run_info,
)

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PROJECT_ROOT = _SRC.parent
DATA_DIR = data_dir()
LEARNED_RULES_MD = DATA_DIR / "learned_rules.md"
RULE_LEARNER_SCRIPT = _SRC / "rule_learner.py"
DEFAULT_CLASSIFICATION_JSON = DATA_DIR / "column_classification.json"
DEFAULT_WEIGHTS_JSON = DATA_DIR / "attribute_weights.json"
DEFAULT_LEARNED_THRESHOLD_JSON = DATA_DIR / "learned_threshold.json"


def display_input_path(path: Path) -> str:
    """Prefer a short path relative to cwd when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def print_phase_banner(name: str, elapsed_s: float) -> None:
    line = "=" * 72
    print(f"\n{line}")
    print(f"  Phase complete: {name}")
    print(f"  Elapsed:      {elapsed_s:.2f}s")
    print(f"{line}\n")


def count_raw_records(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def prepare_pipeline_csv(source: Path) -> tuple[Path, int]:
    """Return (path for all downstream phases, data row count)."""
    with source.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return source, len(rows)


def _csv_fieldnames(delta_csv: Path) -> list[str]:
    with delta_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
    return [str(n) for n in names if n is not None and str(n).strip() != ""]


def _fieldnames_ci_lookup(fieldnames: list[str]) -> dict[str, str]:
    """Lowercase header → original spelling from the CSV."""
    out: dict[str, str] = {}
    for name in fieldnames:
        key = str(name).strip().lower()
        if key and key not in out:
            out[key] = str(name).strip()
    return out


def _classification_schema_columns_ci(classification: dict[str, Any]) -> set[str]:
    known: set[str] = set()
    rid = classification.get("record_id_column")
    if rid is not None and str(rid).strip():
        known.add(str(rid).strip().lower())
    for col in classification.get("content_columns") or []:
        if str(col).strip():
            known.add(str(col).strip().lower())
    for col in classification.get("metadata_columns") or []:
        if str(col).strip():
            known.add(str(col).strip().lower())
    st = classification.get("semantic_types")
    if isinstance(st, dict):
        for k in st:
            if str(k).strip():
                known.add(str(k).strip().lower())
    return known


def validate_reuse_weights_vs_delta(
    classification: dict[str, Any], delta_csv: Path
) -> None:
    """
    Compare ``content_columns`` from disk classification to delta CSV headers;
    warn on schema drift (missing content columns or unknown new columns).
    """
    fieldnames = _csv_fieldnames(delta_csv)
    ci_lookup = _fieldnames_ci_lookup(fieldnames)
    delta_ci = set(ci_lookup.keys())
    known_ci = _classification_schema_columns_ci(classification)

    for col in classification.get("content_columns") or []:
        c = str(col).strip()
        if not c:
            continue
        if c.lower() not in delta_ci:
            print(
                f"Warning: column {c} was in original schema but missing "
                "from delta file — weights may be less accurate"
            )

    for ci in sorted(delta_ci):
        if ci not in known_ci:
            orig = ci_lookup.get(ci, ci)
            print(
                f"Warning: new column {orig} found in delta — run without "
                "--reuse-weights to reclassify schema"
            )


def assert_reuse_weights_artifacts_present() -> None:
    missing: list[str] = []
    if not DEFAULT_CLASSIFICATION_JSON.is_file():
        missing.append(str(DEFAULT_CLASSIFICATION_JSON))
    if not DEFAULT_WEIGHTS_JSON.is_file():
        missing.append(str(DEFAULT_WEIGHTS_JSON))
    if not DEFAULT_LEARNED_THRESHOLD_JSON.is_file():
        missing.append(str(DEFAULT_LEARNED_THRESHOLD_JSON))
    if missing:
        raise SystemExit(
            "--reuse-weights requires existing weight artifacts:\n  "
            + "\n  ".join(missing)
        )


def _fresh_protected_dirs() -> list[Path]:
    """Folders ``--fresh`` must never empty: Verity's own tree and the folder above it."""
    host_root = PROJECT_ROOT.parent
    return [_SRC, PROJECT_ROOT, host_root, host_root / "data"]


def _data_dir_entries(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.name != DATA_DIR_MARKER)


def _write_data_dir_marker(folder: Path) -> None:
    (folder / DATA_DIR_MARKER).write_text(
        "Created by Verity. --fresh may empty this folder.\n", encoding="utf-8"
    )


def prepare_data_dir(*, full_run: bool, fresh: bool, input_csv: Path) -> None:
    """
    Make sure a full run starts from an empty data folder, so no file from an
    earlier run can leak in. Delta runs (--reuse-weights / --incremental) build on
    what is already there and are not checked.
    """
    folder = DATA_DIR
    if folder.exists() and not folder.is_dir():
        raise SystemExit(f"Data folder is not a directory: {folder}")

    if fresh:
        if not full_run:
            raise SystemExit(
                "--fresh cannot be combined with --reuse-weights or --incremental: "
                "delta runs need the existing data folder."
            )
        resolved = folder.resolve()
        for protected in _fresh_protected_dirs():
            p = protected.resolve()
            if resolved == p or resolved in p.parents:
                raise SystemExit(f"--fresh refuses to empty {resolved}: protected folder.")
        if folder.exists():
            if not (folder / DATA_DIR_MARKER).is_file():
                raise SystemExit(
                    f"--fresh refuses to empty {resolved}: it has no {DATA_DIR_MARKER} "
                    "marker, so Verity did not create it. Empty it by hand or set "
                    f"{DATA_DIR_ENV} to a new folder."
                )
            if resolved in input_csv.resolve().parents:
                raise SystemExit(
                    f"--fresh refuses to empty {resolved}: the input CSV is inside it."
                )
            entries = _data_dir_entries(folder)
            for entry in entries:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            print(f"--fresh: removed {len(entries)} item(s) from {resolved}")

    if not folder.exists():
        if not full_run:
            raise SystemExit(
                f"Data folder does not exist: {folder}. A delta run needs the folder "
                "from an earlier full run."
            )
        folder.mkdir(parents=True)
        _write_data_dir_marker(folder)
        print(f"Created data folder: {folder}")
        return

    if not full_run:
        return
    entries = _data_dir_entries(folder)
    if entries:
        shown = "\n  ".join(e.name + ("/" if e.is_dir() else "") for e in entries[:10])
        more = f"\n  … and {len(entries) - 10} more" if len(entries) > 10 else ""
        raise SystemExit(
            f"Data folder is not empty: {folder}\n  {shown}{more}\n"
            "A full run must start from an empty folder so no earlier output leaks in. "
            f"Set {DATA_DIR_ENV} to a new folder, or pass --fresh to empty this one "
            f"(only folders Verity created, marked with {DATA_DIR_MARKER})."
        )
    if not (folder / DATA_DIR_MARKER).is_file():
        _write_data_dir_marker(folder)


def _git_info() -> dict[str, Any]:
    """Commit of the checkout holding Verity, and whether verity/src has uncommitted changes."""
    def _git(*argv: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *argv], cwd=str(PROJECT_ROOT), capture_output=True,
                text=True, check=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", str(_SRC))
    return {
        "git_commit": commit,
        "git_src_uncommitted_changes": None if status is None else bool(status),
    }


def write_run_info(
    *, nn_requested: str, nn_source: str, has_annoy: bool, input_csv: Path, mode: str
) -> None:
    """Start this invocation's entry in run_info.json (match adds resolved mode and n)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    from embed import MODEL_NAME

    try:
        st_version = version("sentence-transformers")
    except PackageNotFoundError:
        st_version = None
    start_run_info(
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        run_mode=mode,
        input_csv=str(input_csv),
        nn_mode_requested=nn_requested,
        nn_mode_source=nn_source,
        annoy_installed=has_annoy,
        nn_mode_resolved=None,
        n_records_in_index=None,
        **_git_info(),
        python_version=platform.python_version(),
        sentence_transformers_version=st_version,
        embedding_model=MODEL_NAME,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
    )


def run_phase(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    print_phase_banner(label, time.perf_counter() - t0)
    return result


def run_rule_learner_on_csv(csv_path: Path) -> None:
    """Invoke rule_learner.py with the given CSV (same interpreter, repo root cwd)."""
    if not csv_path.is_file():
        raise SystemExit(f"Rule learner input missing: {csv_path}")
    if not RULE_LEARNER_SCRIPT.is_file():
        raise SystemExit(f"Rule learner script missing: {RULE_LEARNER_SCRIPT}")
    subprocess.run(
        [sys.executable, str(RULE_LEARNER_SCRIPT), str(csv_path.resolve())],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def print_learned_rules_markdown() -> None:
    """Echo data/learned_rules.md so the user sees learned weights/threshold before matching."""
    if not LEARNED_RULES_MD.is_file():
        print(f"\n(No report file at {LEARNED_RULES_MD})\n")
        return
    text = LEARNED_RULES_MD.read_text(encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"  Contents of {LEARNED_RULES_MD}")
    print("=" * 72)
    print(text)
    if not text.endswith("\n"):
        print()


def print_final_report(merge_stats: dict, n_raw: int) -> None:
    sizes = merge_stats.get("community_sizes") or []
    avg_comm = statistics.mean(sizes) if sizes else 0.0

    print("\n" + "=" * 72)
    print("  PIPELINE SUMMARY")
    print("=" * 72)
    print(f"  Total raw records:           {n_raw}")
    print(f"  Total golden records:        {merge_stats.get('n_golden', 0)}")
    if merge_stats.get("pairwise_eval_skipped"):
        if merge_stats.get("pairwise_eval_skip_reason") == "incremental_merge":
            pr_msg = "(skipped — incremental merge mode)"
        else:
            pr_msg = "(skipped — no ground-truth column in classification)"
        print(f"  Precision / Recall / F1:     {pr_msg}")
    else:
        print(
            "  Precision / Recall / F1:     "
            f"{merge_stats.get('precision', 0.0):.4f} / "
            f"{merge_stats.get('recall', 0.0):.4f} / "
            f"{merge_stats.get('f1', 0.0):.4f}"
        )
    print(f"  Average community size:    {avg_comm:.2f}")

    top = merge_stats.get("top_merged_by_sources") or []
    print("  Top golden records by source-system diversity (most systems):")
    if not top:
        print("    (none)")
    else:
        for rank, row in enumerate(top, start=1):
            print(
                f"    {rank}. golden_id={row.get('golden_id')}  "
                f"systems={row.get('n_sources')}  "
                f"[{row.get('sources_display', '')}]  "
                f"members={row.get('member_count')}"
            )
    print("=" * 72 + "\n")


def _csv_record_ids(pipeline_csv: Path, classification: dict[str, Any]) -> list[str]:
    """Same record ids ``build_graph`` uses (``record_id_column`` on each data row)."""
    from build_graph import row_get_ci

    rid_col = classification["record_id_column"]
    with pipeline_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: list[str] = []
    for r in rows:
        rid = row_get_ci(r, rid_col)
        if not rid:
            raise SystemExit(f"Empty record id for column {rid_col!r}")
        out.append(rid)
    return out


def print_store_load_plan(
    *,
    project_root: Path,
    pipeline_csv: Path,
    source_csv: Path,
    n_records: int,
    record_label: str,
    classification: dict[str, Any],
) -> bool:
    """
    Check the record store for the current record count (``record_label``) and whether this CSV is already loaded.

    Returns True if ``build_graph`` should be skipped (all record ids already in the graph).
    """
    from record_store import open_store

    store = open_store()
    record_ids = _csv_record_ids(pipeline_csv, classification)
    input_name = source_csv.name

    try:
        x = store.record_count(record_label)
        skip_build = bool(record_ids) and store.missing_record_ids(
            record_ids, record_label
        ) == 0
        if skip_build:
            graph_attr_types = store.distinct_attr_types(record_label)
            meta_skip = {
                str(x).strip().lower()
                for x in (classification.get("metadata_columns") or [])
            }
            expected_lower: set[str] = set()
            for col in classification.get("content_columns") or []:
                if str(col).strip().lower() in meta_skip:
                    continue
                expected_lower.add(column_to_attr_type(str(col)).lower())
            if expected_lower - graph_attr_types:
                skip_build = False
                print(
                    "Schema change detected — new columns found, re-running "
                    "build_graph to load new attribute values"
                )
    finally:
        store.close()

    y = n_records
    if skip_build:
        print(
            f"Records from {input_name} already in graph — "
            "skipping build_graph, re-running embed and match"
        )
    else:
        print(f"Record store currently contains {x} existing records")
        print(f"Adding {y} new records from {input_name}")
        print(f"Total after this run: {x + y} records")

    return skip_build


def resolve_input_path(cli_input: Path | None) -> Path:
    if cli_input is not None:
        p = cli_input.expanduser()
        if not p.is_file():
            raise SystemExit(f"Input file not found: {p.resolve()}")
        return p.resolve()

    while True:
        line = input("Enter the path to your input CSV file: ").strip().strip('"')
        if not line:
            print("Please enter a non-empty path.")
            continue
        p = Path(line).expanduser()
        if not p.is_file():
            print(f"File not found: {p.resolve()}")
            continue
        return p.resolve()


def main(
    new_record_ids: set[str] | None = None,
    incremental: bool = False,
) -> None:
    parser = argparse.ArgumentParser(
        description="Run MDM knowledge-graph pipeline on an input customer CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to input CSV (if omitted, you will be prompted)",
    )
    parser.add_argument(
        "--reuse-weights",
        action="store_true",
        help="Skip rule learning; use the weight and classification JSON in the data folder.",
    )
    parser.add_argument(
        "--retrain-weights",
        type=Path,
        default=None,
        metavar="PATH",
        help="Run Phase 1 (rule_learner) on this reference CSV instead of --input.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Delta merge: incremental golden assignment from match_candidates "
            "(stable golden_id); skips full community detection in merge."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Full runs only: empty the data folder first. Refused unless Verity "
            f"created the folder ({DATA_DIR_MARKER} marker)."
        ),
    )
    parser.add_argument(
        "--nn-mode",
        choices=NN_MODES,
        default=None,
        help=(
            f"Neighbor search: exact, approximate (Annoy) or auto. Overrides "
            f"{NN_MODE_ENV}; default auto (Annoy at 10,000+ records when installed)."
        ),
    )
    args = parser.parse_args()
    incremental_requested = bool(incremental) or bool(args.incremental)
    full_run = not args.reuse_weights and not incremental_requested
    # Pin the resolved folder for child processes (rule_learner runs with cwd=verity/,
    # where a relative VERITY_DATA_DIR would resolve differently).
    os.environ[DATA_DIR_ENV] = str(DATA_DIR)
    print(f"Data folder: {DATA_DIR}", flush=True)

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != "0":
        print(
            "\n" + "!" * 72 + "\n"
            f"  WARNING: PYTHONHASHSEED is {hash_seed!r}, not '0'. Scores can shift "
            "between runs\n  (up to 0.0075 seen). Set PYTHONHASHSEED=0 for comparable "
            "results.\n" + "!" * 72 + "\n",
            flush=True,
        )

    if args.nn_mode is not None:
        os.environ[NN_MODE_ENV] = args.nn_mode
        nn_requested, nn_source = args.nn_mode, "--nn-mode"
    else:
        nn_requested, nn_source = nn_mode()
    has_annoy = annoy_installed()
    print(
        f"Neighbor search mode: {nn_requested} (from {nn_source}); "
        f"Annoy installed: {'yes' if has_annoy else 'no'}",
        flush=True,
    )
    if nn_requested == "approximate" and not has_annoy:
        raise SystemExit(
            f"Neighbor search mode 'approximate' (from {nn_source}) needs Annoy, which "
            "is not installed. Install verity/requirements-ann.txt or use --nn-mode exact."
        )

    if args.reuse_weights and args.retrain_weights is not None:
        raise SystemExit(
            "Cannot combine --reuse-weights with --retrain-weights (choose one)."
        )

    if args.reuse_weights:
        print(
            "Weight mode: REUSING existing weights (--reuse-weights)",
            flush=True,
        )
    elif args.retrain_weights is not None:
        ref = args.retrain_weights.expanduser()
        if not ref.is_file():
            raise SystemExit(f"--retrain-weights file not found: {ref.resolve()}")
        print(
            "Weight mode: RETRAINING from reference file (--retrain-weights)",
            flush=True,
        )
        print(f"  Reference CSV: {display_input_path(ref.resolve())}", flush=True)
    else:
        print(
            "Weight mode: LEARNING from delta file (default)",
            flush=True,
        )

    source_csv = resolve_input_path(args.input)
    prepare_data_dir(full_run=full_run, fresh=args.fresh, input_csv=source_csv)
    write_run_info(
        nn_requested=nn_requested,
        nn_source=nn_source,
        has_annoy=has_annoy,
        input_csv=source_csv,
        mode="full" if full_run else "delta",
    )
    pipeline_csv, n_records = prepare_pipeline_csv(source_csv)

    record_label = record_label_from_csv(source_csv)

    print(
        f"Using input file: {display_input_path(source_csv)}  ({n_records} records found)"
    )
    print(f"Record label: {record_label}")

    from build_graph import load_classification_at_start

    classification = load_classification_at_start(pipeline_csv)
    if incremental_requested and new_record_ids is None:
        new_record_ids = set(_csv_record_ids(pipeline_csv, classification))
    skip_build_graph = print_store_load_plan(
        project_root=PROJECT_ROOT,
        pipeline_csv=pipeline_csv,
        source_csv=source_csv,
        n_records=n_records,
        record_label=record_label,
        classification=classification,
    )

    from build_graph import main as run_build_graph
    from embed import main as run_embed
    from explore_matches import run_explore
    from match import main as run_match
    from merge import main as run_merge

    t_all = time.perf_counter()

    t_rules = time.perf_counter()
    if args.reuse_weights:
        assert_reuse_weights_artifacts_present()
        with DEFAULT_CLASSIFICATION_JSON.open(encoding="utf-8") as _cf:
            disk_classification: dict[str, Any] = json.load(_cf)
        validate_reuse_weights_vs_delta(disk_classification, pipeline_csv)
        print(
            "\n(Phase 1 rule_learner skipped — using existing weights "
            "and classification from the data folder.)\n"
        )
        print_phase_banner(
            "1. Learn rules from data (skipped)", time.perf_counter() - t_rules
        )
    else:
        rule_learner_csv = (
            args.retrain_weights.expanduser().resolve()
            if args.retrain_weights is not None
            else pipeline_csv
        )
        run_rule_learner_on_csv(rule_learner_csv)
        print_phase_banner("1. Learn rules from data", time.perf_counter() - t_rules)
    print_learned_rules_markdown()

    if skip_build_graph:
        print("\n(Phase 2 build_graph skipped — data already in the record store.)\n")
    else:
        run_phase(
            "2. build_graph",
            lambda: run_build_graph(
                input_path=pipeline_csv,
                record_label_from_csv_path=source_csv,
            ),
        )
    new_record_ids = run_phase(
        "3. embed",
        lambda: run_embed(
            input_path=pipeline_csv,
            record_label_from_csv_path=source_csv,
        ),
    )
    new_record_ids = {str(rid) for rid in (new_record_ids or set())}
    incremental = bool(new_record_ids) and bool(args.reuse_weights)
    run_phase(
        "4. match",
        lambda: run_match(
            input_path=pipeline_csv,
            record_label_from_csv_path=source_csv,
            new_record_ids=new_record_ids,
        ),
    )

    t_merge = time.perf_counter()
    merge_stats = run_merge(
        input_path=pipeline_csv,
        record_label_from_csv_path=source_csv,
        incremental=incremental,
        new_record_ids=new_record_ids,
    )
    print_phase_banner("5. merge", time.perf_counter() - t_merge)

    def _phase6_explore_matches() -> None:
        try:
            run_explore(input_path=pipeline_csv)
        except Exception as exc:
            print(
                f"\nWarning: explore matches failed ({type(exc).__name__}: {exc}) — "
                "continuing pipeline.\n"
            )
            return
        print(f"Match explorer saved to: {DATA_DIR / 'match_explorer.html'}")
        print("Open this file in your browser to review matches.")

    run_phase("6. explore matches", lambda: _phase6_explore_matches())

    def _phase6b_match_summary() -> None:
        from explore_matches import run_match_summary
        run_match_summary(input_path=pipeline_csv)
        print(f"Match summary saved to: {DATA_DIR / 'match_summary.csv'}")

    run_phase("6b. match summary", lambda: _phase6b_match_summary())

    n_raw = count_raw_records(pipeline_csv)
    print_final_report(merge_stats, n_raw)

    print(f"Total wall time (all phases): {time.perf_counter() - t_all:.2f}s")


if __name__ == "__main__":
    main()

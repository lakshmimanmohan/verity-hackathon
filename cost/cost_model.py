"""
Cost model for the options (A threshold only, B frontier LLM, C classical ML, D hybrid, E SLM) at 100K, 1M and
10M records per day. Reads measured numbers from the repo's result files, never calls an API.

  python cost/cost_model.py [--effort-log-complete] [--run results/dev_2026_run11] [--out cost]

Writes (all under --out, default cost/):
  assumptions.csv     every input, marked measured (with its source file) or assumed
  cost_model.csv      monthly running cost, one-time cost and totals per option, volume, hourly rate, label scenario;
                      cost to Precisely and cost to the customer under BYOK shown separately
  break_even.csv      the volume where each option's total cost over the horizon equals B's
  sensitivity.csv     one factor at a time around the base case
  summary.txt         hours by bucket, ledger classification, key results, plain statements
  chart_monthly_cost.png, chart_break_even.png

Model (all formulas are in the functions below and are tested in cost/test_cost_model.py)
  LLM calls per day   B: records x band pairs per record (measured 20.1; sensitivity 5, 10, 20), every band pair
                      gets a call; D: B's calls x the routed share (measured 38 / 44,882; sensitivity 0.001, 0.01, 0.05, 0.20)
  cost per call       from measured tokens: input x (1 - hit ratio) at the input price + input x hit ratio at the cached
                      price + output at the output price; batch at the listed batch prices
  hosting             core-seconds per day from per-pair CPU time (C, D) and the shared Verity pipeline per record
                      (every option, B included); cost = the cheaper of always-on (whole cores, 50% utilisation) and
                      serverless (core-seconds billed). Hosting prices are ASSUMED.
  operations          hours per month x hourly rate (ASSUMED hours; retraining cadence for C and D)
  one-time            charged to options: the Verity pre-existing build (80 h) and productization (100 h), shared by every
                      option, and labeling for C and D (hours per 1,000 pairs measured from the "Hand check of 100
                      pairs" row). The PoC's own spend (effort-log hours, API spend from the ledger) is NOT charged to
                      any option or scenario: it is the separate block "cost to build the evidence"
  total over horizon  one-time + horizon months x monthly running cost
  who pays            three deployment scenarios (assumed; who pays is a commercial decision this PoC can inform but not
                      settle): S1 Precisely-hosted (Precisely pays hosting, operations and LLM tokens; the customer pays
                      the price passed through, not modeled), S2 Precisely-hosted matcher with the customer's own LLM key
                      (Precisely pays hosting and operations, the customer pays LLM tokens), S3 customer self-hosted or
                      on-prem (the customer pays hardware, operations and LLM tokens if they connect a key; Precisely pays
                      development and support, not modeled). B is not feasible in S3 when data cannot leave the environment;
                      D there runs with the LLM fallback off (quality not measured). Option E is "not run": no numbers.
Without --effort-log-complete every output starts with the DRAFT line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DAYS_PER_MONTH = 30.4
DRAFT = "DRAFT: effort log incomplete, one-time costs for B and D undercounted"
OPTIONS = ("A", "B", "C", "D", "E")
RATES = (50.0, 100.0, 200.0)
VOLUMES = (100_000, 1_000_000, 10_000_000)
LABEL_SCENARIOS = (500, 1000, 2000, 63665)
BASE_LABELS = 2000
BASE_RATE = 100.0
BASE_VOLUME = 1_000_000


# ---------------------------------------------------------------------------------------------- parameters

@dataclass
class Params:
    # LLM (measured on the 5,000-pair sample run, B)
    band_pairs_per_record: float = 20.0854
    routed_share: float = 38 / 44882
    in_tokens: float = 2103.0
    out_tokens: float = 19.0
    hit_ratio: float = 0.849
    price_in: float = 2.5            # USD per million tokens
    price_cached: float = 0.25
    price_out: float = 15.0
    batch_in: float = 1.25
    batch_cached: float = 0.13
    batch_out: float = 7.5
    llm_price_factor: float = 1.0
    # CPU
    c_cpu_s_per_pair: float = 0.000186          # measured mean (8.346 s / 44,882 pairs, one core)
    d_cpu_s_per_pair: float = 0.000186          # C plus a threshold test (derived)
    verity_cpu_s_per_record: float = 0.04197    # run10_1core, verity.total 419.67 s / 10,000 records, one thread
    # hosting prices (ASSUMED)
    core_hour_always_on: float = 0.05
    core_second_serverless: float = 0.12 / 3600
    utilization: float = 0.5
    # operations (ASSUMED hours per month)
    monitor_hours: float = 2.0
    retrain_hours_per_run: float = 4.0
    retrains_per_month: float = 1.0
    llm_upkeep_hours: float = 4.0               # model retirements, prompt re-validation
    a_ops_hours: float = 1.0
    # labeling (hours per 1,000 pairs measured from the effort log)
    label_hours_per_1000: float = 5.0
    # one-time pieces filled from files
    effort_hours: dict = field(default_factory=dict)        # bucket -> production hours (production_needed = yes)
    verity_preexisting_hours: float = 80.0
    verity_productization_hours: float = 100.0
    b_dev_api_usd: float = 0.0
    horizon_months: int = 12
    verity_quotable: bool = False
    latency_loaded: bool = False


def llm_cost_per_call(p: Params, mode: str = "standard", hit_ratio: float | None = None) -> float:
    h = p.hit_ratio if hit_ratio is None else hit_ratio
    if mode == "standard":
        usd = (p.in_tokens * (1 - h) * p.price_in + p.in_tokens * h * p.price_cached + p.out_tokens * p.price_out) / 1e6
    elif mode == "standard_nocache":
        usd = (p.in_tokens * p.price_in + p.out_tokens * p.price_out) / 1e6
    elif mode == "batch":
        usd = (p.in_tokens * (1 - h) * p.batch_in + p.in_tokens * h * p.batch_cached + p.out_tokens * p.batch_out) / 1e6
    elif mode == "batch_nohit":
        usd = (p.in_tokens * p.batch_in + p.out_tokens * p.batch_out) / 1e6
    else:
        raise ValueError(mode)
    return usd * p.llm_price_factor


def calls_per_day(p: Params, option: str, records_per_day: float) -> float:
    b = records_per_day * p.band_pairs_per_record
    return {"B": b, "D": b * p.routed_share}.get(option, 0.0)


def llm_cost_month(p: Params, option: str, records_per_day: float, mode: str = "standard") -> float:
    return calls_per_day(p, option, records_per_day) * DAYS_PER_MONTH * llm_cost_per_call(p, mode)


def hosting_component(core_seconds_per_day: float, p: Params, always_on_min_cores: int = 1) -> dict:
    """Always-on: whole cores at the target utilisation, billed 24 h a day; serverless: core-seconds billed.
    The cheaper one is used."""
    if core_seconds_per_day <= 0:
        return {"cores": 0, "always_on": 0.0, "serverless": 0.0, "used": 0.0}
    cores = max(always_on_min_cores, math.ceil(core_seconds_per_day / 86400 / p.utilization))
    always_on = cores * 24 * DAYS_PER_MONTH * p.core_hour_always_on
    serverless = core_seconds_per_day * DAYS_PER_MONTH * p.core_second_serverless
    return {"cores": cores, "always_on": always_on, "serverless": serverless, "used": min(always_on, serverless)}


def model_core_seconds(p: Params, option: str, records_per_day: float) -> float:
    pairs = records_per_day * p.band_pairs_per_record
    return {"C": pairs * p.c_cpu_s_per_pair, "D": pairs * p.d_cpu_s_per_pair}.get(option, 0.0)


def hosting_month(p: Params, option: str, records_per_day: float) -> dict:
    model = hosting_component(model_core_seconds(p, option, records_per_day), p)
    verity = hosting_component(records_per_day * p.verity_cpu_s_per_record, p)
    return {"model": model, "verity": verity, "used": model["used"] + verity["used"]}


def ops_hours_month(p: Params, option: str, retrains_per_month: float | None = None) -> float:
    r = p.retrains_per_month if retrains_per_month is None else retrains_per_month
    retrain = p.retrain_hours_per_run * r
    return {"A": p.a_ops_hours,
            "B": p.monitor_hours + p.llm_upkeep_hours,
            "C": p.monitor_hours + retrain,
            "D": p.monitor_hours + retrain + p.llm_upkeep_hours}.get(option, 0.0)


def label_hours(p: Params, labels: int) -> float:
    return labels / 1000 * p.label_hours_per_1000


def one_time_lines(p: Params, option: str, rate: float, labels: int, include_verity: bool = True) -> list[tuple[str, float, str]]:
    """(line, USD, kind) for the one-time cost charged to an option. kind is "development" (the Verity pre-existing build
    and productization, shared by every option; include_verity False drops them: the "B without the shared Verity build"
    variant) or "labeling" (C and D). The PoC's effort-log hours and ledger API spend are not here: they are the separate
    block "cost to build the evidence"."""
    lines = ([("verity pre-existing build (80 h, your estimate)", p.verity_preexisting_hours * rate, "development"),
              ("verity productization (100 h, your figure)", p.verity_productization_hours * rate, "development")] if include_verity else [])
    if option in ("C", "D"):
        lines.append((f"labeling, {labels:,} labels x {p.label_hours_per_1000:g} h per 1,000", label_hours(p, labels) * rate, "labeling"))
    return lines


def one_time_total(p: Params, option: str, rate: float, labels: int, include_verity: bool = True) -> float:
    return sum(v for _, v, _ in one_time_lines(p, option, rate, labels, include_verity))


def one_time_split(p: Params, option: str, rate: float, labels: int, include_verity: bool = True) -> dict:
    out = {"development": 0.0, "labeling": 0.0}
    for _, v, kind in one_time_lines(p, option, rate, labels, include_verity):
        out[kind] += v
    return out


def monthly(p: Params, option: str, records_per_day: float, rate: float, labels: int, llm_mode: str = "standard",
            retrains_per_month: float | None = None, include_verity_one_time: bool = True) -> dict:
    if option == "E":
        return {"option": "E", "status": "not run"}
    llm = llm_cost_month(p, option, records_per_day, llm_mode)
    host = hosting_month(p, option, records_per_day)
    ops = ops_hours_month(p, option, retrains_per_month) * rate
    split = one_time_split(p, option, rate, labels, include_verity_one_time)
    ot = split["development"] + split["labeling"]
    amort = ot / p.horizon_months
    running = llm + host["used"] + ops
    return {"option": option, "status": "modelled", "llm": llm, "hosting_model": host["model"]["used"],
            "hosting_verity": host["verity"]["used"], "hosting": host["used"], "hosting_always_on": host["model"]["always_on"] + host["verity"]["always_on"],
            "hosting_serverless": host["model"]["serverless"] + host["verity"]["serverless"], "operations": ops,
            "running": running, "one_time": ot, "one_time_amortized": amort, "total_month": running + amort,
            "total_horizon": ot + p.horizon_months * running,
            "development": split["development"] / p.horizon_months, "labeling": split["labeling"] / p.horizon_months,
            "calls_per_day": calls_per_day(p, option, records_per_day), "cores_model": host["model"]["cores"],
            "cores_verity": host["verity"]["cores"]}


def total_horizon(p: Params, option: str, volume: float, rate: float, labels: int, **kw) -> float:
    return monthly(p, option, volume, rate, labels, **kw)["total_horizon"]


B_VARIANTS = {"price-70": "B, LLM price -70% (mini-class, price only, no quality claim)",
              "no-verity-build": "B without the shared Verity build in its one-time costs"}
B_VARIANT_SHORT = {"price-70": "B, price -70%", "no-verity-build": "B, no Verity build"}


def monthly_b_variant(p: Params, variant: str, records_per_day: float, rate: float, labels: int) -> dict:
    """Variants of B for cost only: the LLM price multiplied by 0.3 (a smaller model's price; nothing is claimed about its
    quality), or B's one-time costs without the shared Verity build and productization lines."""
    if variant == "price-70":
        return monthly(replace(p, llm_price_factor=p.llm_price_factor * 0.3), "B", records_per_day, rate, labels)
    if variant == "no-verity-build":
        return monthly(p, "B", records_per_day, rate, labels, include_verity_one_time=False)
    raise ValueError(variant)


# ------------------------------------------------------------------ deployment scenarios: who pays
SCENARIOS = {
    "S1": "Precisely-hosted: Precisely pays hosting, operations and LLM tokens; the customer pays the price passed through (not modeled)",
    "S2": "Precisely-hosted matcher with the customer's own LLM key (BYOK): Precisely pays hosting and operations; the customer pays LLM tokens",
    "S3": "Customer self-hosted or on-prem: the customer pays hardware, operations and LLM tokens only if they connect their own key; "
          "Precisely pays development and support (not modeled)",
}
COMPONENTS = ("llm", "hosting", "operations", "development", "labeling")
COMPONENT_TEXT = {"llm": "LLM tokens", "hosting": "hosting (the matcher or model, and the shared Verity pipeline)", "operations": "operations (monitoring, retraining, LLM upkeep)",
                  "development": "development (Verity build and productization, amortized)", "labeling": "labeling for C and D (amortized)"}
PARTY = {
    "S1": {"llm": "Precisely", "hosting": "Precisely", "operations": "Precisely", "development": "Precisely", "labeling": "Precisely"},
    "S2": {"llm": "customer", "hosting": "Precisely", "operations": "Precisely", "development": "Precisely", "labeling": "Precisely"},
    "S3": {"llm": "customer", "hosting": "customer", "operations": "customer", "development": "Precisely (not modeled)", "labeling": "customer"},
}
S3_FEASIBLE = {"A": "yes", "B": "no", "C": "yes", "D": "yes, with the LLM fallback off"}


def scenario_rows(p: Params, option: str, records_per_day: float, rate: float, labels: int) -> list[dict]:
    """One row per scenario for an option (two for D in S3: the LLM fallback off, and with the customer's own key
    connected; B in S3 is costed with the customer's own key and flagged as not feasible when data cannot leave).
    Each row splits the monthly cost by party: Precisely, customer, and their total. The development share in S3 is
    Precisely's and is not modeled (left out of the total)."""
    if option == "E":
        return [{"option": "E", "label": "E", "scenario": "", "status": "not run"}]
    m = monthly(p, option, records_per_day, rate, labels)
    base = {c: m[c] for c in COMPONENTS}
    rows = []
    for sc in ("S1", "S2", "S3"):
        variants = [(option, base, "as evaluated")]
        if sc == "S3":
            if option == "B":
                variants = [("B (own key connected)", base, "as evaluated")]
            elif option == "D":
                variants = [("D (LLM fallback off)", {**base, "llm": 0.0}, "not measured"), ("D (own key connected)", base, "as evaluated")]
        for label, comps, quality in variants:
            parties = PARTY[sc]
            precisely = sum(v for c, v in comps.items() if parties[c] == "Precisely")
            customer = sum(v for c, v in comps.items() if parties[c] == "customer")
            rows.append({"option": option, "label": label, "scenario": sc, "status": "modelled", "components": comps, "parties": dict(parties),
                         "precisely": precisely, "customer": customer, "total": precisely + customer,
                         "feasible_data_stays": S3_FEASIBLE[option] if sc == "S3" else "n/a (hosted)", "quality": quality,
                         "not_modeled": "Precisely development and support" if sc == "S3" else ("price passed through to the customer" if sc == "S1" else ""),
                         "calls_per_day": m["calls_per_day"] if comps["llm"] > 0 else 0.0, "cores_model": m["cores_model"], "cores_verity": m["cores_verity"],
                         "hosting_model": m["hosting_model"], "hosting_verity": m["hosting_verity"]})
    return rows


def scenario_total_horizon(p: Params, option: str, volume: float, rate: float, labels: int, scenario: str, d_variant: str = "off") -> float:
    """Total over the horizon in a scenario (S3 leaves out Precisely's development, which is not modeled)."""
    rows = [r for r in scenario_rows(p, option, volume, rate, labels) if r["scenario"] == scenario]
    if scenario == "S3" and option == "D":
        rows = [r for r in rows if r["label"] == ("D (LLM fallback off)" if d_variant == "off" else "D (own key connected)")]
    return rows[0]["total"] * p.horizon_months


# ------------------------------------------------------------------ protocol bars
BAR_F1_MARGIN = -0.02
BAR_PRECISION = 0.98
BAR_RECALL_OVER_A = 0.05
BAR_ECE = 0.05
BAR_LATENCY_C_P95_MS = 20.0
BAR_LLM_SHARE = 0.20


def f1_bar_verdict(lo: float, hi: float) -> str:
    """Option minus B in band F1: pass when the interval's lower end is above -0.02, fail when its upper end is below,
    otherwise inconclusive (protocol.md, borderline bar, judged on the paired 5,000-pair sample)."""
    return "pass" if lo > BAR_F1_MARGIN else ("fail" if hi < BAR_F1_MARGIN else "inconclusive")


def parse_overall(text: str) -> tuple[float, float]:
    m = re.search(r"Overall \([^\n]*\n\s+precision ([\d.]+)\s+recall ([\d.]+)", text)
    return float(m.group(1)), float(m.group(2))


def parse_band_f1(text: str) -> float:
    return float(re.search(r"Band \([^\n]*\n[^\n]*\n\s+F1 ([\d.]+)", text).group(1))


def parse_ece(text: str):
    m = re.search(r"ECE\s+([\d.]+)\s+\(bar", text)
    return float(m.group(1)) if m else None


def parse_f1_diff(text: str) -> tuple[float, float, float]:
    """(difference, lower, upper) of band F1 from a compare_options.py output (first minus second)."""
    blk = text.split("Difference (first minus second)", 1)[1]
    m = re.search(r"band F1\s+([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", blk)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def load_bar_inputs(run: Path, p: Params, latency: dict | None) -> dict:
    """Per option: the numbers each protocol bar needs, read from the committed score and comparison reports."""
    o = run / "options"
    rd = lambda path: path.read_text(encoding="utf-8")
    a_full, c_full, b_samp = rd(o / "a_threshold" / "score_report.txt"), rd(o / "c_model" / "c_v1_lgbm" / "score_report.txt"), \
        rd(o / "b_llm" / "openai" / "sample5000" / "score_report_sample5000.txt")
    d_full = rd(o / "d_hybrid" / "score_report_fullband.txt")
    a_vs_b = parse_f1_diff(rd(o / "compare" / "b_vs_a.txt"))                       # B first: negate for A minus B
    a_vs_b = (-a_vs_b[0], -a_vs_b[2], -a_vs_b[1])
    c_vs_b = parse_f1_diff(rd(o / "compare" / "c_v1_lgbm_vs_b.txt"))
    d_vs_b = parse_f1_diff(rd(o / "d_hybrid" / "compare_d_vs_b.txt"))
    c_lat = None
    if latency:
        c_lat = latency["C"]["median_of_repeats"]["p95_ms"]
    else:
        ct = json.loads((o / "c_model" / "c_v1_lgbm" / "timings.json").read_text(encoding="utf-8"))
        c_lat = [t for t in ct["timings"] if t["stage"] == "optionC.per_pair"][0]["per_item_ms"]["p95"]
    out = {
        "A": {"band_f1": parse_band_f1(a_full), "f1_diff": a_vs_b, "overall": parse_overall(a_full), "ece": None, "basis": "full band; F1 difference on the sample"},
        "B": {"band_f1": parse_band_f1(b_samp), "f1_diff": None, "overall": parse_overall(b_samp), "ece": parse_ece(b_samp),
              "basis": "sample estimate (weighted); B is the baseline of the F1 bar"},
        "C": {"band_f1": parse_band_f1(c_full), "f1_diff": c_vs_b, "overall": parse_overall(c_full), "ece": parse_ece(c_full), "latency_p95_ms": c_lat,
              "basis": "full band; F1 difference on the sample"},
        "D": {"band_f1": parse_band_f1(d_full), "f1_diff": d_vs_b, "overall": parse_overall(d_full), "ece": parse_ece(d_full), "llm_share": p.routed_share,
              "basis": "exact full band; F1 difference on the sample"},
    }
    if latency and "D" in latency:
        out["D"]["latency_p95_ms"] = latency["D"]["median_of_repeats"]["p95_ms"]
    return out


def bars_row(option: str, d: dict, a_recall: float) -> dict:
    """The verdict per bar and whether every bar is met."""
    v = {}
    if d["f1_diff"] is None:
        v["f1"] = "reference (B is the baseline)"
    else:
        v["f1"] = f1_bar_verdict(d["f1_diff"][1], d["f1_diff"][2])
    v["precision"] = "pass" if d["overall"][0] >= BAR_PRECISION else "fail"
    if option == "A":
        v["recall"] = "reference (A is the baseline)"
    else:
        v["recall"] = "pass" if d["overall"][1] >= a_recall + BAR_RECALL_OVER_A - 1e-12 else "fail"
    v["ece"] = ("pass" if d["ece"] <= BAR_ECE else "fail") if option in ("C", "D") else "n/a (bar is for C and D)"
    v["latency"] = ("pass" if d["latency_p95_ms"] <= BAR_LATENCY_C_P95_MS else "fail") if option == "C" else "n/a (bar is for C)"
    v["llm_share"] = ("pass" if d["llm_share"] <= BAR_LLM_SHARE else "fail") if option == "D" else "n/a (bar is for D)"
    checked = [x for x in v.values() if x in ("pass", "fail", "inconclusive")]
    if "fail" in checked or (option == "A"):
        v["every"] = "no"
    elif "inconclusive" in checked:
        v["every"] = "not shown (inconclusive)"
    else:
        v["every"] = "yes"
    return v


def break_even(f_option, f_b, lo: float = 1e3, hi: float = 1e8, steps: int = 400) -> dict:
    """Records per day where f_option(V) = f_b(V). Scans a log grid for the first sign change of f_option - f_b and
    bisects it. f_option cheaper above that volume: kind 'option cheaper above'; cheaper below: 'option cheaper below'.
    No change: 'option cheaper everywhere in range' or 'B cheaper everywhere in range'."""
    grid = [lo * (hi / lo) ** (i / steps) for i in range(steps + 1)]
    d = [f_option(v) - f_b(v) for v in grid]
    for i in range(steps):
        if d[i] == 0:
            return {"volume": grid[i], "kind": "equal at grid point"}
        if d[i] * d[i + 1] < 0:
            a, b = grid[i], grid[i + 1]
            for _ in range(80):
                m = math.sqrt(a * b)
                if (f_option(m) - f_b(m)) * d[i] > 0:
                    a = m
                else:
                    b = m
            return {"volume": math.sqrt(a * b), "kind": "option cheaper above" if d[i] > 0 else "option cheaper below"}
    return {"volume": None, "kind": "option cheaper everywhere in range" if d[0] < 0 else "B cheaper everywhere in range"}


# ---------------------------------------------------------------------------------------------- file inputs

def parse_date(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised date {s!r} (use yyyy-mm-dd or m/d/yyyy)")


def read_effort_log(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("hours") or "").strip():
                continue                                     # a row in progress, no hours yet
            rows.append({"date": parse_date(r["date"]), "hours": float(r["hours"]), "activity": r["activity"].strip(),
                         "bucket": r["bucket"].strip(), "production_needed": r["production_needed"].strip().lower() == "yes"})
    return rows


def hours_by_bucket(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        b = out.setdefault(r["bucket"], {"total": 0.0, "production": 0.0, "other": 0.0})
        b["total"] += r["hours"]
        b["production" if r["production_needed"] else "other"] += r["hours"]
    return out


LEDGER_DEV_PREFIXES = ("tune", "cache_diagnostic", "smoke", "concurrency")
LEDGER_EVAL_LABELS = ("sample5000", "decoycheck", "dfull")
LEDGER_EVAL_PREFIXES = ("final", "dec4", "sample")


def classify_ledger_label(label: str) -> str:
    """tuning rounds, caching diagnostics, the smoke test (and the concurrency test) = B one-time development;
    sample5000, decoy check, D routed-pair runs and the Dec 4 runs = evaluation, counted once, charged to no option."""
    if label.startswith(LEDGER_DEV_PREFIXES):
        return "B development"
    if label in LEDGER_EVAL_LABELS or label.startswith(LEDGER_EVAL_PREFIXES):
        return "evaluation"
    return "unclassified"


def ledger_spend(path: Path) -> dict:
    """USD settled per run label and class, from the append-only ledger."""
    by_label: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "settle":
            lab = str(e.get("run", "")).rsplit("/", 1)[-1]
            by_label[lab] = by_label.get(lab, 0.0) + float(e.get("usd", 0.0))
    classes: dict = {}
    for lab, usd in by_label.items():
        classes[classify_ledger_label(lab)] = classes.get(classify_ledger_label(lab), 0.0) + usd
    return {"by_label": by_label, "by_class": classes}


def label_hours_per_1000_from_log(rows: list[dict]) -> float:
    r = next((x for x in rows if x["activity"].lower().startswith("hand check of")), None)
    if r is None:
        raise SystemExit("effort log has no 'Hand check of N pairs' row to measure labeling hours from")
    n = int(re.search(r"(\d[\d,]*)\s+pairs", r["activity"]).group(1).replace(",", ""))
    return r["hours"] / n * 1000


def load_inputs(repo: Path, run: Path, out_rows: list, productization_hours: float) -> tuple[Params, dict]:
    """Build Params from the repo's result files and record every input in out_rows (name, value, unit, status,
    source, note)."""
    def add(name, value, unit, status, source, note=""):
        out_rows.append({"name": name, "value": value, "unit": unit, "status": status, "source": source, "note": note})

    p = Params(verity_productization_hours=productization_hours)
    b = run / "options" / "b_llm" / "openai" / "sample5000"
    meta = json.loads((b / "run_meta.json").read_text(encoding="utf-8"))
    tim = json.loads((b / "timings.json").read_text(encoding="utf-8"))
    rep = (b / "sample_report.txt").read_text(encoding="utf-8")
    m = re.search(r"tokens: input ([\d,]+), cached ([\d,]+), output ([\d,]+)", rep)
    n_calls = meta["calls"]
    p.in_tokens = int(m.group(1).replace(",", "")) / n_calls
    p.out_tokens = int(m.group(3).replace(",", "")) / n_calls
    p.hit_ratio = meta["prompt_cache"]["cached_tokens"] / meta["prompt_cache"]["input_tokens"]
    src = b.relative_to(repo).as_posix() if b.is_relative_to(repo) else b.as_posix()
    add("B calls in the final run", n_calls, "calls", "measured", f"{src}/run_meta.json")
    add("B input tokens per call", round(p.in_tokens, 1), "tokens", "measured", f"{src}/sample_report.txt")
    add("B output tokens per call", round(p.out_tokens, 2), "tokens", "measured", f"{src}/sample_report.txt")
    add("B prompt-cache hit ratio (cached / input tokens)", round(p.hit_ratio, 4), "ratio", "measured", f"{src}/run_meta.json",
        "sensitivity 0 and 0.85")
    add("B cost per call, from the ledger run", round(meta["spent_usd"] / n_calls, 6), "USD", "measured", f"{src}/run_meta.json")
    pc = [t for t in tim["timings"] if t["stage"].endswith("llm_call")][0]
    add("B latency p50", round(pc["per_item_ms"]["p50"] / 1000, 3), "s", "measured", f"{src}/timings.json")
    add("B latency p95", round(pc["per_item_ms"]["p95"] / 1000, 3), "s", "measured", f"{src}/timings.json")
    add("B throughput in the final run (8 keys, 12 requests per minute per key, 2 workers)",
        round(n_calls / meta["run_time_minutes"], 1), "calls per minute", "measured", f"{src}/run_meta.json",
        "limited by the per-key limiter, not the API")
    for k, v, note in (("input price", p.price_in, ""), ("cached input price", p.price_cached, ""), ("output price", p.price_out, ""),
                       ("batch input price", p.batch_in, "batch cache hits are undocumented"), ("batch cached input price", p.batch_cached, ""),
                       ("batch output price", p.batch_out, "")):
        add(f"gpt-5.4 {k}", v, "USD per million tokens", "measured", "options/prices.json (provider price page, 2026-11-24)", note)
    add("LLM price factor", 1.0, "x", "assumed", "model parameter", "sensitivity 0.5 and 1.5")

    c_dec = run / "options" / "c_model" / "c_v1_lgbm" / "decisions.csv"
    n_band = sum(1 for _ in c_dec.open(encoding="utf-8")) - 1
    routed = sum(1 for _ in (run / "options" / "d_hybrid" / "routed_fullband_pairs.csv").open(encoding="utf-8")) - 1
    p.routed_share = routed / n_band
    add("D routed share (exact, full test band)", round(p.routed_share, 6), "ratio", "measured",
        "results/.../options/d_hybrid/routed_fullband_pairs.csv and c_v1_lgbm/decisions.csv", f"{routed} of {n_band:,} band pairs; sensitivity 0.001, 0.01, 0.05, 0.20")
    readme = (run / "README.txt").read_text(encoding="utf-8")
    band_pairs = int(re.search(r"review ([\d,]+)", readme).group(1).replace(",", ""))
    records = int(re.search(r"([\d,]+) records", readme).group(1).replace(",", ""))
    p.band_pairs_per_record = band_pairs / records
    add("band pairs per record", round(p.band_pairs_per_record, 4), "pairs", "measured", f"{run.name}/README.txt", f"{band_pairs:,} / {records:,}; sensitivity 5, 10, 20")

    ct = json.loads((run / "options" / "c_model" / "c_v1_lgbm" / "timings.json").read_text(encoding="utf-8"))
    pp = [t for t in ct["timings"] if t["stage"] == "optionC.per_pair"][0]
    p.c_cpu_s_per_pair = pp["seconds"] / pp["items"]
    p.d_cpu_s_per_pair = p.c_cpu_s_per_pair
    add("C per-pair CPU time, mean (first measurement, one repeat)", round(p.c_cpu_s_per_pair * 1000, 4), "ms", "measured", "options/c_model/c_v1_lgbm/timings.json", "one core, one pair at a time; features + predict + calibrate; replaced by the 3-repeat measurement when present")
    add("C per-pair CPU time, p95 (first measurement)", pp["per_item_ms"]["p95"], "ms", "measured", "options/c_model/c_v1_lgbm/timings.json")
    add("D per-pair CPU time (before the D measurement)", round(p.d_cpu_s_per_pair * 1000, 4), "ms", "assumed", "derived from C",
        "D runs C on every band pair; the routing test is negligible; replaced by the measurement when present")
    add("A per-pair CPU time", 0.0, "ms", "assumed", "not measured", "a threshold test; treated as zero")

    vt_path = repo / "results" / "dev_2026_timing_summary" / "verity_timing.json"
    latency = None
    if vt_path.exists():
        vt = json.loads(vt_path.read_text(encoding="utf-8"))
        for setting in ("1thread", "allthreads"):
            st = vt["settings"][setting]
            tot = st["stage"]["verity.total"]
            add(f"Verity pipeline total, {setting}, per record (median of {len(st['repeats'])} repeats)", round(tot["median"] / vt["records"] * 1000, 2), "ms", "measured",
                vt_path.relative_to(repo).as_posix(),
                f"min {tot['min']:.1f} s, max {tot['max']:.1f} s, cold {tot['cold']:.1f} s, warm {tot['warm']:.1f} s; spread (max-min)/median {st['total_spread_pct']:.1f}%; "
                + ("the repeats agree within 10%: label removed" if st["total_within_10pct"] else "NOT FOR QUOTING: the repeats differ by more than 10% on total time")
                + "; includes startup; shared by every option, B included")
        one = vt["settings"]["1thread"]
        p.verity_cpu_s_per_record = one["stage"]["verity.total"]["median"] / vt["records"]
        p.verity_quotable = bool(one["total_within_10pct"])
    else:
        ri = json.loads((repo / "results" / "dev_2026_run10_1core" / "verity_data" / "run_info.json").read_text(encoding="utf-8"))["runs"][-1]
        total = [t for t in ri["timings"] if t["stage"] == "verity.total"][0]["seconds"]
        p.verity_cpu_s_per_record = total / 10000
        add("Verity pipeline time per record, one thread", round(p.verity_cpu_s_per_record * 1000, 2), "ms", "measured",
            "results/dev_2026_run10_1core/verity_data/run_info.json (verity.total / 10,000 records)",
            "NOT FOR QUOTING until a controlled run exists: uncontrolled laptop state; includes 14 s of startup; shared by every option, B included")
    lat_path = run / "options" / "latency" / "latency_summary.json"
    if lat_path.exists():
        latency = json.loads(lat_path.read_text(encoding="utf-8"))
        for opt in ("C", "D"):
            m = latency[opt]["median_of_repeats"]
            add(f"{opt} per-pair latency, one core, median of {len(latency[opt]['repeats'])} repeats", f"p50 {m['p50_ms']} ms, p95 {m['p95_ms']} ms, max {m['max_ms']} ms, mean {m['mean_ms']} ms",
                "ms", "measured", lat_path.relative_to(repo).as_posix(), f"{latency['pairs']:,} test band pairs, features included, one pair at a time"
                + ("; plus the routing step" if opt == "D" else ""))
        pl = latency["parallel"]
        add("C and D under load: 10 processes, one thread each", f"{pl['pairs_per_second_total']} pairs per second in total, per-pair p95 {pl['p95_ms_under_load']} ms (C); D {pl.get('D', {}).get('pairs_per_second_total', 'n/a')} pairs per second, p95 {pl.get('D', {}).get('p95_ms_under_load', 'n/a')} ms",
            "", "measured", lat_path.relative_to(repo).as_posix(), "disjoint slices of the band; this machine has 12 cores, 14 threads")
        p.c_cpu_s_per_pair = latency["C"]["median_of_repeats"]["mean_ms"] / 1000
        p.d_cpu_s_per_pair = latency["D"]["median_of_repeats"]["mean_ms"] / 1000
        p.latency_loaded = True

    log = read_effort_log(repo / "cost" / "effort_log.csv")
    p.label_hours_per_1000 = label_hours_per_1000_from_log(log)
    add("labeling hours per 1,000 pairs", p.label_hours_per_1000, "hours", "measured", "cost/effort_log.csv row 'Hand check of 100 pairs' (0.5 h)",
        "sensitivity 2 and 10; the hand check was blind agreement checking, not a labeling workflow")
    buckets = hours_by_bucket(log)
    p.effort_hours = {k: v["production"] for k, v in buckets.items()}
    for k, v in sorted(buckets.items()):
        add(f"effort-log hours, bucket {k}", f"{v['total']:g} total, {v['production']:g} production needed", "hours", "measured",
            "cost/effort_log.csv", "the PoC's own effort: goes into the block 'cost to build the evidence', charged to no option or scenario")
    pre = [r for r in csv.reader((repo / "cost" / "verity_preexisting_cost.csv").open(encoding="utf-8-sig"))][-1]
    p.verity_preexisting_hours = float(pre[1])
    add("Verity pre-existing build", p.verity_preexisting_hours, "hours", "assumed", "cost/verity_preexisting_cost.csv (your estimate)")
    add("Verity productization", productization_hours, "hours", "assumed", "stated by you (--productization-hours)")

    led = ledger_spend(repo / "results" / "llm_ledger.jsonl")
    p.b_dev_api_usd = led["by_class"].get("B development", 0.0)
    add("ledger classification rule", "tuning rounds (tune*), caching diagnostics (cache_diagnostic*), the smoke test (smoke*) and "
        "the concurrency test = B development; sample5000, decoycheck, dfull (D routed pairs) and the "
        "coming Dec 4 runs (final*, dec4*) = evaluation, counted once. Both classes are the PoC's API spend: they are shown in the block "
        "'cost to build the evidence' and charged to no option or scenario", "", "assumed", "your instruction; cost/cost_model.py classify_ledger_label",
        "the concurrency test is my classification (a capacity measurement for B)")
    for lab, usd in sorted(led["by_label"].items()):
        add(f"ledger spend, run {lab}", round(usd, 4), "USD", "measured", "results/llm_ledger.jsonl", classify_ledger_label(lab))
    add("ledger: B development API spend", round(p.b_dev_api_usd, 4), "USD", "measured", "results/llm_ledger.jsonl", "cost to build the evidence; charged to no option or scenario")
    add("ledger: evaluation API spend (counted once)", round(led["by_class"].get("evaluation", 0.0), 4), "USD", "measured", "results/llm_ledger.jsonl", "cost to build the evidence; charged to no option or scenario")
    if led["by_class"].get("unclassified"):
        add("ledger: UNCLASSIFIED spend", round(led["by_class"]["unclassified"], 4), "USD", "measured", "results/llm_ledger.jsonl", "add a rule in classify_ledger_label")

    cc = run / "options" / "b_llm" / "openai" / "concurrency" / "concurrency.json"
    conc = None
    if cc.exists():
        c = conc = json.loads(cc.read_text(encoding="utf-8"))
        for lv in c["levels"]:
            add(f"concurrency test, {lv['workers']} workers: calls per minute", lv["calls_per_minute"], "calls per minute", "measured", cc.relative_to(repo).as_posix(),
                f"latency p50 {lv['latency_p50_s']} s, p95 {lv['latency_p95_s']} s, max {lv['latency_max_s']} s; hit ratio excl. first calls {lv['hit_ratio_excl_first']}; errors {lv['errors']}; 16 keys, no limiter")
        add("concurrency test: last clean level", c.get("last_clean_workers"), "workers", "measured", cc.relative_to(repo).as_posix(), c.get("stopped") or "no error at any level")

    add("hosting: always-on price", p.core_hour_always_on, "USD per core-hour", "assumed", "generic cloud list price, replace with a quote")
    add("hosting: serverless price", round(p.core_second_serverless * 3600, 4), "USD per core-hour of use", "assumed", "generic cloud list price, replace with a quote")
    add("hosting: always-on utilisation", p.utilization, "ratio", "assumed", "model parameter")
    for k, v in (("monitoring", p.monitor_hours), ("retraining, per run", p.retrain_hours_per_run), ("retraining runs per month", p.retrains_per_month),
                 ("LLM upkeep (model retirements, prompt re-validation)", p.llm_upkeep_hours), ("option A operations", p.a_ops_hours)):
        add(f"operations: {k}", v, "hours per month" if "runs" not in k and "per run" not in k else ("runs" if "runs" in k else "hours"), "assumed", "model parameter",
            "sensitivity: weekly 4.33, monthly 1, quarterly 0.33" if "retraining" in k else "")
    for v in VOLUMES:
        add(f"volume {v:,} records per day", v, "records per day", "assumed", "protocol.md")
    add("hourly rates", "50, 100, 200", "USD per hour", "assumed", "your instruction")
    add("days per month", DAYS_PER_MONTH, "days", "assumed", "model parameter")
    add("horizon", p.horizon_months, "months", "assumed", "one-time cost (Verity build and productization, labeling) is amortized and totals are taken over this horizon")
    note = ("who pays is a commercial decision this PoC can inform but not settle; the three scenarios and the party for each cost component "
            "are assumptions, not findings")
    for sc, text in SCENARIOS.items():
        add(f"scenario {sc}", text, "", "assumed", "your instruction", note)
    for sc in SCENARIOS:
        add(f"scenario {sc}: who pays each component", "; ".join(f"{COMPONENT_TEXT[c]}: {PARTY[sc][c]}" for c in COMPONENTS), "", "assumed", "your instruction",
            "hosting and operations assumptions are unchanged; only the party is assigned. Labeling is assumed to be done by Precisely when it hosts the matcher "
            "and by the customer when the matcher runs in their environment (their data stays there)")
    add("scenario S3: feasible when data cannot leave the environment", "A yes; B no; C yes; D yes with the LLM fallback off", "", "assumed", "your instruction",
        "B needs an external LLM for every band pair; D's quality with the LLM fallback off was never measured (it is C's decisions for the routed pairs, "
        "which the evaluation did not score)")
    add("scenario S3: B and D cost basis", "B and D(own key connected) are costed with the customer's own LLM key; D(LLM fallback off) has no LLM line", "", "assumed", "model parameter")
    add("cost to build the evidence", "separate block: effort-log hours x hourly rate and the ledger API spend", "", "measured", "cost/effort_log.csv and results/llm_ledger.jsonl",
        "not charged to any option or scenario; undercounted for B and D until the effort log is complete")
    add("option E", "not run", "", "measured", "options/e_slm.py has no results", "no numbers are invented")
    lc = list(csv.DictReader((run / "options" / "c_model" / "lc_v2" / "learning_curve.csv").open(encoding="utf-8")))
    f1 = {int(r["n"]): float(r["f1_mean"]) for r in lc}
    return p, {"log": log, "buckets": buckets, "ledger": led, "lc_f1": f1, "full_f1": 0.9653, "n_band": n_band, "conc": conc,
               "latency": latency}


# ---------------------------------------------------------------------------------------------- outputs

def write_csv(path: Path, header: list[str], rows: list[list], draft: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if draft:
            f.write(f"# {DRAFT}\n")
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt_usd(x) -> str:
    return "" if x is None else f"{x:,.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cost model for the options.")
    ap.add_argument("--run", type=Path, default=REPO / "results" / "dev_2026_run11")
    ap.add_argument("--out", type=Path, default=REPO / "cost")
    ap.add_argument("--effort-log-complete", action="store_true",
                    help="the effort log covers B and D; without it every output starts with the DRAFT line")
    ap.add_argument("--productization-hours", type=float, default=100.0)
    args = ap.parse_args()
    draft = not args.effort_log_complete
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows_a: list = []
    p, ctx = load_inputs(REPO, args.run, rows_a, args.productization_hours)
    L: list[str] = []
    say = L.append
    if draft:
        say(DRAFT)
        say("")
    log = ctx["log"]
    say(f"Effort log: {len(log)} rows with hours, {min(r['date'] for r in log):%Y-%m-%d} to {max(r['date'] for r in log):%Y-%m-%d}")
    say("Hours by bucket (total / production needed / other):")
    for k, v in sorted(ctx["buckets"].items()):
        say(f"  {k:<8}{v['total']:>6g} / {v['production']:>5g} / {v['other']:>5g}")
    say(f"  These hours are the PoC's own effort. They go into the block 'cost to build the evidence' below and are not charged to any option or scenario. "
        f"B and D are only partly in the log (logged production hours: B {ctx['buckets'].get('B', {}).get('production', 0):g}, "
        f"D {ctx['buckets'].get('D', {}).get('production', 0):g}), so that block is undercounted until the log is completed (use --effort-log-complete once it is).")
    led = ctx["ledger"]
    ev_rows = []
    say("\nCost to build the evidence (the PoC's own spend; NOT charged to any option or scenario)")
    say(f"  {'effort-log hours':<30}{'hours':>8}" + "".join(f"{'$' + format(r, 'g') + '/h':>12}" for r in RATES))
    tot_h = 0.0
    for k, v in sorted(ctx["buckets"].items()):
        tot_h += v["total"]
        say(f"  {'bucket ' + k:<30}{v['total']:>8g}" + "".join(f"{v['total'] * r:>12,.0f}" for r in RATES))
        ev_rows.append(["effort-log hours", f"bucket {k}", f"{v['total']:g}"] + [f"{v['total'] * r:.2f}" for r in RATES])
    say(f"  {'hours total':<30}{tot_h:>8g}" + "".join(f"{tot_h * r:>12,.0f}" for r in RATES))
    ev_rows.append(["effort-log hours", "total", f"{tot_h:g}"] + [f"{tot_h * r:.2f}" for r in RATES])
    api_total = sum(led["by_class"].values())
    say(f"  API spend from the ledger: " + ", ".join(f"{k} ${v:.4f}" for k, v in sorted(led["by_class"].items())) + f"; total ${api_total:.4f}")
    for lab, usd in sorted(led["by_label"].items()):
        say(f"    {lab:<28} ${usd:>8.4f}  {classify_ledger_label(lab)}")
        ev_rows.append(["API spend (ledger)", f"{lab} ({classify_ledger_label(lab)})", ""] + [f"{usd:.4f}"] * 3)
    ev_rows.append(["API spend (ledger)", "total", ""] + [f"{api_total:.4f}"] * 3)
    say(f"  {'evidence total (hours x rate + API spend)':<44}" + "".join(f"{tot_h * r + api_total:>12,.0f}" for r in RATES))
    ev_rows.append(["total", "hours x rate + API spend", ""] + [f"{tot_h * r + api_total:.2f}" for r in RATES])
    write_csv(out / "evidence_cost.csv", ["kind", "item", "hours", "usd_at_50_per_hour", "usd_at_100_per_hour", "usd_at_200_per_hour"], ev_rows, draft)

    # labeling table
    say("\nLabeling for C and D, next to B's zero labels (hours at "
        f"{p.label_hours_per_1000:g} h per 1,000 pairs, measured from the hand check; sensitivity 2 and 10)")
    say(f"  {'labels':>8}{'mean band F1':>14}{'hours':>8}" + "".join(f"{'$' + format(r, 'g') + '/h':>12}" for r in RATES))
    for n in LABEL_SCENARIOS:
        f1 = ctx["full_f1"] if n == 63665 else ctx["lc_f1"][n]
        say(f"  {n:>8,}{f1:>14.4f}{label_hours(p, n):>8.1f}" + "".join(f"{label_hours(p, n) * r:>12,.0f}" for r in RATES))
    say(f"  {'B: 0':>8}{'':>14}{0:>8}" + "".join(f"{0:>12}" for _ in RATES) + "   (B needs no labeled training pairs to run)")
    say("  The 63,665 row is what the dev result for C used (every train band pair, F1 0.9653); it is not a recommended budget. "
        "On the dev data the labels came free from the generator. The 500 to 2,000 rows are the mean band F1 over 5 sampling seeds of "
        "the learning curve (lc_v2); the spread between seeds is large (see learning_curve.csv). B's development also used labeled train pairs "
        "(8 few-shot examples and 2,000 tuning pairs, free from the generator); that is not charged as labeling here.")

    # who pays: every option, volume and scenario
    def grid_row(lab, rate, vol, r):
        if r["status"] == "not run":
            return [lab, rate, vol, "E", "E", "", "not run"] + [""] * 23
        c, pt = r["components"], r["parties"]
        h = p.horizon_months
        return [lab, rate, vol, r["option"], r["label"], r["scenario"], "modelled", r["feasible_data_stays"], r["quality"],
                f"{r['precisely']:.2f}", f"{r['customer']:.2f}", f"{r['total']:.2f}",
                f"{r['precisely'] * h:.2f}", f"{r['customer'] * h:.2f}", f"{r['total'] * h:.2f}",
                f"{c['llm']:.2f}", pt["llm"], f"{r['hosting_model']:.2f}", f"{r['hosting_verity']:.2f}", pt["hosting"], f"{c['operations']:.2f}", pt["operations"],
                f"{c['development']:.2f}", pt["development"], f"{c['labeling']:.2f}", pt["labeling"], f"{r['calls_per_day']:.0f}",
                f"{r['cores_model']}", f"{r['cores_verity']}", r["not_modeled"]]

    grid = []
    for lab in LABEL_SCENARIOS:
        for rate in RATES:
            for vol in VOLUMES:
                for opt in OPTIONS:
                    for r in scenario_rows(p, opt, vol, rate, lab):
                        grid.append(grid_row(lab, rate, vol, r))
    write_csv(out / "cost_model.csv",
              ["labels_for_C_and_D", "hourly_rate_usd", "records_per_day", "option", "option_row", "scenario", "status",
               "feasible_when_data_cannot_leave_the_environment", "quality", "precisely_usd_month", "customer_usd_month", "total_usd_month",
               "precisely_usd_12_months", "customer_usd_12_months", "total_usd_12_months", "llm_usd_month", "llm_paid_by", "hosting_model_usd_month",
               "hosting_verity_usd_month", "hosting_paid_by", "operations_usd_month", "operations_paid_by", "development_amortized_usd_month",
               "development_paid_by", "labeling_amortized_usd_month", "labeling_paid_by", "llm_calls_per_day", "cores_model", "cores_verity",
               "not_modeled"], grid, draft)

    say("\nDeployment scenarios (assumed; who pays is a commercial decision this PoC can inform but not settle)")
    for sc, text in SCENARIOS.items():
        say(f"  {sc}: {text}")
    say("  Who pays each cost component (hosting and operations assumptions are unchanged; only the party is assigned):")
    say(f"    {'component':<62}{'S1':<12}{'S2':<12}S3")
    for c in COMPONENTS:
        say(f"    {COMPONENT_TEXT[c]:<62}{PARTY['S1'][c]:<12}{PARTY['S2'][c]:<12}{PARTY['S3'][c]}")
    say("    Labeling is assumed to be done by Precisely when it hosts the matcher and by the customer when the matcher runs in their environment.")
    say("  S3 feasibility when data cannot leave the environment: A yes, B no, C yes, D yes with the LLM fallback off (its quality in that case is not measured).")

    say(f"\nMonthly cost by component at the base case (hourly rate ${BASE_RATE:g}, labels {BASE_LABELS:,}, standard prices with cache, hit ratio {p.hit_ratio:.3f}), "
        f"USD per month, all parties together")
    say(f"  {'records/day':>12}{'option':>7}{'LLM':>12}{'hosting':>10}{'operations':>12}{'one-time/12':>13}{'total':>12}")
    for vol in VOLUMES:
        for opt in OPTIONS:
            m = monthly(p, opt, vol, BASE_RATE, BASE_LABELS)
            if opt == "E":
                say(f"  {vol:>12,}{'E':>7}  not run")
                continue
            say(f"  {vol:>12,}{opt:>7}{m['llm']:>12,.0f}{m['hosting']:>10,.0f}{m['operations']:>12,.0f}{m['one_time_amortized']:>13,.0f}{m['total_month']:>12,.0f}")
    say(f"  (hosting is the cheaper of always-on and serverless; shared Verity pipeline per record is in every row: "
        f"{p.verity_cpu_s_per_record * 1000:.1f} ms of one core per record{'' if p.verity_quotable else ', NOT FOR QUOTING'}; "
        f"one-time = the Verity build and productization shared by every option, plus labeling for C and D)")

    for vol in (BASE_VOLUME, 100_000, 10_000_000):
        say(f"\nWho pays at {vol:,} records per day, USD per month (hourly rate ${BASE_RATE:g}; C and D with {LABEL_SCENARIOS[-1]:,} labels, as evaluated; "
            f"development and labeling amortized over {p.horizon_months} months)")
        say(f"  {'option':<24}{'scen.':<6}{'Precisely':>13}{'customer':>13}{'total':>13}   {'feasible if data cannot leave (S3)':<36}quality")
        for opt in OPTIONS:
            for r in scenario_rows(p, opt, vol, BASE_RATE, LABEL_SCENARIOS[-1]):
                if r["status"] == "not run":
                    say(f"  {'E':<24}      not run (no results; no numbers)")
                    continue
                pre = f"{r['precisely']:,.0f}" if r["scenario"] != "S3" else f"{r['precisely']:,.0f}*"
                say(f"  {r['label']:<24}{r['scenario']:<6}{pre:>13}{r['customer']:>13,.0f}{r['total']:>13,.0f}   "
                    f"{r['feasible_data_stays'] if r['scenario'] == 'S3' else '':<36}{r['quality'] if opt == 'D' and r['scenario'] == 'S3' else ''}")
        say("  * S3: Precisely's development and support are not modeled (shown as 0); S1: the price passed through to the customer is not modeled. "
            "B in S3 is costed with the customer's own key and is not feasible when data cannot leave the environment.")
    make_who_pays_chart(p, out, draft)
    cpm = calls_per_day(p, "B", 100_000) / 1440
    say(f"\nThroughput: B at 100K records/day needs {calls_per_day(p, 'B', 100_000):,.0f} calls a day = {cpm:,.0f} calls per minute "
        f"({cpm / 60:.1f} per second; about {cpm / 60 * 0.95:.0f} requests in flight at the measured p50 latency of 0.95 s). "
        f"1M and 10M records/day need 10x and 100x that. D needs {calls_per_day(p, 'D', 10_000_000) / 1440:,.0f} calls per minute at 10M records/day.")

    # options against the protocol bars, with cost at 1M records per day
    bar_in = load_bar_inputs(args.run, p, ctx["latency"])
    a_recall = bar_in["A"]["overall"][1]
    bar_rows = []
    say(f"\nOptions against the protocol bars (protocol.md), with cost at {BASE_VOLUME:,} records per day (hourly rate ${BASE_RATE:g}; C and D as evaluated: "
        f"{LABEL_SCENARIOS[-1]:,} labels)")
    say("  F1 bar: band F1 within 2 points of B, judged on the paired 5,000-pair sample (pass: interval lower end above -0.02; fail: upper end below; else inconclusive).")
    say("  Precision bar: overall precision >= 0.98. Recall bar: overall recall at least 5 points above A. ECE <= 0.05 (C, D). C latency p95 <= 20 ms. D LLM share <= 20%.")
    say(f"  {'option':<8}{'F1 vs B [95% CI]':<30}{'F1 bar':<14}{'prec':>7}{'bar':>6}{'recall':>8}{'bar':>6}{'ECE bar':>9}{'latency':>9}{'LLM share':>10}  every bar?          cost/month   12-month total")
    for opt in "ABCD":
        d = bar_in[opt]
        v = bars_row(opt, d, a_recall)
        m = monthly(p, opt, BASE_VOLUME, BASE_RATE, LABEL_SCENARIOS[-1])
        diff = "baseline" if d["f1_diff"] is None else f"{d['f1_diff'][0]:+.4f} [{d['f1_diff'][1]:+.4f}, {d['f1_diff'][2]:+.4f}]"
        say(f"  {opt:<8}{diff:<30}{v['f1'][:12]:<14}{d['overall'][0]:>7.4f}{v['precision']:>6}{d['overall'][1]:>8.4f}{v['recall'][:5]:>6}{v['ece'][:4]:>9}{v['latency'][:4]:>9}{v['llm_share'][:4]:>10}"
            f"  {v['every']:<18}{m['total_month']:>12,.0f}{m['total_horizon']:>16,.0f}")
        bar_rows.append([opt, d["basis"], d["band_f1"], "" if d["f1_diff"] is None else f"{d['f1_diff'][0]:.4f}", "" if d["f1_diff"] is None else f"{d['f1_diff'][1]:.4f}",
                         "" if d["f1_diff"] is None else f"{d['f1_diff'][2]:.4f}", v["f1"], d["overall"][0], v["precision"], d["overall"][1], v["recall"],
                         "" if d["ece"] is None else d["ece"], v["ece"], d.get("latency_p95_ms", ""), v["latency"], d.get("llm_share", ""), v["llm_share"], v["every"],
                         f"{m['total_month']:.2f}", f"{m['total_horizon']:.2f}", LABEL_SCENARIOS[-1] if opt in "CD" else 0])
    say(f"  {'E':<8}not run (no results; no numbers)")
    bar_rows.append(["E", "not run"] + [""] * 19)
    say("  B's overall figures are sample estimates; A, C and D are exact on the full test band. A is the reference for the recall bar and fails the F1 bar; "
        "C's F1 bar is inconclusive at this sample size (its interval straddles -0.02).")
    c2 = monthly(p, "C", BASE_VOLUME, BASE_RATE, 2000)
    d2 = monthly(p, "D", BASE_VOLUME, BASE_RATE, 2000)
    say(f"\nLabel rows for C and D: the label cost is only ever shown next to the F1 measured with that label count")
    say(f"  {'option':<6}{'row':<14}{'labels':>8}{'label hours':>12}{'label cost at $50/$100/$200':>30}{'band F1':>26}{'12-month total at 1M/day':>26}")
    for opt, d_ in (("C", bar_in["C"]), ("D", bar_in["D"])):
        m = monthly(p, opt, BASE_VOLUME, BASE_RATE, 63665)
        say(f"  {opt:<6}{'as evaluated':<14}{63665:>8,}{label_hours(p, 63665):>12.1f}{'/'.join(f'{label_hours(p, 63665) * r:,.0f}' for r in RATES):>30}"
            f"{d_['band_f1']:>20.4f} (full band){m['total_horizon']:>14,.0f}")
    say(f"  {'C':<6}{'at 2,000':<14}{2000:>8,}{label_hours(p, 2000):>12.1f}{'/'.join(f'{label_hours(p, 2000) * r:,.0f}' for r in RATES):>30}"
        f"{ctx['lc_f1'][2000]:>14.4f} (mean of 5 seeds){c2['total_horizon']:>9,.0f}")
    say(f"  {'D':<6}{'at 2,000':<14}{2000:>8,}{label_hours(p, 2000):>12.1f}{'/'.join(f'{label_hours(p, 2000) * r:,.0f}' for r in RATES):>30}"
        f"{'not measured':>26}{d2['total_horizon']:>14,.0f}")
    say("  'As evaluated' is what the dev results used (every train band pair, labels free from the generator); it is not a recommended budget. "
        "D's F1 was never measured with fewer labels, so no F1 is shown for D at 2,000. The C figure at 2,000 is the mean over 5 sampling seeds of the "
        "learning curve (individual seeds ranged widely), not a paired test result.")
    write_csv(out / "options_vs_bars.csv",
              ["option", "basis", "band_f1", "f1_diff_vs_B", "f1_diff_lo", "f1_diff_hi", "f1_bar", "overall_precision", "precision_bar", "overall_recall", "recall_bar",
               "ece", "ece_bar", "latency_p95_ms", "latency_bar", "llm_share", "llm_share_bar", "meets_every_bar", "total_usd_month_at_1M_per_day",
               "total_usd_12_months_at_1M_per_day", "labels"], bar_rows, draft)

    # B variants (cost only)
    say(f"\nB variants (cost only; no quality claim for either) at {BASE_VOLUME:,} records per day, ${BASE_RATE:g}/h, USD")
    say(f"  {'variant':<72}{'LLM / month':>14}{'total / month':>15}{'12-month total':>16}")
    for name, m in [("B as measured (gpt-5.4, frozen prompt)", monthly(p, "B", BASE_VOLUME, BASE_RATE, 0))] + \
            [(B_VARIANTS[k], monthly_b_variant(p, k, BASE_VOLUME, BASE_RATE, 0)) for k in B_VARIANTS]:
        say(f"  {name:<72}{m['llm']:>14,.0f}{m['total_month']:>15,.0f}{m['total_horizon']:>16,.0f}")
    var_rows = []
    for k in B_VARIANTS:
        for rate in RATES:
            for vol in VOLUMES:
                m = monthly_b_variant(p, k, vol, rate, 0)
                var_rows.append([k, rate, vol, f"{m['llm']:.2f}", f"{m['running']:.2f}", f"{m['one_time']:.2f}", f"{m['total_month']:.2f}", f"{m['total_horizon']:.2f}"])
    write_csv(out / "b_variants.csv", ["variant", "hourly_rate_usd", "records_per_day", "llm_usd_month", "running_usd_month", "one_time_usd",
                                       "total_usd_month", "total_usd_over_horizon"], var_rows, draft)
    say("  Break-even of A, C and D against each B variant (labels 2,000; range 1,000 to 100,000,000 records per day):")
    for k in B_VARIANTS:
        for opt in ("A", "C", "D"):
            for rate in RATES:
                r = break_even(lambda v, o=opt, rt=rate: total_horizon(p, o, v, rt, 2000), lambda v, kk=k, rt=rate: monthly_b_variant(p, kk, v, rt, 2000)["total_horizon"])
                tail = "no crossing: " + r["kind"] if r["volume"] is None else f"{r['volume']:,.0f} records per day ({r['kind']})"
                say(f"    {B_VARIANT_SHORT[k]:<20}{opt} at ${rate:g}/h: {tail}")

    conc = ctx["conc"]
    if conc:
        clean = [lv for lv in conc["levels"] if lv["workers"] == conc["last_clean_workers"]]
        say("\nMeasured concurrency (16 cache keys, per-key limiter off, 100 calls per level, Balanced power plan, on AC):")
        for lv in conc["levels"]:
            say(f"  {lv['workers']:>2} workers: {lv['calls_per_minute']:>6.1f} calls per minute, latency p50 {lv['latency_p50_s']} s, p95 {lv['latency_p95_s']} s, "
                f"max {lv['latency_max_s']} s, hit ratio excluding first calls {lv['hit_ratio_excl_first']}, errors {lv['errors']}")
        if clean:
            top = clean[0]
            need = calls_per_day(p, "B", 100_000) / 1440
            say(f"  Capacity line: B at 100K records/day needs about {calls_per_day(p, 'B', 100_000) / 1440:,.0f} calls/min, measured only up to about "
                f"{top['calls_per_minute']:.0f} ({top['workers']} workers); the rest is extrapolated.")
            say(f"  Highest clean level: {top['workers']} workers = {top['calls_per_minute']:.0f} calls per minute (no 429 or other error at any level; the SDK retries "
                f"a 429 twice before it would show). B at 100K records/day needs {need:,.0f} calls per minute, {need / top['calls_per_minute']:.1f}x that; at the "
                f"measured near-linear scaling (about {top['calls_per_minute'] / top['workers']:.0f} calls per minute per worker) that is about {need / (top['calls_per_minute'] / top['workers']):.0f} "
                f"workers, UNTESTED above {top['workers']}. 1M and 10M records/day need 10x and 100x more and would need a higher rate limit from the provider (not measured).")

    # break-even
    be_rows, be_text = [], []
    for lab in LABEL_SCENARIOS:
        for rate in RATES:
            for opt in ("A", "C", "D"):
                r = break_even(lambda v, o=opt: total_horizon(p, o, v, rate, lab), lambda v: total_horizon(p, "B", v, rate, lab))
                be_rows.append(["S1 and S2 (same total)", lab, rate, opt, "" if r["volume"] is None else f"{r['volume']:.0f}", r["kind"]])
                if lab == BASE_LABELS:
                    be_text.append((opt, rate, r))
                r3 = break_even(lambda v, o=opt: scenario_total_horizon(p, o, v, rate, lab, "S3"), lambda v: scenario_total_horizon(p, "B", v, rate, lab, "S3"))
                be_rows.append(["S3 (D with the LLM fallback off; B with the customer's key, not feasible if data cannot leave)", lab, rate, opt,
                                "" if r3["volume"] is None else f"{r3['volume']:.0f}", r3["kind"]])
    write_csv(out / "break_even.csv", ["scenario", "labels_for_C_and_D", "hourly_rate_usd", "option", "break_even_records_per_day", "result"], be_rows, draft)
    say(f"\nBreak-even against B (total over {p.horizon_months} months equal; labels {BASE_LABELS:,}; volume range 1,000 to 100,000,000 records per day)")
    for opt, rate, r in be_text:
        if r["volume"] is None:
            say(f"  {opt} at ${rate:g}/h: {r['kind']}")
        else:
            say(f"  {opt} at ${rate:g}/h: {r['volume']:,.0f} records per day ({r['kind']})")
    for opt, rate, r in be_text:
        if r["kind"] == "B cheaper everywhere in range":
            say(f"  -> {opt} at ${rate:g}/h never breaks even in range: B is cheaper overall at every volume from 1,000 to 100,000,000 records per day")
        elif r["kind"] == "option cheaper above":
            say(f"  -> {opt} at ${rate:g}/h: B is cheaper overall below {r['volume']:,.0f} records per day; {opt} is cheaper above")
        elif r["kind"] == "option cheaper below":
            say(f"  -> {opt} at ${rate:g}/h: {opt} is cheaper overall below {r['volume']:,.0f} records per day; B is cheaper above")
    nb = [(o, rt) for o, rt, r in be_text if r["kind"] == "option cheaper everywhere in range"]
    say("  Where no break-even is listed for an option, it is cheaper than B overall at every volume in the range: " +
        (", ".join(f"{o} at ${rt:g}/h" for o, rt in nb) if nb else "none") + ". At the measured price B costs about "
        f"${llm_cost_per_call(p) * p.band_pairs_per_record * 1000:,.0f} per 1,000 records in LLM calls alone, so any one-time saving is spent within days.")

    # sensitivity
    sens = []

    def sens_rows(factor, value, p2, rate=BASE_RATE, labels=BASE_LABELS, retr=None, mode="standard"):
        for opt in ("A", "B", "C", "D"):
            m = monthly(p2, opt, BASE_VOLUME, rate, labels, llm_mode=mode, retrains_per_month=retr)
            sens.append([factor, value, opt, f"{m['llm']:.2f}", f"{m['hosting']:.2f}", f"{m['operations']:.2f}", f"{m['running']:.2f}",
                         f"{m['one_time']:.2f}", f"{m['total_horizon']:.2f}"])

    sens_rows("base case", "", p)
    for r in RATES:
        sens_rows("hourly rate", r, p, rate=r)
    for v in (5, 10, 20, round(p.band_pairs_per_record, 2)):
        sens_rows("band pairs per record", v, replace(p, band_pairs_per_record=v))
    for v in (0.001, 0.01, 0.05, 0.20):
        sens_rows("D routed share", v, replace(p, routed_share=v))
    for v in (0.5, 1.0, 1.5):
        sens_rows("LLM price factor", v, replace(p, llm_price_factor=v))
    for v in (0.0, 0.85):
        sens_rows("hit ratio", v, replace(p, hit_ratio=v))
    for v in (4.33, 1.0, 0.33):
        sens_rows("retrainings per month", v, p, retr=v)
    for v in (2, 5, 10):
        sens_rows("labeling hours per 1,000", v, replace(p, label_hours_per_1000=v))
    sens_rows("batch prices (cache hits)", "", p, mode="batch")
    sens_rows("batch prices (no hits)", "", p, mode="batch_nohit")
    sens_rows("standard, no caching", "", p, mode="standard_nocache")
    write_csv(out / "sensitivity.csv", ["factor", "value", "option", "llm_usd_month", "hosting_usd_month", "operations_usd_month",
                                        "running_usd_month", "one_time_usd", "total_usd_over_horizon"], sens, draft)
    say(f"\nSensitivity at {BASE_VOLUME:,} records per day, ${BASE_RATE:g}/h, {BASE_LABELS:,} labels: total over {p.horizon_months} months, USD "
        "(sensitivity.csv has the pieces)")
    say(f"  {'factor':<28}{'value':>8}" + "".join(f"{o:>14}" for o in "ABCD"))
    last = None
    for r in sens:
        key = (r[0], r[1])
        if key != last:
            line = {x[2]: x[8] for x in sens if (x[0], x[1]) == key}
            say(f"  {r[0]:<28}{str(r[1]):>8}" + "".join(f"{float(line[o]):>14,.0f}" for o in "ABCD"))
            last = key
    say("\nOption E: not run (no results; no numbers).")
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    write_csv(out / "assumptions.csv", ["name", "value", "unit", "status", "source", "note"],
              [[r["name"], r["value"], r["unit"], r["status"], r["source"], r["note"]] for r in rows_a], draft)
    make_charts(p, out, draft)
    print(text)
    return 0


def make_who_pays_chart(p: Params, out: Path, draft: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, sc in enumerate(("S1", "S2", "S3")):
        rows = [r for opt in "ABCD" for r in scenario_rows(p, opt, BASE_VOLUME, BASE_RATE, LABEL_SCENARIOS[-1]) if r["scenario"] == sc]
        for i, show_b in enumerate((True, False)):
            ax = axes[i][j]
            sel = [r for r in rows if show_b or r["option"] != "B"]
            short = {"B (own key connected)": "B\nown key", "D (LLM fallback off)": "D\nfallback off", "D (own key connected)": "D\nown key"}
            names = [short.get(r["label"], r["label"]) for r in sel]
            pre = [r["precisely"] for r in sel]
            cus = [r["customer"] for r in sel]
            ax.bar(names, pre, color="#1f77b4", label="Precisely")
            ax.bar(names, cus, bottom=pre, color="#ff7f0e", label="customer")
            for k, r in enumerate(sel):
                ax.text(k, pre[k] + cus[k], f"${pre[k] + cus[k]:,.0f}" + ("*" if sc == "S3" else ""), ha="center", va="bottom", fontsize=7)
                if sc == "S3" and r["option"] == "B":
                    ax.text(k, 0, "not feasible if\ndata cannot leave", ha="center", va="bottom", fontsize=6, color="white")
            ax.set_title(f"{sc}" + ("" if show_b else " (B removed, zoom)"), fontsize=10)
            ax.set_ylabel("USD per month")
            ax.tick_params(axis="x", labelsize=7)
            ax.margins(y=0.15)
    axes[0][0].legend(fontsize=8)
    fig.text(0.5, 0.005, f"{BASE_VOLUME:,} records/day, ${BASE_RATE:g}/h, C and D with {LABEL_SCENARIOS[-1]:,} labels (as evaluated). E: not run. "
             "S1: price passed through to the customer not modeled. S3: * Precisely's development and support not modeled. Who pays is a commercial decision this PoC "
             "can inform but not settle.", ha="center", fontsize=7)
    fig.suptitle(("DRAFT: effort log incomplete, one-time costs for B and D undercounted\n" if draft else "") + "Who pays, monthly cost per option and scenario")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "chart_who_pays.png", dpi=120)
    plt.close(fig)


def make_charts(p: Params, out: Path, draft: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"llm": "#1f77b4", "hosting": "#2ca02c", "operations": "#ff7f0e", "one_time_amortized": "#9467bd"}
    names = {"llm": "LLM", "hosting": "hosting", "operations": "operations", "one_time_amortized": "one-time / 12 months"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, vol in enumerate(VOLUMES):
        for i, show_b in enumerate((True, False)):
            ax = axes[i][j]
            opts = [o for o in "ABCD" if show_b or o != "B"]
            bottoms = [0.0] * len(opts)
            for comp in colors:
                vals = [monthly(p, o, vol, BASE_RATE, BASE_LABELS)[comp] for o in opts]
                ax.bar(opts, vals, bottom=bottoms, color=colors[comp], label=names[comp] if (i == 0 and j == 0) else None)
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            for k, o in enumerate(opts):
                ax.text(k, bottoms[k], f"${bottoms[k]:,.0f}", ha="center", va="bottom", fontsize=8)
            ax.set_title(f"{vol:,} records/day" + ("" if show_b else " (B removed, zoom)"), fontsize=10)
            ax.set_ylabel("USD per month")
            ax.margins(y=0.15)
    axes[0][0].legend(fontsize=8)
    fig.text(0.5, 0.005, f"E: not run. Hourly rate ${BASE_RATE:g}, {BASE_LABELS:,} labels for C and D, standard prices with cache. "
             "Verity pipeline per record is in every option" + ("" if p.verity_quotable else " (NOT FOR QUOTING)") + ". One-time includes the shared Verity build and productization hours in every option.", ha="center", fontsize=8)
    fig.suptitle(("DRAFT: effort log incomplete, one-time costs for B and D undercounted\n" if draft else "") + "Monthly cost per option")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "chart_monthly_cost.png", dpi=120)
    plt.close(fig)

    vols = [10 ** (3 + 5 * i / 100) for i in range(101)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, rate in zip(axes, RATES):
        for o, c in zip("ABCD", ("#7f7f7f", "#d62728", "#1f77b4", "#2ca02c")):
            ax.plot(vols, [total_horizon(p, o, v, rate, BASE_LABELS) for v in vols], label=o, color=c)
        for o in "ACD":
            r = break_even(lambda v, o=o: total_horizon(p, o, v, rate, BASE_LABELS), lambda v: total_horizon(p, "B", v, rate, BASE_LABELS))
            if r["volume"]:
                ax.axvline(r["volume"], color="#555555", ls=":", lw=0.8)
                ax.annotate(f"{o}: {r['volume']:,.0f}", (r["volume"], total_horizon(p, "B", r["volume"], rate, BASE_LABELS)), fontsize=8, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"${rate:g} per hour")
        ax.set_xlabel("records per day")
        ax.set_ylabel(f"total USD over {p.horizon_months} months (one-time + running)")
        ax.legend(fontsize=8)
    fig.text(0.5, 0.005, "Dotted lines: break-even volume against B. E: not run. A does not meet the quality bars (see summary.txt).", ha="center", fontsize=8)
    fig.suptitle(("DRAFT: effort log incomplete, one-time costs for B and D undercounted\n" if draft else "") + f"Break-even against B ({BASE_LABELS:,} labels for C and D)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out / "chart_break_even.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

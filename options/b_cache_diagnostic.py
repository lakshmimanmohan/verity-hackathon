"""
Prompt-caching diagnostic for the OpenAI path of option B. Spends real money (hard cap --max-usd, via the
ledger). TRAIN band pairs only: never test pairs, never a few-shot pair or a pair touching a few-shot
record, never a pair of an earlier smoke test or diagnostic round.

  python options/b_cache_diagnostic.py results/dev_2026_run11 --round 1 --max-usd 0.20
  python options/b_cache_diagnostic.py results/dev_2026_run11 --round 2 --max-usd 0.25

ROUND 1 (6 different pairs per variant, calls one after another, 4 s apart):
  V1  one fixed prompt_cache_key for every call ("verity-b-v1"), instructions in a developer message
  V2  no prompt_cache_key at all, instructions in a developer message
  V3  as V1, but the instructions are the top-level "instructions" parameter

ROUND 2 (5 different pairs per variant): the fixed prompt_cache_key "verity-b-v1", default retention,
instructions in the developer message, but the shared prefix is padded before the pair with a fixed generic
reference appendix (legal-suffix and street-type equivalences, phone and website formatting variations;
no pair data, identical on every call) to about 2,400 / 3,400 / 4,500 tokens (3.5 characters per token):
  P1 about 2,400 tokens   P2 about 3,400   P3 about 4,500
  P4 (run only if P1 to P3 all fail) = P3 plus prompt_cache_retention "24h"

Stops after the first variant whose calls from the second onward show cached tokens above 0, and at once
on any error. For the first call of each variant the raw usage object (token fields only) is printed.
The report is appended to <run>/options/b_llm/cache_diagnostic.txt. No headers and no key are printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b_llm_baseline as B  # noqa: E402
from llm_ledger import Ledger, LedgerRefused  # noqa: E402

KEY = "verity-b-v1"
CHARS_PER_TOKEN = 3.5
# (name, description, prompt_cache_key, where the instructions go, retention)
ROUND1 = (("V1", "fixed prompt_cache_key, developer message", KEY, "developer", None),
          ("V2", "no prompt_cache_key, developer message", None, "developer", None),
          ("V3", "fixed prompt_cache_key, top-level instructions", KEY, "instructions", None))
ROUND2 = (("P1", "shared prefix about 2,400 tokens", KEY, "developer", None),
          ("P2", "shared prefix about 3,400 tokens", KEY, "developer", None),
          ("P3", "shared prefix about 4,500 tokens", KEY, "developer", None),
          ("P4", "P3 plus prompt_cache_retention 24h", KEY, "developer", "24h"))
PAD_TARGET_TOKENS = {"P1": 2400, "P2": 3400, "P3": 4500, "P4": 4500}
USER_AND_OVERHEAD_TOKENS = 150     # measured: about 1,821 input tokens for a 1,671-token (estimated) system prompt


def appendix_lines() -> list[str]:
    """A fixed, generic reference appendix: no pair data, deterministic, identical on every call."""
    out = ["", "REFERENCE APPENDIX (general background; it does not change the task above)", "",
           "Legal-suffix equivalences (the same suffix written differently):"]
    groups = [("Incorporated", "Inc", "Inc.", "INC"), ("Corporation", "Corp", "Corp.", "CORP"),
              ("Limited", "Ltd", "Ltd.", "LTD"), ("Limited Liability Company", "LLC", "L.L.C.", "llc"),
              ("Limited Liability Partnership", "LLP", "L.L.P."), ("Company", "Co", "Co.", "CO"),
              ("Public Limited Company", "PLC", "plc", "P.L.C."),
              ("Gesellschaft mit beschraenkter Haftung", "GmbH", "G.m.b.H."), ("Aktiengesellschaft", "AG", "A.G."),
              ("Societe Anonyme", "SA", "S.A."), ("Proprietary Limited", "Pty Ltd", "Pty. Ltd."),
              ("Besloten Vennootschap", "BV", "B.V."), ("Naamloze Vennootschap", "NV", "N.V."),
              ("Sociedad Anonima", "SA", "S.A."), ("Sociedad de Responsabilidad Limitada", "SRL", "S.R.L."),
              ("Societa a responsabilita limitata", "Srl", "S.r.l."), ("Aktiebolag", "AB", "A.B."),
              ("Osakeyhtio", "Oy"), ("Anonim Sirketi", "AS", "A.S."), ("Kabushiki Kaisha", "KK", "K.K.")]
    out += ["  " + " = ".join(g) for g in groups]
    out += ["", "Street-type and address-part equivalences (US style):"]
    street = [("Street", "St", "St."), ("Avenue", "Ave", "Ave."), ("Boulevard", "Blvd", "Blvd."), ("Road", "Rd", "Rd."),
              ("Drive", "Dr", "Dr."), ("Lane", "Ln", "Ln."), ("Court", "Ct", "Ct."), ("Place", "Pl", "Pl."),
              ("Square", "Sq", "Sq."), ("Terrace", "Ter", "Terr."), ("Parkway", "Pkwy", "Pky"), ("Highway", "Hwy", "Hwy."),
              ("Circle", "Cir", "Cir."), ("Trail", "Trl", "Tr."), ("Way", "Wy"), ("Plaza", "Plz", "Plz."),
              ("Suite", "Ste", "Ste."), ("Floor", "Fl", "Fl."), ("Building", "Bldg", "Bldg."), ("North", "N", "N."),
              ("South", "S", "S."), ("East", "E", "E."), ("West", "W", "W."), ("Post Office Box", "PO Box", "P.O. Box", "POB")]
    out += ["  " + " = ".join(g) for g in street]
    out += ["", "Phone number formats (a fictional number written in different ways):"]
    for i in range(60):
        a, b, c = "555", f"{100 + i:04d}", "212"
        out.append(f"  ({c}) {a}-{b} = {c}-{a}-{b} = {c}.{a}.{b} = {c}{a}{b} = +1 {c} {a} {b} = 1-{c}-{a}-{b} = ({c}){a}{b}")
    out += ["", "Website formatting variations (a fictional site written in different ways):"]
    for i in range(60):
        host = f"example-{i:02d}.com"
        out.append(f"  https://www.{host}/ = http://{host} = www.{host} = {host} = {host.upper()} = https://{host}/about")
    out += ["", "Name punctuation variations:"]
    for i in range(60):
        out.append(f"  Example-{i:02d} Holdings, Inc. = Example {i:02d} Holdings Inc = EXAMPLE-{i:02d} HOLDINGS INC. = Example-{i:02d} Hldgs Inc")
    return out


def padded_system(system: str, target_tokens: int) -> str:
    """system plus whole appendix lines until about target_tokens (3.5 characters per token)."""
    want = int(target_tokens * CHARS_PER_TOKEN)
    text = system.rstrip("\n")
    for line in appendix_lines():
        if len(text) >= want:
            break
        text += "\n" + line
    return text + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--max-usd", type=float, default=0.20)
    ap.add_argument("--spacing-s", type=float, default=4.0)
    ap.add_argument("--round", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    per_variant = 6 if args.round == 1 else 5
    variants = ROUND1 if args.round == 1 else ROUND2
    run = args.run
    provider = B.make_provider("openai", None, live=True)
    provider.require_approved()
    rd = B.RunData(run)
    truth = B.TrainTruth(run, rd.side)
    bdir = run / "options" / "b_llm"
    fewshot = json.loads((bdir / "fewshot.json").read_text(encoding="utf-8"))
    fs_ids = {(e["ids"][0], e["ids"][1]) for e in fewshot}
    fs_records = {r for p in fs_ids for r in p}
    system = B.build_system_prompt(fewshot, rd.cols)
    smoke = {(r["record_a"], r["record_b"]) for r in B.read_csv(bdir / "openai" / "smoke5" / "decisions.csv")}
    pool = B.stratified_train_sample(rd, truth, rd.band("train"), fs_ids, 60, fs_records)
    avail = [p for p in pool if (p[0], p[1]) not in smoke]
    round1_pairs = avail[:18]                      # what round 1 used (6 x 3)
    pairs = avail if args.round == 1 else avail[18:]
    need = per_variant * len(variants)
    assert len(pairs) >= need and not any(p[0] in fs_records or p[1] in fs_records for p in pairs)
    pairs = pairs[:need]
    if args.round == 2:
        assert not ({(p[0], p[1]) for p in pairs} & {(p[0], p[1]) for p in round1_pairs})
    assert not ({(p[0], p[1]) for p in pairs} & smoke)
    ledger = Ledger(B.DEFAULT_LEDGER)
    secrets = [os.environ.get("OPENAI_API_KEY", "")]
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    def finish(code: int) -> int:
        report = bdir / "cache_diagnostic.txt"
        prior = report.read_text(encoding="utf-8") if report.exists() else ""
        report.write_text(prior + ("\n\n" if prior else "") + "\n".join(lines) + "\n", encoding="utf-8")
        return code

    out(f"ROUND {args.round}. Caching diagnostic, {provider.model} ({provider.settings_label()}), "
        f"{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    out(f"cap ${args.max_usd:.2f}; TRAIN band pairs only; {per_variant} different pairs per variant, {args.spacing_s:g} s apart; "
        f"base system prompt {len(system):,} characters")
    run_id = f"{run.name}/b_llm/openai/cache_diagnostic_r{args.round}"
    cached_from_second = False
    n = 0
    for v, (name, desc, key, where, retention) in enumerate(variants):
        sysv = padded_system(system, PAD_TARGET_TOKENS[name]) if args.round == 2 else system
        out(f"\n{name}: {desc}" + (f"; system text {len(sysv):,} characters" if args.round == 2 else ""))
        out(f"  {'call':<5}{'cache key sent':<15}{'input':>7}{'cached':>8}{'write':>7}{'output':>8}{'cost $':>11}{'latency s':>11}")
        first_input = None
        for i in range(per_variant):
            a, b = pairs[v * per_variant + i][:2]
            user = B.user_turn(rd.records[a], rd.records[b], rd.cols)
            body = provider.request(sysv, user, "live")
            body.pop("prompt_cache_key", None)
            if key:
                body["prompt_cache_key"] = key
            if retention:
                body["prompt_cache_retention"] = retention
            if where == "instructions":
                body["instructions"] = sysv
                body["input"] = [{"role": "user", "content": user}]
            if i:
                time.sleep(args.spacing_s)
            try:
                rid = ledger.reserve(provider.upper_bound_usd(sysv, user, "live"), run=run_id, provider="openai",
                                     model=provider.model, run_cap_usd=args.max_usd, note=f"cache diagnostic {name}")
            except LedgerRefused as e:
                out(f"  STOPPED: ledger refused: {e}")
                return finish(2)
            t0 = time.perf_counter()
            try:
                resp = provider.client.responses.create(**body)
            except Exception as e:  # noqa: BLE001
                ledger.release(rid, note=f"error {type(e).__name__}")
                out(f"  STOPPED at once, call {i + 1}: {B._scrub(f'{type(e).__name__}: {e}', secrets)}")
                return finish(1)
            lat = time.perf_counter() - t0
            res = B.OpenAIProvider.from_body(resp.model_dump(), lat)
            usd = provider.cost_of(res.usage, "live")
            ledger.settle(rid, usd, input_tokens=res.usage["input_tokens"], output_tokens=res.usage["output_tokens"])
            n += 1
            det = resp.usage.input_tokens_details if resp.usage is not None else None
            write = (getattr(det, "cache_write_tokens", 0) or 0) if det else 0
            raw_cached = (getattr(det, "cached_tokens", 0) or 0) if det else 0
            ok = "" if res.answered else f"   NOT ANSWERED: {res.error}"
            out(f"  {i + 1:<5}{(key or 'none'):<15}{res.usage['input_tokens']:>7}{res.usage['cached_tokens']:>8}{write:>7}"
                f"{res.usage['output_tokens']:>8}{usd:>11.6f}{lat:>11.2f}{ok}")
            if i == 0:
                first_input = res.usage["input_tokens"]
                out(f"  raw usage object of call 1 (token fields only): {json.dumps(resp.usage.model_dump() if resp.usage else None)}")
            if raw_cached != res.usage["cached_tokens"]:
                out(f"  PARSING BUG: raw cached_tokens {raw_cached} but the parser recorded {res.usage['cached_tokens']}")
            if i and res.usage["cached_tokens"] > 0:
                cached_from_second = True
            if not res.answered:
                out("  STOPPED at once: unusable answer")
                return finish(1)
        if first_input:
            prefix = max(first_input - USER_AND_OVERHEAD_TOKENS, 0)
            cached = prefix // 128 * 128
            exp = (cached * provider.price("cache_read") + (first_input - cached) * provider.price("input")
                   + 18 * provider.price("output")) / 1e6
            out(f"  real input tokens (call 1): {first_input}; if caching worked, about {cached} tokens (the prefix of about "
                f"{prefix} rounded down to 128) would be read from the cache: expected cost per call about ${exp:.6f} "
                f"(cached at 0.25, rest at 2.50, 18 output tokens at 15.00 per million) against about "
                f"${(first_input * provider.price('input') + 18 * provider.price('output')) / 1e6:.6f} uncached")
        if cached_from_second:
            out(f"\n{name}: cached tokens above 0 from the second call onward. Later variants not run.")
            break
    else:
        out("\nNo variant showed cached tokens above 0. Stopped; nothing further spent.")
    st = ledger.state()
    out(f"\n{n} calls. Ledger total ${st.total_micro / 1e6:.4f} (this round ${st.by_run_micro.get(run_id, 0) / 1e6:.4f}), "
        f"remaining ${(ledger.cap_micro - st.total_micro) / 1e6:.4f}")
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())

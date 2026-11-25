# Head to head: options A to D (freeze run, seed 7919)

Everything below is read from files under `results/eval_7919` and `results/freeze_2026-11-25` (see the sources section). `missing` means the frozen outputs do not hold the figure; the expected source is named. Nothing was rerun or invented.

## Bases

Quality columns (band F1 and its interval, band precision and recall, overall precision and recall, ECE, F1 versus B) are on the fixed 5,000-pair sample. Cost, break-even and latency columns come from the cost model and benchmarks and are not sample figures.

Notes:

- Cost outputs read from `results/eval_7919/cost_final` (effort log: 36 rows).
- Labels as evaluated for C and D in this run: 66,031 (`assumptions.csv` of the cost outputs). Hourly rate $100 and scenario S1 unless a column says otherwise.
- 'LLM $' columns are the LLM cost only (LLM share x B's measured cost per call); the total cost is in the monthly table.
- Table 2 (full band) is exact for A, C and D and is **not comparable with B's sample figures**. The secondary row D (orig) is in its own table.

### Table 1.1: Quality, sample basis

| option | band F1 [95% CI] | F1 vs B [95% CI] | F1 verdict | band precision | band recall | overall precision | overall recall | ECE |
|---|---|---|---|---|---|---|---|---|
| A | 0.1058 [0.0666, 0.1516] | -0.8798 [-0.9268, -0.8270] | fail | 1.0000 | 0.0559 | 1.0000 | 0.5120 | not reported (bar is for C and D) |
| B | 0.9856 [0.9587, 1.0000] | baseline | reference (B is the baseline) | 0.9743 | 0.9972 | 0.9971 | 0.5713 | 0.0142 |
| C | 0.9627 [0.9315, 0.9836] | -0.0229 [-0.0641, +0.0118] | inconclusive | 0.9732 | 0.9525 | 0.9971 | 0.5685 | 0.0004 |
| D | 0.9730 [0.9363, 1.0000] | -0.0126 [-0.0400, +0.0028] | inconclusive | 0.9499 | 0.9972 | 0.9942 | 0.5713 | 0.0006 |

### Table 1.2: Protocol bars (computed from the frozen reports with the protocol rules)

| option | meets every bar (sample basis) | meets every bar (full-band basis, A, C and D only) |
|---|---|---|
| A | no | no |
| B | yes | n/a (B exists only on the sample) |
| C | not shown (inconclusive) | not shown (inconclusive) |
| D | not shown (inconclusive) | not shown (inconclusive) |

Bars (protocol.md): F1 within 2 points of B on the paired sample (from `bars_table.csv`); overall precision >= 0.98; overall recall at least 5 points above A; ECE <= 0.05 (C, D); C latency p95 <= 20 ms; D LLM share <= 20% (from `bars_table.csv`). The two columns apply the same rules to the sample-basis reports and to the full-band reports; the F1 verdict, the latency and the LLM share are the same in both.

The two columns agree for A, C and D (B has no full-band value): no bar's verdict differs between the sample-basis and the full-band reports.

### Table 1.3: Cost per pair and per record (not sample figures)

| option | LLM $ per 1,000 band pairs | LLM $ per record | band pairs per record |
|---|---|---|---|
| A | 0.0000 | 0.00000 | 20.0499 |
| B | 1.5234 | 0.03054 | 20.0499 |
| C | 0.0000 | 0.00000 | 20.0499 |
| D | 0.0147 | 0.00029 | 20.0499 |

### Table 1.4: Monthly cost (S1, $100/h, labels as evaluated; cost model, not sample figures)

| option | monthly $ at 100K records/day (Precisely / customer / total) | monthly $ at 1M records/day (Precisely / customer / total) | monthly $ at 10M records/day (Precisely / customer / total) |
|---|---|---|---|
| A | 1,643.50 / 0.00 / 1,643.50 | 1,659.96 / 0.00 / 1,659.96 | 1,824.07 / 0.00 / 1,824.07 |
| B | 95,003.87 / 0.00 / 95,003.87 | 930,723.45 / 0.00 / 930,723.45 | 9,287,918.66 / 0.00 / 9,287,918.66 |
| C | 2,139.54 / 2,751.29 / 4,890.83 | 2,157.88 / 2,751.29 / 4,909.17 | 2,340.77 / 2,751.29 / 5,092.06 |
| D | 3,459.25 / 2,751.29 / 6,210.54 | 11,564.68 / 2,751.29 / 14,315.97 | 92,618.48 / 2,751.29 / 95,369.77 |

### Table 1.5: Latency (benchmarks, not sample figures)

| option | latency per pair (p50, p95) | D fallback tail |
|---|---|---|
| A | not measured (threshold comparison) | n/a (no LLM fallback) |
| B | p50 1.045 s, p95 1.330 s (per LLM call) | n/a (no LLM fallback) |
| C | p50 0.0826 ms, p95 0.2791 ms (one core, features included) | n/a (no LLM fallback) |
| D | p50 0.083 ms, p95 0.2803 ms (one core, features included, plus routing) | p50 0.0592 ms, p95 0.1323 ms, p99 0.2271 ms, p99.9 0.5671 ms, p99.95 921.2648 ms, max 4028.0308 ms [simulation from measured distributions] |

### Table 1.6: Break-even against B (S1 and S2 total; cost model, not sample figures)

| option | break-even vs B at $50/h, labels as evaluated | break-even vs B at $50/h, 2,000 labels | break-even vs B at $100/h, labels as evaluated | break-even vs B at $100/h, 2,000 labels | break-even vs B at $200/h, labels as evaluated | break-even vs B at $200/h, 2,000 labels |
|---|---|---|---|---|---|---|
| A | option cheaper everywhere in range | option cheaper everywhere in range | option cheaper everywhere in range | option cheaper everywhere in range | option cheaper everywhere in range | option cheaper everywhere in range |
| B | baseline | baseline | baseline | baseline | baseline | baseline |
| C | option cheaper above 1,477 records/day | option cheaper everywhere in range | option cheaper above 2,954 records/day | option cheaper everywhere in range | option cheaper above 5,908 records/day | option cheaper everywhere in range |
| D | option cheaper above 1,720 records/day | option cheaper everywhere in range | option cheaper above 3,441 records/day | option cheaper everywhere in range | option cheaper above 6,881 records/day | option cheaper above 1,078 records/day |

## Table 2: full band, not comparable with B's sample figures

| option | band F1 [95% CI] | band precision | band recall |
|---|---|---|---|
| A | 0.1058 [0.0652, 0.1472] | 1.0000 | 0.0559 |
| C | 0.9633 [0.9491, 0.9765] | 0.9743 | 0.9525 |
| D | 0.9862 [0.9770, 0.9943] | 0.9754 | 0.9972 |

## Secondary row: D (orig) (the original routing rule, lo 0.5, hi 0.9)

Same figures as Table 1, one per line. `missing` marks a figure the frozen outputs do not hold, with its expected source.

| figure | D (orig) |
|---|---|
| band F1 [95% CI] | 0.9656 [0.9336, 0.9865] |
| F1 vs B [95% CI] | -0.0200 [-0.0601, +0.0132] |
| F1 verdict | inconclusive |
| band precision | 0.9733 |
| band recall | 0.9581 |
| overall precision | 0.9971 |
| overall recall | 0.5688 |
| ECE | 0.0003 |
| meets every bar (sample basis) | not shown (inconclusive) |
| meets every bar (full-band basis, A, C and D only) | not computed (column is for A, C and D only) |
| LLM $ per 1,000 band pairs | 0.0006 |
| LLM $ per record | 0.00001 |
| band pairs per record | 20.0499 |
| monthly $ at 100K records/day (Precisely / customer / total) | missing (expected source: cost/cost_model.csv (it has no D (orig) rows; cost/options_vs_bars.csv holds only the 1M total)) |
| monthly $ at 1M records/day (Precisely / customer / total) | missing / missing / 5,667.81 (total only) |
| monthly $ at 10M records/day (Precisely / customer / total) | missing (expected source: cost/cost_model.csv (it has no D (orig) rows; cost/options_vs_bars.csv holds only the 1M total)) |
| latency per pair (p50, p95) | missing (expected source: options/latency/latency_summary.json (measures D with the v2 setting only)) |
| D fallback tail | missing (expected source: cost/assumptions.csv row 'D per-pair latency with the LLM fallback (simulation)') |
| break-even vs B at $50/h, labels as evaluated | missing (expected source: cost/break_even.csv (no row for D (orig))) |
| break-even vs B at $50/h, 2,000 labels | missing (expected source: cost/break_even.csv (no row for D (orig))) |
| break-even vs B at $100/h, labels as evaluated | missing (expected source: cost/break_even.csv (no row for D (orig))) |
| break-even vs B at $100/h, 2,000 labels | missing (expected source: cost/break_even.csv (no row for D (orig))) |
| break-even vs B at $200/h, labels as evaluated | missing (expected source: cost/break_even.csv (no row for D (orig))) |
| break-even vs B at $200/h, 2,000 labels | missing (expected source: cost/break_even.csv (no row for D (orig))) |

Full band (exact, not comparable with B's sample figures): band F1 [95% CI] 0.9689 [0.9563, 0.9806]; band precision 0.9800; band recall 0.9581

## Decoy outcomes (same name, different entity; test group)

| option | basis | decoy pairs | wrongly matched | source |
|---|---|---|---|---|
| A | 5,000-pair sample (weighted estimate) | 157.9 | 0.0 (0.00%) | results/eval_7919/options/a_threshold/score_report_sample5000.txt |
| A | full band (exact) | 111 | 0 (0.00%) | results/eval_7919/options/a_threshold/score_report.txt |
| B | 5,000-pair sample (weighted estimate) | 157.9 | 0.0 (0.00%) | results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt |
| C | 5,000-pair sample (weighted estimate) | 157.9 | 0.0 (0.00%) | results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt |
| C | full band (exact) | 111 | 0 (0.00%) | results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt |
| D | 5,000-pair sample (weighted estimate) | 157.9 | 0.0 (0.00%) | results/eval_7919/options/d_hybrid/score_report_sample5000.txt |
| D | full band (exact) | 111 | 0 (0.00%) | results/eval_7919/options/d_hybrid/score_report_fullband.txt |
| D (orig) | 5,000-pair sample (weighted estimate) | 157.9 | 0.0 (0.00%) | results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| D (orig) | full band (exact) | 111 | 0 (0.00%) | results/eval_7919/options/d_hybrid/score_report_fullband_original.txt |

B's decoy check: 66 test band decoy pairs, 0 false matches; one-sided 95% bound on the false-match rate <= 0.04438; 52 live calls and 14 cache hits, $0.0785 (sources: results/eval_7919/options/b_llm/openai/decoycheck/decoy_report.txt, results/eval_7919/options/b_llm/openai/decoycheck/run_meta.json).

## Cost numbers for the slides

- Effort log of the freeze run's stage 12: sha256 `1c09589013a42775707175347c9db8e8268e2343ba0a5d53050a117ce758704a` (source: results/freeze_2026-11-25/freeze_report.txt). The cost outputs read here are from `results/eval_7919/cost_final`, which used an effort log of 36 rows.
- First line of results/eval_7919/cost_final/summary.txt: "Effort log: 36 rows with hours, 2026-11-23 to 2026-11-25". No DRAFT banner.

### One-time build cost per option (USD; before = Verity pre-existing build and productization only; after = plus the option's own hours)

| option | $/h | build before | build after | labeling at 2,000 labels ($100/h) | labeling as evaluated ($100/h) |
|---|---|---|---|---|---|
| A | 50 | 9,000.00 | 9,250.00 | 0.00 | 0.00 |
| A | 100 | 18,000.00 | 18,500.00 | 0.00 | 0.00 |
| A | 200 | 36,000.00 | 37,000.00 | 0.00 | 0.00 |
| B | 50 | 9,000.00 | 9,278.71 | 0.00 | 0.00 |
| B | 100 | 18,000.00 | 18,553.71 | 0.00 | 0.00 |
| B | 200 | 36,000.00 | 37,103.71 | 0.00 | 0.00 |
| C | 50 | 9,000.00 | 9,225.00 | 1,000.00 | 33,015.50 |
| C | 100 | 18,000.00 | 18,450.00 | 1,000.00 | 33,015.50 |
| C | 200 | 36,000.00 | 36,900.00 | 1,000.00 | 33,015.50 |
| D | 50 | 9,000.00 | 9,353.71 | 1,000.00 | 33,015.50 |
| D | 100 | 18,000.00 | 18,703.71 | 1,000.00 | 33,015.50 |
| D | 200 | 36,000.00 | 37,403.71 | 1,000.00 | 33,015.50 |

Source: results/eval_7919/cost_final/one_time_cost.csv. Labeling at $50 and $200 per hour (from results/eval_7919/cost_final/summary.txt, 'label cost at $50/$100/$200'):

| option | labels | label hours | cost at $50 / $100 / $200 |
|---|---|---|---|
| C | as evaluated (66,031) | 330.2 | 16,508 / 33,016 / 66,031 |
| C | at 2,000 (2,000) | 10.0 | 500 / 1,000 / 2,000 |
| D | as evaluated (66,031) | 330.2 | 16,508 / 33,016 / 66,031 |
| D | at 2,000 (2,000) | 10.0 | 500 / 1,000 / 2,000 |

### Who pays at 1M records/day (USD per month, $100/h, labels as evaluated)

| option row | scenario | Precisely | customer | total | LLM paid by | hosting paid by | operations paid by | development paid by | labeling paid by |
|---|---|---|---|---|---|---|---|---|---|
| A | S1 | 1,659.96 | 0.00 | 1,659.96 | Precisely | Precisely | Precisely | Precisely | customer |
| A | S2 | 1,659.96 | 0.00 | 1,659.96 | customer | Precisely | Precisely | Precisely | customer |
| A | S3 | n/m | 118.29 | 118.29 | customer | customer | customer | Precisely (not modeled) | customer |
| B | S1 | 930,723.45 | 0.00 | 930,723.45 | Precisely | Precisely | Precisely | Precisely | customer |
| B | S2 | 2,164.44 | 928,559.01 | 930,723.45 | customer | Precisely | Precisely | Precisely | customer |
| B (own key connected) | S3 | n/m | 929,177.30 | 929,177.30 | customer | customer | customer | Precisely (not modeled) | customer |
| C | S1 | 2,157.88 | 2,751.29 | 4,909.17 | Precisely | Precisely | Precisely | Precisely | customer |
| C | S2 | 2,157.88 | 2,751.29 | 4,909.17 | customer | Precisely | Precisely | Precisely | customer |
| C | S3 | n/m | 3,371.67 | 3,371.67 | customer | customer | customer | Precisely (not modeled) | customer |
| D | S1 | 11,564.68 | 2,751.29 | 14,315.97 | Precisely | Precisely | Precisely | Precisely | customer |
| D | S2 | 2,579.03 | 11,736.94 | 14,315.97 | customer | Precisely | Precisely | Precisely | customer |
| D (LLM fallback off) | S3 | n/m | 3,771.68 | 3,771.68 | customer | customer | customer | Precisely (not modeled) | customer |
| D (own key connected) | S3 | n/m | 12,757.33 | 12,757.33 | customer | customer | customer | Precisely (not modeled) | customer |

Source: results/eval_7919/cost_final/cost_model.csv.

### Volume below which B is cheaper overall (S1 and S2 total over 12 months; break-even.csv)

| option vs B | $/h | labels | file's result | break-even records/day | B is cheaper overall |
|---|---|---|---|---|---|
| A | 50 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| C | 50 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| D | 50 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| A | 100 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| C | 100 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| D | 100 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| A | 200 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| C | 200 | 2,000 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| D | 200 | 2,000 | option cheaper above | 1,078 | below 1,078 records/day |
| A | 50 | 66,031 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| C | 50 | 66,031 | option cheaper above | 1,477 | below 1,477 records/day |
| D | 50 | 66,031 | option cheaper above | 1,720 | below 1,720 records/day |
| A | 100 | 66,031 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| C | 100 | 66,031 | option cheaper above | 2,954 | below 2,954 records/day |
| D | 100 | 66,031 | option cheaper above | 3,441 | below 3,441 records/day |
| A | 200 | 66,031 | option cheaper everywhere in range |  | nowhere in 1,000 to 100,000,000 records/day |
| C | 200 | 66,031 | option cheaper above | 5,908 | below 5,908 records/day |
| D | 200 | 66,031 | option cheaper above | 6,881 | below 6,881 records/day |

Where the result is 'option cheaper above N', B is cheaper below N records/day. Where it is 'option cheaper everywhere in range', B is not cheaper anywhere in the range 1,000 to 100,000,000 records/day. Source: results/eval_7919/cost_final/break_even.csv.

## Sources

Per column, the file each option's cell was read from (the full cell-by-cell list is `head_to_head_sources.csv`). 'derived' cells are plain arithmetic or the protocol's rules applied to cells read from files.

| table | column | source file per option |
|---|---|---|
| Table 1 and secondary row | band F1 [95% CI] | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| Table 1 and secondary row | F1 vs B [95% CI] | A, B, C, D, D (orig): results/eval_7919/bars_table.csv |
| Table 1 and secondary row | F1 verdict | A, B, C, D, D (orig): results/eval_7919/bars_table.csv |
| Table 1 and secondary row | band precision | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| Table 1 and secondary row | band recall | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| Table 1 and secondary row | overall precision | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| Table 1 and secondary row | overall recall | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| Table 1 and secondary row | ECE | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt |
| Table 1 and secondary row | meets every bar (sample basis) | A: results/eval_7919/options/a_threshold/score_report_sample5000.txt; results/eval_7919/options/a_threshold/score_report_sample5000.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the sample-basis report]; B: results/eval_7919/options/b_llm/openai/sample5000/score_report_sample5000.txt; results/eval_7919/options/a_threshold/score_report_sample5000.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the sample-basis report]; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report_sample5000.txt; results/eval_7919/options/a_threshold/score_report_sample5000.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the sample-basis report]; D: results/eval_7919/options/d_hybrid/score_report_sample5000.txt; results/eval_7919/options/a_threshold/score_report_sample5000.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the sample-basis report; LLM share from bars_table.csv]; D (orig): results/eval_7919/options/d_hybrid/score_report_sample5000_original.txt; results/eval_7919/options/a_threshold/score_report_sample5000.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the sample-basis report; LLM share from bars_table.csv] |
| Table 1 and secondary row | meets every bar (full-band basis, A, C and D only) | A: results/eval_7919/options/a_threshold/score_report.txt; results/eval_7919/options/a_threshold/score_report.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the full-band report]; B, D (orig):  [constant text]; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt; results/eval_7919/options/a_threshold/score_report.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the full-band report]; D: results/eval_7919/options/d_hybrid/score_report_fullband.txt; results/eval_7919/options/a_threshold/score_report.txt; results/eval_7919/bars_table.csv; results/eval_7919/options/latency/latency_summary.json [derived: the protocol's bars applied to the full-band report; LLM share from bars_table.csv] |
| Table 1 and secondary row | LLM $ per 1,000 band pairs | A, B, C, D, D (orig): results/eval_7919/results_table.csv |
| Table 1 and secondary row | LLM $ per record | A, B, C, D, D (orig): results/eval_7919/results_table.csv; results/eval_7919/cost_final/assumptions.csv [derived: per 1,000 band pairs x 20.0499 band pairs per record / 1,000] |
| Table 1 and secondary row | band pairs per record | A, B, C, D, D (orig): results/eval_7919/cost_final/assumptions.csv |
| Table 1 and secondary row | monthly $ at 100K records/day (Precisely / customer / total) | A, B, C, D: results/eval_7919/cost_final/cost_model.csv; D (orig): cost/cost_model.csv (it has no D (orig) rows; cost/options_vs_bars.csv holds only the 1M total) [missing] |
| Table 1 and secondary row | monthly $ at 1M records/day (Precisely / customer / total) | A, B, C, D: results/eval_7919/cost_final/cost_model.csv; D (orig): results/eval_7919/cost_final/options_vs_bars.csv [read (total only: the cost model has no Precisely/customer split for D (orig))] |
| Table 1 and secondary row | monthly $ at 10M records/day (Precisely / customer / total) | A, B, C, D: results/eval_7919/cost_final/cost_model.csv; D (orig): cost/cost_model.csv (it has no D (orig) rows; cost/options_vs_bars.csv holds only the 1M total) [missing] |
| Table 1 and secondary row | latency per pair (p50, p95) | A:  [constant text]; B: results/eval_7919/options/b_llm/openai/sample5000/timings.json; C, D: results/eval_7919/options/latency/latency_summary.json; D (orig): options/latency/latency_summary.json (measures D with the v2 setting only) [missing] |
| Table 1 and secondary row | D fallback tail | A, B, C:  [constant text]; D: results/eval_7919/cost_final/assumptions.csv; D (orig): cost/assumptions.csv row 'D per-pair latency with the LLM fallback (simulation)' [missing] |
| Table 1 and secondary row | break-even vs B at $50/h, labels as evaluated | A, B, C, D: results/eval_7919/cost_final/break_even.csv; D (orig): cost/break_even.csv (no row for D (orig)) [missing] |
| Table 1 and secondary row | break-even vs B at $50/h, 2,000 labels | A, B, C, D: results/eval_7919/cost_final/break_even.csv; D (orig): cost/break_even.csv (no row for D (orig)) [missing] |
| Table 1 and secondary row | break-even vs B at $100/h, labels as evaluated | A, B, C, D: results/eval_7919/cost_final/break_even.csv; D (orig): cost/break_even.csv (no row for D (orig)) [missing] |
| Table 1 and secondary row | break-even vs B at $100/h, 2,000 labels | A, B, C, D: results/eval_7919/cost_final/break_even.csv; D (orig): cost/break_even.csv (no row for D (orig)) [missing] |
| Table 1 and secondary row | break-even vs B at $200/h, labels as evaluated | A, B, C, D: results/eval_7919/cost_final/break_even.csv; D (orig): cost/break_even.csv (no row for D (orig)) [missing] |
| Table 1 and secondary row | break-even vs B at $200/h, 2,000 labels | A, B, C, D: results/eval_7919/cost_final/break_even.csv; D (orig): cost/break_even.csv (no row for D (orig)) [missing] |
| Table 2 | band F1 [95% CI] | A: results/eval_7919/options/a_threshold/score_report.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt; D: results/eval_7919/options/d_hybrid/score_report_fullband.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_fullband_original.txt |
| Table 2 | band precision | A: results/eval_7919/options/a_threshold/score_report.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt; D: results/eval_7919/options/d_hybrid/score_report_fullband.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_fullband_original.txt |
| Table 2 | band recall | A: results/eval_7919/options/a_threshold/score_report.txt; C: results/eval_7919/options/c_model/c_v1_lgbm/score_report.txt; D: results/eval_7919/options/d_hybrid/score_report_fullband.txt; D (orig): results/eval_7919/options/d_hybrid/score_report_fullband_original.txt |

# Verity hackathon demo prep

One file for the whole demo: positioning, running order and timing, the hardest questions, numbers to know, a glossary, and notes for each of the 30 slides. Slide numbers match the deck Verity_hackathon_deck.pptx.

## Contents

- 1. Positioning: what Verity is and is not
- 2. Running order and timing
- 3. The hardest questions, short answers
- 4. Numbers to know by heart
- 5. Before you record
- 6. Glossary
- 7. Slide-by-slide notes (slides 1 to 30)

## 1. Positioning: what Verity is and is not

**One sentence.** Verity is a services accelerator. It finds which fields to compare and how much each counts, so setup does not depend on a customer expert supplying match rules. It is not a new match engine.

**What DIS and MDM already do.** They match records with their own engines. Those engines may work better than Verity's score, and this test did not compare against them.

**What Verity adds.**

- Which fields to compare, learned from the data (34 seconds on 10,000 records).
- How much each field counts, as weights learned without labels.
- A match threshold, with a floor of 0.82.
- A field-by-field explanation of each match.
- A cheap, local way to settle borderline pairs, instead of paying an LLM for every pair.

**Three ways to use it** (slide 4).

| Mode | Who matches | What Verity does | Status |
|---|---|---|---|
| 1. Hand off | DIS or MDM | Learns fields, weights and a threshold; a consultant approves | Files and an HTML explorer exist; export into the products is not built |
| 2. Stand-alone | Verity, on the customer's desktop | Learns the setup, matches, explains | Built; options A to D tested on synthetic company data; run on address and product data with no mapping, checked by blind reviews of sampled pairs |
| 3. In tandem | DIS or MDM decides the clear pairs | Decides the borderline pairs (decision model, optional LLM) and shares results back | Decision logic tested (options C and D); the hand-off and return path are not built |

**How the hackathon test fits.** The four options are four ways to decide borderline pairs only: A the Verity score, B a frontier LLM, C a decision model, D a hybrid. The learned setup (fields, weights, threshold) and the decision on clear pairs are the same in every option. The test measured quality, cost and time on the same 44,022 borderline test pairs. The accelerator claim (finding fields and weights without a customer expert's rules) is shown on three files, not scored against an alternative. The test did not compare with the DIS or MDM engines.

**If asked "would the product engine do better?"** Say: "It may. I did not test it. Verity's job is to find the fields and weights and to settle the borderline pairs, so it works with the engine rather than replacing it. Stand-alone mode is for customers without an engine, or who need everything on a desktop."

**Say:** services accelerator, proposes, hands off, in tandem, settles borderline pairs.

**Do not say:** Verity beats the DIS or MDM engines; Verity replaces the engine; the integration exists; setup time saved against a customer expert (not measured).

## 2. Running order and timing

| Slide | Title | Section | Speaking time |
|---|---|---|---|
| 1 | Cover | Main flow |  |
| 2 | Problem and key question | Main flow | around 28 s |
| 3 | What Verity does | Main flow | around 13 s |
| 4 | Three ways to use Verity | Main flow | around 21 s |
| 5 | What Verity learns from the data | Main flow | around 15 s |
| 6 | Option A: Verity score only | Main flow | around 21 s |
| 7 | Option B: frontier LLM | Main flow | around 12 s |
| 8 | Option C: decision model | Main flow | around 19 s |
| 9 | Option D: hybrid | Main flow | around 16 s |
| 10 | Three test pairs, picked by rule | Main flow | around 16 s |
| 11 | Head to head on the same test pairs (required slide) | Main flow | around 34 s |
| 12 | How the monthly cost is built | Main flow | around 21 s |
| 13 | Where Verity adds value | Main flow | around 21 s |
| 14 | Cost and who pays | Main flow | around 16 s |
| 15 | Fit, risks and next steps | Main flow | around 19 s |
| 16 | Closing slide | Main flow |  |
| 17 | Backup divider | Backup |  |
| 18 | Protocol, freeze and reruns | Backup |  |
| 19 | Design choices in detail | Backup |  |
| 20 | Explaining one decision | Backup |  |
| 21 | Governance and review | Backup |  |
| 22 | Address data, no mapping | Backup |  |
| 23 | Product data, no mapping | Backup |  |
| 24 | Data realism | Backup |  |
| 25 | 97 of 100 (label check) | Backup |  |
| 26 | Option C needs labeled pairs (learning curve) | Backup |  |
| 27 | Time per decision and scale | Backup |  |
| 28 | Cost model assumptions | Backup |  |
| 29 | Full test set, exact figures | Backup |  |
| 30 | Limits we state | Backup |  |

Slides 2 to 15 add up to around 272 seconds (4:32) at 140 words a minute, plus the 9-second cover line, so around 4:41 in all. The limit is 5:00. Read it aloud once with a timer.

## 3. The hardest questions, short answers

1. **Isn't Verity just another match engine?** No. DIS and MDM keep their engines, which may match better. Verity finds which fields to compare and how much each counts, so setup does not rely on a customer expert's match rules. It can hand off to the product, match on its own on a desktop, or settle borderline pairs next to the product engine. (Slides 2, 4)
2. **Did you test against the DIS or MDM engines?** No. The test compared four ways of settling borderline pairs on the same test pairs. A comparison with the product engines on approved customer data is a next step. (Slides 4, 15, 30)
3. **Why is D "inconclusive" and why do you still recommend it?** D is 1.3 points below the LLM with a 95% range from -4.0 to +0.3 points. The bar is 2 points, so the 5,000-pair sample cannot say pass or fail. The point estimate is inside the bar, D costs 1.5% of the LLM, and every other bar passes. The recommendation is D where a 1.3-point gap is acceptable, the LLM where it is not. (Slide 11)
4. **Is the data realistic? It is synthetic.** No customer data may be used. The generator seeds acronyms, shortened names, PO boxes, typos, missing phones and same-name decoys. Defect rates were designed, not measured. A blind check of 100 pairs agreed with the generator on 97. Approved customer data is the first next step. (Slides 24, 25, 15)
5. **Was anything tuned on the test data?** No. Test data set 7919 was generated once, at the freeze run. The prompt was tuned on training pairs, C's cutoff on training pairs, and D's routing limits on five development data sets. 41 files are fingerprinted and anyone can check them with one command. (Slide 18)
6. **How much would this actually save?** At 1 million records a day, the hybrid costs $14,316 a month against $930,723 for the LLM: a modeled saving of $916,407 after build and labeling. At 100,000 records a day, $88,793. These are model outputs from stated assumptions. (Slides 13, 14, 28)
7. **How much expert time does Verity save?** Not measured. Measured: 34 seconds from input file to a learned setup on 10,000 records, with no match rules supplied. A consultant still reviews and approves. (Slide 13)
8. **Does it plug into DIS and MDM today?** No. The outputs are files and an HTML explorer. Export into DIS and MDM match configuration, and returning results in tandem mode, are next steps. (Slides 3, 4, 15)
9. **Does it work for products and locations?** It was run on company records (labeled), address data and two product files, with no mapping. Only the company records have an answer key. Blind reviews of sampled pairs agreed on 28 of 30 address pairs and 33 of 40 product pairs. These are small, exploratory checks, not accuracy figures. (Slides 15, 22, 23)
10. **How much weight can the blind reviews carry?** Little on their own. The samples are 30 and 40 pairs, the labels are mine, and the identifier path was designed after the product and address runs. They show the pipeline behaves sensibly on other data types, and they expose limits: same-name records with weak address agreement can still match, and near-matches on identifiers go to review. (Slides 22, 23, 30)
11. **Why 0.005 for the routing limit?** A rule committed before any call chose it from a grid of five values by pooled F1 over five development data sets. It is the smallest value in the grid, so it may be an edge effect; the grid was not extended afterwards. (Slide 9)
12. **What if the LLM is not allowed?** D with the LLM off decides exactly like C (0 pairs differ). C is a fixed model file, runs locally and costs $4,909 a month at 1 million records a day, 2.3 points below the LLM (inconclusive). (Slides 8, 9)
13. **How is the monthly cost built?** At 1 million records a day it is the LLM calls (around 610 million a month at $0.0015234 for option B, around 5.9 million for the hybrid), hosting, operations, and build and labeling spread over 12 months. For the LLM option nearly all of it is LLM calls; for the decision model it is mostly build and labeled examples. Operations hours, hosting prices and 5 hours of labeling per 1,000 pairs are assumptions. (Slide 13)
14. **What does 72 numbers per pair mean?** Verity already compares two records field by field. Option C summarizes that comparison as 72 numbers: 18 for the pair as a whole and 9 for each of 6 field types. It reads no record text, and 6 x 9 + 18 = 72 is fixed by the code. (Slide 8)
15. **What is not done or not measured?** Integration into the products, a comparison with the product engines, expert setup time today, a cold start after a reboot, approximate search timing, a rename-every-column test, locations, and the embedding model publisher clearance. (Slide 30)

## 4. Numbers to know by heart

| Number | Meaning |
|---|---|
| 44,022 / 358 | Borderline test pairs / true matches among them |
| 5,000 | Pairs in the sample where the LLM was scored |
| F1: 0.106 / 0.986 / 0.963 / 0.973 | Options A / B / C / D on the sample |
| -1.3 and -2.3 points | D and C against the LLM, both inconclusive against the 2-point bar |
| $1,660 / $930,723 / $4,909 / $14,316 | Monthly cost at 1 million records a day for A / B / C / D |
| $916,407 | Modeled monthly saving, D against the LLM, at 1 million records a day |
| 3,441 / 2,954 | Records a day above which D / C are cheaper than the LLM ($100 an hour, 66,031 labels) |
| 1.05 s / 0.083 ms | Typical time per pair, LLM / decision model |
| 0.97% / 11.5% | Pairs sent to the LLM / records with at least one LLM call (option D) |
| 34 s / 167 s | Learned setup ready / whole Verity run, 10,000 records |
| 66,031 / 2,000 | Labeled pairs in the freeze run / labels for a mean F1 of 0.90 on development data |
| 9 of 3,239 | False merges for C on the full test set (0.28%) |
| 97 of 100 | Blind label check agreement on the test data |
| 28 of 30 / 33 of 40 | Blind-review agreement on sampled address pairs / product pairs (exploratory) |
| 41 of 41 | Frozen files verified |
| $8.22 / $22.15 | Freeze-run LLM spend / project LLM spend |
| 21.5 hours | Logged hands-on effort |

## 5. Before you record

- Read the notes of slides 2 to 15 aloud once with a timer; the target is under 4:45. Slide 6 carries the line that all four options differ only in how they decide borderline pairs.
- Confirm the publisher clearance for all-MiniLM-L6-v2 under AIP3 Section 7, then update the last-but-one limit on slide 30.
- Decide whether to say "approximately 21 hours" (logged 21.5) or "approximately 20 hours" on slide 14.
- Open the deck in PowerPoint and check slides 4, 6 to 9 and 11, which are the densest.
- Keep the message straight: Verity is an accelerator that works with DIS and MDM, not a new match engine.
- No claim on expert setup time saved, no claim that integration exists, no comparison with the product engines, no accuracy claim for the address or product data (blind reviews only).

## 6. Glossary

- **F1 score:** One number from 0 to 1 balancing true matches found against wrong matches made.
- **Precision:** Share of the matches made that are right.
- **Recall:** Share of the true matches that were found.
- **Borderline pair:** Two records that look alike but may not be the same; the test set has 44,022.
- **Match engine:** Software that scores record pairs and decides which are the same. DIS and MDM have their own.
- **Services accelerator:** A tool a consultant uses to cut setup time on a customer project.
- **LLM:** Large language model, a general-purpose AI. Option B uses gpt-5.4.
- **LightGBM:** Method that builds many small decision trees, each correcting the previous ones; used by option C.
- **all-MiniLM-L6-v2:** Small open model that turns text into numbers; runs locally.
- **Threshold:** The score above which Verity calls two records a match; learned, with a floor of 0.82.
- **Freeze run:** The single run on the test data after all settings were fixed.
- **Inconclusive:** The sample cannot tell whether a gap is inside the 2-point bar.
- **DIS, MDM:** Data Integrity Suite; master data management.
- **S1, S2, S3:** Precisely hosts; customer brings its own LLM key; customer hosts.

## 7. Slide-by-slide notes

### Slide 1: Cover

**Section:** Main flow

**What the slide shows**

- Title "Verity", subtitle "Matching setup, learned and explained", presenter name, "Precisely Hackathon 2026".

**What to say**

> Recorded demo. All figures come from the freeze run on test data set 7919, committed under the tag freeze-2026-11-25. Nothing runs live.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| Freeze tag | freeze-2026-11-25 | `results/freeze_2026-11-25/freeze_manifest.json` |
| Test data set | seed 7919, generated once | `protocol.md` |

**Terms in plain words**

- **Verity:** The matching tool built before the hackathon. It learns matching setup from the data.
- **Freeze run:** The single run on the test data after every setting was fixed.

**Questions you may get**

- **Q: Is this a live demo?**
  A: No. Every number is pre-run, from the freeze run. Anyone can check the files with python eval/verify_freeze.py, which reports 41 of 41 files verified.

- **Q: What did the team build for the hackathon?**
  A: The comparison: four options scored on the same sealed test data with a written protocol, plus the cost model. Verity itself existed before. The effort log shows 21.5 hours of hands-on time for the hackathon work.

**Do not claim**

- Do not say the results are live or reproducible by running the video.

**Related slides:** Slide 18 (protocol and freeze) if asked how results were protected.

### Slide 2: Problem and key question

**Section:** Main flow | **Layout:** Three-column layout | **Speaking time:** around 28 seconds

**What the slide shows**

- Precisely product: DIS Data Quality and MDM already have matching engines. They match records that describe the same thing, such as a customer, a product or a location.
- Customer problem: each new data set needs match rules from a customer expert (which fields to compare, how much each counts, where the thresholds go). Pairs the engine cannot settle need review. An LLM can decide them but costs $930,723 a month at 1 million records a day.
- Key question, in two parts: 1. Can Verity find which fields to compare and how much each counts, from the data, without a customer expert writing the rules? 2. Can a local model settle borderline pairs as well as a frontier LLM, at lower cost and with less delay? Quality bar: within 2 points of the LLM on the F1 score.

**What to say**

> DIS Data Quality and MDM already match records, but each new data set needs a customer expert's match rules. A large language model, an LLM, can decide borderline pairs at $930,723 a month for 1 million records a day. Two questions: can Verity find the setup from the data, and can a local model settle borderline pairs within 2 points of the LLM on F1?

> Source: head_to_head_v3 Table 1.4.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| $930,723 a month | Total monthly cost of option B (LLM) at 1 million records a day, hosted by Precisely, $100 an hour, one-time costs spread over 12 months | `results/eval_7919/head_to_head_v3/head_to_head.md, Table 1.4` |
| 2 points | Quality bar written before testing: within 2 points of the LLM on F1 | `protocol.md, amendment on the F1 bar` |

**Terms in plain words**

- **DIS:** Data Integrity Suite, the Precisely product family. Data Quality is the part that matches records.
- **MDM:** Master data management: keeping one trusted record per customer, product or location.
- **Customer expert:** A subject-matter expert at the customer who knows which records are the same, and who normally supplies match rules.
- **LLM:** Large language model: a general-purpose AI such as the ones behind ChatGPT. Option B uses gpt-5.4.
- **F1 score:** One number from 0 to 1 balancing true matches found against wrong matches made. 1.0 is perfect.
- **Borderline pair:** Two records that look alike but may not be the same, so someone or something must decide.

**Questions you may get**

- **Q: Why two questions?**
  A: The accelerator job (finding fields and weights) and the borderline-pair decision are separate parts of the work. Question 1 is shown on three files with the learned setup. Question 2 is the measured head-to-head of options A to D against the LLM.

- **Q: Is Verity another match engine?**
  A: No. DIS and MDM keep their engines, which may match better than Verity's score. Verity finds which fields to compare and how much each counts, so setup does not depend on a customer expert supplying match rules. It can hand that to the product, match on its own, or settle the borderline pairs next to the product engine. See slide 4.

- **Q: If DIS and MDM already match, why Verity?**
  A: Their engines need a setup: fields, weights and thresholds. Today that relies on a customer expert's rules and repeated tuning. Verity proposes the setup from the data in 34 seconds on 10,000 records and explains each match. It can also settle the pairs the engine leaves open without paying an LLM for every pair.

- **Q: Why that quality bar?**
  A: It was set before any option was scored. 2 points on F1 is the tolerance for trading a small quality loss for large cost savings. The protocol records it as an amendment.

- **Q: How long does setup take a customer expert today?**
  A: Not measured in this project, so do not quote a number. What is measured: 34 seconds for Verity's learned setup on 10,000 records, with no match rules supplied.

- **Q: Why compare against an LLM?**
  A: The brief asks for a comparison against a frontier LLM on the same evaluation set. B is that baseline, and it is the tempting way to settle borderline pairs.

- **Q: Does Verity cover products and locations too?**
  A: It was run on customer-style company records (labeled), address data and two product files, with no mapping. Only the company records have a labeled answer key; the others were checked by blind review of sampled pairs (slides 22 and 23). Quality on product and location matching is not measured.

- **Q: Why is the LLM cost so high?**
  A: Each borderline pair needs one call, and a record has 20.05 candidate pairs on average. At $0.0015 a call and 1 million records a day, the calls dominate the bill.

**Do not claim**

- Do not say Verity replaces or beats the DIS or MDM engines.
- Do not state how long setup takes a customer expert today.
- Do not say the LLM is unusable; say it is expensive and slow at this volume.

**Related slides:** Slides 3, 4, 6 to 9 and 11.

### Slide 3: What Verity does

**Section:** Main flow | **Layout:** Journey layout with four icons | **Speaking time:** around 13 seconds

**What the slide shows**

- Four-step flow: reads records and works out what each field holds; learns fields to compare, weights and a threshold; explains each match field by field; hands the setup to DIS or MDM, or matches on its own.
- Text under the flow: a consultant approves the proposal; Verity runs locally on a sample of the customer's data. Pitch: Verity finds which fields to compare and how much each counts, so setup does not wait for a customer expert's rules, and a local model settles borderline pairs at a fraction of the LLM cost.

**What to say**

> Verity reads a customer's records, works out which fields to compare and how much each counts, learns a threshold, and explains each match field by field. A consultant approves the proposal.

> Today the proposal is a set of files and an HTML explorer; export into the products' configuration is a next step.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 34 s | Time from input file to a learned setup (start-up 20.6 s plus rule learning 13.2 s) on 10,000 records | `results/eval_7919/post_freeze/post_freeze_config.md, section 2` |
| 167 s | Whole Verity run on 10,000 records, all threads | `results/eval_7919/verity_data/run_info.json` |
| 0.82 | Fixed floor under the match threshold; the learned threshold was 0.4556 | `results/eval_7919/verity_data/learned_threshold.json` |

**Terms in plain words**

- **Threshold:** The score above which two records count as a match. Verity learns a value from the data and never lets it fall below 0.82.
- **Consultant:** A services or customer-success person who reviews and approves the proposed setup.
- **Learned setup:** The fields to compare, how much each counts, and the threshold, all derived from the data.

**Questions you may get**

- **Q: How does it find which fields matter?**
  A: It classifies each column from its values (name, street address, identifier-like, general text), then weights each field by how unique and how complete it is. Slide 5 shows the result.

- **Q: Does it need column mappings or match rules?**
  A: No columns were mapped and no match rules supplied on any of the three files. Slides 22 and 23 show two other schemas.

- **Q: Does it run locally?**
  A: Yes. Verity stores records in SQLite and runs on one machine. Only option B and the optional fallback in option D call an outside LLM.

- **Q: Is the hand-off into DIS and MDM built?**
  A: No. Outputs today are column_classification.json, attribute_weights.json, learned_threshold.json, learned_rules.md and an HTML match explorer. Export into DIS and MDM match configuration is a next step (slide 15).

- **Q: Is it deterministic?**
  A: Options A and C are: a fixed score or a fixed model file gives the same answer for the same input. Runs use PYTHONHASHSEED=0 and rehearsals reproduced earlier decisions with 0 differences.

**Do not claim**

- Do not say the integration exists.
- Do not say "never reads a column name": the column name text is part of the embedded text, and the rename test has not been run.

**Related slides:** Slides 4, 5, 19 and 22.

### Slide 4: Three ways to use Verity

**Section:** Main flow | **Layout:** Three process lanes plus banner and note | **Speaking time:** around 21 seconds

**What the slide shows**

- Three rows. 1. Hand off: Verity learns the setup, a consultant approves, DIS or MDM runs the matching, results stay in the product.
- 2. Stand-alone: customer records stay on the customer's desktop; Verity learns the setup, matches (options A to D) and returns matches with field-by-field evidence.
- 3. In tandem: DIS or MDM matches and decides the clear pairs; the borderline pairs go to Verity (decision model, with an optional LLM); decisions and evidence are shared back to DIS or MDM.
- Banner: Verity is not a new match engine. DIS and MDM keep theirs, and their engines may match better than Verity's score. Verity finds which fields to compare and how much each counts, so setup does not depend on a customer expert's match rules.
- Note: purple is Verity, dark purple is DIS or MDM. Tested here: how well Verity settles borderline pairs (options A to D). The hand-off and sharing results back are designed, not built.

**What to say**

> Verity is not a new match engine. DIS and MDM keep theirs. Verity can hand fields and weights to DIS or MDM, match on its own on the customer's desktop, or work in tandem: the product matches, and Verity settles the borderline pairs and shares the results back.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 3 modes | Hand off, stand-alone, in tandem | `This slide` |
| Options A to D | Four ways to decide borderline pairs, scored on the same 44,022 test pairs | `results/eval_7919/head_to_head_v3/head_to_head.md` |
| 0 | Comparisons run against the DIS or MDM engines in this test | `protocol.md` |

**Terms in plain words**

- **Hand off:** Verity proposes fields, weights and a threshold; the product's engine does the matching.
- **Stand-alone:** Verity does everything on the customer's own desktop, with no product engine involved.
- **In tandem:** The product matches; Verity settles the pairs the product leaves borderline and returns the results.
- **Match engine:** The software that scores record pairs and decides which are the same. DIS and MDM have their own.

**Questions you may get**

- **Q: Isn't this just another match engine?**
  A: No. DIS and MDM keep their engines, which may match better than Verity's score. The main job is finding which fields to compare and how much each counts, without relying on a customer expert's match rules. Verity can also match by itself where there is no product engine, or where everything must stay on a desktop, and it can settle borderline pairs next to the product engine.

- **Q: Did you compare Verity with the DIS or MDM engines?**
  A: No. The test measured how Verity settles borderline pairs against a frontier LLM. A comparison with the product engines on approved customer data is a next step, and those engines may do better.

- **Q: What exists today?**
  A: Verity's own matching and decisions (options A to D), the learned-setup files and an HTML explorer. The export into DIS and MDM configuration, and the return of results in tandem mode, are designed but not built.

- **Q: How do options A to D relate to the three modes?**
  A: They are four ways to decide borderline pairs. In stand-alone mode, Verity's own score decides the clear pairs in every option, and the option decides the borderline ones. In tandem mode, the product decides the clear pairs and the option decides the borderline ones. In hand-off mode none of them is used.

- **Q: Which option would you use in tandem?**
  A: Option D where a small quality gap is acceptable, or option C if no LLM is allowed. With the LLM off, D decides exactly like C.

- **Q: What if the product engine already handles borderline pairs well?**
  A: Then use it, and take only the fields, weights and explanations from Verity. The value is faster setup without relying on a customer expert.

- **Q: Why show a mode you have not built?**
  A: The slide marks it as designed, not built. It explains where the pieces would fit; the test measured the decision part.

**Do not claim**

- Do not say Verity is better than the product matching engines.
- Do not say the hand-off or the tandem return path exists.
- Do not call Verity a replacement for DIS or MDM matching.

**Related slides:** Slides 2, 3, 11 and 15.

### Slide 5: What Verity learns from the data

**Section:** Main flow | **Layout:** Table, 7 rows | **Speaking time:** around 15 seconds

**What the slide shows**

- A table of the seven fields on the test data: field, what Verity found, share unique, share blank, weight.
- Weights: company_name 28.3%, address_line 25.7%, phone 24.5%, postal_code 11.6%, website 9.8%, city 0.1%, state 0.1%.
- Footnote: weight is how much a field counts when two records are compared. Verity left out record_id and source_system by itself.

**What to say**

> Verity worked out each field's type from its values, and set weights from how unique and complete each one is. Name, address and phone count most. It dropped the ID and source columns itself.

> The score uses these weights, adjusted for the fields both records have. Source: post_freeze_config.md and attribute_weights.json from the freeze run.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 28.3% / 25.7% / 24.5% | Weights for name, address, phone | `results/eval_7919/verity_data/attribute_weights.json (normalized_discriminative_weight)` |
| 79.1% unique | Share of company_name values that are unique | `results/eval_7919/verity_data/attribute_weights.json` |
| 8.3% / 18.6% blank | Phone and website blank shares | `results/eval_7919/data_profile.txt` |

**Terms in plain words**

- **Share unique:** How many of the values in a field are distinct. A field where every record differs separates records well.
- **Weight:** Uniqueness times completeness, shown as a share of the total.
- **Identifier-like:** Verity's name for fields such as phone that behave like an identifier.

**Questions you may get**

- **Q: How are weights calculated?**
  A: Discriminative score = share unique times (1 minus share blank). Shares are normalized to sum to 1.

- **Q: Does the score use exactly these numbers?**
  A: For a pair, the weights are renormalized over the fields both records have. Pair 2 has no phone or website, so its address weight is 0.391 instead of 0.257 (slide 20). A second weight that blends cross-field reliability is computed but is display-only.

- **Q: Why do city and state get almost nothing?**
  A: Only 0.3% and 0.2% of their values are unique: nearly every record shares a city or state, so they hardly separate records.

- **Q: Why were record_id and source_system dropped?**
  A: record_id has 100% unique values (a primary key). source_system has 4 values, so it is near-constant. The classifier gives its reason for each in column_classification.json.

- **Q: Could a consultant change a weight?**
  A: That is the proposed review step. Pinning weights is a next step; it is not built.

- **Q: Do the weights come from labels?**
  A: No. They come from the data alone. Labels are used only for option C.

**Do not claim**

- Do not say the composite weight drives the score.
- Do not present these weights as tuned to the test labels; they are learned without labels.

**Related slides:** Slides 20, 22, 23 and 30.

### Slide 6: Option A: Verity score only

**Section:** Main flow | **Layout:** Process diagram in two groups plus three cards | **Speaking time:** around 21 seconds

**What the slide shows**

- Two group labels above the flow: "Same in every option" over steps 1 to 3, and "This option decides the borderline pairs" over steps 4 and 5.
- Steps 1 to 3 (grey, the same on all four option slides): learn the setup (fields to compare, weights, threshold); find and score pairs (all-MiniLM-L6-v2 finds each record's 30 closest look-alikes, Verity scores each pair); split the pairs (clear pairs are decided here; borderline pairs, scores 0.77 to 0.87, go on).
- Steps 4 and 5 (colored): Verity's score decides (match if the score clears the threshold, 0.82, and the field evidence supports it); answer (match or no match, with field-by-field evidence).
- Cards: benefits (no labels, no AI service, no per-pair fee; one processor core; explains each match), what it needs (the records and the open model), what it delivers (precision 1.000 but only 5.6% of true borderline pairs found; F1 0.106 on borderline pairs, fails the bar; $1,660 a month at 1 million records a day).

**What to say**

> In all four options, Verity learns the setup and decides the clear pairs. The options differ only in how they decide borderline pairs. Option A lets Verity's score decide them. It finds only 5.6% of the true borderline pairs, so F1 is 0.106 and it fails the bar.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.106 | F1 on the 5,000-pair sample; 95% range [0.067, 0.152]; fails the bar by -0.880 | `results/eval_7919/head_to_head_v3/head_to_head.md, Table 1.1` |
| 1.000 / 0.512 | Overall precision and recall | `results/eval_7919/head_to_head_v3/head_to_head.md` |
| 5.6% | Band recall: share of true borderline pairs found | `results/eval_7919/options/a_threshold/score_report.txt` |
| 0 of 2,909 | False merges among A's predicted matches on the full test set | `results/eval_7919/post_freeze/post_freeze_extras.md, section 1` |
| $1,660 | Monthly cost at 1 million records a day | `results/eval_7919/cost_final/cost_model.csv` |

**Terms in plain words**

- **F1 score:** Balances true matches found against wrong matches made. 1.0 is perfect.
- **Borderline pair:** Two records that look alike but may not be the same. 44,022 test pairs are in this group, 358 of them true matches.
- **all-MiniLM-L6-v2:** A small open text model (Apache 2.0, safetensors, pinned revision) that turns each record into numbers so look-alikes can be found. It runs locally and is used by every option to find the candidate pairs; options A, C and D also use it in the score.

**Questions you may get**

- **Q: What is the same in every option, and what changes?**
  A: Same: the learned setup (fields, weights, threshold), the look-alike search with all-MiniLM-L6-v2, Verity's score, and the decision on clear pairs. What changes is only how borderline pairs are decided: A by Verity's score, B by the LLM, C by the decision model, D by the decision model with the LLM for the unsure 0.97%. The quality figures and costs on these slides are for that borderline step.

- **Q: If A is perfectly precise, why fail?**
  A: A passes the precision bar but misses most true borderline pairs. Its F1 on that group is 0.106, which fails the 2-point bar by 0.88 points of F1.

- **Q: Why does A miss them?**
  A: The borderline group is, by construction, pairs whose score is close to the threshold. Most fail the evidence check (insufficient evidence). The hard cases are acronyms, shortened names, PO boxes and typos.

- **Q: What is the overall recall of 0.512?**
  A: The share of all true duplicate pairs that A matched, including ones far from the threshold. Overall recall for C and D is around 5.7 points higher.

- **Q: Why is A's latency "not measured"?**
  A: A is a threshold comparison on a score Verity already computes, so there is no separate model to time. The Verity pipeline itself took 167 s for 10,000 records.

- **Q: Is A just a baseline?**
  A: It is Verity's own decision, so it is both the starting point and the zero-label option.

**Do not claim**

- Do not say A is useless: it makes no wrong matches on this test (0 of 2,909).

**Related slides:** Slides 11 and 29.

### Slide 7: Option B: frontier LLM

**Section:** Main flow | **Layout:** Process diagram in two groups plus three cards | **Speaking time:** around 12 seconds

**What the slide shows**

- Two group labels above the flow: "Same in every option" over steps 1 to 3, and "This option decides the borderline pairs" over steps 4 and 5.
- Steps 1 to 3 (grey, the same on all four option slides): learn the setup (fields to compare, weights, threshold); find and score pairs (all-MiniLM-L6-v2 finds each record's 30 closest look-alikes, Verity scores each pair); split the pairs (clear pairs are decided here; borderline pairs, scores 0.77 to 0.87, go on).
- Steps 4 and 5 (colored): write the question (a fixed prompt: instructions, 8 solved examples, then the two records of the pair being checked); the LLM decides (gpt-5.4 answers same or different, with a probability; magenta marks the outside LLM).
- Cards: benefits (highest quality, F1 0.986; no labels; nothing to train), what it needs (an LLM account and network; records leave the customer's environment; a prompt tuned on training pairs, then frozen), what it delivers (F1 0.986 on borderline pairs, the reference; around 1 second a pair and 20 pairs a record; $930,723 a month).

**What to say**

> Option B, the baseline: gpt-5.4 decides every borderline pair, from one fixed prompt with 8 solved examples. F1 is 0.986, at 1.05 seconds a pair and $930,723 a month.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.986 | F1 on the sample; 95% range [0.959, 1.000] | `results/eval_7919/head_to_head_v3/head_to_head.md` |
| 1.05 s / 1.33 s | Typical and slow (95th percentile) time per call; 4,996 live calls | `results/eval_7919/latency_final/latency_final.md` |
| $0.001523 | Cost per call, measured in the freeze run; 85% of the prompt reused from cache | `results/eval_7919/options/b_llm/openai/sample5000 (timings.json, calls.jsonl)` |
| 20.05 | Candidate pairs per record, so around 2.1 s a record at 10 workers | `results/eval_7919/latency_final/latency_final.md, section 3` |
| $930,723 | Monthly cost at 1 million records a day | `results/eval_7919/cost_final/cost_model.csv` |
| 1 wrong match, 1 miss | LLM errors on the 5,000-pair sample | `results/eval_7919/post_freeze/post_freeze_extras.md` |

**Terms in plain words**

- **F1 score:** Balances true matches found against wrong matches made. 1.0 is perfect.
- **Borderline pair:** Two records that look alike but may not be the same. 44,022 test pairs are in this group, 358 of them true matches.
- **all-MiniLM-L6-v2:** A small open text model (Apache 2.0, safetensors, pinned revision) that turns each record into numbers so look-alikes can be found. It runs locally and is used by every option to find the candidate pairs; options A, C and D also use it in the score.
- **gpt-5.4:** The model version gpt-5.4-2026-03-05, with extended reasoning off, a 64-token answer limit and a fixed prompt (hash 857de291325e).
- **Prompt cache:** The repeated part of the prompt is billed at a lower price; the measured reuse is 85%.

**Questions you may get**

- **Q: What are "the two records" in the prompt?**
  A: The two records of the single look-alike pair being checked. Each call covers one pair: the same instructions and 8 solved examples, then record A and record B with their fields (name, address, city, state, ZIP, phone, website). Check b_system_prompt.txt for the exact layout.

- **Q: Which LLM and settings?**
  A: gpt-5.4-2026-03-05, reasoning effort none, one fixed prompt with 8 solved examples, answers saved. The prompt was tuned in two rounds on training pairs, then frozen before the test data existed.

- **Q: Could a better setting score higher?**
  A: Possibly. Extended reasoning was off for cost and latency. A higher setting could score higher and costs more. This is a stated limit.

- **Q: Whose account was used?**
  A: A personal API key, with synthetic data only. The champion confirmed this was acceptable. A customer deployment would use a customer or Precisely account.

- **Q: Does customer data leave the environment?**
  A: Yes in option B, and in D for the 0.97% of pairs sent to the LLM. Option A and C stay local.

- **Q: How much did the test cost?**
  A: The freeze run spent $8.22 on LLM calls (5,406 calls). All project LLM spend is $22.15, under the $45 cap.

- **Q: Why is B the quality reference if it made errors?**
  A: It is the baseline the brief asks for, scored on the same sample. Its errors on the sample are 1 wrong match and 1 miss.

- **Q: What do the options change?**
  A: Only how borderline pairs are decided. Steps 1 to 3 are the same on all four slides. See slide 6 for the full answer.

**Do not claim**

- Do not say B is the ground truth; the generator's labels are the truth.

**Related slides:** Slides 11, 14 and 27.

### Slide 8: Option C: decision model

**Section:** Main flow | **Layout:** Process diagram in two groups plus three cards | **Speaking time:** around 19 seconds

**What the slide shows**

- Two group labels above the flow: "Same in every option" over steps 1 to 3, and "This option decides the borderline pairs" over steps 4 and 5.
- Steps 1 to 3 (grey, the same on all four option slides): learn the setup (fields to compare, weights, threshold); find and score pairs (all-MiniLM-L6-v2 finds each record's 30 closest look-alikes, Verity scores each pair); split the pairs (clear pairs are decided here; borderline pairs, scores 0.77 to 0.87, go on).
- Steps 4 and 5 (colored): measure the pair (72 numbers per pair: Verity's own score and evidence figures, and the highest, average and lowest similarity of each type of field); the decision model decides (LightGBM, a decision-tree model, gives a probability; same if it clears a cutoff set on training pairs).
- Cards: benefits (no per-pair fee, one processor core; a fixed model file gives the same answer; the probability shows how sure it is), what it needs (labeled examples, 66,031 here, 2,000 gave a mean F1 of 0.90 on development data; the open text model), what it delivers (F1 0.963 on borderline pairs, 2.3 points below the LLM, inconclusive; 0.083 ms a pair; $4,909 a month).

**What to say**

> Option C turns each borderline pair into 72 numbers, mostly Verity's own similarity and evidence figures by field type, and feeds them to LightGBM, a decision-tree model trained on labeled pairs: 66,031 here. F1 is 0.963, in 0.083 milliseconds a pair, for $4,909 a month.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.963 | F1 on the sample; 95% range [0.932, 0.984]; gap to the LLM -0.023 [-0.064, +0.012], inconclusive | `results/eval_7919/head_to_head_v3/head_to_head.md` |
| 66,031 | Labeled training pairs in the freeze run | `results/eval_7919/cost_final/assumptions.csv` |
| 0.083 ms / 0.279 ms | Typical and slow (95th percentile) time per pair, one core, features included | `results/eval_7919/latency_final/latency_final.md` |
| 0.9 at 2,000 labels | Mean F1 on development data; 0.76 at 250; 0.51 at 100 | `results/dev_2026_run11/options/c_model/lc_v2/learning_curve.csv` |
| 9 of 3,239 | False merges on the full test set (0.28%) | `results/eval_7919/post_freeze/post_freeze_extras.md` |
| $4,909 | Monthly cost at 1 million records a day | `results/eval_7919/cost_final/cost_model.csv` |

**Terms in plain words**

- **F1 score:** Balances true matches found against wrong matches made. 1.0 is perfect.
- **Borderline pair:** Two records that look alike but may not be the same. 44,022 test pairs are in this group, 358 of them true matches.
- **all-MiniLM-L6-v2:** A small open text model (Apache 2.0, safetensors, pinned revision) that turns each record into numbers so look-alikes can be found. It runs locally and is used by every option to find the candidate pairs; options A, C and D also use it in the score.
- **LightGBM:** An open-source machine-learning method that builds many small decision trees, each one correcting the mistakes of the ones before. It learns from labeled examples and outputs a score for each pair, which Verity calibrates into a probability.
- **Labeled example:** A pair a person has marked "same" or "different".
- **Cutoff:** The probability above which C says "same", chosen on training pairs only.
- **72 numbers:** The model inputs for each pair. 18 are pair-level: Verity's own score, text similarity, evidence and conflict figures, the match type, and counts of fields compared. 54 are by field type (6 types x 9): the highest, average and lowest similarity, the highest, average and lowest evidence similarity, the type's total weight, and how many fields of that type were compared or missing. No record text is read.

**Questions you may get**

- **Q: What does "72 numbers per pair" mean, and why 72?**
  A: It is not 72 ways of reading the text. For each pair, Verity has already compared the two records field by field. Option C summarizes that comparison as 72 numbers: 18 on the pair as a whole (Verity's score, the text similarity, the evidence and conflict figures, the match type, how many fields were compared or missing) and 54 by field type (6 types, 9 numbers each: highest, average and lowest similarity, the same three for the evidence check, the type's total weight, and how many fields of that type were compared or missing). The count is fixed by the code (6 x 9 + 18). On the freeze data 58 of the 72 were ever non-empty and 48 varied, because the data has no ID or date fields. The point is to give the model more than the single score, for example a high overall score with a weak street-address similarity.

- **Q: Does C read the record text?**
  A: No. It sees only the numbers Verity already computed. C was frozen before the identifier comparison existed, so it does not know the code_signal or code_conflict labels.

- **Q: Where do the labels come from?**
  A: From the synthetic data generator's truth file. In a customer setting, labels would be pairs an expert approves. The cost model assumes 5 hours per 1,000 labeled pairs, paid by the customer.

- **Q: Is 66,031 labels realistic?**
  A: No customer would label that many. The learning curve (slide 26) shows 2,000 labels reach a mean F1 of 0.90 and 250 reach 0.76 on development data, with wide spread at small sizes.

- **Q: Why is C inconclusive and not a fail?**
  A: The quality bar is within 2 points. C is 2.3 points below with a 95% range from -6.4 to +1.2, so the sample can neither confirm nor reject the bar.

- **Q: Could C overfit the synthetic data?**
  A: Yes, that is a risk. It was trained on one record-level split with no test records in training, but all data is synthetic. A test on approved customer data is the first next step.

- **Q: Why is the answer the same every time?**
  A: C is a fixed model file. The same measurements always give the same probability.

- **Q: How was the cutoff chosen?**
  A: On training pairs only; the exact rule is in options/c_decision_model.py. On the test data the cutoff was 0.75.

- **Q: What do the options change?**
  A: Only how borderline pairs are decided. Steps 1 to 3 are the same on all four slides. See slide 6 for the full answer.

**Do not claim**

- Do not call C "AI" without saying it is a small decision-tree model trained on labeled pairs.
- Do not say the 2,000-label result is from the freeze run: it is a development result.

**Related slides:** Slides 9, 11, 14 and 26.

### Slide 9: Option D: hybrid

**Section:** Main flow | **Layout:** Process diagram in two groups plus three cards | **Speaking time:** around 16 seconds

**What the slide shows**

- Two group labels above the flow: "Same in every option" over steps 1 to 3, and "This option decides the borderline pairs" over steps 4 and 5.
- Steps 1 to 3 (grey, the same on all four option slides): learn the setup (fields to compare, weights, threshold); find and score pairs (all-MiniLM-L6-v2 finds each record's 30 closest look-alikes, Verity scores each pair); split the pairs (clear pairs are decided here; borderline pairs, scores 0.77 to 0.87, go on).
- Step 4 (colored): model C predicts (it turns the pair into 72 numbers, as in option C, and LightGBM gives the probability that the pair is the same). Step 5 splits in two: "C is sure: it decides" (probability below 0.005, or 0.9 and above: 99.03% of pairs) and "C is unsure: the LLM decides" (in between: 0.97% of pairs, magenta).
- Cards: benefits (close to the LLM at 1.5% of its monthly cost; the LLM sees only the hard pairs; with the LLM off, D decides like C), what it needs (everything C needs, plus an LLM account for the unsure pairs), what it delivers (F1 0.973 on borderline pairs, 1.3 points below the LLM, inconclusive; 11.5% of records include one LLM call; $14,316 a month).

**What to say**

> Option D lets C decide most borderline pairs and sends the 0.97% it can't settle to the LLM. With the LLM off, it equals C. F1 is 0.973, 1.3 points below the LLM, for $14,316 a month.

> IF ASKED: 11.5% of test records include at least one LLM call.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.973 | F1 on the sample; 95% range [0.936, 1.000]; gap -0.013 [-0.040, +0.003], inconclusive | `results/eval_7919/head_to_head_v3/head_to_head.md` |
| 426 of 44,022 | Pairs sent to the LLM (0.968%) | `results/eval_7919/options/d_hybrid/routed_fullband_pairs.csv` |
| 518 of 4,500 | Test records that include at least one LLM call (11.5%) | `results/eval_7919/latency_final/latency_final.md, section 3` |
| $14,316 | Monthly cost at 1 million records a day | `results/eval_7919/cost_final/cost_model.csv` |
| lo 0.005, hi 0.9 | Routing limits, frozen before the test data existed | `results/eval_7919/options/d_hybrid/frozen_settings_v2.json` |
| 2 errors | LLM errors on the 426 routed pairs | `results/eval_7919/post_freeze/post_freeze_extras.md, section 2` |

**Terms in plain words**

- **F1 score:** Balances true matches found against wrong matches made. 1.0 is perfect.
- **Borderline pair:** Two records that look alike but may not be the same. 44,022 test pairs are in this group, 358 of them true matches.
- **all-MiniLM-L6-v2:** A small open text model (Apache 2.0, safetensors, pinned revision) that turns each record into numbers so look-alikes can be found. It runs locally and is used by every option to find the candidate pairs; options A, C and D also use it in the score.
- **Routing:** Deciding which pairs go to the LLM: those where C's probability is from 0.005 up to, but not including, 0.9.

**Questions you may get**

- **Q: How were the routing limits chosen?**
  A: A rule written and committed before any call picked the lower limit from five candidates (0.005, 0.01, 0.05, 0.1, 0.5) by pooled F1 over five development data sets, with an LLM share of at most 1%. It chose 0.005, the smallest candidate, and the upper limit stayed 0.9.

- **Q: Is 0.005 an edge effect?**
  A: Yes, it is the smallest value in the grid, and F1 rose each time the limit went down. The grid was not extended after seeing results, because that would be a new selection. This is stated openly.

- **Q: Why did the original rule route fewer pairs?**
  A: The original rule (lower limit 0.5) sent 16 pairs, 0.036%, and scored 0.969 on the full test set against 0.986 for the final rule.

- **Q: What if the LLM is switched off?**
  A: D decides exactly like C. The freeze run verified 0 differing pairs.

- **Q: Why is D's sample precision 0.95 while the full-set precision is 0.975?**
  A: The sample counts a wrong match with a weight of 9.4, so one or two sampled errors move precision a lot. Use the full-set figures for precision and the paired sample difference for the comparison with the LLM.

- **Q: Why 11.5% of records?**
  A: Pairs routed to the LLM cluster on certain records. 518 of the 4,500 test records have at least one, with up to 12 on one record.

- **Q: Would more LLM routing help?**
  A: On development data, a lower limit caught more errors per pair, but it costs more calls. The routed share on the test data (0.97%) was four times the development average (0.26%).

- **Q: What do the options change?**
  A: Only how borderline pairs are decided. Steps 1 to 3 are the same on all four slides. See slide 6 for the full answer.

**Do not claim**

- Do not say the LLM is only used rarely at any scale: at 10 million records a day the hybrid costs $95,370 a month.

**Related slides:** Slides 11, 21 and 27.

### Slide 10: Three test pairs, picked by rule

**Section:** Main flow | **Layout:** Table, 7 rows by 4 columns | **Speaking time:** around 16 seconds

**What the slide shows**

- A table of three pairs. Pair 1: Falcon Evergreen Manufacturing Co, Tampa, same address, phone and website written two ways; decided by Verity alone (score 0.970); 0.23 ms, $0.
- Pair 2: Redstone Granite Medical Supply, Milwaukee, a PO box in one record and a street address in the other; decided by the decision model (Verity 0.846, C says 1.00); 0.083 ms, $0.
- Pair 3: the same PO box record against a shortened name, "Redstone Granite Me"; decided by the LLM (Verity 0.814, C 0.09, LLM 0.88); 0.96 s, $0.0015. All three answers are same, and all are correct.

**What to say**

> Three test pairs, picked by rule. Verity decides the first alone, in 0.23 milliseconds. The decision model settles the second in 0.083 milliseconds. The LLM decides the third in 0.96 seconds, for $0.0015. All 3 are correct.

> Source: demo_pairs.md.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.2257 ms | Verity's median scoring time per pair over 216,092 scored pairs (pair 1) | `results/eval_7919/demo_pairs/demo_pairs.md` |
| 0.0826 ms | Option C median over all test pairs (pair 2); not this pair's own time | `results/eval_7919/demo_pairs/demo_pairs.md` |
| 0.964 s, $0.001505 | This pair's own live call (pair 3) | `results/eval_7919/demo_pairs/demo_pairs.md` |
| 0.970414 / 0.846448 / 0.813628 | Verity scores for pairs 1, 2, 3 | `results/eval_7919/verity_data/scored_pairs.csv` |

**Terms in plain words**

- **Verity score:** A number from 0 to 1 for how alike two records are. The match threshold is 0.82.
- **Evidence check:** A second test that field-level evidence supports the match. Pairs 2 and 3 fail it, so Verity alone holds them back.
- **PO box:** A postal box address, which looks different from a street address.

**Questions you may get**

- **Q: Were these pairs hand-picked?**
  A: No. The rule: the first pair by record ID on each route (Verity decides outside the borderline group; C decides without the LLM; routed to the LLM). Selection read no truth file. The truth file only marked each one correct.

- **Q: Why do pairs 2 and 3 share a record?**
  A: The rule produced it: both are the first pair on their route and happen to involve the same record, R009d5c8244. It is a good illustration of the PO box and shortened name cases.

- **Q: Is "correct" from the answer key?**
  A: Yes, the generator's truth file, used only to mark the result after selection.

- **Q: Why did pair 3 need the LLM?**
  A: C gave it a probability of 0.086 (probably different), which is below 0.9 and above 0.005, so the rule sent it to the LLM, which said same with probability 0.88.

- **Q: Does this show the system is accurate?**
  A: No. Three pairs illustrate the routes. The accuracy evidence is the head-to-head on slide 11.

**Do not claim**

- Do not say pair 2's 0.083 ms is its own measurement.

**Related slides:** Slide 20 (explaining pair 2) and slide 11.

### Slide 11: Head to head on the same test pairs (required slide)

**Section:** Main flow | **Layout:** Table, 5 rows by 7 columns, plus banner and footnote | **Speaking time:** around 34 seconds

**What the slide shows**

- The required results table: one row per option (A, B, C, D) with match quality (F1, 95% range, gap to the LLM), precision and recall on the whole test set, LLM cost per 1,000 pairs, monthly cost at 100K, 1M and 10M records a day, time per pair, and the volume above which the option is cheaper than the LLM.
- Banner: recommendation to use option D where a 1.3-point gap to the LLM is acceptable; it costs 1.5% of the LLM at 1 million records a day; use the LLM where the gap is not acceptable.
- Footnote: definitions of precision, recall, F1 and "inconclusive", and the basis (5,000-pair sample of 44,022 borderline test pairs, 358 true; $100 an hour; 66,031 labeled examples).

**What to say**

> On the same 44,022 borderline pairs, the LLM scores an F1 of 0.986. Verity's score alone scores 0.106 and fails the bar. The decision model scores 0.963 and the hybrid 0.973, 1.3 points below the LLM, and both are inconclusive against the 2-point bar. At 1 million records a day the LLM costs $930,723 a month, the decision model $4,909 and the hybrid $14,316. I recommend the hybrid where a 1.3-point gap is acceptable, and the LLM where it isn't.

> IF ASKED: F1 balances precision and recall. The hybrid costs $0.015 per 1,000 pairs against $1.52 for the LLM. 11.5% of test records (518 of 4,500) include at least one LLM call. Sources: head_to_head_v3 Tables 1.1 to 1.6; verdict words come from bars_table.csv.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| F1 | A 0.106; B 0.986; C 0.963; D 0.973 (sample basis) | `results/eval_7919/head_to_head_v3/head_to_head.md, Table 1.1` |
| Gap to the LLM | A -0.880 (fails); C -0.023 [-0.064, +0.012]; D -0.013 [-0.040, +0.003] (both inconclusive) | `results/eval_7919/bars_table.txt (stage 11 output)` |
| Overall precision, recall | A 1.000, 0.512; B 0.997, 0.571; C 0.997, 0.569; D 0.994, 0.571 | `results/eval_7919/head_to_head_v3/head_to_head.md` |
| Cost per 1,000 pairs | B $1.52; D $0.015; A and C $0 | `results/eval_7919/head_to_head_v3/head_to_head.md, Table 1.3` |
| Monthly cost 100K / 1M / 10M | A $1.6K, $1.7K, $1.8K; B $95.0K, $931K, $9.29M; C $4.9K, $4.9K, $5.1K; D $6.2K, $14.3K, $95.4K | `results/eval_7919/cost_final/cost_model.csv` |
| Cheaper than the LLM above | C 2,954 records a day; D 3,441 (at $100 an hour, 66,031 labels) | `results/eval_7919/cost_final/break_even.csv` |

**Terms in plain words**

- **95% range:** The interval that would contain the true value in 95 of 100 repeats of the sample.
- **Inconclusive:** The 5,000-pair sample cannot tell whether the gap is inside the 2-point bar: the range runs from beyond -2 to above 0.
- **Precision:** Share of the matches made that are right.
- **Recall:** Share of the true matches that were found.
- **Break-even:** The volume above which an option is cheaper overall than the LLM, counting build and labeling.

**Questions you may get**

- **Q: Why is the verdict "inconclusive" and not pass?**
  A: For D, the gap to the LLM is -1.3 points with a 95% range from -4.0 to +0.3. The bar is -2.0. The range crosses it, so the sample cannot say pass or fail. Scoring the LLM on all 44,022 pairs would cost around $67 in calls, above the project spending cap.

- **Q: Why recommend D if the verdict is inconclusive?**
  A: The point estimate is inside the bar, the cost is 1.5% of the LLM, and every other bar passes. The recommendation says use D where a 1.3-point gap is acceptable and the LLM where it is not.

- **Q: What are the other bars and did they pass?**
  A: Overall precision at least 98% (pass), overall recall at least 5 points above A (C +5.65, D +5.93: pass), calibration error at most 0.05 (pass), time per pair for C under 20 ms (pass, 0.28 ms), LLM share for D at most 20% (pass, 0.97%).

- **Q: Why is recall only 0.57 overall?**
  A: Overall recall counts every true pair, and most of them sit outside the borderline group, where Verity alone decides. On development data, thousands of true pairs above the borderline group failed Verity's evidence check, and none of the options revisits them; the same pattern is likely here, so check the per-slice recall in the freeze report before answering in detail. It is the largest recall gain still available.

- **Q: Why are the figures on a sample for B but exact for others?**
  A: The LLM was scored only on a fixed 5,000-pair sample (every true pair plus random non-matches, weighted) because scoring all pairs would exceed the cap. Slide 29 shows exact full-set figures for A, C and D, which are not comparable with B.

- **Q: How are the monthly costs computed?**
  A: The cost model adds LLM calls, hosting, operations, and one-time build and labeling spread over 12 months, in scenario S1 (Precisely hosts). See slides 14 and 28.

- **Q: Why does D's cost jump at 10M records a day?**
  A: Its LLM share scales with volume: 0.97% of pairs at $0.0015 a call gives $92,618 of the $95,370 total.

- **Q: Is A's time "not measured"?**
  A: A is a threshold on Verity's score, so there is no separate model to time.

**Do not claim**

- Do not say D "matches" the LLM; say it is 1.3 points below and the verdict is inconclusive.
- Do not compare the sample F1 for B with full-set F1 for the others.

**Related slides:** Slides 6 to 9, 13, 14, 21, 29.

### Slide 12: How the monthly cost is built

**Section:** Main flow | **Layout:** Table, 7 rows by 5 columns, plus formula note | **Speaking time:** around 21 seconds

**What the slide shows**

- A table at 1 million records a day, one column per option and one row per cost item: LLM calls ($0, $928,559, $0, $8,986), hosting ($18, $18, $20, $20), operations ($100, $600, $600, $1,000), build spread over 12 months ($1,542, $1,546, $1,538, $1,559), labeled examples spread over 12 months ($0, $0, $2,751, $2,751), and the monthly total ($1,660, $930,723, $4,909, $14,316).
- Formula note: LLM calls = records a day x 30.4 days x 20.05 pairs per record x share of pairs sent to the LLM (B: all, D: 0.97%) x $0.0015 a call; build = hours x $100 / 12; labeled examples = 66,031 pairs x 5 hours per 1,000 x $100 / 12; hosting is the cheaper of an always-on server and pay-per-use.
- Measured versus assumed: pairs per record, share sent to the LLM, tokens and price per call are measured; the hourly rate, hosting prices, operations hours and 5 hours of labeling per 1,000 pairs are assumed.

**What to say**

> Here is how the monthly cost is built, at 1 million records a day. For the LLM option it is almost all LLM calls: around 610 million a month at approximately $0.0015 each. For the decision model, build and labeled examples, spread over 12 months, are most of the cost.

> IF ASKED: the hybrid sends 1% of pairs to the LLM. 30.4 days a month. Cost per call comes from 2,103 input tokens (85% cached) and 19 output tokens at list prices: $0.0015234, against $0.0015234 measured over 4,996 calls. Operations hours: A 1, B 6, C 6, D 10, at $100. Hosting assumes $0.05 per always-on core-hour or $0.12 per pay-per-use core-hour. Verity's 18 ms per record was timed on a 10,000-record run and scaled linearly. At 100,000 records a day: LLM option $95,004, hybrid $6,211. Source: cost_model.csv and cost/cost_model.py.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 609,516,960 calls | Option B: 1,000,000 records x 30.4 days x 20.0499 pairs per record | `cost_final/cost_model.csv; cost/cost_model.py` |
| $0.0015234 | Cost per call from 2,103 input tokens (85% cached) and 19 output tokens at list prices; measured spend over 4,996 calls was $7.611, the same to four digits | `results/eval_7919/options/b_llm/openai/sample5000` |
| 5,898,269 calls | Option D: 194,022 routed pairs a day (0.9677% of band pairs) x 30.4 | `cost_final/cost_model.csv` |
| 185 / 185.5 / 184.5 / 187 hours | Build hours for A / B / C / D: own hours + 4 shared Verity + 80 earlier build + 100 productization | `results/eval_7919/cost_final/one_time_cost.csv` |
| $2,751.29 | Labeling: 66,031 pairs x 5 hours per 1,000 x $100, spread over 12 months | `results/eval_7919/cost_final/summary.txt` |

**Terms in plain words**

- **Always-on server:** A server that runs and bills all month, priced per core-hour.
- **Pay-per-use:** Billing only for the processing time used. At these volumes it is cheaper.
- **Spread over 12 months:** One-time costs divided by 12 and added to each month.
- **Core-hour:** One processor core running for one hour.

**Questions you may get**

- **Q: Where does the LLM line come from?**
  A: Volume times price. At 1 million records a day, a month has 30.4 million records, each with 20.05 candidate pairs, so option B makes around 610 million calls at $0.0015234: $928,559. Option D sends 0.97% of pairs, around 5.9 million calls: $8,986.

- **Q: Is $0.0015 a call measured?**
  A: The model computes it from tokens and list prices (2,103 input tokens, 85% cached, and 19 output tokens). The measured spend over 4,996 live calls was $7.611, or $0.0015234 a call, so the two agree to four digits. The list prices are a quote dated 2026-11-24, not a measurement.

- **Q: What is hosting and how is it priced?**
  A: The cheaper of an always-on server ($0.05 per core-hour at 50% utilization) and pay-per-use ($0.12 per core-hour). At these volumes pay-per-use wins. Verity's 18.05 ms per record was timed on a 10,000-record development run and scaled linearly. Exact neighbor search grows faster than that, so hosting could be understated at large scale, but it is $18 to $20 of the total.

- **Q: Where do the operations hours come from?**
  A: They are assumptions: A 1 hour a month, B 6, C 6 and D 10 (monitoring, retraining and LLM upkeep), at $100 an hour.

- **Q: How is the build cost counted?**
  A: Own hours plus 4 shared Verity hours plus 80 hours of earlier Verity build plus 100 hours of productization, at $100, divided by 12. B and D also carry $3.71 of development API spend. D has more own hours (1 + 1.5 + 0.5) than C (0.5).

- **Q: Why do C and D carry labeling and B does not?**
  A: C and D need labeled pairs: 66,031 here, at an assumed 5 hours per 1,000, paid by the customer. B needs none.

- **Q: What changes at 100,000 records a day?**
  A: LLM option $95,004 and hybrid $6,211 a month. The LLM share falls with volume, and the fixed lines (operations, build, labeling) stay the same.

- **Q: Was the cost model checked?**
  A: Its 47 tests pass, and the four totals were recomputed by hand from the formulas: they match, with a 1-cent rounding difference on option B.

**Do not claim**

- Do not call the operations hours, hosting prices or the 5 hours per 1,000 labels measured.

**Related slides:** Slides 11, 13, 14 and 28.

### Slide 13: Where Verity adds value

**Section:** Main flow | **Layout:** Three-column layout | **Speaking time:** around 21 seconds

**What the slide shows**

- Three columns. Setup time: 34 seconds from input file to a learned setup on 10,000 records; no columns mapped and no match rules supplied by a customer expert, on customer, address and product data; a consultant reviews instead of building.
- Cost: $916,407 a month less than the LLM at 1 million records a day (option D); $88,793 a month less at 100,000 records a day; after counting build effort and labeled examples.
- Speed and explanation: a typical pair in 0.083 ms against 1.05 s; every match comes with its evidence field by field; 1.3 points below the LLM on F1, inconclusive against the 2-point bar.

**What to say**

> Here is where Verity adds value. Setup: 34 seconds from input file to a learned setup, with no match rules from a customer expert. Cost: the hybrid saves $916,407 a month at 1 million records a day. Speed: 0.083 milliseconds a pair against 1.05 seconds, with evidence for every match.

> IF ASKED: the saving is after build and labeling. At 100,000 records a day it is $88,793 a month. The 34 seconds is start-up plus rule learning on the freeze run (33.8 s). Savings are monthly totals from the cost model, scenario S1, $100 an hour, with one-time costs spread over 12 months. The time a specialist spends on setup today was not measured.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 34 s | 33.8 s = start-up 20.6 + rule learning 13.2, 10,000 records | `results/eval_7919/post_freeze/post_freeze_config.md, section 2` |
| $916,407 | B $930,723.45 minus D $14,315.97, per month at 1M records a day | `results/eval_7919/cost_final/cost_model.csv` |
| $88,793 | B $95,003.87 minus D $6,210.54 at 100K records a day | `results/eval_7919/cost_final/cost_model.csv` |
| 0.083 ms vs 1.05 s | Typical time per pair, option C or D vs the LLM | `results/eval_7919/latency_final/latency_final.md` |

**Terms in plain words**

- **Learned setup:** Which fields count, how much each counts, and the match threshold, all derived from the data.
- **Monthly saving:** The difference between two modeled monthly totals; it is not a measured saving.

**Questions you may get**

- **Q: How much time does this save a specialist?**
  A: Not measured, so no claim is made. What is measured: 34 seconds for the learned setup on 10,000 records, with no mapping or tuning by hand. The consultant still reviews and approves the proposal.

- **Q: Is the $916,407 a real saving?**
  A: It is the difference between two modeled monthly costs at 1 million records a day, after build and labeling spread over 12 months. It depends on the assumptions on slide 28. At 100,000 records a day it is $88,793.

- **Q: Does the saving include labeling?**
  A: Yes. Labeling for C and D ($33,016 for 66,031 labeled pairs, spread over 12 months) is inside the totals.

- **Q: Is 34 seconds the whole run?**
  A: No. The whole Verity run took 167 s for 10,000 records. 34 s is when the learned setup (types, weights, threshold) is ready.

- **Q: Why compare against the LLM only?**
  A: The brief asks for the frontier LLM baseline. Option A has no model cost but fails the quality bar.

- **Q: Does Verity replace the product matching engine?**
  A: No. It proposes the setup the engine runs, can settle the borderline pairs next to it, or can match by itself where there is no engine (slide 4).

**Do not claim**

- Do not claim time saved against a specialist.
- Do not describe the $916,407 as measured.
- Do not say the saving applies to a customer expert's time.

**Related slides:** Slides 11, 14 and 28.

### Slide 14: Cost and who pays

**Section:** Main flow | **Layout:** Three-column layout | **Speaking time:** around 16 seconds

**What the slide shows**

- Cost to build: approximately 21 hours of hands-on hackathon time; approximately $18,500 per option (Verity build 80 hours and productization 100 hours); labeled examples for C and D $33,016 for 66,031 pairs, paid by the customer.
- Who pays at 1 million records a day: with Precisely hosting the totals are $930,723 (LLM), $4,909 (decision model), $14,316 (hybrid); with the customer's own LLM key the customer pays the LLM fees ($928,559 for the LLM option) and each total stays the same.
- Where it runs: if the customer hosts, the decision model costs $3,372 a month; the LLM option cannot run where data cannot leave; below 3,441 records a day the LLM is cheaper overall (hybrid, $100 an hour).

**What to say**

> Hackathon effort was approximately 21 hours of hands-on time. Each option carries approximately $18,500 of build, and the customer pays for labeled examples: $33,016 for 66,031 pairs. Below 3,441 records a day the LLM is cheaper overall.

> Customers who can't send data out run the decision model alone. Sources: cost_final one_time_cost.csv, cost_model.csv, break_even.csv and effort_log.csv (21.5 hours). The labeling payer is an assumption.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 21.5 hours | Hands-on effort: Verity 4.5, A 1.0, B 3.75, C 1.5, D 1.0, evaluation 9.75 | `cost/effort_log.csv` |
| $18,500 | One-time build per option at $100 an hour: A $18,500; B $18,554; C $18,450; D $18,704 | `results/eval_7919/cost_final/one_time_cost.csv` |
| $33,016 | 66,031 labels, 330.2 hours at 5 hours per 1,000, $100 an hour; $1,000 at 2,000 labels | `results/eval_7919/cost_final/summary.txt` |
| 3,441 / 2,954 | Break-even volumes for D and C at $100 an hour and 66,031 labels; at 2,000 labels both are cheaper at every volume | `results/eval_7919/cost_final/break_even.csv` |
| $928,559 | LLM fees the customer pays in the customer-key scenario | `results/eval_7919/cost_final/cost_model.csv` |

**Terms in plain words**

- **Scenarios:** S1: Precisely hosts everything. S2: the customer brings its own LLM key. S3: the customer hosts everything.
- **Productization:** Turning the prototype into a supported product; estimated at 100 hours.
- **Spread over 12 months:** One-time costs are divided by 12 and added to each monthly total.

**Questions you may get**

- **Q: Where does 21 hours come from?**
  A: The effort log counts hands-on time only: 21.5 hours, split Verity 4.5, the four options 7.25 (A 1.0, B 3.75, C 1.5, D 1.0) and the evaluation 9.75. Waiting for runs and review time are not counted. Verity itself took an estimated 80 hours earlier.

- **Q: Why does each option carry $18,500?**
  A: 80 hours of Verity build and 100 hours of productization at $100 an hour, shared by every option, plus the option's own hours.

- **Q: Where does the 5 hours per 1,000 labels come from?**
  A: An assumption taken from a hand-check row in the effort log (0.5 hours for 100 pairs). It was blind agreement checking, not a labeling workflow, so treat it as a placeholder. At 2,000 labels the cost is $1,000.

- **Q: Why does the customer pay for labels?**
  A: An assumption: customer experts know which records are the same. The cost model reports Precisely and customer costs separately.

- **Q: What if the customer hosts?**
  A: Scenario S3: the decision model costs the customer $3,372 a month, and the LLM option costs $929,177 only if data may leave, which is why it is marked as unable to run where data cannot leave.

- **Q: Do the rankings change under other assumptions?**
  A: On the development run, no. The tests covered $50 to $200 an hour, pairs per record, LLM price plus or minus 50%, cache reuse, and shared setup cost 1 to 100 times.

**Do not claim**

- Do not present 5 hours per 1,000 labels as measured.

**Related slides:** Slides 11, 13 and 28.

### Slide 15: Fit, risks and next steps

**Section:** Main flow | **Layout:** Three-column layout | **Speaking time:** around 19 seconds

**What the slide shows**

- Where it fits: a services accelerator that finds which fields to compare and how much each counts; tested on customer, address and product data with no columns mapped; works three ways (hand off to DIS or MDM, stand-alone on the customer's desktop, or in tandem settling borderline pairs); not a replacement match engine.
- Key risks: all test data is synthetic; the DIS and MDM engines were not tested here and may match better than Verity's score; address and product data were checked on sampled pairs, not a full labeled test.
- Next steps: test on approved customer data, including product and location data; compare with DIS and MDM matching, then export the proposal into their match configuration; let a consultant review and pin field weights and review rules; productization approximately 100 hours.

**What to say**

> Verity fits as a services accelerator that finds which fields to compare and how much each counts, tested on customer, address and product data. The main risk is synthetic data, so the next step is approved customer data and a comparison with the product engines.

> Productization is approximately 100 hours.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 100 hours | Estimated productization effort | `results/eval_7919/cost_final/assumptions.csv` |
| 28 of 30 and 33 of 40 | Blind-review agreement on sampled address pairs and sampled product pairs | `results/second_schema_v5_review and results/product_v5_review (local, gitignored)` |
| 0.97% vs 0.09% to 0.52% | LLM-routed share on the test data vs on five development data sets | `results/eval_7919/options/d_hybrid/routed_fullband_pairs.csv; results/dstudy_summary` |

**Terms in plain words**

- **Services accelerator:** A tool a consultant uses to cut setup time on a customer project, rather than a new product.
- **Pin a weight:** Fix a field's weight by hand instead of using the learned one.
- **Review rules:** How a consultant wants near-matching pairs treated, for example sent to a person instead of merged.

**Questions you may get**

- **Q: Why call it an accelerator and not a product?**
  A: Verity proposes setup and explanations; the products can run the matching. Productization would add export, support and testing, estimated at 100 hours.

- **Q: What has been tested beyond company records?**
  A: Address data (14,570 records, 7 columns) and two product files, all with no mapping. They have no labeled answer key, so the checks are blind reviews of sampled pairs: 28 of 30 address pairs and 33 of 40 product pairs agreed with my labels. These are small samples and exploratory (slides 22 and 23).

- **Q: What is the biggest risk?**
  A: Synthetic data: the defect rates were designed and may flatter every option. The second risk is that the product engines were not tested and may match better.

- **Q: What would a pilot look like?**
  A: Run Verity on a sample of one customer's records, review the proposed fields and weights with a consultant, then compare with the existing DIS or MDM configuration. This is a proposal, not a plan with dates.

- **Q: Who owns export into the product?**
  A: Not decided. It is a next step; no work has started.

**Do not claim**

- Do not say Verity is integrated, or that it has been compared with the product engines.
- Do not call the blind reviews accuracy results.

**Related slides:** Slides 4, 22, 23 and 30.

### Slide 16: Closing slide

**Section:** Main flow

**What the slide shows**

- Precisely logo slide.

**What to say**

> Backup slides follow.

**Questions you may get**

- **Q: What is the one-sentence summary?**
  A: Verity learns which fields to match on and how much each counts, explains each match, and a local decision model settles most borderline pairs at around 1.5% of the LLM's monthly cost, within 1.3 points of its quality (inconclusive against the 2-point bar).

**Related slides:** Slides 11 and 13.

### Slide 17: Backup divider

**Section:** Backup

**What the slide shows**

- Section divider with the title "Backup".

**What to say**

> Not spoken in the 4 to 5 minute run.

**Questions you may get**

- **Q: When do I use backup slides?**
  A: Only for questions after the recording, or if the format allows a longer session.

**Related slides:** Slides 18 to 30.

### Slide 18: Protocol, freeze and reruns

**Section:** Backup

**What the slide shows**

- Eight bullets: quality bars and borderline test pairs defined before scoring; 9 dated amendments; test data set 7919 generated once on Nov 25; 41 files fingerprinted and tagged freeze-2026-11-25 before any test data existed; 14 stages, no deviation, 40 safety checks none fired; freeze run cost $8.22, project LLM spend $22.15; labeled evaluation found two scoring flaws (precision 77% to 100% at the same recall); one-command verification, and two later fixes that left the test-data outputs byte-identical.

**What to say**

> The manifest lists every frozen file with its hash, and the tag points to the manifest commit. Running eval/verify_freeze.py from a fresh clone reports 41 of 41 files verified.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 41 of 41 | Frozen files that match the manifest | `results/freeze_2026-11-25/VERIFY.md` |
| 9 | Dated amendments in the protocol | `protocol.md` |
| 0.7695 to 1.0000 | Precision on development data, run 1 vs run 9, at recall 0.5115 vs 0.5114 | `results/dev_2026_run9; protocol notes` |
| $8.22 / $22.15 | Freeze-run LLM spend and project total | `results/llm_ledger.jsonl` |

**Terms in plain words**

- **Freeze:** Fixing every setting and fingerprinting the code before any test data is created.
- **Amendment:** A dated, append-only change to the written protocol.
- **Hash:** A fingerprint of a file; if the file changes, the fingerprint changes.

**Questions you may get**

- **Q: What were the two scoring flaws?**
  A: A bonus for records from different source systems, and matches based on text similarity alone. Both showed up on development data once labeled evaluation existed. Removing them raised precision from 0.77 to 1.00 at the same recall.

- **Q: Did anything change after the freeze?**
  A: Yes, one fix for column names with spaces (commit d51c5b9). The test-data outputs stayed byte-identical (same hashes for scored_pairs.csv and match_candidates.csv). Frozen files are unchanged: verify_freeze reports 41 of 41.

- **Q: Was the prompt or any threshold tuned on the test data?**
  A: No. The prompt was tuned on training pairs and frozen. The C cutoff came from training pairs. The routing limits were chosen on five development data sets. The test data was generated once, at the freeze.

- **Q: What if a stage had failed?**
  A: The runbook allows a rerun only for infrastructure failures, with identical inputs. None occurred.

- **Q: What are the 9 amendments?**
  A: Evaluation seed change; exact neighbor search; three clarifications on the band; record-level split; F1 bar instead of accuracy; LLM options on a 5,000-pair sample; budget correction; hybrid routing rule v2; scope narrowed to options A to D.

- **Q: What changed after the freeze, and how do you know it did not affect the results?**
  A: Two fixes to Verity: one key function so column names with spaces work (commit d51c5b9), and a comparison path for pairs with no multi-word text (code path, revised three times after the product and address runs). Both were checked by rerunning the test data: scored_pairs.csv and match_candidates.csv have the same sha256 as the freeze run, and the code path scored 0 pairs on the test data. The 41 frozen files still verify.

- **Q: Is the code path part of the protocol result?**
  A: No. It was designed after the product and address runs, so all of it is exploratory. It cannot change the test-data results, which was checked.

**Do not claim**

- Do not say the code never changed after the freeze; say the frozen files did not, and one later fix left the outputs identical.

**Related slides:** Slides 19 and 30.

### Slide 19: Design choices in detail

**Section:** Backup

**What the slide shows**

- Left: records in SQLite; text-to-numbers with all-MiniLM-L6-v2; 30 closest look-alikes per record by exact search; score blends text similarity with field comparison (text share capped at 0.65), threshold learned from sampled pairs with a floor of 0.82; borderline pairs are scores within 0.05 of the threshold (0.77 to 0.87); pairs with no multi-word text are compared field by field on identifiers, and a differing identifier sends the pair to review.
- Right: A, B, C, D in one line each; evaluation: split by record, exact search, test data generated once.

**What to say**

> Sources: protocol.md, scored_pairs_columns.json, frozen_settings_v2.json, frozen_prompt.json.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.65 / 0.35 | Cap on the text share of the score; the field share is the remainder. On the test data the cap applied (the data-derived value was 0.79) | `results/eval_7919/post_freeze/post_freeze_config_corrections.md` |
| 0.455581 and 0.82 | Learned threshold and the effective threshold after the floor | `results/eval_7919/verity_data/learned_threshold.json` |
| 0.77 to 0.87 | The borderline range around 0.82, plus or minus 0.05 | `protocol.md` |
| 30 | Nearest neighbors per record | `verity/src/match.py` |

**Terms in plain words**

- **SQLite:** A small database stored in one file on the local machine.
- **Exact search:** Comparing every record with every other to find the closest, as opposed to an approximate shortcut.

**Questions you may get**

- **Q: Why SQLite?**
  A: It is a single local file with no database server to install, which suits a tool that runs next to the customer's data.

- **Q: How is the text share set?**
  A: From how evenly the field weights are spread (an entropy ratio), clamped between 0.12 and 0.65. On this data the ratio was 0.79, so the 0.65 cap applied.

- **Q: Why exact search and not approximate?**
  A: Exact search made evaluation reproducible. Approximate search timing was not measured, so no claim is made on scale beyond 10,000 records.

- **Q: Why a floor of 0.82?**
  A: A fixed floor that stops a low learned threshold (0.4556 here) from matching too many pairs. It applied on all three data files.

- **Q: Why is the embedding model safe to use?**
  A: Apache 2.0 license, safetensors format, pinned revision (1110a243fdf4706b3f48f1d95db1a4f5529b4d41), no remote code. Publisher clearance under AIP3 Section 7 is not yet confirmed.

- **Q: How are pairs of codes handled, such as part numbers?**
  A: The text score needs multi-word text in both records, because records made only of codes look nearly identical to the text model and would match falsely. That guard has been in Verity since the original repo (added 2026-06-12 for exactly that reason). Pairs that fail it are now compared field by field on identifiers, with no text model. An identifying field that differs sends the pair to review instead of merging. Identifying means a code-like column whose learned weight is at least 1/(2n) for n scoring columns.

- **Q: Is the conflict rule new?**
  A: The evidence and conflict check has been in Verity since before the hackathon: a pair cannot match if disagreeing fields carry too much weight (limits unchanged since the original repo). The hard rule for identifier fields is new, and extends that check.

**Do not claim**

- Do not say the 0.65 text share is learned; it is a cap on a data-derived value.

**Related slides:** Slides 3, 6 and 30.

### Slide 20: Explaining one decision

**Section:** Backup

**What the slide shows**

- A table for demo pair 2: company_name (text similarity 0.944, weight 0.430, evidence 0.986), address_line (0.158, 0.391, 0.148), postal_code (0.714, 0.176, 0.000), city (1.000, 0.001, 1.000), state (1.000, 0.001, 1.000).
- Note: Verity alone holds the pair back (evidence check 0.002) because a PO box does not match a street address and a ZIP+4 does not match a five-digit ZIP; the decision model gives 1.00 and the answer, same, is correct.

**What to say**

> Pair 2 from the demo. Evidence similarity is what the evidence check reads; the score also blends in text similarity.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.001717 | Evidence mass for pair 2 (the evidence check fails) | `results/eval_7919/post_freeze/post_freeze_extras.md, section 3` |
| 0.657507 | Evidence mass for pair 1, which passes | `results/eval_7919/post_freeze/post_freeze_extras.md` |

**Terms in plain words**

- **Evidence similarity:** A stricter field similarity the evidence check reads: a ZIP+4 against a five-digit ZIP scores 0.000.
- **Field weight:** How much the field counts for this pair, adjusted for the fields both records have.

**Questions you may get**

- **Q: Why are the weights different from slide 5?**
  A: Pair 2 has no phone or website. The weights are renormalized over the fields both records have, so address_line rises from 0.257 to 0.391.

- **Q: What would let Verity decide pairs like this on its own?**
  A: Normalizing ZIP+4 and phone formats in the evidence check. On development data, 166 true pairs with the same phone digits in different formats failed the check. This is a next step, not built.

- **Q: Is this an official result?**
  A: It is an exhibit built from stored values, marked exploratory. No model was rerun.

**Do not claim**

- Do not say the explanation comes from the LLM or the decision model; it comes from Verity's stored field evidence.

**Related slides:** Slides 5 and 10.

### Slide 21: Governance and review

**Section:** Backup

**What the slide shows**

- Left: false merges (two different records wrongly matched) on the full test set: A 0 of 2,909, C 9 of 3,239 (0.28%), D 9 of 3,255 (0.28%); 0 of 111 same-name different-company pairs matched wrongly; the LLM made 2 errors on the 426 routed pairs, and a perfect human reviewer would lift F1 from 0.986 to 0.989.
- Right: if data stewards took D's routed pairs instead of the LLM, the queue is 9.7 per 1,000 borderline pairs (approximately 19,400 pairs a day at 100K records, 194,000 at 1 million); the original rule sends 0.36 per 1,000 and scores 0.969 against 0.986. Marked post-freeze and exploratory.

**What to say**

> Source: post_freeze_extras.md sections 1 and 2.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 9 of 3,239 | C false merges; 95% range 0.13% to 0.53% | `results/eval_7919/post_freeze/post_freeze_extras.md` |
| 9.677 per 1,000 | Routed pairs per 1,000 borderline pairs for D | `results/eval_7919/post_freeze/post_freeze_extras.md` |
| 194,022 | Pairs a day if stewards took D's routed pairs at 1M records a day | `results/eval_7919/post_freeze/post_freeze_extras.md` |
| 0.9862 to 0.9889 | Full-set F1 for D with a perfect steward (upper bound) | `results/eval_7919/post_freeze/post_freeze_extras.md` |

**Terms in plain words**

- **False merge:** Two different real-world records wrongly matched into one.
- **Data steward:** The person who reviews uncertain pairs.
- **Decoy pairs:** Different companies given the same name on purpose to test false merges.

**Questions you may get**

- **Q: Is a false-merge rate a hard limit?**
  A: It is reported, not set as a limit. C and D each made 9 on the full test set (0.28%).

- **Q: Could stewards replace the LLM?**
  A: The queue would be 194,022 pairs a day at 1 million records, which is unrealistic. That is why the LLM fallback exists. A narrower rule shrinks the queue to 7,287 pairs a day but scores 0.969 instead of 0.986.

- **Q: Why does D have the same 9 false merges as C?**
  A: The LLM changed the decisions on 20 pairs and made 2 errors among the 426 routed. The sample basis (18.8 for D vs 9.4 for C) shows 2 sampled errors at weight 9.4; use the full-set figures.

**Do not claim**

- Do not call this a protocol result.

**Related slides:** Slides 9 and 11.

### Slide 22: Address data, no mapping

**Section:** Backup

**What the slide shows**

- A table of six fields from address data (14,570 records, 7 columns): Alias Account (name, 36.3%), Postal Code (25.9%), Street Name (13.9%), City Name (12.3%), House Number1 (11.5%), Box1 Number (0.2%).
- Note: no mapping; Id left out because every value is unique; 322,969 pairs scored on text and 40,578 through the identifier comparison; a blind review of 30 sampled pairs, 28 agree with my labels, and both disagreements were same-name pairs with weak address agreement that the text comparison still matched; later fixes left the test-data outputs byte-identical.

**What to say**

> I ran the same code on address data: 14,570 records with columns such as Alias Account and Postal Code, and no mapping. Verity typed the fields, left out the Id column, and gave the name the most weight. After fixing column names with spaces, 322,969 pairs were scored on text and 40,578 through the identifier comparison.

> I reviewed 30 sampled pairs blind and 28 agree. The 2 disagreements were same-name pairs with weak address agreement.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 322,969 / 40,578 | Pairs scored on the text path / the identifier comparison (3 skipped) | `results/second_schema_v5/summary/second_schema_report.md` |
| 145 / 104 | Identifier-path matches / conflicts held for review | `results/second_schema_v5/summary/second_schema_report.md` |
| 28 of 30 | Blind review: all 20 identifier-path matches agree; 8 of 10 text-path matches agree | `results/second_schema_v5_review (local, gitignored)` |
| 0.414924 / 0.82 | Learned and effective threshold | `results/second_schema_v5/summary/second_schema_report.md` |
| 125.5 s | Verity run time on this file | `results/second_schema_v5/summary/second_schema_report.md` |

**Terms in plain words**

- **No mapping:** No column was mapped to a standard name by hand.
- **Same-name pair:** Two records with an identical name at different addresses.
- **Identifier comparison:** Field-by-field comparison used when records have no multi-word text.

**Questions you may get**

- **Q: What was the first problem on this file?**
  A: Column names with spaces such as "Alias Account" did not match the record store's keys, so every pair looked unusable and all 363,550 were skipped. One shared key function fixed it, and the test-data outputs stayed byte-identical.

- **Q: What do the two disagreements show?**
  A: Both are text-path matches (unchanged since the freeze) where the name is identical and carries 36% of the weight, while city, house number and street are weak. They scored 0.88 and 0.87 against a threshold of 0.82. It is a real limit: same-name records at different addresses can still match, which is the co-location case. It is shown on the limits slide.

- **Q: Is 28 of 30 an accuracy figure?**
  A: No. It is the agreement of my blind labels with Verity on 30 sampled pairs. The sample is small, the file has no answer key, and the identifier path was designed after seeing this file, so treat it as an exploratory check.

- **Q: Why do the group counts disagree between the outputs?**
  A: golden_records.csv and match_summary.csv come from different clustering code and disagree (5,825 against 7,032 groups). It was not investigated, so no group counts are quoted.

- **Q: Where did the file come from?**
  A: A sample file confirmed as approved for this use. No record values appear in any committed file.

- **Q: Why is 90% of the output in the review range?**
  A: On this file 292,442 of 363,583 scored pairs are in review, mostly for insufficient evidence. With no answer key, nothing is said on whether those pairs match.

**Do not claim**

- Do not quote group counts or any accuracy for this file.
- Do not say the same-name disagreements come from the new identifier path; they come from the text path.

**Related slides:** Slides 5, 19 and 23.

### Slide 23: Product data, no mapping

**Section:** Backup

**What the slide shows**

- A table of five fields from one of two product files: member_part_number (identifier, 43.7%), upc_ean (identifier-like, 28.1%), manufacturer_part_number (general text, 24.9%), manufacturer_code (1.9%), brand_code (1.4%).
- Note: two product files, no mapping; the primary key and a near-constant source column left out by itself; pairs with no multi-word text are compared field by field on identifiers, and a differing identifier sends the pair to review; a blind review of 40 sampled pairs, 33 agree with my labels, all 15 matches were right, and the 7 disagreements were near-matches held for review that looked like the same product.

**What to say**

> I ran the same code on two product files, with no mapping. Verity typed the fields, left out the primary key and a near-constant source column, and put the most weight on the part-number and barcode columns.

> Product codes have no multi-word text, so those pairs are compared field by field on identifiers, and a differing identifier sends the pair to review. I reviewed 40 sampled pairs blind: 33 agree, all 15 matches were right, and the 7 disagreements were near-matches held for review.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 11,584 pairs | File with only code columns: all compared on identifiers; 0 new matches; 11 near-matches held for review; 321 exact duplicates matched | `results/product_v5_raw/summary/product_schema_report.md` |
| 9,075 + 1,655 pairs | File with a description column: text path and identifier path; 104 identifier-path matches; 44 near-matches held for review | `results/product_v5_semantic/summary/product_schema_report.md` |
| 33 of 40 | Blind review: 15 of 15 matches right, 15 of 15 random pairs agree, 3 of 10 held pairs agree | `results/product_v5_review (local, gitignored)` |
| 43.7% / 28.1% / 24.9% | Weights for part number, barcode and manufacturer part number (file with only code columns) | `results/product_v5_raw/summary/attribute_weights.json` |

**Terms in plain words**

- **Identifier:** A field that behaves like an ID, such as a part number or barcode.
- **Near-match held for review:** A pair whose scores pass the threshold but whose identifying field differs slightly; it goes to a person instead of merging.

**Questions you may get**

- **Q: Why did one file produce no new matches?**
  A: All its content columns are codes, so any difference in a part number or barcode puts the pair in review. Only the 321 exact duplicates matched, and 11 near-matches wait for review. The file may simply hold few duplicates beyond the exact ones; the review cannot say.

- **Q: Why did the identifier comparison not exist at the freeze?**
  A: The text score has needed multi-word text since the original repo, because code-only records embed almost identically and would match falsely. That guard skipped every pair on product codes. The identifier path was added after the freeze, checked to leave the test-data outputs byte-identical, and is exploratory.

- **Q: Why do 7 of the 10 held pairs look like the same product to you?**
  A: They share an exact barcode and a part number that differs by a character or two, so the rule held them for review. Nothing was merged wrongly. A consultant could choose to treat such near-matches as matches; that is listed as a next step, and changing it now would be a fourth change shaped by these files.

- **Q: Is 33 of 40 an accuracy figure?**
  A: No. It is the agreement of my blind labels with Verity on 40 sampled pairs from the two files. The set is small and exploratory.

- **Q: Does Verity match products?**
  A: On two product files it identified the fields and weights, and the identifier comparison produced matches and review pairs that a blind review mostly agrees with. There is no answer key for product data, so quality on product matching is not measured.

**Do not claim**

- Do not say Verity matches product records accurately.
- Do not give a recall or precision figure for product data.

**Related slides:** Slides 15, 19 and 30.

### Slide 24: Data realism

**Section:** Backup

**What the slide shows**

- Left: 10,000 records for 4,025 companies; 5,975 duplicate records, 54.83% carry a defect beyond a legal-suffix change and 12.79% carry two or more; 673 records in 191 same-name groups; 8.26% lack a phone and 18.57% lack a website.
- Right: defect counts (shortened name 675, acronym 660, PO box instead of street address 645, typo 607, case change 604, missing phone 566, address typo 361); US companies and English names only; defect rates chosen by design.

**What to say**

> Source: data_profile.txt from the freeze run.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 4,025 | Entities in the test data | `results/eval_7919/data_profile.txt` |
| 54.83% and 12.79% | Duplicates with a defect beyond a suffix change; with two or more | `results/eval_7919/data_profile.txt` |
| 191 | Same-name groups (decoys) | `results/eval_7919/data_profile.txt` |

**Terms in plain words**

- **Decoy:** A different company deliberately given the same name.
- **Legal suffix:** Inc, LLC and similar endings.

**Questions you may get**

- **Q: Why synthetic data?**
  A: No customer data may be used. A generator gives known truth and a sealed test set. Defect types mirror common real problems: acronyms, shortened names, PO boxes, typos.

- **Q: Is it realistic?**
  A: The defect mix was designed, not measured on customer data, and covers US English company names only. A 100-pair blind check agreed with the generator on 97 (slide 25).

- **Q: How were hard cases chosen?**
  A: They follow typical matching problems. Say this only as a design choice, not as measured prevalence.

**Do not claim**

- Do not say the defect rates reflect real customer data.

**Related slides:** Slides 25 and 30.

### Slide 25: 97 of 100 (label check)

**Section:** Backup

**What the slide shows**

- "97 of 100": I labeled 100 test pairs without seeing the answer key; 97 agree and none were marked unsure.
- Of the 3 disagreements, 1 was a slip; the other 2 were pairs where names or the website overlap but address and phone differ. The development data scored 98 of 100.

**What to say**

> Blind agreement is the reported figure. Reviewer notes written after the key was seen are in hand_check_review.md and change no label.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 97 of 100 | Blind agreement on the test data | `results/eval_7919/hand_check/hand_check_score.txt` |
| 98 of 100 | Blind agreement on the development data | `results/dev_2026_run9/hand_check_score.txt` |
| Pairs 49, 50, 69 | The three disagreements | `results/eval_7919/hand_check/hand_check_review.md` |

**Terms in plain words**

- **Blind check:** Labeling pairs without seeing the generator's answer, then comparing.

**Questions you may get**

- **Q: What were the disagreements?**
  A: Pair 49 was a slip (same address, phone and website; labeled different by mistake). Pair 50 had overlapping names but different address, phone and website. Pair 69 was a same-name, same-website decoy with a different address and phone.

- **Q: Does this prove the labels are right?**
  A: It shows the generator's labels agree with a careful human on 97 of 100 pairs. It samples 8 groups of pairs, so it is a check, not a proof.

- **Q: Did you change a label after seeing the key?**
  A: No. The blind figure is reported. Notes written afterwards are in a separate file.

**Do not claim**

- Do not report 98 of 100 for the test data; that is the development figure.

**Related slides:** Slides 24 and 30.

### Slide 26: Option C needs labeled pairs (learning curve)

**Section:** Backup

**What the slide shows**

- A line chart of mean F1 against the number of labeled training pairs: 100 labels 0.51; 250 0.76; 500 0.78; 1,000 0.84; 2,000 0.90; 63,665 0.97. Five random samples per size. Development data, not the freeze run.

**What to say**

> Source: lc_v2/learning_curve.csv on development data. The freeze run trained C on 66,031 labeled pairs.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.5097 / 0.7648 / 0.7771 / 0.8376 / 0.8999 / 0.9653 | Mean F1 at 100, 250, 500, 1,000, 2,000 and 63,665 labels | `results/dev_2026_run11/options/c_model/lc_v2/learning_curve.csv` |
| 0.22 to 0.85 | Range at 100 labels across the five samples | ` same file` |

**Terms in plain words**

- **Mean F1:** The average F1 over five random draws of the training pairs.

**Questions you may get**

- **Q: How many labels does a customer need?**
  A: On development data, 250 labels (around 1.3 hours at 5 hours per 1,000) gave 0.76 on average with wide spread, 2,000 labels (10 hours) gave 0.90. The 5 hours per 1,000 is an assumption.

- **Q: Why is the last point a single run?**
  A: All 63,665 training pairs is one run, so there is no spread to report.

- **Q: Is this the freeze result?**
  A: No, it is development data, labeled that way on the slide.

**Do not claim**

- Do not quote the curve as a result on the test data.

**Related slides:** Slides 8 and 14.

### Slide 27: Time per decision and scale

**Section:** Backup

**What the slide shows**

- Left: LLM call typical 1.05 s, slow 1.33 s (95th percentile); decision model typical 0.083 ms, slow 0.279 ms on one processor core; hybrid 0.97% of pairs call the LLM and 11.5% of records include at least one call; Verity setup and scoring 167 s for 10,000 records.
- Right: development run, LLM 95, 234 and 399 calls a minute with 2, 5 and 10 parallel workers, no errors; options C and D 74,084 pairs a second across 10 processes; a cold start after a reboot was not measured.

**What to say**

> Freeze-run sources: latency_final.md, latency_summary.json, run_info.json. Concurrency figures come from the development run.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.0826 ms / 0.2791 ms | C median and 95th percentile per pair | `results/eval_7919/options/latency/latency_summary.json` |
| 1.045 s / 1.330 s | LLM call median and 95th percentile (4,996 calls) | `results/eval_7919/latency_final/latency_final.md` |
| 2.10 s | Time per record for the LLM at 10 workers (20.05 calls) | `results/eval_7919/latency_final/latency_final.md` |
| 74,084 pairs a second | C and D across 10 processes (development run) | `results/dev_2026_run11/options/latency` |

**Terms in plain words**

- **95th percentile:** The time that 95 of 100 decisions beat; it shows the slow end.
- **Cold start:** The extra delay when a service starts from nothing.

**Questions you may get**

- **Q: Does the 0.083 ms include everything?**
  A: It includes computing the measurements, one pair at a time, on one core. It excludes the embedding step and the neighbor search, which belong to the Verity run (167 s for 10,000 records).

- **Q: Why is there no tail simulation for D?**
  A: The freeze run did not save per-pair latencies for C, so no tail figure was simulated. The measured facts are 0.97% of pairs at around 1 s and 11.5% of records with one such call.

- **Q: How does it scale?**
  A: Only the development concurrency runs are available. Verity was timed up to 10,000 records, and approximate search timing was not measured.

**Do not claim**

- Do not quote a cold-start or large-scale latency.

**Related slides:** Slides 9, 11 and 30.

### Slide 28: Cost model assumptions

**Section:** Backup

**What the slide shows**

- Left: one-time costs at $100 an hour (Verity build 80 hours; productization 100 hours; shared by every option); option hours on top (A $500, B $554, C $450, D $704); labeled examples for C and D at 5 hours per 1,000 pairs, paid by the customer.
- Right: LLM $0.0015 a call with 85% of the prompt reused; hosting, operations and who pays follow the deployment; tested with $50 to $200 an hour, pairs per record, LLM price plus or minus 50%, reuse rate, shared setup cost 1 to 100 times.

**What to say**

> Sources: cost_final one_time_cost.csv, assumptions.csv, sensitivity.csv.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 80 and 100 hours | Verity build and productization | `cost_final/assumptions.csv` |
| $500 / $554 / $450 / $704 | Option hours on top of $18,000 | `cost_final/one_time_cost.csv` |
| 5 hours per 1,000 | Assumed labeling effort | `cost_final/assumptions.csv` |
| 0.85 | Prompt cache reuse measured in the freeze run | `results/eval_7919/options/b_llm/openai/sample5000 (timings.json, calls.jsonl)` |

**Terms in plain words**

- **Sensitivity:** Re-running the cost model with a changed assumption to see whether rankings change.

**Questions you may get**

- **Q: Which assumptions are weakest?**
  A: The 5 hours per 1,000 labels (a placeholder), the $100 hourly rate, and the 80-hour Verity build estimate. On the development run, the option ranking did not change under any tested sensitivity.

- **Q: Why 12 months?**
  A: One-time costs are spread over 12 months to give a monthly total.

- **Q: Does the model include LLM price changes?**
  A: It tests plus or minus 50% on the LLM price.

**Do not claim**

- Do not present the hours as measured; they are assumptions or logged hands-on time.

**Related slides:** Slides 13 and 14.

### Slide 29: Full test set, exact figures

**Section:** Backup

**What the slide shows**

- A table for all 44,022 borderline test pairs: A F1 0.106 [0.065, 0.147], precision 1.000, recall 0.056; C 0.963 [0.949, 0.977], 0.974, 0.953; D 0.986 [0.977, 0.994], 0.975, 0.997; D with the original rule 0.969 [0.956, 0.981], 0.980, 0.958.
- Note: these figures are not comparable with the LLM figures on slide 11, because the LLM was scored only on the 5,000-pair sample.

**What to say**

> Source: head_to_head_v3 Table 2 and the secondary row for D with the original routing rule.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 0.986 | D's F1 on the full test set (0.973 on the sample) | `results/eval_7919/head_to_head_v3/head_to_head.md, Table 2` |
| 0.969 | D with the original routing rule | `results/eval_7919/head_to_head_v3/head_to_head.md` |

**Terms in plain words**

- **Full test set:** All 44,022 borderline pairs, with exact counts and no sampling.

**Questions you may get**

- **Q: Why is D 0.986 here but 0.973 on slide 11?**
  A: Different basis. The sample F1 weights a few sampled errors heavily; the full-set F1 uses exact counts. Compare the options on one basis only.

- **Q: Why not score the LLM on all pairs?**
  A: It would cost around $67, above the project cap.

- **Q: What does the original rule show?**
  A: Routing only 16 pairs gives 0.969 instead of 0.986, a smaller review queue but lower quality.

**Do not claim**

- Do not set this table next to B's sample F1.

**Related slides:** Slides 9, 11 and 21.

### Slide 30: Limits we state

**Section:** Backup

**What the slide shows**

- Nine limits: synthetic data (US companies, English names, designed defect rates); the LLM ran with extended reasoning off; the 5,000-pair sample cannot prove "within 2 points"; the LLM made 1 wrong match and missed 1 true pair on the sample; column names affect the text-to-numbers step and renaming, cold start and approximate search timing were not run or measured; field weights come from uniqueness and completeness, with a second display-only weight; the embedding model's publisher clearance is not yet confirmed; address and product data were checked on sampled pairs, not a full labeled test, and same-name records with weak address agreement can still match; the DIS and MDM matching engines were not tested here and may match better than Verity's score.

**What to say**

> Update the embedding model line once the publisher clearance is confirmed.

**Numbers on the slide and where they come from**

| Number | Meaning | Source file |
|---|---|---|
| 5 points | Smallest gap the 5,000-pair sample can reliably show | `protocol.md; sampling-noise check (eval/sample_noise.py)` |
| All-MiniLM-L6-v2 | Apache 2.0; safetensors; pinned revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 | `verity/src/embed.py` |

**Terms in plain words**

- **Rename test:** Rerunning with every column renamed to see whether results hold; it would show how much column names influence the embeddings.
- **AIP3 Section 7:** The Precisely legal guideline listing approved model publishers.

**Questions you may get**

- **Q: Why is the publisher clearance open?**
  A: The model is Apache 2.0, safetensors, pinned and run locally, but the sentence-transformers publisher has not been checked against the approved list in Section 7. If it is not listed, the model needs a legal review; swapping it would change every result.

- **Q: What is the rename test?**
  A: Verity embeds the column name text with each value. Renaming every column would show whether quality depends on the names. It has not been run, so the claim is "no hardcoded column names" and not "never reads a column name".

- **Q: What was not measured?**
  A: Cold start after a reboot, approximate neighbor search timing, specialist setup time today, and larger scale than 10,000 records.

- **Q: Did you test the product matching engines?**
  A: No. They may match better than Verity's score. The test compared four ways of settling borderline pairs. A comparison with DIS and MDM on approved customer data is a next step.

- **Q: Which limit worries you most?**
  A: Synthetic data: the defect rates were designed, and every option may look better on them than on customer data. Next is the same-name, different-address case, seen in 2 of 10 sampled address matches.

**Do not claim**

- Do not remove a limit to shorten the slide.

**Related slides:** Slides 3, 4, 15, 19, 22, 23 and 27.

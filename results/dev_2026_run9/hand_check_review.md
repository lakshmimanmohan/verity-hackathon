# Hand check of generator labels, dev_2026_run9

## Blind result

- **Agreement: 98/100 (98.0%)**, 0 unsure. This is the figure reported.
- Data: seed 2026, a development seed. The labels come from the same generator code
  (`data/generate_data_org.py`) that produces the evaluation set (seed 7919).
- Sample: 100 pairs drawn by `eval/hand_check.py --sample` with seed 0, in eight strata
  (25 true and 25 false pairs in the band, 15 decoy pairs, 15 true hard-case pairs above the
  band, 10 false pairs above the band, 10 random true pairs). The reviewer labelled
  `hand_check_pairs.csv` without seeing ids, truth, scores, match types or tags.
- Full output, with both disagreements side by side: `hand_check_score.txt`.

## Reviewer notes (written after the key was seen)

- **Pair 58** (blind label "same", key "different"): a data-entry slip. The intended label
  was "different".
- **Pair 53** (blind label "different", key "same"): the reviewer would now mark it
  "unsure". One record has a PO box and the other a street address, and the phone and
  website are missing on one side.

## Post-review

- 99/99 decided pairs agree, 1 unsure (pair 53).
- The blind figure, 98/100, is the one reported. `hand_check_pairs.csv` holds the blind
  labels unchanged.

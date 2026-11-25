EXPLORATORY: not a protocol result; written after post_freeze_config.md; no setting was chosen from these numbers.

# Corrections to post_freeze_config.md

post_freeze_config.md is unchanged. These are the two places where it is wrong or incomplete.

1. Field weights. post_freeze_config.md says the score used normalized_discriminative_weight as the field weights. The score uses discriminative_score (uniqueness ratio x (1 - null rate)), renormalized for each pair over the fields both records have (verity/src/match.py lines 467-474, where discriminative_score is read, and lines 1355-1378, where the weights of a pair are renormalized). The two agree only when a pair has all 7 fields. Example: demo pair 2 has no phone or website, so its address_line weight is 0.3912, against 0.2572 in attribute_weights.json. composite_weight is display-only (verity/src/explore_matches.py:261); the matching code never reads it.

2. Embedding blend. post_freeze_config.md marks the 0.65 / 0.35 blend as a fixed setting. The embedding share is derived from the entropy of the weight shares and clamped to the range 0.12 to 0.65 (verity/src/match.py lines 485-494). On this run w_cos = 0.7913, so the 0.65 cap applies and the field share is 0.35. The within-field blend of 0.65 cosine and 0.35 character Jaccard is a fixed constant (verity/src/match.py:108-109).

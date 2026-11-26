# Learned matching rules
Source file: `(local path removed)`  
Rows analyzed: `500`

> **Excluded from learning** (hold-out / label-like columns): `product_id`, `source_system`.  

## Schema discovery
Columns were typed from value patterns (dates, emails, numeric parse rates, name-like strings) and cardinality tiers (identifiers vs quasi-identifiers vs descriptive fields). The weight table lists **columns used for learning** (evaluation / label-like fields removed).

| Column | Semantic type | Physical kind | Uniq. ratio | Null % | Weight (disc.) |
|--------|----------------|---------------|------------|-------|----------------|
| `member_part_number` | IDENTIFIER | STRING | 0.924 | 0.0% | 0.437 |
| `upc_ean` | QUASI_IDENTIFIER | STRING | 0.678 | 12.4% | 0.281 |
| `manufacturer_part_number` | DESCRIPTIVE | STRING | 0.526 | 0.0% | 0.249 |
| `manufacturer_code` | DESCRIPTIVE | STRING | 0.040 | 0.0% | 0.019 |
| `brand_code` | DESCRIPTIVE | STRING | 0.030 | 0.0% | 0.014 |

## Discriminative weights
Each column receives `discriminative_score = uniqueness_ratio * (1 - null_rate)`. Values are normalized to sum to **1.0** so they act as data-driven attribute weights **without manual tuning**.

### Strongest signals
- **`member_part_number`** normalized weight **0.437**: 92.4% unique values, 0.0% null — classified as **IDENTIFIER**.
- **`upc_ean`** normalized weight **0.281**: 67.8% unique values, 12.4% null — classified as **QUASI_IDENTIFIER**.
- **`manufacturer_part_number`** normalized weight **0.249**: 52.6% unique values, 0.0% null — classified as **DESCRIPTIVE**.
- **`manufacturer_code`** normalized weight **0.019**: 4.0% unique values, 0.0% null — classified as **DESCRIPTIVE**.
- **`brand_code`** normalized weight **0.014**: 3.0% unique values, 0.0% null — classified as **DESCRIPTIVE**.

## Similarity aggregation
- DESCRIPTIVE / NAME string columns: 65% cosine (record embedding) + 35% character-set Jaccard; falls back to Jaccard when embeddings are missing  
- Other strings: character-set Jaccard similarity  
- Numbers: min–max normalization per column; similarity `1 − |Δ_scaled|`  
- Dates: ordinal distance scaled to ~10 years  
- Email: separate Jaccard on local-part and domain, then averaged  

Random **1000** row pairs were scored with the normalized weights.

## Learned match threshold
Pooled weighted similarity was clustered with **`KMeans(n_clusters=2)`** (separating a low cluster from a high cluster).

- Lower cluster: **`572`** pairs, mean **`0.1374`**, sd **`0.0335`**  
- Lower cluster center: **`0.1374`**  
- Upper cluster center: **`0.2358`**  
- **`threshold = low mean + 2 x sd = 0.2043`** — pairs above this are treated as plausible matches.  
- Centroid midpoint (earlier rule, for reference): **`0.1866`**

See **`data/threshold_similarity_distribution.png`** for the histogram with the cutoff.

## Field reliability
When pairs already agree on strongly weighted columns but disagree elsewhere, the conflicting field accumulates penalties; fields that consistently co-agree with anchors gain higher **`reliability_score`**.

| Column | Reliability |
|--------|------------|
| `manufacturer_code` | 1.000 |
| `brand_code` | 0.967 |
| `manufacturer_part_number` | 0.399 |
| `member_part_number` | 0.100 |
| `upc_ean` | 0.100 |

## Composite suggestion
`composite_weight` (normalized) blends discriminative uniqueness with reliability — useful when some fields drift (addresses) while others stay stable (email/phone).

# Learned matching rules
Source file: `(local path removed)`  
Rows analyzed: `508`

> **Excluded from learning** (hold-out / label-like columns): `product_id`, `source_system`.  

## Schema discovery
Columns were typed from value patterns (dates, emails, numeric parse rates, name-like strings) and cardinality tiers (identifiers vs quasi-identifiers vs descriptive fields). The weight table lists **columns used for learning** (evaluation / label-like fields removed).

| Column | Semantic type | Physical kind | Uniq. ratio | Null % | Weight (disc.) |
|--------|----------------|---------------|------------|-------|----------------|
| `description` | DESCRIPTIVE | STRING | 0.913 | 8.3% | 0.530 |
| `manufacturer_part_number` | DESCRIPTIVE | STRING | 0.384 | 1.2% | 0.240 |
| `upc_ean` | DESCRIPTIVE | STRING | 0.423 | 14.0% | 0.230 |

## Discriminative weights
Each column receives `discriminative_score = uniqueness_ratio * (1 - null_rate)`. Values are normalized to sum to **1.0** so they act as data-driven attribute weights **without manual tuning**.

### Strongest signals
- **`description`** normalized weight **0.530**: 91.3% unique values, 8.3% null — classified as **DESCRIPTIVE**.
- **`manufacturer_part_number`** normalized weight **0.240**: 38.4% unique values, 1.2% null — classified as **DESCRIPTIVE**.
- **`upc_ean`** normalized weight **0.230**: 42.3% unique values, 14.0% null — classified as **DESCRIPTIVE**.

## Similarity aggregation
- DESCRIPTIVE / NAME string columns: 65% cosine (record embedding) + 35% character-set Jaccard; falls back to Jaccard when embeddings are missing  
- Other strings: character-set Jaccard similarity  
- Numbers: min–max normalization per column; similarity `1 − |Δ_scaled|`  
- Dates: ordinal distance scaled to ~10 years  
- Email: separate Jaccard on local-part and domain, then averaged  

Random **1000** row pairs were scored with the normalized weights.

## Learned match threshold
Pooled weighted similarity was clustered with **`KMeans(n_clusters=2)`** (separating a low cluster from a high cluster).

- Lower cluster: **`507`** pairs, mean **`0.3631`**, sd **`0.0654`**  
- Lower cluster center: **`0.3631`**  
- Upper cluster center: **`0.5107`**  
- **`threshold = low mean + 2 x sd = 0.4939`** — pairs above this are treated as plausible matches.  
- Centroid midpoint (earlier rule, for reference): **`0.4369`**

See **`data/threshold_similarity_distribution.png`** for the histogram with the cutoff.

## Field reliability
When pairs already agree on strongly weighted columns but disagree elsewhere, the conflicting field accumulates penalties; fields that consistently co-agree with anchors gain higher **`reliability_score`**.

| Column | Reliability |
|--------|------------|
| `manufacturer_part_number` | 1.000 |
| `upc_ean` | 0.615 |
| `description` | 0.100 |

## Composite suggestion
`composite_weight` (normalized) blends discriminative uniqueness with reliability — useful when some fields drift (addresses) while others stay stable (email/phone).

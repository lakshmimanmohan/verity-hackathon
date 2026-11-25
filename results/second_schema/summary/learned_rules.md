# Learned matching rules
Source file: `(local path removed)`  
Rows analyzed: `14570`

> **Excluded from learning** (hold-out / label-like columns): `Id`.  

## Schema discovery
Columns were typed from value patterns (dates, emails, numeric parse rates, name-like strings) and cardinality tiers (identifiers vs quasi-identifiers vs descriptive fields). The weight table lists **columns used for learning** (evaluation / label-like fields removed).

| Column | Semantic type | Physical kind | Uniq. ratio | Null % | Weight (disc.) |
|--------|----------------|---------------|------------|-------|----------------|
| `Alias Account` | NAME | STRING | 0.626 | 0.0% | 0.363 |
| `Postal Code` | DESCRIPTIVE | STRING | 0.447 | 0.0% | 0.259 |
| `Street Name` | DESCRIPTIVE | STRING | 0.240 | 0.4% | 0.139 |
| `City Name` | DESCRIPTIVE | STRING | 0.212 | 0.0% | 0.123 |
| `House Number1` | DESCRIPTIVE | NUMBER | 0.202 | 2.2% | 0.115 |
| `Box1 Number` | DESCRIPTIVE | NUMBER | 0.033 | 91.4% | 0.002 |

## Discriminative weights
Each column receives `discriminative_score = uniqueness_ratio * (1 - null_rate)`. Values are normalized to sum to **1.0** so they act as data-driven attribute weights **without manual tuning**.

### Strongest signals
- **`Alias Account`** normalized weight **0.363**: 62.6% unique values, 0.0% null — classified as **NAME**.
- **`Postal Code`** normalized weight **0.259**: 44.7% unique values, 0.0% null — classified as **DESCRIPTIVE**.
- **`Street Name`** normalized weight **0.139**: 24.0% unique values, 0.4% null — classified as **DESCRIPTIVE**.
- **`City Name`** normalized weight **0.123**: 21.2% unique values, 0.0% null — classified as **DESCRIPTIVE**.
- **`House Number1`** normalized weight **0.115**: 20.2% unique values, 2.2% null — classified as **DESCRIPTIVE**.

## Similarity aggregation
- DESCRIPTIVE / NAME string columns: 65% cosine (record embedding) + 35% character-set Jaccard; falls back to Jaccard when embeddings are missing  
- Other strings: character-set Jaccard similarity  
- Numbers: min–max normalization per column; similarity `1 − |Δ_scaled|`  
- Dates: ordinal distance scaled to ~10 years  
- Email: separate Jaccard on local-part and domain, then averaged  

Random **1000** row pairs were scored with the normalized weights.

## Learned match threshold
Pooled weighted similarity was clustered with **`KMeans(n_clusters=2)`** (separating a low cluster from a high cluster).

- Lower cluster: **`492`** pairs, mean **`0.3088`**, sd **`0.0531`**  
- Lower cluster center: **`0.3088`**  
- Upper cluster center: **`0.4432`**  
- **`threshold = low mean + 2 x sd = 0.4149`** — pairs above this are treated as plausible matches.  
- Centroid midpoint (earlier rule, for reference): **`0.3760`**

See **`data/threshold_similarity_distribution.png`** for the histogram with the cutoff.

## Field reliability
When pairs already agree on strongly weighted columns but disagree elsewhere, the conflicting field accumulates penalties; fields that consistently co-agree with anchors gain higher **`reliability_score`**.

| Column | Reliability |
|--------|------------|
| `Street Name` | 1.000 |
| `City Name` | 0.581 |
| `House Number1` | 0.578 |
| `Postal Code` | 0.370 |
| `Alias Account` | 0.100 |
| `Box1 Number` | 0.100 |

## Composite suggestion
`composite_weight` (normalized) blends discriminative uniqueness with reliability — useful when some fields drift (addresses) while others stay stable (email/phone).

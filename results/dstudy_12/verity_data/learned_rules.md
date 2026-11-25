# Learned matching rules
Source file: `C:\verity-hackathon\results\dstudy_12\orgs_raw.csv`  
Rows analyzed: `10000`

> **Excluded from learning** (hold-out / label-like columns): `record_id`, `source_system`.  

## Schema discovery
Columns were typed from value patterns (dates, emails, numeric parse rates, name-like strings) and cardinality tiers (identifiers vs quasi-identifiers vs descriptive fields). The weight table lists **columns used for learning** (evaluation / label-like fields removed).

| Column | Semantic type | Physical kind | Uniq. ratio | Null % | Weight (disc.) |
|--------|----------------|---------------|------------|-------|----------------|
| `company_name` | NAME | STRING | 0.785 | 0.0% | 0.280 |
| `address_line` | STREET_ADDRESS | STRING | 0.720 | 0.0% | 0.257 |
| `phone` | QUASI_IDENTIFIER | STRING | 0.752 | 8.0% | 0.247 |
| `postal_code` | DESCRIPTIVE | STRING | 0.325 | 0.0% | 0.116 |
| `website` | DESCRIPTIVE | STRING | 0.341 | 18.7% | 0.099 |
| `city` | DESCRIPTIVE | STRING | 0.003 | 0.0% | 0.001 |
| `state` | DESCRIPTIVE | STRING | 0.002 | 0.0% | 0.001 |

## Discriminative weights
Each column receives `discriminative_score = uniqueness_ratio * (1 - null_rate)`. Values are normalized to sum to **1.0** so they act as data-driven attribute weights **without manual tuning**.

### Strongest signals
- **`company_name`** normalized weight **0.280**: 78.5% unique values, 0.0% null — classified as **NAME**.
- **`address_line`** normalized weight **0.257**: 72.0% unique values, 0.0% null — classified as **STREET_ADDRESS**.
- **`phone`** normalized weight **0.247**: 75.2% unique values, 8.0% null — classified as **QUASI_IDENTIFIER**.
- **`postal_code`** normalized weight **0.116**: 32.5% unique values, 0.0% null — classified as **DESCRIPTIVE**.
- **`website`** normalized weight **0.099**: 34.1% unique values, 18.7% null — classified as **DESCRIPTIVE**.

## Similarity aggregation
- DESCRIPTIVE / NAME string columns: 65% cosine (record embedding) + 35% character-set Jaccard; falls back to Jaccard when embeddings are missing  
- Other strings: character-set Jaccard similarity  
- Numbers: min–max normalization per column; similarity `1 − |Δ_scaled|`  
- Dates: ordinal distance scaled to ~10 years  
- Email: separate Jaccard on local-part and domain, then averaged  

Random **1000** row pairs were scored with the normalized weights.

## Learned match threshold
Pooled weighted similarity was clustered with **`KMeans(n_clusters=2)`** (separating a low cluster from a high cluster).

- Lower cluster: **`470`** pairs, mean **`0.3628`**, sd **`0.0479`**  
- Lower cluster center: **`0.3625`**  
- Upper cluster center: **`0.4791`**  
- **`threshold = low mean + 2 x sd = 0.4587`** — pairs above this are treated as plausible matches.  
- Centroid midpoint (earlier rule, for reference): **`0.4208`**

See **`data/threshold_similarity_distribution.png`** for the histogram with the cutoff.

## Field reliability
When pairs already agree on strongly weighted columns but disagree elsewhere, the conflicting field accumulates penalties; fields that consistently co-agree with anchors gain higher **`reliability_score`**.

| Column | Reliability |
|--------|------------|
| `state` | 1.000 |
| `city` | 0.832 |
| `company_name` | 0.100 |
| `address_line` | 0.100 |
| `postal_code` | 0.100 |
| `phone` | 0.100 |
| `website` | 0.100 |

## Composite suggestion
`composite_weight` (normalized) blends discriminative uniqueness with reliability — useful when some fields drift (addresses) while others stay stable (email/phone).

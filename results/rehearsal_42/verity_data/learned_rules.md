# Learned matching rules
Source file: `C:\verity-hackathon\results\rehearsal_42\orgs_raw.csv`  
Rows analyzed: `10000`

> **Excluded from learning** (hold-out / label-like columns): `record_id`, `source_system`.  

## Schema discovery
Columns were typed from value patterns (dates, emails, numeric parse rates, name-like strings) and cardinality tiers (identifiers vs quasi-identifiers vs descriptive fields). The weight table lists **columns used for learning** (evaluation / label-like fields removed).

| Column | Semantic type | Physical kind | Uniq. ratio | Null % | Weight (disc.) |
|--------|----------------|---------------|------------|-------|----------------|
| `company_name` | NAME | STRING | 0.788 | 0.0% | 0.280 |
| `address_line` | STREET_ADDRESS | STRING | 0.719 | 0.0% | 0.256 |
| `phone` | QUASI_IDENTIFIER | STRING | 0.752 | 7.5% | 0.248 |
| `postal_code` | DESCRIPTIVE | STRING | 0.325 | 0.0% | 0.116 |
| `website` | DESCRIPTIVE | STRING | 0.343 | 19.3% | 0.098 |
| `city` | DESCRIPTIVE | STRING | 0.003 | 0.0% | 0.001 |
| `state` | DESCRIPTIVE | STRING | 0.002 | 0.0% | 0.001 |

## Discriminative weights
Each column receives `discriminative_score = uniqueness_ratio * (1 - null_rate)`. Values are normalized to sum to **1.0** so they act as data-driven attribute weights **without manual tuning**.

### Strongest signals
- **`company_name`** normalized weight **0.280**: 78.8% unique values, 0.0% null — classified as **NAME**.
- **`address_line`** normalized weight **0.256**: 71.9% unique values, 0.0% null — classified as **STREET_ADDRESS**.
- **`phone`** normalized weight **0.248**: 75.2% unique values, 7.5% null — classified as **QUASI_IDENTIFIER**.
- **`postal_code`** normalized weight **0.116**: 32.5% unique values, 0.0% null — classified as **DESCRIPTIVE**.
- **`website`** normalized weight **0.098**: 34.3% unique values, 19.3% null — classified as **DESCRIPTIVE**.

## Similarity aggregation
- DESCRIPTIVE / NAME string columns: 65% cosine (record embedding) + 35% character-set Jaccard; falls back to Jaccard when embeddings are missing  
- Other strings: character-set Jaccard similarity  
- Numbers: min–max normalization per column; similarity `1 − |Δ_scaled|`  
- Dates: ordinal distance scaled to ~10 years  
- Email: separate Jaccard on local-part and domain, then averaged  

Random **1000** row pairs were scored with the normalized weights.

## Learned match threshold
Pooled weighted similarity was clustered with **`KMeans(n_clusters=2)`** (separating a low cluster from a high cluster).

- Lower cluster: **`489`** pairs, mean **`0.3636`**, sd **`0.0452`**  
- Lower cluster center: **`0.3636`**  
- Upper cluster center: **`0.4794`**  
- **`threshold = low mean + 2 x sd = 0.4540`** — pairs above this are treated as plausible matches.  
- Centroid midpoint (earlier rule, for reference): **`0.4215`**

See **`data/threshold_similarity_distribution.png`** for the histogram with the cutoff.

## Field reliability
When pairs already agree on strongly weighted columns but disagree elsewhere, the conflicting field accumulates penalties; fields that consistently co-agree with anchors gain higher **`reliability_score`**.

| Column | Reliability |
|--------|------------|
| `state` | 1.000 |
| `city` | 0.881 |
| `company_name` | 0.100 |
| `address_line` | 0.100 |
| `postal_code` | 0.100 |
| `phone` | 0.100 |
| `website` | 0.100 |

## Composite suggestion
`composite_weight` (normalized) blends discriminative uniqueness with reliability — useful when some fields drift (addresses) while others stay stable (email/phone).

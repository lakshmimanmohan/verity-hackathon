"""
Option C features, built only from one scored_pairs.csv row plus scored_pairs_columns.json.

Every per-field value is grouped by its column's semantic type (from the sidecar's
field_semantic_types); no column name is used. The type list is Verity's fixed set, so the
feature columns are the same for any dataset. All per-type aggregates are order-free
(max / mean / min / sum / count), so renaming or reordering columns, with the same values,
gives identical features (see options/test_c_features.py).

Limitation: scored_pairs.csv lists only the fields both records have, so a column missing
on one side cannot be told apart from one missing on both; T_n_absent counts either.
"""

from __future__ import annotations

import json
import math
from typing import Any

SEMANTIC_TYPES = ("IDENTIFIER", "QUASI_IDENTIFIER", "NAME", "STREET_ADDRESS", "DESCRIPTIVE", "DATE")
MATCH_TYPES = ("full_signal", "weak_signal", "embedding_only", "below_threshold",
               "insufficient_evidence", "overlap_rejected")
PAIR_FEATURES = [
    "score", "score_minus_threshold", "cosine", "field_sim", "evidence_mass", "conflict_mass",
    "evidence_passes", "has_shared_keys",
    *[f"mt_{m}" for m in MATCH_TYPES],
    "n_fields_compared", "n_fields_absent", "weight_top_share", "n_types_compared",
]
TYPE_SUFFIXES = ("jac_max", "jac_mean", "jac_min", "ev_max", "ev_mean", "ev_min",
                 "weight", "n_compared", "n_absent")
TYPE_FEATURES = [f"{t}_{s}" for t in SEMANTIC_TYPES for s in TYPE_SUFFIXES]
FEATURE_NAMES = PAIR_FEATURES + TYPE_FEATURES
NAN = float("nan")


class FeatureBuilder:
    def __init__(self, columns_doc: dict[str, Any]):
        st = columns_doc["field_semantic_types"]
        self.type_of = {str(k): str(v).strip().upper() for k, v in st.items()}
        self.n_cols_of_type = {t: 0 for t in SEMANTIC_TYPES}
        for t in self.type_of.values():
            if t in self.n_cols_of_type:
                self.n_cols_of_type[t] += 1
        self.n_content_cols = len(self.type_of)
        self.learned_threshold = float(columns_doc["learned_threshold"])
        self.effective_threshold = float(columns_doc["effective_threshold"])

    def features(self, row: dict[str, str]) -> list[float]:
        score = float(row["score"])
        eff = float(row["effective_threshold"])
        applied = float(row["applied_threshold"]) if row["applied_threshold"] else eff
        jac = json.loads(row["field_similarities"]) if row["field_similarities"] else {}
        wts = json.loads(row["field_weights"]) if row["field_weights"] else {}
        ev = json.loads(row["evidence_field_similarities"]) if row["evidence_field_similarities"] else {}
        # Sorted before summing so the float result does not depend on key order.
        field_sim = sum(sorted(float(wts[k]) * float(jac.get(k, 0.0)) for k in wts))
        out = [
            score,
            score - eff,
            float(row["cosine"]) if row["cosine"] else NAN,
            field_sim,
            float(row["evidence_mass"]) if row["evidence_mass"] else NAN,
            float(row["conflict_mass"]) if row["conflict_mass"] else NAN,
            1.0 if row["evidence_passes"] == "true" else 0.0,
            1.0 if abs(applied - eff) < 1e-9 else 0.0,
            *[1.0 if row["match_type"] == m else 0.0 for m in MATCH_TYPES],
            float(len(wts)),
            float(self.n_content_cols - len(wts)),
            max((float(v) for v in wts.values()), default=0.0),
            float(len({self.type_of.get(k) for k in wts})),
        ]
        by_type: dict[str, list[str]] = {t: [] for t in SEMANTIC_TYPES}
        for k in wts:
            t = self.type_of.get(k)
            if t in by_type:
                by_type[t].append(k)
        for t in SEMANTIC_TYPES:
            keys = by_type[t]
            if keys:
                # Values sorted so means and sums do not depend on column order.
                j = sorted(float(jac.get(k, 0.0)) for k in keys)
                e = sorted(float(ev[k]) for k in keys if k in ev)
                out += [j[-1], sum(j) / len(j), j[0],
                        e[-1] if e else NAN, (sum(e) / len(e)) if e else NAN, e[0] if e else NAN,
                        sum(sorted(float(wts[k]) for k in keys)), float(len(keys)),
                        float(self.n_cols_of_type[t] - len(keys))]
            else:
                out += [NAN] * 7 + [0.0, float(self.n_cols_of_type[t])]
        return out


def same_features(a: list[float], b: list[float]) -> bool:
    return all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b)) and len(a) == len(b)

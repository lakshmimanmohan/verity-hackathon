"""
Option C v1: classical decision model for test band pairs.

  python options/c_decision_model.py results/<run> --model lgbm|lr --version <name>
      [--n N] [--true-share 0.25] [--k 5]

Trains on TRAIN band pairs only (decision_status == review, both records in the train group of
split_records.csv); truth labels are read for train pairs only. Features: options/c_features.py
(semantic types only). Steps:
  1. grouped CV inside train: train-side split components dealt into k folds (seed 0, balanced
     by records); for fold f, evaluate pairs with both records in f, train on pairs with both
     records outside f. Straddling pairs get no out-of-fold (OOF) prediction.
  2. model selection (LightGBM grid of 8; logistic regression has none) by weighted average
     precision on the pooled OOF predictions
  3. isotonic calibration fitted on the OOF predictions
  4. cutoff t*: maximise weighted OOF band F1 subject to estimated overall train-group
     precision >= 0.98 (rule in cv_report.txt); ties go to the larger t; if no t qualifies,
     every band pair is "different"
  5. final model refit on all training pairs, applied to the test band pairs
Weights: OOF pairs carry coverage weights N_c(subset) / N_c(OOF) per class c, and with --n a
population weight N_c(train band) / N_c(subset), so estimates refer to the whole train band.
--n draws a nested, seed-0 subset of train band pairs with round(true_share * n) true pairs
(at most all true pairs) and the rest false.

Writes results/<run>/options/c_model/<version>/: decisions.csv (record_a, record_b, decision,
confidence, model), model.txt or model.json, calibrator.json, features.json, cv_report.txt,
oof_reliability.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings

# LightGBM's sklearn wrapper names numpy columns Column_i and then warns on every numpy
# predict; the inputs are always the same numpy layout, so the warning carries no information.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c_features import FEATURE_NAMES, FeatureBuilder  # noqa: E402

SEED = 0
HALF_WIDTH_MICRO = 50_000
WRITE_TYPES = {"full_signal", "weak_signal", "embedding_only"}
PRECISION_FLOOR = 0.98
LGBM_GRID = [
    {"num_leaves": nl, "min_child_samples": mcs, "scale_pos_weight": spw}
    for nl in (15, 31) for mcs in (20, 50) for spw in ("1", "sqrt")
]
LGBM_FIXED = {"objective": "binary", "learning_rate": 0.05, "n_estimators": 400,
              "subsample": 1.0, "colsample_bytree": 1.0, "random_state": SEED,
              "deterministic": True, "force_col_wise": True, "verbose": -1}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def micro(x: str) -> int:
    return round(float(x) * 1_000_000)


def components(nodes, edges) -> dict[str, str]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return {n: find(n) for n in nodes}


def make_model(kind: str, cfg: dict, n_pos: int, n_neg: int):
    if kind == "lgbm":
        import lightgbm as lgb
        spw = 1.0 if cfg["scale_pos_weight"] == "1" else float(np.sqrt(n_neg / max(n_pos, 1)))
        return lgb.LGBMClassifier(**LGBM_FIXED, num_leaves=cfg["num_leaves"],
                                  min_child_samples=cfg["min_child_samples"], scale_pos_weight=spw)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                              max_iter=5000, random_state=SEED))


def lr_matrix(X: np.ndarray) -> np.ndarray:
    """Logistic regression input: NaN -> 0 plus one missing-indicator column per feature with NaN."""
    nan = np.isnan(X)
    return np.hstack([np.nan_to_num(X, nan=0.0), nan.astype(float)])


def weighted_f1_curve(p: np.ndarray, y: np.ndarray, w: np.ndarray):
    """For thresholds at the distinct p values (descending): weighted TP and predicted positives."""
    order = np.argsort(-p, kind="mergesort")
    ps, ys, ws = p[order], y[order], w[order]
    ctp = np.cumsum(ws * ys)
    cpred = np.cumsum(ws)
    last = np.r_[ps[1:] != ps[:-1], True]  # last index of each distinct value
    return ps[last], ctp[last], cpred[last]


def reliability(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> list[dict]:
    rows = []
    b = np.minimum((p * 10).astype(int), 9)
    for i in range(10):
        m = b == i
        W = float(w[m].sum())
        rows.append({"bin": f"[{i / 10:.1f}, {(i + 1) / 10:.1f}{']' if i == 9 else ')'}",
                     "weight": W, "n": int(m.sum()),
                     "mean_p": float((w[m] * p[m]).sum() / W) if W else None,
                     "frac_true": float((w[m] * y[m]).sum() / W) if W else None})
    return rows


def ece(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    tot = float(w.sum())
    return float(sum(r["weight"] / tot * abs(r["frac_true"] - r["mean_p"])
                     for r in reliability(p, y, w) if r["weight"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Option C v1: classical decision model.")
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", choices=("lgbm", "lr"), required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--n", type=int, default=None, help="Train on a nested subset of n band pairs")
    ap.add_argument("--true-share", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sample-seed", type=int, default=SEED,
                    help="Seed for the --n subset only (folds, model and grid stay at seed 0)")
    args = ap.parse_args()
    csv.field_size_limit(10**8)
    t_start = time.perf_counter()
    run, data = args.run, args.run / "verity_data"
    out = run / "options" / "c_model" / args.version
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads((data / "scored_pairs_columns.json").read_text(encoding="utf-8"))
    fb = FeatureBuilder(doc)
    side = {r["record_id"]: r["side"] for r in read_csv(run / "split_records.csv")}
    rows = read_csv(data / "scored_pairs.csv")

    # Train labels only: a lookup that refuses any pair not wholly in the train group.
    entity = {r["record_id"]: r["true_entity_id"] for r in read_csv(run / "orgs_raw_truth.csv")
              if side.get(r["record_id"]) == "train"}

    def label(a: str, b: str) -> int:
        if side[a] != "train" or side[b] != "train":
            raise RuntimeError("attempt to read a label outside the train group")
        return int(entity[a] == entity[b])

    train_band, test_band = [], []
    m_out = tp_out = 0
    edges = []
    for r in rows:
        a, b = r["record_a"], r["record_b"]
        sa, sb = side[a], side[b]
        st = r["decision_status"]
        if micro(r["score"]) > micro(r["effective_threshold"]) + HALF_WIDTH_MICRO:
            edges.append((a, b))                       # above-band edge (split graph)
        if st == "match":
            edges.append((a, b))                       # matched edge (split graph)
        if sa != sb:
            continue
        if st == "review":
            (train_band if sa == "train" else test_band).append(r)
        elif st == "match" and sa == "train":
            m_out += 1
            tp_out += label(a, b)

    train_records = sorted(n for n, s in side.items() if s == "train")
    comp = components(train_records, edges)

    y_all = np.array([label(r["record_a"], r["record_b"]) for r in train_band])
    N_pos_all, N_neg_all = int(y_all.sum()), int(len(y_all) - y_all.sum())

    # Nested subset for the learning curve.
    idx = np.arange(len(train_band))
    pop_w = {1: 1.0, 0: 1.0}
    if args.n is not None:
        rng = random.Random(args.sample_seed)
        pos = [i for i in idx if y_all[i] == 1]
        neg = [i for i in idx if y_all[i] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_true = min(len(pos), round(args.true_share * args.n))
        n_false = min(len(neg), args.n - n_true)
        idx = np.array(sorted(pos[:n_true] + neg[:n_false]))
        pop_w = {1: N_pos_all / max(n_true, 1), 0: N_neg_all / max(n_false, 1)}
    sub = [train_band[i] for i in idx]
    y = y_all[idx]
    X = np.array([fb.features(r) for r in sub], dtype=float)

    # Folds: components dealt to the fold with the fewest records (seed-0 order).
    def make_folds(k: int) -> np.ndarray:
        comps = sorted({comp[n] for n in train_records})
        random.Random(SEED).shuffle(comps)
        size = Counter(comp[n] for n in train_records)
        load = [0] * k
        fold_of = {}
        for c in comps:
            f = min(range(k), key=lambda j: (load[j], j))
            fold_of[c] = f
            load[f] += size[c]
        fa = np.array([fold_of[comp[r["record_a"]]] for r in sub])
        fbb = np.array([fold_of[comp[r["record_b"]]] for r in sub])
        return np.where(fa == fbb, fa, -1), fa, fbb

    k = args.k
    while True:
        fold, fa, fbb = make_folds(k)
        if all(int(y[fold == f].sum()) >= 2 for f in range(k)) or k <= 2:
            break
        k -= 1

    configs = LGBM_GRID if args.model == "lgbm" else [{}]
    Xm = X if args.model == "lgbm" else lr_matrix(X)
    oof_mask = fold >= 0
    cov_w = {c: (int((y == c).sum()) / max(int(((y == c) & oof_mask).sum()), 1)) for c in (0, 1)}
    w_oof_all = np.array([cov_w[int(c)] * pop_w[int(c)] for c in y])
    from sklearn.metrics import average_precision_score

    grid_results = []
    best = None
    for cfg in configs:
        oof = np.full(len(y), np.nan)
        for f in range(k):
            tr = (fa != f) & (fbb != f)
            te = fold == f
            mdl = make_model(args.model, cfg, int(y[tr].sum()), int((1 - y[tr]).sum()))
            mdl.fit(Xm[tr], y[tr])
            oof[te] = mdl.predict_proba(Xm[te])[:, 1]
        ap_w = float(average_precision_score(y[oof_mask], oof[oof_mask], sample_weight=w_oof_all[oof_mask]))
        grid_results.append((cfg, ap_w))
        if best is None or ap_w > best[1]:
            best = (cfg, ap_w, oof.copy())
    cfg, best_ap, oof = best

    # Calibration on OOF.
    from sklearn.isotonic import IsotonicRegression
    po, yo, wo = oof[oof_mask], y[oof_mask], w_oof_all[oof_mask]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(po, yo, sample_weight=wo)
    pc = iso.predict(po)

    # Cutoff rule.
    tvals, ctp, cpred = weighted_f1_curve(pc, yo, wo)
    pos_w = float((wo * yo).sum())
    prec_band = np.where(cpred > 0, ctp / np.maximum(cpred, 1e-12), 0.0)
    rec_band = ctp / pos_w if pos_w else np.zeros_like(ctp)
    f1 = np.where(prec_band + rec_band > 0, 2 * prec_band * rec_band / np.maximum(prec_band + rec_band, 1e-12), 0.0)
    overall_p = (tp_out + ctp) / (m_out + cpred)
    ok = overall_p >= PRECISION_FLOOR
    if ok.any():
        cand = np.where(ok)[0]
        best_f1 = f1[cand].max()
        i_star = cand[np.isclose(f1[cand], best_f1, rtol=0, atol=1e-12)].min()  # tvals descending: min index = largest t
        t_star = float(tvals[i_star])
        rule_note = "max OOF band F1 with estimated overall precision >= 0.98"
        f1_star, p_star, r_star, op_star = float(f1[i_star]), float(prec_band[i_star]), float(rec_band[i_star]), float(overall_p[i_star])
    else:
        t_star = 1.0 + 1e-9
        rule_note = "no threshold met the precision floor: every band pair is 'different'"
        f1_star = p_star = r_star = 0.0
        op_star = tp_out / m_out if m_out else 0.0

    # Final model on all training pairs.
    import importlib.metadata as im
    final = make_model(args.model, cfg, int(y.sum()), int((1 - y).sum()))
    final.fit(Xm, y)
    lib = f"lightgbm {im.version('lightgbm')}" if args.model == "lgbm" else f"scikit-learn {im.version('scikit-learn')}"
    model_str = f"{args.version} ({'LightGBM' if args.model == 'lgbm' else 'logistic regression'}, {lib})"

    Xt = np.array([fb.features(r) for r in test_band], dtype=float)
    pt = iso.predict(final.predict_proba(Xt if args.model == "lgbm" else lr_matrix(Xt))[:, 1])
    with (out / "decisions.csv").open("w", newline="", encoding="utf-8") as g:
        wcsv = csv.writer(g)
        wcsv.writerow(["record_a", "record_b", "decision", "confidence", "model"])
        for r, p in zip(test_band, pt):
            wcsv.writerow([r["record_a"], r["record_b"], "same" if p >= t_star else "different",
                           f"{p:.6f}", model_str])

    if args.model == "lgbm":
        final.booster_.save_model(str(out / "model.txt"))
    else:
        scaler, lrm = final.named_steps["standardscaler"], final.named_steps["logisticregression"]
        (out / "model.json").write_text(json.dumps({
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": lrm.coef_[0].tolist(), "intercept": float(lrm.intercept_[0])}, indent=1), encoding="utf-8")
    (out / "calibrator.json").write_text(json.dumps({
        "type": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}), encoding="utf-8")
    (out / "features.json").write_text(json.dumps({
        "model": model_str, "kind": args.model, "feature_names": FEATURE_NAMES,
        "lr_input": "features with NaN set to 0, plus one missing-indicator per feature" if args.model == "lr" else None,
        "config": cfg, "fixed_params": LGBM_FIXED if args.model == "lgbm" else
        {"C": 1.0, "class_weight": "balanced", "standardized": True},
        "cutoff": t_star, "seed": SEED, "k_folds": k, "n": args.n, "true_share": args.true_share if args.n else None,
        "sample_seed": args.sample_seed if args.n else None,
    }, indent=1), encoding="utf-8")
    rel = reliability(pc, yo, wo)
    (out / "oof_reliability.json").write_text(json.dumps(rel, indent=1), encoding="utf-8")

    lines = [
        f"Option C cv_report: {args.version}",
        f"Model: {model_str}",
        f"Run: {run.resolve()}",
        "",
        f"Train band pairs: {len(train_band)} ({N_pos_all} true, {N_neg_all} false)",
        f"Training set: {len(sub)} pairs ({int(y.sum())} true, {int(len(y) - y.sum())} false)"
        + (f"; nested subset n={args.n} (sample seed {args.sample_seed}), true share {args.true_share:.0%}; population weights "
           f"true {pop_w[1]:.3f}, false {pop_w[0]:.3f}" if args.n else "; all train band pairs"),
        f"Test band pairs scored: {len(test_band)} (no test labels read)",
        "",
        f"Grouped CV: {k} folds over {len(set(comp.values()))} train-side split components (seed {SEED}, balanced by records)",
        "  fold  eval pairs  eval true  train pairs  train true",
    ]
    for f in range(k):
        te, tr = fold == f, (fa != f) & (fbb != f)
        lines.append(f"  {f:>4}  {int(te.sum()):>10}  {int(y[te].sum()):>9}  {int(tr.sum()):>11}  {int(y[tr].sum()):>10}")
    lines += [
        f"  OOF coverage: {int(oof_mask.sum())} of {len(y)} pairs ({oof_mask.mean():.1%}); true "
        f"{int((oof_mask & (y == 1)).sum())} of {int(y.sum())}, false {int((oof_mask & (y == 0)).sum())} of {int((y == 0).sum())}",
        f"  OOF weights per pair: true {cov_w[1] * pop_w[1]:.4f}, false {cov_w[0] * pop_w[0]:.4f} "
        "(coverage x population)",
        "",
        "Model selection (weighted average precision on OOF)",
    ]
    for c, a in grid_results:
        lines.append(f"  {json.dumps(c) if c else '(logistic regression, no grid)'}  AP {a:.4f}"
                     + ("   <- chosen" if c == cfg else ""))
    lines += [
        "",
        "Calibration: isotonic regression on OOF predictions (weighted)",
        f"  OOF ECE (10 equal-width bins, weighted): {ece(pc, yo, wo):.4f}",
        "  bin          weighted n      n   mean p   frac true",
    ]
    for r in rel:
        if r["n"]:
            lines.append(f"  {r['bin']:<12} {r['weight']:>10.1f} {r['n']:>6}   {r['mean_p']:.4f}   {r['frac_true']:.4f}")
    lines += [
        "",
        "Cutoff rule",
        "  candidates t: the distinct calibrated OOF probabilities; predicted same when p >= t",
        "  band F1(t): weighted OOF band F1",
        f"  P(t) = (TP_out + TPw(t)) / (M_out + PREDw(t)), train group: M_out = {m_out} Verity matches "
        f"outside the band, TP_out = {tp_out}",
        f"  t* = argmax band F1(t) subject to P(t) >= {PRECISION_FLOOR}; ties to the larger t; none -> all different",
        f"  result: t* = {t_star:.6f} ({rule_note})",
        f"  at t*: OOF band F1 {f1_star:.4f}, precision {p_star:.4f}, recall {r_star:.4f}; "
        f"estimated overall train precision {op_star:.4f}",
        "",
        f"Test band: {int((pt >= t_star).sum())} of {len(test_band)} predicted same",
        f"Wall time: {time.perf_counter() - t_start:.1f}s",
    ]
    (out / "cv_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

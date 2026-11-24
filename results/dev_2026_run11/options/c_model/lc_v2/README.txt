Option C learning curve v2 (run dev_2026_run11)
================================================

What this is
------------
LightGBM (the c_v1_lgbm recipe) trained on subsets of the TRAIN band pairs of run
dev_2026_run11 and scored on the full test band, for 5 sampling seeds (0 to 4) at 5 sizes
(100, 250, 500, 1,000, 2,000 labelled pairs; 25% true labels, at most all 418 true train band
pairs). Subsets are nested within each sampling seed. Folds, model seed and the LightGBM grid
stay at seed 0. The last point is the all-train-pairs model (c_v1_lgbm), one model.

The test band was scored once per seed and size for reporting only. No choice was made from
these numbers.

Files
-----
  learning_curve.csv          per size: mean, min, max of band F1 and band average precision,
                              mean overall recall, lowest overall precision
  learning_curve_points.csv   every point (seed, size, labels, cutoff, band F1 with 95% CI,
                              band average precision, overall precision and recall)
  learning_curve.png          mean band F1 with min-max range, mean band AP, reference line for
                              option A; labels like "500 (125 true)"; the caption states the
                              seeds and the 25% true share
  all_score_report.txt        score report of the all-train-pairs model
  s<seed>_n<size>/            one folder per point: score_report.txt and cv_report.txt are
                              committed; features.json and oof_reliability.json are small
                              and not committed

Not committed (about 100 MB; listed in .gitignore): decisions.csv, model.txt and
calibrator.json in each s<seed>_n<size>/ folder.

Regenerating the uncommitted decisions and model files
------------------------------------------------------
From the repo root, with the run's verity_data/ (scored_pairs.csv and its sidecar, from the
run11 recipe in results/dev_2026_run11/README.txt) and split_records.csv in place, and with
options/c_model/c_v1_lgbm/ present (lc_v2 reads it for the all-train-pairs point):

  export PYTHONHASHSEED=0 PYTHONUTF8=1
  python options/c_learning_curve.py results/dev_2026_run11 --seeds 0 1 2 3 4 --subdir lc_v2

This retrains all 25 subset models (about 3 minutes per seed on the development laptop) and
rewrites every file in this folder. Training is deterministic: a retrain of c_v1_lgbm gave
byte-identical model.txt, calibrator.json and decisions (the model column carries the version
name). Sampling seed 0 reproduces the v1 learning curve (options/c_model/lc_n*).

Summary (band F1 = same-entity as the positive class, on the 44,882 test band pairs)
-------------------------------------------------------------------------------------
  labels (true)       seeds   F1 mean   min     max     AP mean   min     max
     100 ( 25 true)     5     0.5097   0.2234  0.8525  0.3745   0.1146  0.7687
     250 ( 62 true)     5     0.7648   0.5255  0.9427  0.6426   0.3541  0.8974
     500 (125 true)     5     0.7771   0.5504  0.9103  0.6532   0.3770  0.8424
   1,000 (250 true)     5     0.8376   0.6093  0.9569  0.7445   0.4381  0.9280
   2,000 (418 true)     5     0.8999   0.8507  0.9489  0.8256   0.7417  0.9123
  63,665 (418 true)     1     0.9653                    0.9799

All 25 subset models chose the cutoff t* = 1.0 (with so few out-of-fold pairs the models
separate perfectly in cross-validation); the all-train-pairs model chose 0.8014.

Timings in any timings.json under options/c_model were recorded on an uncontrolled laptop
state and are not for quoting.

#!/usr/bin/env bash
# Verity timing repeats on the seed-2026 development data (exact neighbor search): 3 repeats with all threads, then 3
# with OMP_NUM_THREADS=1, back to back, each in a fresh run folder results/dev_2026_timing_<setting>_r<n>. After each run
# its numbers are saved and its large files deleted (eval/timing_repeats.py). Run it from the repo root on an idle machine:
#
#   bash eval/run_timing_repeats.sh
#
# Every run's match_candidates.csv is compared byte for byte with run 11's, and so is each generated input csv.
set -u
export PYTHONHASHSEED=0 PYTHONUTF8=1
PY=.venv/Scripts/python.exe
REF=results/dev_2026_run11
for setting in allthreads 1thread; do
  for n in 1 2 3; do
    F=results/dev_2026_timing_${setting}_r${n}
    if [ -e "$F" ]; then echo "$F already exists; refusing"; exit 1; fi
    mkdir -p "$F"
    if [ "$setting" = "1thread" ]; then export OMP_NUM_THREADS=1; else unset OMP_NUM_THREADS; fi
    echo "=== $F  $(date '+%Y-%m-%d %H:%M:%S')  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
    $PY data/generate_data_org.py --seed 2026 --output "$F/orgs_raw.csv" > "$F/generate.log" 2>&1
    if cmp -s "$F/orgs_raw.csv" "$REF/orgs_raw.csv"; then IN=identical; else IN=DIFFERENT; fi
    VERITY_DATA_DIR="$F/verity_data" $PY verity/src/main.py --nn-mode exact --input "$PWD/$F/orgs_raw.csv" > "$F/verity_run.log" 2>&1
    echo "verity exit code $?"
    if cmp -s "$F/verity_data/match_candidates.csv" "$REF/verity_data/match_candidates.csv"; then MC=identical; else MC=DIFFERENT; fi
    echo "match_candidates.csv: $MC; input csv: $IN" > "$F/match_candidates_check.txt"
    echo "vs run 11: $(cat "$F/match_candidates_check.txt")"
    $PY eval/timing_repeats.py save-and-clean "$F"
  done
done
$PY eval/timing_repeats.py summarize
echo "=== done $(date '+%Y-%m-%d %H:%M:%S')"

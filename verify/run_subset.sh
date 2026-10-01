#!/usr/bin/env bash
# Quick check (minutes, not hours): rerun the main Eindhoven configuration on N test tracks and
# compare every per-event row, plus every calibration-fitted quantity, with the paper's final run.
# --test-subset scores a seeded sample of test tracks; calibration and model fits are unchanged.
#   verify/run_subset.sh [N=5] [path/to/reference/records_seed0_n1200_vw1_uncapped.csv]
set -euo pipefail
cd "$(dirname "$0")/../experiments"
export PYTHONPATH="$(cd .. && pwd)"
: "${TPP_DIR:?set TPP_DIR (third_party/setup_trajectron.sh)}"
PY=${PYTHON:-python}
N=${1:-5}
REF=${2:-../reference/eindhoven_floorA/subset_records.csv}
OUT=$(mktemp -d)
A=../artifacts
"$PY" exp10_information_react.py --seed 0 --jobs 8 --tests exact --no-still-cap --lead1-levels 4.2 \
  --entropy-margin-levels 4.2 --distortions 0.1 0.3 --no-tpp-gaussian --tpp-dir "$TPP_DIR" \
  --tracks 1200 --gru-dir $A/eindhoven/gru_10hz --gru-at5-dir $A/eindhoven/gru_10hz_at5 \
  --tpp-model $A/eindhoven/trajectron_run1 --imm-params $A/eindhoven/imm_fit.json \
  --noise-floor A --levels 3 4.2 6 --test-subset "$N" --out "$OUT" > "$OUT/run.log" 2>&1
"$PY" ../verify/compare.py "$OUT" "$REF" ../reference/eindhoven_floorA/summary_seed0_n1200_vw1_uncapped.json

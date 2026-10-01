#!/usr/bin/env bash
# Reproduce every number and figure in the paper into outputs/final/.
# About 2.5 h on an Apple M1 Pro with 8 workers. Requires get_data.sh and TPP_DIR (third_party/setup_trajectron.sh).
set -euo pipefail
cd "$(dirname "$0")/experiments"
export PYTHONPATH="$(cd .. && pwd)"
: "${TPP_DIR:?set TPP_DIR to the patched Trajectron++ clone (third_party/setup_trajectron.sh)}"
PY=${PYTHON:-python}
OUT=../outputs/final
A=../artifacts

COMMON=(--seed 0 --jobs 8 --tests exact --no-still-cap --lead1-levels 4.2 --entropy-margin-levels 4.2
        --distortions 0.1 0.3 --no-tpp-gaussian --tpp-dir "$TPP_DIR")
EIND=(--tracks 1200 --gru-dir $A/eindhoven/gru_10hz --gru-at5-dir $A/eindhoven/gru_10hz_at5
      --tpp-model $A/eindhoven/trajectron_run1 --imm-params $A/eindhoven/imm_fit.json)
DITL=(--dataset day-in-the-life --tracks 570 --gru-dir $A/ditl/gru_10hz --gru-at5-dir $A/ditl/gru_10hz_at5
      --tpp-model $A/ditl/trajectron_run1 --tpp-ckpt 34 --imm-params $A/ditl/imm_fit.json)

run() { mkdir -p "$1"; echo "[$(date +%H:%M)] $1"; "$PY" exp10_information_react.py "${@:2}" --out "$1" > "$1/run.log" 2>&1; }

# 1. REACT runs (Eindhoven: main + no-floor ablation; Day in the Life: 1 cm floor + 2 cm sensitivity)
run $OUT/eindhoven_floorA  "${COMMON[@]}" "${EIND[@]}" --noise-floor A   --levels 3 4.2 6
run $OUT/eindhoven_nofloor "${COMMON[@]}" "${EIND[@]}" --noise-floor none --levels 4.2
run $OUT/ditl_floor1cm     "${COMMON[@]}" "${DITL[@]}" --noise-floor 1.0 --levels 3 4.2 6
run $OUT/ditl_floor2cm     "${COMMON[@]}" "${DITL[@]}" --noise-floor 2.0 --levels 4.2

E=$(ls $OUT/eindhoven_floorA/summary_*.json); ER=$(ls $OUT/eindhoven_floorA/records_*.csv)
D=$(ls $OUT/ditl_floor1cm/summary_*.json);    DR=$(ls $OUT/ditl_floor1cm/records_*.csv)

# 2. Classical accuracy (ADE, FDE, log-likelihood) on the same test frames
"$PY" eval_accuracy.py "$E" $OUT/eindhoven_floorA/accuracy.json
"$PY" eval_accuracy.py "$D" $OUT/ditl_floor1cm/accuracy.json

# 3. Covariance-scale sweep (Fig 9) and share of frames behind climatology
"$PY" exp11_react_vs_calibration.py "$E" --lead 5  --info 4.2 --jobs 8
"$PY" exp11_react_vs_calibration.py "$E" --lead 10 --info 4.2 --jobs 8
"$PY" realized_behind.py "$E" "$ER" $OUT/eindhoven_floorA/realized_behind.json
"$PY" realized_behind.py "$D" "$DR" $OUT/ditl_floor1cm/realized_behind.json

# 4. Figures
"$PY" paper_figures.py --paper
echo "done: compare with reference/ (see README)"

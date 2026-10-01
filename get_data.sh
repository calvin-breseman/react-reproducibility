#!/usr/bin/env bash
# Fetch the two public datasets and build the Eindhoven CSV the runs read.
#   Eindhoven Centraal (Zenodo 10.5281/zenodo.13784588, CC-BY-4.0): day-2 shard -> data/eindhoven_20000tracks.csv
#   Day in the Life (huggingface.co/datasets/standard-cognition/day-in-the-life): canonical_tracks.parquet
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
PY=${PYTHON:-python}

"$PY" -c "from reactmetric.datasets.eindhoven import download_eindhoven; print(download_eindhoven(days='day02'))"
"$PY" data/make_eindhoven_csv.py          # defaults: day02, 20000 tracks, min length 40, seed 0, stratified by hour

DITL=~/.cache/reactmetric/day_in_the_life
mkdir -p "$DITL"
"$PY" -c "from huggingface_hub import hf_hub_download; hf_hub_download('standard-cognition/day-in-the-life', 'canonical_tracks.parquet', repo_type='dataset', local_dir='$DITL')"

shasum -a 256 -c data/inputs.sha256

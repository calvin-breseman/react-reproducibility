# REACT: reproducibility package

Code and inputs that reproduce every figure and number in *REACT: Measuring Regime Adaptation in
Trajectories* (Breseman and Woollard, TGF 2026) from the public data and the shipped model
checkpoints.

`reactmetric/` is a frozen subset of the reactmetric library: only the modules this pipeline
imports, unchanged from the code that produced the paper. The released library may differ; use
this copy to reproduce the paper.

## Contents

| Path | What it is |
| --- | --- |
| `experiments/exp10_information_react.py` | Main driver: event detection, REACT onset/recovery/relapse per event, model and forecast |
| `experiments/react_information.py`, `bayes_changepoints.py`, `trajectron_forecaster.py`, `day_in_the_life.py` | Its components: posterior tests, changepoint events, Trajectron++ adapter, Day in the Life loader |
| `experiments/eval_accuracy.py` | ADE, FDE and log-likelihood on the same test frames |
| `experiments/exp11_react_vs_calibration.py` | Covariance-scale sweep (Fig 9) |
| `experiments/realized_behind.py` | Share of frames each forecaster scores below climatology |
| `experiments/paper_figures.py` | All figures (`--paper` writes the 11 manuscript figures) |
| `artifacts/` | Trained GRU, GRU@5 and Trajectron++ checkpoints and fitted IMM parameters, per dataset |
| `data/sim/` | Known-truth simulation records (Figs 6 and 10) |
| `data/make_eindhoven_csv.py` | Builds the 20,000-track Eindhoven sample from the Zenodo shard |
| `third_party/` | Trajectron++ pin (commit `1031c7bd`) and our patch |
| `reference/` | The paper's outputs: run summaries, accuracy, q-sweeps, figures, and SHA-256 of every output file |
| `verify/` | Subset rerun and comparator |

## Reproduce

Tested with CPython 3.13.2 on an Apple M1 Pro (macOS). `requirements.txt` pins the exact versions.

```bash
pip install -r requirements.txt
./get_data.sh                                   # Zenodo + Hugging Face downloads, builds data/eindhoven_20000tracks.csv
third_party/setup_trajectron.sh                 # clone + patch Trajectron++
export TPP_DIR="$PWD/third_party/Trajectron-plus-plus"
shasum -a 256 -c artifacts.sha256 data/inputs.sha256
```

**Quick check (about 5 min):** rerun the main Eindhoven configuration on a seeded subset of
test tracks and compare each per-event row with the paper's run.

```bash
verify/run_subset.sh 5
```

`--test-subset` leaves calibration and every fit unchanged, so rows for those tracks must match
the full run. The comparator requires equal strings, integers and booleans, floats within
relative 1e-9, and identical calibration fits. `reference/eindhoven_floorA/subset_records.csv`
holds the reference rows for subsets of up to 20 tracks.

**Full run (about 2.5 h):** `./run_all.sh` writes `outputs/final/`. Compare it with
`reference/final_outputs.sha256` and the JSON files in `reference/`. The full per-event records
(1.1 GB) are not in this repository.

## Data

- Eindhoven Centraal pedestrian trajectories, Pouw et al., Zenodo
  [10.5281/zenodo.13784588](https://zenodo.org/records/13784588), CC-BY-4.0. The day-2 shard is
  sampled to 20,000 tracks (seed 0, stratified by hour); the result is checked against
  `data/inputs.sha256`.
- Day in the Life, Standard AI,
  [huggingface.co/datasets/standard-cognition/day-in-the-life](https://huggingface.co/datasets/standard-cognition/day-in-the-life),
  `canonical_tracks.parquet`.

All splits are derived with seed 0: Eindhoven 1,200 tracks (360 calibration / 840 test), Day in
the Life 570 pieces.

## Licence

Apache-2.0 (`LICENSE`). Trajectron++ is MIT-licensed and is fetched from its upstream repository.

## Verification status

- `verify/run_subset.sh 5` with this repository: PASS (1,278 per-event rows and all calibration
  fits identical to the paper's run), 5 min on an Apple M1 Pro.
- `eval_accuracy.py` on the Day in the Life summary reproduces the paper's accuracy table exactly;
  it also reports the moment-matched `Trajectron` row, which the paper does not use.
- The 20,000-track Eindhoven CSV rebuilt by `get_data.sh` is byte-identical to the one used.
- The full `run_all.sh` has not been rerun in this repository.

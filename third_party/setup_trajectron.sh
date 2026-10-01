#!/usr/bin/env bash
# Clone Trajectron++ at the commit used for the paper and apply our inference-side changes.
# Afterwards: export TPP_DIR="$(pwd)/third_party/Trajectron-plus-plus"
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=1031c7bd1a444273af378c1ec1dcca907ba59830
[ -d Trajectron-plus-plus ] || git clone https://github.com/StanfordASL/Trajectron-plus-plus.git
git -C Trajectron-plus-plus checkout -q "$COMMIT"
git -C Trajectron-plus-plus diff --quiet && git -C Trajectron-plus-plus apply ../trajectron_pp.patch   # skip if already applied
echo "TPP_DIR=$(pwd)/Trajectron-plus-plus"

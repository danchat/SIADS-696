#!/bin/bash
set -euo pipefail
DEST="/nfs/turbo/si-acastel/mimic-project/data_raw/mimiciv_3_1"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading into: $DEST"
date

wget -r -N -c -np \
  --wait=1 --random-wait --tries=10 --timeout=30 \
  https://physionet.org/files/mimiciv/3.1/

echo "Done."
date


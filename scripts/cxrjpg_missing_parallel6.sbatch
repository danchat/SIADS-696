#!/bin/bash

#SBATCH --job-name=cxrjpg_missing_p6

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=6

#SBATCH --mem=8G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_missing_p6_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_missing_p6_%j.err



set -euo pipefail



OUTDIR="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0"

CHUNKDIR="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0/missing_chunks"

PARALLEL=6



mkdir -p "$OUTDIR"

cd "$OUTDIR"



echo "Downloading missing JPGs using $PARALLEL workers"

echo "OUTDIR=$OUTDIR"

echo "CHUNKDIR=$CHUNKDIR"

echo "Total missing URLs (should be 37787): $(cat "$OUTDIR/missing_urls.txt" | wc -l)"



# Sanity check: chunks exist

ls -l "$CHUNKDIR"/chunk_* | head



pids=()

i=0

for chunk in "$CHUNKDIR"/chunk_*; do

  i=$((i+1))

  echo "Worker $i using $(basename "$chunk") (lines: $(wc -l < "$chunk"))"



  # Note: no -N here; these URLs are filtered to missing files.

  # Keep -c to resume partial downloads safely.

  wget --netrc -c --tries=10 --timeout=30 --wait=1 --random-wait -i "$chunk" \

    > "/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_missing_${SLURM_JOB_ID}_worker_${i}.out" \

    2> "/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_missing_${SLURM_JOB_ID}_worker_${i}.err" &



  pids+=($!)



  if [ "$i" -ge "$PARALLEL" ]; then

    break

  fi

done



fail=0

for pid in "${pids[@]}"; do

  if ! wait "$pid"; then

    fail=1

  fi

done



echo "Workers finished. Quick error scan:"

grep -Hn "ERROR\|failed\|403\|429\|503\|timed out\|unable to" \

  /nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_missing_${SLURM_JOB_ID}_worker_*.err || true



if [ "$fail" -ne 0 ]; then

  echo "One or more workers failed; see worker err logs in /nfs/turbo/si-acastel/mimic-project/logs/" >&2

  exit 1

fi



echo "Completed successfully."



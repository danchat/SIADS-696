#!/bin/bash

#SBATCH --job-name=cxrjpg_miss_p6

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=6

#SBATCH --mem=8G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_miss_p6_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_miss_p6_%j.err



set -euo pipefail



OUTDIR="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0"

CHUNKDIR="${OUTDIR}/missing_chunks"

LOGDIR="/nfs/turbo/si-acastel/mimic-project/logs"



# Robust job id (never empty)

JOBID="${SLURM_JOB_ID:-manual_$$}"

WORKLOG="${LOGDIR}/cxrjpg_miss_${JOBID}"

mkdir -p "$WORKLOG"



cd "$OUTDIR"



echo "Truly-parallel missing download"

echo "JOBID=$JOBID"

echo "OUTDIR=$OUTDIR"

echo "CHUNKDIR=$CHUNKDIR"

echo "WORKLOG=$WORKLOG"

echo "Chunks:"

ls -l "$CHUNKDIR"/chunk_*



# Start 6 workers deterministically: chunk_00 .. chunk_05

for i in 1 2 3 4 5 6; do

  chunk=$(printf "%s/chunk_%02d" "$CHUNKDIR" $((i-1)))

  out="${WORKLOG}/worker_${i}.out"

  err="${WORKLOG}/worker_${i}.err"



  echo "Loop i=$i chunk=$chunk" >> "${WORKLOG}/launcher.log"



  if [ ! -s "$chunk" ]; then

    echo "ERROR: chunk missing or empty: $chunk" >&2

    exit 1

  fi



  : > "$out"

  : > "$err"



  echo "Worker $i starting $(basename "$chunk") (lines: $(wc -l < "$chunk"))"



  wget --netrc -c --tries=10 --timeout=30 --wait=1 --random-wait -i "$chunk" \

    >> "$out" 2>> "$err" &

done



wait



echo "All workers finished. Quick error scan:"

grep -Hn "ERROR\|failed\|403\|429\|503\|timed out\|unable to" \

  "${WORKLOG}"/worker_*.err || true



echo "Completed."


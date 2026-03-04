#!/bin/bash

#SBATCH --job-name=cxrjpg_50k_p6

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=6

#SBATCH --mem=8G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_p6_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_p6_%j.err



set -euo pipefail



LIST="/nfs/turbo/si-acastel/mimic-project/manifests/mimic-cxr-jpg_2_1_0/files_50k_urls.txt"

OUTDIR="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0"

PARALLEL=6



mkdir -p "$OUTDIR"

cd "$OUTDIR"



# Use SLURM-provided temp space if available; otherwise /tmp

WORKDIR="${SLURM_TMPDIR:-/tmp}/cxrjpg_50k_${SLURM_JOB_ID}"

mkdir -p "$WORKDIR"



# Clean list (remove blank lines; strip CR if any)

CLEAN_LIST="${WORKDIR}/urls_clean.txt"

sed '/^[[:space:]]*$/d' "$LIST" | tr -d '\r' > "$CLEAN_LIST"



LINES=$(wc -l < "$CLEAN_LIST")

if [ "$LINES" -eq 0 ]; then

  echo "ERROR: Cleaned URL list is empty: $CLEAN_LIST" >&2

  exit 1

fi



# Portable split: compute lines per chunk (ceil division)

CHUNK=$(( (LINES + PARALLEL - 1) / PARALLEL ))

echo "Starting ${PARALLEL} parallel wget workers..."

echo "WORKDIR=${WORKDIR}"

echo "OUTDIR=${OUTDIR}"

echo "Total URLs: $LINES"

echo "Lines per chunk: $CHUNK"



# Create chunk_00, chunk_01, ..., chunk_05 (and possibly one extra depending on rounding)

split -l "$CHUNK" -d -a 2 "$CLEAN_LIST" "${WORKDIR}/chunk_"



# Launch up to PARALLEL independent wget workers

pids=()

i=0

for chunk in "${WORKDIR}"/chunk_*; do

  i=$((i+1))

  echo "Worker ${i} downloading from $(basename "$chunk")"

  echo "  chunk path: $chunk  (lines: $(wc -l < "$chunk"))"



  # IMPORTANT: keep this wget on ONE line to avoid any line-continuation weirdness

  wget --netrc -N -c --tries=10 --timeout=30 --wait=1 --random-wait -i "$chunk" \

    > "${WORKDIR}/worker_${i}.out" 2> "${WORKDIR}/worker_${i}.err" &



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



echo "All workers finished. Error summary (if any):"

grep -Hn "ERROR\|failed\|Missing\|403\|429\|503\|timed out\|unable to" "${WORKDIR}"/worker_*.err || true



# Copy worker logs to Turbo so you can inspect them after the job ends

LOGCOPY="/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_p6_${SLURM_JOB_ID}_workers"

mkdir -p "$LOGCOPY"

cp -f "${WORKDIR}"/worker_*.out "${WORKDIR}"/worker_*.err "$LOGCOPY"/ 2>/dev/null || true



if [ "$fail" -ne 0 ]; then

  echo "One or more workers exited with an error. See ${LOGCOPY}/worker_*.err" >&2

  exit 1

fi



echo "Completed successfully."



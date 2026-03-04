#!/bin/bash

#SBATCH --job-name=cxrjpg_50k_p

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=8G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_p_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_p_%j.err



set -euo pipefail



LIST="/nfs/turbo/si-acastel/mimic-project/manifests/mimic-cxr-jpg_2_1_0/files_50k_urls.txt"

OUTDIR="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0"



# Number of simultaneous downloads inside ONE job.

# Start with 6. If logs are clean and stable, try 8.

PARALLEL=6



mkdir -p "$OUTDIR"

cd "$OUTDIR"



# Parallelize downloads:

# - remove blank/whitespace-only lines

# - strip CR if present (harmless even if not)

# - run up to PARALLEL wget processes concurrently

# - IMPORTANT: -r prevents running wget with no URL (fixes "wget: missing URL")

sed '/^[[:space:]]*$/d' "$LIST" | tr -d '\r' | \

  xargs -r -n 1 -P "$PARALLEL" \

  wget --netrc -N -c --tries=10 --timeout=30 --wait=1 --random-wait



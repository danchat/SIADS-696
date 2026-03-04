#!/bin/bash

#SBATCH --job-name=cxrjpg_50k

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=8G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/cxrjpg_50k_%j.err



set -euo pipefail



LIST="/nfs/turbo/si-acastel/mimic-project/manifests/mimic-cxr-jpg_2_1_0/files_50k_urls.txt"

OUTDIR="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0"



mkdir -p "$OUTDIR"

cd "$OUTDIR"



wget --netrc -N -c --tries=10 --timeout=30 --wait=1 --random-wait -i "$LIST"



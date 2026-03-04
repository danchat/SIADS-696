#!/bin/bash

#SBATCH --job-name=gcs_cxr

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=16G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/gcs_cxr_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/gcs_cxr_%j.err



set -euo pipefail



# Use your local SDK install

export PATH="$HOME/tools/google-cloud-sdk/bin:$PATH"



BILLPROJ="mimic-transfer"

SRC="gs://mimic-cxr-jpg-2.1.0.physionet.org"

DEST="/nfs/turbo/si-acastel/mimic-project/data_raw/mimic-cxr-jpg_2_1_0_gcs"



mkdir -p "$DEST"



echo "JobID=$SLURM_JOB_ID"

echo "Host=$(hostname -f)"

echo "Billing project: $BILLPROJ"

echo "Source: $SRC"

echo "Destination: $DEST"

date



echo "Testing requester-pays access..."

gsutil -u "$BILLPROJ" ls "$SRC" | head -n 20



echo "Starting parallel recursive copy..."

# This pulls EVERYTHING in the bucket (including metadata CSVs + the files/ tree)

gsutil -m -u "$BILLPROJ" cp -r "$SRC"/* "$DEST"/



echo "Done."

date


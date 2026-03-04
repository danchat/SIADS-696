#!/bin/bash

#SBATCH --job-name=ehr_labpanel_v2

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/ehr_labpanel_v2_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/ehr_labpanel_v2_%j.err



set -euo pipefail

module load python/3.10

source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate



python /nfs/turbo/si-acastel/mimic-project/scripts/build_ehr_features_labpanel_v2.py \

  --window_hours 24 \

  --max_itemids 60 \

  --out_name ehr_study_labpanel_v2_24h_meanminmax_exact.parquet

#!/bin/bash
#SBATCH --job-name=ehr_top10_exact
#SBATCH --account=siads696w26_class
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/ehr_top10_exact_%j.out
#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/ehr_top10_exact_%j.err

set -euo pipefail

module load python/3.10
source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate

python /nfs/turbo/si-acastel/mimic-project/scripts/build_ehr_features_top10labs_meanminmax_exact.py \
  --window_hours 24 \
  --chunksize 2000000 \
  --out /nfs/turbo/si-acastel/mimic-project/derived/ehr_study_top10labs_24h_meanminmax_exact.parquet

#!/bin/bash
#SBATCH --job-name=ehr_top10_test
#SBATCH --account=siads696w26_class
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/ehr_top10_test_%j.out
#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/ehr_top10_test_%j.err

set -euo pipefail
module load python/3.10
source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate

python /nfs/turbo/si-acastel/mimic-project/scripts/build_ehr_features_top10labs.py --max_studies 20000

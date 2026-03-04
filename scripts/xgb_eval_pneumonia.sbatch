#!/bin/bash

#SBATCH --job-name=xgb_eval_pnx
#SBATCH --account=siads696w26_class
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/xgb_eval_pna_%j.out
#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/xgb_eval_pna_%j.err

set -euo pipefail

echo "Starting evaluation job at $(date)"
echo "Running on node: $(hostname)"

module load python/3.10

source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate

echo "Python version:"
python --version

echo "Running evaluation script..."

python /nfs/turbo/si-acastel/mimic-project/scripts/evaluate_xgb_pneumonia_best.py

echo "Job completed at $(date)"

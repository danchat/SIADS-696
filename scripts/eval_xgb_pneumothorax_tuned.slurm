#!/bin/bash

#SBATCH --job-name=xgb_pnx_eval

#SBATCH --account=siads696w26_class

#SBATCH --partition=standard

#SBATCH --time=01:00:00

#SBATCH --nodes=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=16G

#SBATCH --output=/nfs/turbo/si-acastel/mimic-project/logs/xgb_pnx_eval_%j.out

#SBATCH --error=/nfs/turbo/si-acastel/mimic-project/logs/xgb_pnx_eval_%j.err



set -euo pipefail

module load python/3.10

source /nfs/turbo/si-acastel/mimic-project/venv/bin/activate



python /nfs/turbo/si-acastel/mimic-project/scripts/eval_xgb_pneumothorax_tuned.py
